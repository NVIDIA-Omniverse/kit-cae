# %%
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
# tutorial-requires: gpu

# %% [markdown]
# # A3. An operator built on VTK
#
# Tutorial 5 wrote an operator in Warp, on the device the dataset lives on.
# This appendix wraps a VTK filter as an OVCAE operator:
# **clip by scalar**, which keeps the part of a volume where a field is above
# a value and returns its surface.
#
# The dataset arrives on the compute device, while VTK works in host memory.
# The operator converts the arrays to a `vtkUnstructuredGrid`, clips it,
# extracts the surface, and returns the polygons on the compute device as
# a Mesh product.
#
# The data is appendix A2's: the Exodus file through the Data Source A2
# wrote, packaged in `exodus_source.py` beside this notebook. The live view
# is `ovcae.stream.LiveView` from tutorial 2. This appendix requires an
# RTX-capable NVIDIA GPU.

# %% [markdown]
# ## Initialize, and register two schemas
#
# Two codeless schemas are registered before the first stage: A2's
# `VtkExternalDatasetAPI`, so the descriptor can be authored, and this
# appendix's `CaeVizVtkClipAPI` in `vtk_clip_schema/`, which carries the
# operator's one setting, `cae:viz:vtk_clip:value`. The clip schema goes
# through `ovcae.register_schema_paths`, as tutorial 5 explained, so usd-core
# and OVStage both know it. The descriptor schema needs only usd-core,
# because only the Data Source reads it.

# %%
import ovrtx

ovrtx.register_schema_paths()

import numpy as np
import ovstage
import warp as wp
from exodus_source import API as DESCRIPTOR_API
from exodus_source import FILE_NAMES, ExodusDataSource
from support import DATA_DIR, OUTPUT_DIR, REPO_ROOT, cuda_gpu_index, tutorial_device

import ovcae
from ovcae.schemas import register_usd_plugin

ovcae.setup()
wp.config.log_level = wp.LOG_WARNING

SCHEMA = "CaeVizVtkClipAPI"
VALUE = "cae:viz:vtk_clip:value"
register_usd_plugin(REPO_ROOT / "tutorials" / "vtk_external_dataset_schema")
ovcae.register_schema_paths(REPO_ROOT / "tutorials" / "vtk_clip_schema")

device = tutorial_device()
gpu_index = cuda_gpu_index(device) or "0"
renderer = ovrtx.Renderer(ovrtx.RendererConfig(active_cuda_gpus=gpu_index))
print("Compute device:", device, " render GPU:", gpu_index)

# %% [markdown]
# ## Write the operator
#
# The declaration is tutorial 5's shape: the operator's own schema, a
# dataset selection in the `source` role, a field selection in the `clip`
# role for the scalar to clip by, and an optional `colors` selection. The
# body has three parts:
#
# 1. **Down to the host.** The dataset is a Warp SimData unstructured grid,
#    whose handle keeps points, cell types, offsets, and connectivity in
#    exactly VTK's layout, so `numpy()` on each and a `vtkCellArray` built
#    from the offsets and connectivity make a `vtkUnstructuredGrid` without
#    reinterpreting anything. Fields go on as point data under their role
#    names.
# 2. **VTK.** `vtkTableBasedClipDataSet` keeps the cells where the clip
#    array is at or above the value, cutting the ones it crosses;
#    `vtkDataSetSurfaceFilter` extracts the resulting volume's surface as
#    polygons with the point data interpolated onto it.
# 3. **Back up.** The polygons' offsets become face vertex counts, their
#    connectivity the face vertex indices, and the interpolated color field
#    a vertex primvar, reduced to its magnitude when it is a vector. The
#    output arrays are created on the compute device for publication.
#
# `numpy()` reads the input arrays into host memory,
# `numpy_to_vtk(..., deep=1)` copies them into VTK, and
# `wp.array(..., device=)` creates the output arrays on the compute device.
# These transfers and conversions add to the cost of the VTK filters.

# %%
import vtk
from vtk.util import numpy_support
from warp_simdata import AssociationType

from ovcae import operators
from ovcae.operators import utils as op_utils


def to_vtk(dataset, fields: dict[str, np.ndarray]) -> vtk.vtkUnstructuredGrid:
    """Copy a Warp SimData unstructured grid and the given host arrays into a vtkUnstructuredGrid."""
    handle = dataset.handle
    grid = vtk.vtkUnstructuredGrid()
    points = vtk.vtkPoints()
    points.SetData(numpy_support.numpy_to_vtk(handle.points.numpy(), deep=1))
    grid.SetPoints(points)
    cells = vtk.vtkCellArray()
    cells.SetData(
        numpy_support.numpy_to_vtkIdTypeArray(handle.cell_offsets.numpy().astype(np.int64), deep=1),
        numpy_support.numpy_to_vtkIdTypeArray(handle.cell_connectivity.numpy().astype(np.int64), deep=1),
    )
    grid.SetCells(
        numpy_support.numpy_to_vtk(handle.cell_types.numpy(), deep=1, array_type=vtk.VTK_UNSIGNED_CHAR), cells
    )
    for name, values in fields.items():
        array = numpy_support.numpy_to_vtk(np.ascontiguousarray(values), deep=1)
        array.SetName(name)
        grid.GetPointData().AddArray(array)
    return grid


@operators.operator(
    required_schemas={SCHEMA, "CaeVizDatasetSelectionAPI:source", "CaeVizFieldSelectionAPI:clip"},
    optional_schemas={"CaeVizFieldSelectionAPI:colors"},
    dataset_components={"source": {"topology", "geometry"}},
)
class VtkClip:
    """Keep the part of a volume where the clip field is at or above a value, and publish its surface."""

    def execute(self, context):
        if context.prim.type_name != "Mesh":
            raise operators.InvalidRecipe("VTK Clip requires a Mesh recipe")
        value = float(context.prim.get_attribute(VALUE).get())
        dataset = op_utils.get_input_dataset(context, role="source")

        fields = {}
        for role in ("clip", "colors"):
            if dataset.has_field(role):
                field = dataset.get_field(role)
                if field.association != AssociationType.NODE:
                    raise operators.InvalidRecipe(f"{role} must be a per-node field")
                fields[role] = field.to_array().numpy()
        if fields["clip"].ndim != 1:
            raise operators.InvalidRecipe("clip must be a scalar field")

        clip = vtk.vtkTableBasedClipDataSet()
        clip.SetInputData(to_vtk(dataset, fields))
        clip.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "clip")
        clip.SetValue(value)
        surface = vtk.vtkDataSetSurfaceFilter()
        surface.SetInputConnection(clip.GetOutputPort())
        surface.Update()
        polygons = surface.GetOutput()
        if polygons.GetNumberOfPolys() == 0:
            return op_utils.empty(f"nothing at or above {value}")

        offsets = numpy_support.vtk_to_numpy(polygons.GetPolys().GetOffsetsArray())
        connectivity = numpy_support.vtk_to_numpy(polygons.GetPolys().GetConnectivityArray())
        primvars = []
        if "colors" in fields:
            colors = numpy_support.vtk_to_numpy(polygons.GetPointData().GetArray("colors"))
            if colors.ndim == 2:  # a vector field colors by its magnitude
                colors = np.linalg.norm(colors, axis=1)
            primvars.append(
                op_utils.primvar("colors", wp.array(colors.astype(np.float32), device=context.device), "vertex")
            )
        return op_utils.mesh(
            points=wp.array(
                numpy_support.vtk_to_numpy(polygons.GetPoints().GetData()), dtype=wp.vec3f, device=context.device
            ),
            face_vertex_counts=wp.array(np.diff(offsets).astype(np.int32), device=context.device),
            face_vertex_indices=wp.array(connectivity.astype(np.int32), device=context.device),
            primvars=primvars,
        )


registry = operators.OperatorRegistry.with_builtins()
registry.register(VtkClip)
print("Registered:", VtkClip.__name__)

# %% [markdown]
# `op_utils.empty` is how an operator reports a legitimate empty result, a
# clip value above the field's maximum here, without failing the recipe.

# %% [markdown]
# ## Author the scene
#
# The descriptor is A2's, and so are the three Faces recipes that show the
# heated disk. The clip recipe is a `Mesh` assembled from the typed
# wrappers as in tutorial 5: the clip schema with its value, the element
# block as the dataset, `Temp` as the clip field, and velocity for coloring.
# At 450 K the kept volume is the hot gas wrapping the disk, and its surface
# shows how fast that gas moves.

# %%
import os

from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdRender

import ovcae.authoring as author
from ovcae.authoring import schemas as S
from ovcae.stream import OrbitCamera

exodus_path = DATA_DIR / "disk_out_ref.ex2"
DATASET = "/World/DiskOut"
VOLUME = "/Root/ElementBlocks/UnnamedblockID1"
DISK = {
    "Top": "/Root/SideSets/UnnamedsetID4",
    "Rim": "/Root/SideSets/UnnamedsetID5",
    "Shaft": "/Root/SideSets/UnnamedsetID6",
}
CLIP = "/World/Recipes/HotGas"
CAMERA = "/World/Camera"
RENDER_PRODUCT = "/Render/Camera"
EYE, TARGET = (26.0, -34.0, 8.0), (0.0, 0.0, 0.0)
TEMPERATURE = (293.0, 913.0)

scene_path = OUTPUT_DIR / "A3_hot_gas.usda"
scene_path.parent.mkdir(parents=True, exist_ok=True)
scene_path.unlink(missing_ok=True)

stage = Usd.Stage.CreateNew(str(scene_path))
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
stage.SetDefaultPrim(stage.DefinePrim("/World", "Xform"))

descriptor = stage.DefinePrim(DATASET)
descriptor.ApplyAPI(DESCRIPTOR_API)
descriptor.CreateAttribute(FILE_NAMES, Sdf.ValueTypeNames.AssetArray).Set(
    [Sdf.AssetPath(os.path.relpath(exodus_path, scene_path.parent))]
)

for name, side_set in DISK.items():
    author.create_faces(
        stage,
        f"/World/Recipes/Disk{name}",
        source=DATASET,
        identifier=side_set,
        colors="Temp",
        colormap="coolwarm",
        color_range=TEMPERATURE,
    )

mesh = UsdGeom.Mesh.Define(stage, CLIP)
mesh.CreatePointsAttr([])
mesh.CreateFaceVertexCountsAttr([])
mesh.CreateFaceVertexIndicesAttr([])
mesh.CreateSubdivisionSchemeAttr("none")
recipe = mesh.GetPrim()
S.OperatorAPI.apply(recipe).enabled = True
S.schema_from_registry(SCHEMA).apply(recipe).value = 450.0
selection = S.DatasetSelectionAPI.apply(recipe, "source")
selection.target = DATASET
selection.identifier = VOLUME
S.FieldSelectionAPI.apply(recipe, "clip").field_names = ["Temp"]
S.FieldSelectionAPI.apply(recipe, "colors").field_names = ["V"]
author.setup_scalar_materials(recipe, colormap="turbo", color_range=(0.0, 22.0))

camera = UsdGeom.Camera.Define(stage, CAMERA)
camera.CreateFocalLengthAttr(20.0)
camera.CreateClippingRangeAttr(Gf.Vec2f(0.1, 500.0))
camera.AddTransformOp().Set(Gf.Matrix4d(*OrbitCamera(EYE, TARGET).matrix().ravel().tolist()))
UsdLux.DomeLight.Define(stage, "/World/Light").CreateIntensityAttr(1000.0)

product = UsdRender.Product.Define(stage, RENDER_PRODUCT)
product.CreateResolutionAttr(Gf.Vec2i(1280, 720))
product.CreateCameraRel().SetTargets([camera.GetPath()])
color_var = UsdRender.Var.Define(stage, f"{RENDER_PRODUCT}/LdrColor")
color_var.CreateSourceNameAttr("LdrColor")
color_var.GetPrim().CreateAttribute("omni:rtx:aov", Sdf.ValueTypeNames.Opaque)
product.CreateOrderedVarsRel().SetTargets([color_var.GetPath()])

stage.Save()
print("Applied schemas:", list(recipe.GetAppliedSchemas()))
print("Saved", scene_path)

# %% [markdown]
# ## Evaluate
#
# As in tutorial 5, the operator registry goes to the Session, here through
# the update loop that creates it, and the population description already
# covers the clip schema. The Data
# Source is A2's over the authored stage. The first update is timed as a
# whole, file read included, so the edit below has a number to compare to.

# %%
import time

from support import read_attribute

from ovcae.app import UpdateLoop

ov_stage = ovstage.Stage("tutorial-A3")
ovstage.population.open_usd_with_desc(
    ov_stage,
    str(scene_path),
    ordinal=1,
    time_code=0.0,
    descs=ovcae.ovstage_population_desc(domains=ovstage.PopulationDomain.ALL),
)
ov_stage.advance_write_floor(1).wait()

data_source = ExodusDataSource(stage)
loop = UpdateLoop(ov_stage, data_source, ordinal=1, device=device, operators=registry)

started = time.perf_counter()
report = loop.update()
elapsed = time.perf_counter() - started
for diagnostic in report.diagnostics:
    print(f"  [{diagnostic.severity}] {diagnostic.code}: {diagnostic.message}")
for result in report.operators:
    print(f"{result.operator_path:<30} {result.status:<10} {result.reason or ''}")
counts = np.asarray(read_attribute(ov_stage, CLIP, "faceVertexCounts", loop.ordinal))
print(f"Hot gas surface: {len(counts)} polygons, first update {elapsed:.2f} s including the file read")

# %% [markdown]
# ## Make it live
#
# `LiveView` from tutorial 2 again, with one slider for the clip value.
# The slider writes the schema attribute on the recipe prim, the render
# thread runs an update, and the operator reruns: down to the host, VTK,
# back up, every time the value changes. Watch the hot gas peel away from
# the disk as the value rises. The disk surfaces come back unchanged
# throughout.

# %%
import ipywidgets as widgets
from IPython.display import HTML, display
from support import stream_host

from ovcae.stream import LiveView

renderer.attach_ovstage(ov_stage)
view = LiveView(
    loop,
    renderer,
    render_product=RENDER_PRODUCT,
    camera=CAMERA,
    eye=EYE,
    target=TARGET,
    gpu_index=gpu_index,
    public_host=stream_host(),
)
view.start()

clip_slider = widgets.FloatSlider(
    value=450.0, min=300.0, max=900.0, step=10.0, description="Clip K", continuous_update=True, readout_format=".0f"
)
clip_slider.observe(lambda change: view.request(CLIP, VALUE, change["new"]), names="value")

display(HTML(view.html()))
display(clip_slider)

# %% [markdown]
# ## What an edit costs
#
# Set the value from code and wait for the update. The clip recipe runs
# and the disk surfaces are reused. The cell prints the elapsed time and
# per-operator status.

# %%
started = time.perf_counter()
clip_slider.value = 600.0
report = view.wait_for_update()
print(f"Update {report.input_ordinal} -> {report.output_ordinal} after {time.perf_counter() - started:.2f} s")
for result in report.operators:
    print(f"{result.operator_path:<30} {result.status:<10} {result.reason or ''}")
counts = np.asarray(read_attribute(ov_stage, CLIP, "faceVertexCounts", report.output_ordinal))
print(f"Hot gas surface at 600 K: {len(counts)} polygons")

# %% [markdown]
# Wait until the video shows the updated scene, then save the latest frame.
# The snapshot does not wait for rendering to finish.

# %%
from support import save_png

png_path = save_png(view.snapshot(), OUTPUT_DIR / "A3_vtk_operator.png")
print("Saved", png_path)
if "get_ipython" in globals():
    from IPython.display import Image

    display(Image(filename=str(png_path)))

# %% [markdown]
# ## Shut down

# %%
view.stop()
loop.close()
data_source.close()
renderer.detach_ovstage()
ov_stage.destroy()
renderer.destroy()
print("Done")

# %% [markdown]
# ## What just happened
#
# A VTK filter became an OVCAE operator with the same declaration and
# registration as a Warp one. The wrapper converts the dataset to VTK's
# host representation and returns the surface as a Mesh product. The
# Session treated it like any operator: it reran on the setting, left the
# other recipes alone, and reused the file through the Data Source.
#
# Any VTK filter that turns a dataset into a surface, lines, or points fits
# this shape. Filters that produce a new *dataset* with fields, such as a
# resample or a gradient, do not: products are geometry, and a recipe that
# targets one sees only its points. That is the boundary of the operator
# contract today.
