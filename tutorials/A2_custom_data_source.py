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
# # A2. A Data Source of your own
#
# OVCAE reads the formats the CAE OpenUSD plug-ins know, and tutorial 4
# showed how to hand it datasets from memory. Between the two sits a common
# case: a file in a format OVCAE does not read, for which you already have a
# reader. This appendix writes a **Data Source** that serves such files, so
# recipes can select them the way they select anything else, and puts a live
# view on the result with controls for a slice and a set of streamlines.
#
# The reader is VTK, and the file is `disk_out_ref.ex2`, the ParaView
# tutorial dataset: gas flowing around a heated, rotating disk, stored as
# Exodus II. It is checked into `data/` beside this notebook; its provenance
# and license are in [`data/README.md`](data/README.md).
#
# This appendix needs an RTX-capable NVIDIA GPU and a browser that can
# reach the GPU host for the interactive part. Without a browser it still
# runs to completion and saves the frames it would have shown.

# %% [markdown]
# ## Initialize, and register the schema
#
# A recipe selects a dataset by targeting a prim. For a file OVCAE cannot
# compose, that prim is a **descriptor**: an otherwise empty prim carrying a
# schema that says which files hold the dataset. The schema here is the
# codeless `VtkExternalDatasetAPI` in `vtk_external_dataset_schema/`, with
# one property, `asset[] vtk:fileNames`. It is registered with usd-core
# before the first stage, as tutorial 5 did for its operator schema. OVStage
# and OVCAE do not need to know it: the Data Source below reads the
# descriptor from the usd-core stage itself.

# %%
import ovrtx

ovrtx.register_schema_paths()

import numpy as np
import ovstage
import warp as wp
from pxr import Plug
from support import DATA_DIR, OUTPUT_DIR, REPO_ROOT, cuda_gpu_index, tutorial_device

import ovcae

ovcae.setup()
wp.config.log_level = wp.LOG_WARNING

API = "VtkExternalDatasetAPI"
FILE_NAMES = "vtk:fileNames"
Plug.Registry().RegisterPlugins([str(REPO_ROOT / "tutorials" / "vtk_external_dataset_schema")])

device = tutorial_device()
gpu_index = cuda_gpu_index(device) or "0"
renderer = ovrtx.Renderer(ovrtx.RendererConfig(active_cuda_gpus=gpu_index))
print("Compute device:", device, " render GPU:", gpu_index)

# %% [markdown]
# ## Look at the file with VTK
#
# `vtkExodusIIReader` exposes the file's structure before reading any
# arrays: element blocks, and the nodal and element variables. Nothing is
# loaded until an array is enabled and the reader updated. The disk file has
# one hexahedral block and seven nodal variables, one of them the velocity
# vector `V`.

# %%
import vtk
from vtk.util.numpy_support import vtk_to_numpy

exodus_path = DATA_DIR / "disk_out_ref.ex2"

reader = vtk.vtkExodusIIReader()
reader.SetFileName(str(exodus_path))
reader.UpdateInformation()

print(
    "Blocks   :",
    [reader.GetObjectName(reader.ELEM_BLOCK, i) for i in range(reader.GetNumberOfObjects(reader.ELEM_BLOCK))],
)
print(
    "Nodal    :",
    [reader.GetObjectArrayName(reader.NODAL, i) for i in range(reader.GetNumberOfObjectArrays(reader.NODAL))],
)
print("Time steps:", reader.GetNumberOfTimeSteps())

# %% [markdown]
# ## Write the Data Source
#
# A Data Source subclasses `ovcae.DataSource` and answers three questions:
#
# - `resolves(key)`: is this key mine? Here, does the prim at that path
#   carry the descriptor schema. Answering without loading anything is what
#   lets `CompositeDataSource` route keys between several sources.
# - `frame_at_time(time)`: return a **frame provider** bound to one moment.
#   Everything an update asks for goes through the provider, so a
#   time-varying source picks its samples once here and stays consistent
#   for the whole update.
# - `release_resources()`: drop what you hold, called once after `close`
#   when no update still has a lease.
#
# The provider answers two more. `describe_component` returns an opaque
# **identity** for a dataset's topology, geometry, or a field, plus the
# native sample it came from; the Session compares identities between
# updates to decide whether an operator has to run. Here the identity includes
# the file path and its modification time, so unchanged files retain their
# component identities. `resolve_dataset` loads the data and returns a Warp
# SimData dataset on the requested device, with the requested fields attached
# under the **role names** the recipe uses.
#
# A dataset selection has a second channel besides the target prim: the
# **identifier**, a string that qualifies *which* dataset beneath the target
# is meant. The CAE OpenUSD readers use it to pick a zone or boundary beneath
# a holder prim; here it picks one dataset out of a composite file. VTK
# readers return composite data as a multiblock tree or a **partitioned
# dataset collection**, whose **data assembly** names every node by a path
# such as `/Root/SideSets/UnnamedsetID4`. The source converts whatever the
# reader returns to a collection and takes the identifier to be an assembly
# node path that holds exactly one partitioned dataset. A composite file
# with no identifier is an error, not a guess: which block a slice should
# cut is the recipe's decision, and the error lists the valid paths.
#
# The VTK part is ordinary: read the file once, convert to a collection,
# walk the assembly, and keep each dataset's points, cell types, offsets,
# connectivity, and fields as host arrays, in exactly the layout the `vtk`
# data model takes. A partitioned dataset with several partitions is one
# dataset split for parallelism, so its partitions are appended into one
# grid. Devices get copies on demand.

# %%
import os

from pxr import Usd
from warp_simdata import AssociationType, Field
from warp_simdata.data_models.vtk import unstructured_grid

from ovcae.data import DatasetSample, held_time, resolve_asset_path


def _arrays(grid) -> dict:
    """Host arrays of one unstructured grid, in the layout the ``vtk`` data model takes."""
    point_data = grid.GetPointData()
    return {
        "points": vtk_to_numpy(grid.GetPoints().GetData()).astype(np.float32),
        "cell_types": vtk_to_numpy(grid.GetCellTypes()).astype(np.uint8),
        "cell_offsets": vtk_to_numpy(grid.GetCells().GetOffsetsArray()).astype(np.int32),
        "connectivity": vtk_to_numpy(grid.GetCells().GetConnectivityArray()).astype(np.int32),
        "fields": {
            point_data.GetArrayName(i): vtk_to_numpy(point_data.GetArray(i)).astype(np.float32)
            for i in range(point_data.GetNumberOfArrays())
        },
    }


def _one_grid(partitioned) -> vtk.vtkUnstructuredGrid | None:
    """The partitions of one partitioned dataset as a single unstructured grid; None if it is empty."""
    partitions = [partitioned.GetPartition(i) for i in range(partitioned.GetNumberOfPartitions())]
    partitions = [part for part in partitions if part is not None and part.GetNumberOfCells()]
    if not partitions:
        return None
    if len(partitions) == 1 and partitions[0].IsA("vtkUnstructuredGrid"):
        return partitions[0]
    append = vtk.vtkAppendFilter()
    for part in partitions:
        append.AddInputData(part)
    append.Update()
    return append.GetOutput()


def read_exodus(path: str) -> dict[str, dict]:
    """Read an Exodus II file into host arrays per dataset, keyed by data assembly node path."""
    reader = vtk.vtkExodusIIReader()
    reader.SetFileName(path)
    reader.UpdateInformation()
    for kind in (reader.ELEM_BLOCK, reader.SIDE_SET):
        for index in range(reader.GetNumberOfObjects(kind)):
            reader.SetObjectStatus(kind, index, 1)
    for index in range(reader.GetNumberOfObjectArrays(reader.NODAL)):
        reader.SetObjectArrayStatus(reader.NODAL, reader.GetObjectArrayName(reader.NODAL, index), 1)

    convert = vtk.vtkConvertToPartitionedDataSetCollection()
    convert.SetInputConnection(reader.GetOutputPort())
    convert.Update()
    collection = convert.GetOutput()
    assembly = collection.GetDataAssembly()

    datasets = {}
    pending = [assembly.GetRootNode()]
    while pending:  # every assembly node that holds exactly one non-empty partitioned dataset is selectable
        node = pending.pop()
        pending.extend(assembly.GetChildNodes(node, False))
        indices = list(assembly.GetDataSetIndices(node, False))
        if len(indices) != 1:
            continue
        grid = _one_grid(collection.GetPartitionedDataSet(indices[0]))
        if grid is not None:
            datasets[assembly.GetNodePath(node)] = _arrays(grid) | {
                "label": assembly.GetAttributeOrDefault(node, "label", "")
            }
    return datasets


class ExodusDataSource(ovcae.DataSource):
    """Serve ``VtkExternalDatasetAPI`` descriptors on a usd-core stage by reading their files with VTK."""

    def __init__(self, stage):
        super().__init__()
        self._stage = stage
        self._cache: dict[tuple[str, int], dict[str, dict]] = {}

    def resolves(self, key: str) -> bool:
        prim = self._stage.GetPrimAtPath(str(key)) if self._stage else None
        return bool(prim) and prim.HasAPI(API)

    def frame_at_time(self, time_seconds: float):
        return _ExodusFrame(self, float(time_seconds))

    def release_resources(self) -> None:
        self._stage = None
        self._cache.clear()

    # Helpers the frame provider uses.

    def select_file(self, key: str, time_seconds: float) -> tuple[str, DatasetSample]:
        """The file the descriptor names at ``time_seconds``, under held sampling."""
        prim = self._stage.GetPrimAtPath(str(key))
        if not prim or not prim.HasAPI(API):
            raise ovcae.DatasetResolutionError(f"{key} is not a {API} descriptor")
        attribute = prim.GetAttribute(FILE_NAMES)
        rate = self._stage.GetTimeCodesPerSecond()
        sample = held_time(tuple(attribute.GetTimeSamples()), time_seconds * rate)
        values = attribute.Get(Usd.TimeCode(sample) if sample is not None else Usd.TimeCode.Default()) or ()
        if len(values) != 1:
            raise ovcae.DatasetResolutionError(f"{key} must name exactly one file per sample; got {len(values)}")
        path = resolve_asset_path(self._stage.GetRootLayer(), values[0])
        if not path or not os.path.isfile(path):
            raise ovcae.DatasetResolutionError(f"{key} names a missing file {values[0].path!r}")
        return path, DatasetSample(None if sample is None else sample / rate)

    def _datasets(self, path: str) -> dict[str, dict]:
        """The file's datasets, read once per (path, modification time) so a rewritten file is re-read."""
        key = (path, os.stat(path).st_mtime_ns)
        if key not in self._cache:
            self._cache = {k: v for k, v in self._cache.items() if k[0] != path}  # drop the stale version
            self._cache[key] = read_exodus(path)
        return self._cache[key]

    def identifiers(self, path: str) -> tuple[str, ...]:
        """The data assembly node paths the file at ``path`` offers as identifiers."""
        return tuple(sorted(self._datasets(path)))

    def arrays(self, path: str, identifier: str) -> dict:
        identifiers = self.identifiers(path)
        if identifier not in identifiers:
            problem = (
                "is a composite dataset and needs an identifier naming one of its datasets"
                if not identifier
                else f"has no dataset {identifier!r}"
            )
            raise ovcae.DatasetResolutionError(
                f"{path} {problem}. Valid identifiers: {', '.join(repr(i) for i in identifiers)}"
            )
        return self._datasets(path)[identifier]


class _ExodusFrame:
    """One lease on the source: the file selection is fixed at construction."""

    def __init__(self, source: ExodusDataSource, time_seconds: float):
        self._source = source
        self._time = time_seconds

    def describe_component(self, key: str, identifier: str, component: str):
        path, sample = self._source.select_file(key, self._time)
        identity = (path, identifier, component, os.stat(path).st_mtime_ns, sample.time)
        return identity, sample

    def resolve_dataset(self, key: str, *, identifier: str, device: str, fields):
        path, _sample = self._source.select_file(key, self._time)
        arrays = self._source.arrays(path, identifier)
        dataset = unstructured_grid.create_dataset(
            wp.array(arrays["points"], dtype=wp.vec3f, device=device),
            wp.array(arrays["cell_types"], dtype=wp.uint8, device=device),
            wp.array(arrays["cell_offsets"], dtype=wp.int32, device=device),
            wp.array(arrays["connectivity"], dtype=wp.int32, device=device),
        )
        for role, names in fields.items():
            if len(names) != 1 or names[0] not in arrays["fields"]:
                available = ", ".join(sorted(arrays["fields"]))
                raise ovcae.DatasetResolutionError(f"{role!r} must name one of {available}; got {list(names)}")
            values = arrays["fields"][names[0]]
            dtype = wp.vec3f if values.ndim == 2 else wp.float32
            dataset.add_field(
                role, Field.from_array(wp.array(values, dtype=dtype, device=device), AssociationType.NODE)
            )
        return dataset


print("Data Source defined")

# %% [markdown]
# An author needs to know the identifiers before writing recipes, so the
# source exposes them. The volume is the single element block; side sets
# 4 to 6 are the top, rim, and shaft of the heated disk. Asking for the
# file with no identifier is refused with the list of what it holds.

# %%
probe = ExodusDataSource(Usd.Stage.CreateInMemory())
for identifier in probe.identifiers(str(exodus_path)):
    arrays = probe.arrays(str(exodus_path), identifier)
    print(
        f"{identifier:<36} {arrays['label']:<22} {len(arrays['points']):>6} points {len(arrays['cell_types']):>6} cells"
    )

try:
    probe.arrays(str(exodus_path), "")
except ovcae.DatasetResolutionError as error:
    print("\nWithout an identifier:", error)
probe.close()

# %% [markdown]
# ## Author the scene
#
# The descriptor is a plain prim with the schema applied and the file named
# relative to the layer, so the scene stays relocatable with its data. The
# recipes come from the builders of tutorial 3, and all target the
# descriptor. A **planar slice** through the axis of the disk colored by
# temperature, in `x` mode so its translation alone places it, and
# **streamlines** from a sphere of seeds in the inflow above the disk,
# colored by speed, name the element block as their identifier. Three
# **Faces** recipes name side sets 4 to 6, the top, rim, and shaft of the
# heated disk, so the disk itself appears as a surface colored by its
# temperature. A **Glyphs** recipe names side set 7, the x = 0 symmetry
# plane, and draws arrows at sampled nodes: pointed along `V`, their
# length the speed mapped from the file's range onto a readable one, and
# colored by temperature. Nothing beyond the identifier is needed to turn a
# surface of the file into a vector plot. The gas enters at the top,
# sweeps down over the disk at the middle, and leaves through the bottom.
#
# The camera looks at the disk from above and to the side; its eye and
# target are kept for the orbit control later.

# %%
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdRender

import ovcae.authoring as author
from ovcae.authoring import schemas as S
from ovcae.stream import OrbitCamera

DATASET = "/World/DiskOut"
VOLUME = "/Root/ElementBlocks/UnnamedblockID1"
DISK = {
    "Top": "/Root/SideSets/UnnamedsetID4",
    "Rim": "/Root/SideSets/UnnamedsetID5",
    "Shaft": "/Root/SideSets/UnnamedsetID6",
}
SLICE = "/World/Recipes/Slice"
STREAMLINES = "/World/Recipes/Streamlines"
GLYPHS = "/World/Recipes/Glyphs"
SYMMETRY_PLANE = "/Root/SideSets/UnnamedsetID7"
MAX_GLYPHS = "cae:viz:glyphs:maxCount"
SEEDS = "/World/Controls/Seeds"
CAMERA = "/World/Camera"
RENDER_PRODUCT = "/Render/Camera"
EYE, TARGET = (26.0, -34.0, 8.0), (0.0, 0.0, 0.0)

scene_path = OUTPUT_DIR / "A2_disk_out.usda"
scene_path.parent.mkdir(parents=True, exist_ok=True)
scene_path.unlink(missing_ok=True)

stage = Usd.Stage.CreateNew(str(scene_path))
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
stage.SetDefaultPrim(stage.DefinePrim("/World", "Xform"))

descriptor = stage.DefinePrim(DATASET)
descriptor.ApplyAPI(API)
descriptor.CreateAttribute(FILE_NAMES, Sdf.ValueTypeNames.AssetArray).Set(
    [Sdf.AssetPath(os.path.relpath(exodus_path, scene_path.parent))]
)

author.create_planar_slice(
    stage,
    SLICE,
    source=DATASET,
    identifier=VOLUME,
    colors="Temp",
    mode="x",
    origin=(0.0, 0.0, 0.0),
    colormap="coolwarm",
    color_range=(293.0, 913.0),
)
for name, side_set in DISK.items():
    author.create_faces(
        stage,
        f"/World/Recipes/Disk{name}",
        source=DATASET,
        identifier=side_set,
        colors="Temp",
        colormap="coolwarm",
        color_range=(293.0, 913.0),
    )
seeds = author.create_seed_sphere(stage, SEEDS, center=(0.0, 0.0, 4.0), radius=2.0, count=48)
author.create_streamlines(
    stage,
    STREAMLINES,
    source=DATASET,
    identifier=VOLUME,
    seeds=seeds,
    velocities="V",
    colors="V",
    max_steps=400,
    width=0.06,
    colormap="turbo",
    color_range=(0.0, 22.0),
)

glyphs = author.create_glyphs(
    stage,
    GLYPHS,
    source=DATASET,
    identifier=SYMMETRY_PLANE,
    shape="Arrow",
    orientations="V",
    scales="V",
    colors="Temp",
    max_count=300,
    colormap="coolwarm",
    color_range=(293.0, 913.0),
)
S.FieldSelectionAPI(glyphs, "scales").mode = "vector_magnitude"  # arrow length follows speed, not components
scale_mapping = S.FieldMappingAPI.apply(glyphs, "scales")
scale_mapping.domain = (0.0, 22.0)
scale_mapping.range = (0.2, 1.4)

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
for prim in stage.GetPrimAtPath("/World/Recipes").GetChildren():
    selection = S.DatasetSelectionAPI(prim, "source")
    print(f"{prim.GetPath()!s:<28} {selection.identifier:<36}", author.validate_recipe(prim) or "ok")
print("Saved", scene_path)

# %% [markdown]
# ## Evaluate
#
# The OVStage is populated as always; the descriptor arrives as a plain prim
# there, which is fine because nothing on the stage side reads it. The Data
# Source is constructed over the usd-core stage that was just saved, the same
# layer the OVStage was populated from, and handed to the Session through
# `at_time` like any other. Had the scene also contained payloads OVCAE can
# read, `ovcae.CompositeDataSource(UsdDataSource(stage), ExodusDataSource(stage))`
# would route each key to whichever source claims it.

# %%
from ovcae.app import UpdateLoop

ov_stage = ovstage.Stage("tutorial-A2")
ovstage.population.open_usd_with_desc(
    ov_stage,
    str(scene_path),
    ordinal=1,
    time_code=0.0,
    descs=ovcae.ovstage_population_desc(domains=ovstage.PopulationDomain.ALL),
)
ov_stage.advance_write_floor(1).wait()

data_source = ExodusDataSource(stage)
loop = UpdateLoop(ov_stage, data_source, ordinal=1, device=device)

report = loop.update()
for diagnostic in report.diagnostics:
    print(f"  [{diagnostic.severity}] {diagnostic.code}: {diagnostic.message}")
for result in report.operators:
    print(f"{result.operator_path:<30} {result.status:<10} {result.reason or ''}")

# %% [markdown]
# ## Make it live
#
# `ovcae.stream.LiveView` from tutorial 2: the render thread, mailboxes,
# orbit camera, and OVStream server around the update loop. The camera goes
# through the view's presentation mailbox; the sliders below request recipe
# edits, because the seed and slice prims are inputs the operators read.

# %%
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
print("Streaming", view.width, "x", view.height, "on signaling port", view.signal_port)

# %% [markdown]
# ## Play with it
#
# Click the video to take control, left-drag to orbit, wheel to zoom; move
# the pointer off it to give the keyboard back to the notebook. Two
# sliders move the slice along x and the seed sphere along z; both are
# **transform** writes on prims the operators read. The third sets how many
# arrows the Glyphs recipe keeps,
# a plain **setting** on the recipe prim. All three are recipe inputs, so
# unlike the camera they trigger an update. The mailbox keeps the latest
# value per control, skipping intermediate values during a fast drag.

# %%
import ipywidgets as widgets
from IPython.display import HTML, display


def translation(x: float = 0.0, z: float = 0.0) -> np.ndarray:
    matrix = np.eye(4)
    matrix[3, 0], matrix[3, 2] = x, z
    return matrix


slice_slider = widgets.FloatSlider(
    value=0.0, min=-5.5, max=5.5, step=0.25, description="Slice x", continuous_update=True
)
seed_slider = widgets.FloatSlider(
    value=4.0, min=-8.0, max=9.0, step=0.25, description="Seeds z", continuous_update=True
)
slice_slider.observe(lambda change: view.request(SLICE, "omni:xform", translation(x=change["new"])), names="value")
seed_slider.observe(lambda change: view.request(SEEDS, "omni:xform", translation(z=change["new"])), names="value")
glyph_slider = widgets.IntSlider(value=300, min=20, max=543, step=1, description="Arrows", continuous_update=True)
glyph_slider.observe(lambda change: view.request(GLYPHS, MAX_GLYPHS, change["new"]), names="value")

display(HTML(view.html()))
display(widgets.HBox([slice_slider, seed_slider, glyph_slider]))

# %% [markdown]
# ## What an edit costs
#
# Setting a slider from code does what a drag does. Move the seeds and wait
# for the update the render thread runs: the slice, the arrows, and the
# three disk surfaces come back unchanged because nothing they read moved,
# and the Data Source was not asked to read the file again because the
# identities it reported did not change. Then move the slice: it
# recomputes, and so do the streamlines, which evaluate on every update
# because seeds taken from the scene carry no identity to compare, as the
# [troubleshooting guide](../docs/troubleshooting.md) notes. Finally thin
# the arrows: only the Glyphs recipe and the streamlines run.

# %%
seed_slider.value = 1.0
report = view.wait_for_update()
print(f"Update {report.input_ordinal} -> {report.output_ordinal}")
for result in report.operators:
    print(f"{result.operator_path:<30} {result.status:<10} {result.reason or ''}")

slice_slider.value = 2.0
report = view.wait_for_update()
print(f"Update {report.input_ordinal} -> {report.output_ordinal}")
for result in report.operators:
    print(f"{result.operator_path:<30} {result.status:<10} {result.reason or ''}")

glyph_slider.value = 150
report = view.wait_for_update()
print(f"Update {report.input_ordinal} -> {report.output_ordinal}")
for result in report.operators:
    print(f"{result.operator_path:<30} {result.status:<10} {result.reason or ''}")

# %% [markdown]
# Wait until the video shows the updated scene, then save the latest frame.
# The snapshot does not wait for rendering to finish.

# %%
from support import save_png

png_path = save_png(view.snapshot(), OUTPUT_DIR / "A2_custom_data_source.png")
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
# A file OVCAE cannot compose became a dataset recipes can select, through
# a descriptor prim with a schema of your own and a Data Source of about
# a hundred lines: three methods to claim keys and hand out frame
# providers, two on the provider to identify and load data. The reading
# itself stayed in VTK. The Data Source caches the file during evaluation,
# and the Session reuses unaffected products.
#
# The identifier carried the second half of a selection, which dataset of
# a composite file a recipe reads, without a prim per dataset in the scene:
# one descriptor, and recipes that name an assembly node each. Leaving it
# out on a composite file is an error, so a recipe always says what it cut. The same shape serves any reader
# you already have, and time series come for free: author `vtk:fileNames` with time samples, and `select_file`
# already picks the one at or before the requested time. The
# [readers guide](../docs/readers.md) documents the Data Source contract in
# full, and the [integration guide](../docs/integration-guide.md#change-detection)
# explains what the identities buy you. The Data Source is packaged as
# `exodus_source.py` beside this notebook so appendix A3 can build on it.
