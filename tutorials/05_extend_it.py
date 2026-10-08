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

# %% [markdown]
# # 5. Extend it
#
# Tutorial 4 colored a rippling plate with a stock recipe, and the plate
# stayed flat because no built-in operator moves points. This tutorial
# writes the operator that does: **Warp by Vector** moves every point of a
# surface along a per-node vector field, scaled by a factor, and colors the
# result by a scalar. On the way it covers the two things a custom operator
# needs beyond the class itself: a **schema** for its setting, and the
# registrations that make the schema known.
#
# The plate, the ripple, and the coupling loop are tutorial 4's, imported
# from `ripple.py` beside this notebook so the attention stays on the
# operator. This tutorial requires an RTX-capable NVIDIA GPU and uses the
# device selected in tutorial 0 for evaluation and rendering.

# %% [markdown]
# ## Initialize, and register the schema
#
# A setting on a recipe is an attribute, and an attribute needs a schema so
# it has a type and a default wherever it appears. The schema here is a
# codeless one, two files in `warp_by_vector_schema/` beside this notebook:
# `plugInfo.json` names it and `generatedSchema.usda` declares one property,
# `cae:viz:warp_by_vector:scale`. Open them; there is nothing else to it.
#
# Two runtimes have to learn about the schema: usd-core, so authoring can
# apply it and the attribute gets its fallback value, and OVStage, so
# populating a stage copies the attribute in. `ovcae.register_schema_paths`
# registers a schema folder with both and returns the schema identifiers it
# found. That is all OVCAE needs: the Session asks usd-core's schema registry
# what an applied schema declares, so the operator's scene view captures the
# property once usd-core knows the schema, and the population description
# populates every schema registered this way.
#
# Like `ovcae.setup()`, this must run before the process opens its first
# stage. Both runtimes build their schema registry once, on the first schema
# read, and a schema registered after that is never seen. The renderer is
# created before any stage, as in tutorial 4.

# %%
import ovrtx

ovrtx.register_schema_paths()

import numpy as np
import ovstage
import warp as wp
from pxr import Usd
from support import OUTPUT_DIR, REPO_ROOT, cuda_gpu_index, tutorial_device

import ovcae

ovcae.setup()
wp.config.log_level = wp.LOG_WARNING

SCHEMA = "CaeVizWarpByVectorAPI"
SCALE = "cae:viz:warp_by_vector:scale"
SCHEMA_DIR = REPO_ROOT / "tutorials" / "warp_by_vector_schema"

registered = ovcae.register_schema_paths(SCHEMA_DIR)

device = tutorial_device()
render = device != "cpu"
renderer = ovrtx.Renderer(ovrtx.RendererConfig(active_cuda_gpus=cuda_gpu_index(device))) if render else None

print("Registered from the folder:", *registered)
print("usd-core knows the schema :", Usd.SchemaRegistry().FindAppliedAPIPrimDefinition(SCHEMA) is not None)
print("OVCAE will populate       :", SCHEMA in ovcae.registered_schemas())
print("Compute device            :", device)
print("Rendering                 :", "on GPU " + cuda_gpu_index(device) if render else "skipped on CPU")

# %% [markdown]
# ## The plate, again
#
# `ripple.make_plate` is tutorial 4's plate; `ripple.ripple_sample` is its
# solver, with one addition: besides the scalar `height` it can attach a
# `displacement` vector field, the height straight up, for the operator to
# move points along.

# %%
import ripple

plate, points = ripple.make_plate(161, device)
sample = ripple.ripple_sample(plate, points, 0.0, device, displacement=True)
print("Plate :", plate.get_num_nodes(), "nodes,", plate.get_num_elems(), "quads, on", plate.device)
print("Fields:", sample.get_field_names())

# %% [markdown]
# ## Write the operator
#
# An operator is a class with an `execute(context)` method, decorated to
# declare what recipes it matches and what it reads:
#
# - `required_schemas` lists the applied schemas a recipe must carry, here
#   the operator's own schema, a dataset selection in the `source` role, and
#   a field selection in the `displacement` role. `optional_schemas` adds a
#   `colors` selection the recipe may omit.
# - `dataset_components` declares the dataset components used for change
#   detection. Field selections name the fields the operator needs.
#
# Inside `execute`, `context.prim` is a read-only view of the committed
# recipe prim, which is where the scale comes from.
# `get_input_dataset` returns the selected dataset with the selected fields
# attached under their **role names**, `displacement` and `colors`, whatever
# they were called in the source. From there it is plain Warp SimData and
# Warp: `element_faces` turns the plate's quads into a renderable surface
# whose `node_idx` field maps each surface point back to a plate node, three
# small kernels gather the fields and move the points, and `op_utils.mesh`
# describes the result as a standard Mesh product for the Session to publish.

# %%
import warp_simdata as simdata
from warp_simdata.operators import element_faces

from ovcae import operators
from ovcae.operators import utils as op_utils


@wp.kernel
def gather_vec3(values: wp.array(dtype=wp.vec3f), indices: wp.array(dtype=wp.int32), out: wp.array(dtype=wp.vec3f)):
    i = wp.tid()
    out[i] = values[indices[i]]


@wp.kernel
def gather_float(
    values: wp.array(dtype=wp.float32), indices: wp.array(dtype=wp.int32), out: wp.array(dtype=wp.float32)
):
    i = wp.tid()
    out[i] = values[indices[i]]


@wp.kernel
def displace(
    points: wp.array(dtype=wp.vec3f), vectors: wp.array(dtype=wp.vec3f), scale: float, out: wp.array(dtype=wp.vec3f)
):
    i = wp.tid()
    out[i] = points[i] + scale * vectors[i]


@operators.operator(
    required_schemas={SCHEMA, "CaeVizDatasetSelectionAPI:source", "CaeVizFieldSelectionAPI:displacement"},
    optional_schemas={"CaeVizFieldSelectionAPI:colors"},
    dataset_components={"source": {"topology", "geometry"}},
)
class WarpByVector:
    """Move a surface's points along a per-node vector field, scaled."""

    def execute(self, context):
        if context.prim.type_name != "Mesh":
            raise operators.InvalidRecipe("Warp by Vector requires a Mesh recipe")
        scale = float(context.prim.get_attribute(SCALE).get())

        dataset = op_utils.get_input_dataset(context, role="source")
        vectors = dataset.get_field("displacement")
        if vectors.association != simdata.AssociationType.NODE or vectors.dtype != wp.vec3f:
            raise operators.InvalidRecipe("displacement must be a per-node float32 3-vector field")

        surface = element_faces.compute(dataset)
        node_idx = surface.get_field("node_idx").to_array()
        n = surface.get_num_nodes()
        device = context.device

        gathered = wp.empty(n, dtype=wp.vec3f, device=device)
        wp.launch(gather_vec3, dim=n, inputs=[vectors.to_array(), node_idx, gathered], device=device)
        moved = wp.empty(n, dtype=wp.vec3f, device=device)
        wp.launch(displace, dim=n, inputs=[surface.handle.points, gathered, scale, moved], device=device)

        primvars = []
        if dataset.has_field("colors"):
            colors = wp.empty(n, dtype=wp.float32, device=device)
            wp.launch(
                gather_float, dim=n, inputs=[dataset.get_field("colors").to_array(), node_idx, colors], device=device
            )
            primvars.append(op_utils.primvar("colors", colors, "vertex"))

        wp.synchronize_device(device)
        return op_utils.mesh(
            points=moved,
            face_vertex_counts=surface.handle.face_vertex_counts,
            face_vertex_indices=surface.handle.face_vertex_indices,
            primvars=primvars,
        )


registry = operators.OperatorRegistry.with_builtins()
registry.register(WarpByVector)
print("Registered:", WarpByVector.__name__, "alongside the built-in operators")

# %% [markdown]
# Operators are never discovered implicitly: the registry starts from the
# built-in set, the class is registered by hand, and the Session below takes
# the registry. The decorator validates the declaration; had `source` in
# `dataset_components` not appeared in the schemas, this cell would have
# raised.

# %% [markdown]
# ## Author the recipe
#
# The recipe is a `Mesh` prim like any other, with the operator's schema
# applied next to the standard ones. This custom operator has no recipe
# builder, so the recipe is assembled from the typed schema
# wrappers tutorial 3 used to edit one: `OperatorAPI` enables it,
# `DatasetSelectionAPI` in the `source` role targets the key the data will
# be published under, and two `FieldSelectionAPI` instances map the
# dataset's fields onto the operator's roles. `schema_from_registry`
# returns a wrapper for the custom schema, generated from the definition,
# so the scale is set the same way. `setup_scalar_materials` binds a
# colormap over the `colors` primvar the operator emits. Camera, light, and
# render product are tutorial 4's.

# %%
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdRender

import ovcae.authoring as author
from ovcae.authoring import schemas as S

KEY = "/World/Plate"
RECIPE = "/World/Ripple"
RENDER_PRODUCT = "/Render/Camera"

scene_path = OUTPUT_DIR / "05_ripple_scene.usda"
scene_path.parent.mkdir(parents=True, exist_ok=True)
scene_path.unlink(missing_ok=True)

stage = Usd.Stage.CreateNew(str(scene_path))
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
stage.SetDefaultPrim(stage.DefinePrim("/World", "Xform"))
stage.DefinePrim(KEY)

mesh = UsdGeom.Mesh.Define(stage, RECIPE)
mesh.CreatePointsAttr([])
mesh.CreateFaceVertexCountsAttr([])
mesh.CreateFaceVertexIndicesAttr([])
mesh.CreateSubdivisionSchemeAttr("none")
recipe = mesh.GetPrim()

S.OperatorAPI.apply(recipe).enabled = True
WarpByVectorAPI = S.schema_from_registry(SCHEMA)
WarpByVectorAPI.apply(recipe).scale = 2.0
S.DatasetSelectionAPI.apply(recipe, "source").target = KEY
S.FieldSelectionAPI.apply(recipe, "displacement").field_names = ["displacement"]
S.FieldSelectionAPI.apply(recipe, "colors").field_names = ["height"]
author.setup_scalar_materials(recipe, colormap="coolwarm", color_range=ripple.HEIGHT_RANGE)

camera = UsdGeom.Camera.Define(stage, "/World/Camera")
camera.CreateFocalLengthAttr(24.0)
camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
camera.AddTranslateOp().Set(Gf.Vec3d(2.2, -2.6, 1.6))
camera.AddRotateXYZOp().Set(Gf.Vec3f(62.0, 0.0, 40.0))
UsdLux.DomeLight.Define(stage, "/World/Light").CreateIntensityAttr(1000.0)

product = UsdRender.Product.Define(stage, RENDER_PRODUCT)
product.CreateResolutionAttr(Gf.Vec2i(960, 540))
product.CreateCameraRel().SetTargets([camera.GetPath()])
color_var = UsdRender.Var.Define(stage, f"{RENDER_PRODUCT}/LdrColor")
color_var.CreateSourceNameAttr("LdrColor")
color_var.GetPrim().CreateAttribute("omni:rtx:aov", Sdf.ValueTypeNames.Opaque)
product.CreateOrderedVarsRel().SetTargets([color_var.GetPath()])

stage.Save()
print("Applied schemas:", list(recipe.GetAppliedSchemas()))
print("validate_recipe:", author.validate_recipe(recipe) or "ok")
print("Saved", scene_path)

# %% [markdown]
# `validate_recipe` checks built-in operator contracts, so it reports
# "no operator API applied" for this custom operator. The registered custom
# schema is recognized. Evaluate the recipe with the registry below to check
# the custom operator's behavior.

# %% [markdown]
# ## Populate and create the Session
#
# The same steps as tutorial 4, plus the operator registry handed to the
# Session. The population description already includes the schema registered
# above, so nothing else changes: everything the operator needs to find its
# recipe and read its scale is in place.

# %%
ov_stage = ovstage.Stage("tutorial-05")
ovstage.population.open_usd_with_desc(
    ov_stage,
    str(scene_path),
    ordinal=1,
    time_code=0.0,
    descs=ovcae.ovstage_population_desc(domains=ovstage.PopulationDomain.ALL),
)
ov_stage.advance_write_floor(1).wait()

source = ovcae.MemoryDataSource()
session = ovcae.Session(ov_stage, device=device, operators=registry)
if render:
    renderer.attach_ovstage(ov_stage)
print("Ready")

# %% [markdown]
# ## The coupling loop
#
# Tutorial 4's loop, with the solver now producing the displacement too.
# The published points are read back after each step to compare the
# highest point against what the ripple formula predicts: the displacement
# is straight up, so the maximum height is the scale times the largest
# absolute `height` at that time.

# %%
from support import read_attribute, save_png

frames = []


def render_frame(ordinal: int):
    products = renderer.step(render_products={RENDER_PRODUCT}, delta_time=0.0, ordinal=ordinal)
    with (
        products[RENDER_PRODUCT]
        .frames[0]
        .render_vars[f"{RENDER_PRODUCT}/LdrColor"]
        .map(device=ovrtx.Device.CPU) as mapped
    ):
        return np.from_dlpack(mapped).copy()


def evaluate(time: float, ordinal: int):
    """Update from ``ordinal`` to ``ordinal + 1`` at ``time`` and commit; return the new ordinal and report."""
    with source.at_time(time) as frame_data:
        report = session.update(input_ordinal=ordinal, output_ordinal=ordinal + 1, frame_data=frame_data).wait()
    for diagnostic in report.diagnostics:
        print(f"  [{diagnostic.severity}] {diagnostic.code}: {diagnostic.message}")
    if not report.committable:
        raise RuntimeError("update not committable")
    ov_stage.advance_write_floor(ordinal + 1).wait()
    return ordinal + 1, report


ordinal = 1
for t in np.linspace(0.0, 1.0, 6):
    source.publish(KEY, ripple.ripple_sample(plate, points, float(t), device, displacement=True))
    ordinal, report = evaluate(float(t), ordinal)
    result = report.operators[0]
    published_points = np.asarray(read_attribute(ov_stage, RECIPE, "points", ordinal)).reshape(-1, 3)
    highest = float(np.abs(published_points[:, 2]).max())
    expected = 2.0 * float(np.abs(ripple.ripple_height(points, float(t))).max())
    print(f"t={t:.1f}  {result.status:<10} {len(published_points):>6} points", end="  ")
    print(f"max |z| {highest:.4f}  expected {expected:.4f}")
    assert abs(highest - expected) < 1e-4
    if render:
        render_frame(ordinal)  # the first step after a change uploads; keep the second
        frames.append(save_png(render_frame(ordinal), OUTPUT_DIR / f"05_ripple_t{t:.1f}.png"))

# %% [markdown]
# ## Change the setting
#
# The scale is an attribute on the recipe prim, so an application changes
# it the way it changes any recipe setting: by writing the attribute to the
# stage at a new ordinal. Nothing new is published this time, so the only
# change the update sees is the recipe's, and the operator runs again with
# the new value: the plate stands twice as tall.

# %%
from support import write_scalar

ordinal += 1
write_scalar(ov_stage, RECIPE, SCALE, np.float32(4.0), ordinal)
ov_stage.advance_write_floor(ordinal).wait()

ordinal, report = evaluate(1.0, ordinal)
published_points = np.asarray(read_attribute(ov_stage, RECIPE, "points", ordinal)).reshape(-1, 3)
print("After scale = 4:", report.operators[0].status, f"- max |z| {np.abs(published_points[:, 2]).max():.4f}")
if render:
    render_frame(ordinal)
    frames.append(save_png(render_frame(ordinal), OUTPUT_DIR / "05_ripple_scale4.png"))

# %% [markdown]
# ## Look at it
#
# One frame per step, then the last step at the larger scale. The plate now
# undulates, and the coloring follows the height as before.

# %%
if frames and "get_ipython" in globals():
    from IPython.display import Image, display

    for path in frames:
        print(path.name)
        display(Image(filename=str(path), width=480))
elif frames:
    print("Saved", *[path.name for path in frames])
else:
    print("No frames on CPU; the numbers above are the result")

# %% [markdown]
# ## If the schema had not been registered
#
# The registration is easy to forget, and the failure is specific. Had the
# schema folder not been registered before the first stage, the recipe's
# `apiSchemas` token would still reach OVStage, but usd-core would have no
# definition for it. The report then says so twice: a `SCHEMA_NOT_CAPTURED`
# warning names the token, and the operator fails because the scale
# attribute is outside the scene view it is allowed to read. Registration is
# process-wide and happens once, so this notebook cannot show the failure
# after the success; `tests/test_custom_schema.py` does.

# %% [markdown]
# ## Shut down

# %%
session.close()
source.close()
if render:
    renderer.detach_ovstage()
ov_stage.destroy()
if render:
    renderer.destroy()
print("Done")

# %% [markdown]
# ## What just happened
#
# An operator class declared the schemas it matches and the data it reads,
# was registered by hand, and read a setting from a schema of its own after
# that schema was registered with usd-core and OVStage in one call. A
# recipe for it was assembled from the typed schema wrappers, since no
# builder knows it. The Session treated it like a built-in: it ran when the
# data or the setting changed and reported the one thing that was
# misconfigured.
#
# The [operators guide](../docs/operators.md) is the reference for the
# declaration and the execution contract. The remaining extension point, a
# Data Source of your own for a format OVCAE does not read, is the subject
# of appendix A2.
