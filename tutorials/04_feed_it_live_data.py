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
# # 4. Feed it live data
#
# So far every dataset came out of a file. Simulation codes coupled to a
# visualization, and applications that compute their own data, have arrays
# in memory instead, and writing them to disk just to read them back is not
# an option. This tutorial feeds OVCAE from memory: a small stand-in for a
# solver produces a new result every step, publishes it to a **Memory Data
# Source**, and the same Faces recipe you know from tutorial 1 turns each
# result into a picture.
#
# The data is a square plate carrying a ripple that travels outward from
# its center. It is built with a Warp SimData **data model**, which is how
# any in-memory mesh becomes a dataset. Nothing else is new: the recipe, the
# session, and the update loop are the ones from before, which is the point.
#
# This tutorial requires an RTX-capable NVIDIA GPU and uses the device
# selected in tutorial 0 for evaluation and rendering.

# %% [markdown]
# ## Initialize
#
# The usual order: OVRTX's schema paths, `ovcae.setup()`, then the renderer,
# and only after that any stage. The renderer comes before the stage for a
# reason tutorials 1 to 3 met without saying so: OVRTX has to exist when a
# stage is populated for the colormap textures in the materials, which are
# named relative to the material library, to resolve.

# %%
import ovrtx

ovrtx.register_schema_paths()

import numpy as np
import ovstage
import warp as wp
from support import OUTPUT_DIR, cuda_gpu_index, tutorial_device

import ovcae

ovcae.setup()
wp.config.log_level = wp.LOG_WARNING

device = tutorial_device()
render = device != "cpu"
renderer = ovrtx.Renderer(ovrtx.RendererConfig(active_cuda_gpus=cuda_gpu_index(device))) if render else None
print("Compute device:", device)
print("Rendering     :", "on GPU " + cuda_gpu_index(device) if render else "skipped on CPU")

# %% [markdown]
# ## Build a dataset in memory
#
# Warp SimData datasets are built through data models, one per convention
# for describing a mesh. This tutorial uses the EnSight Gold model: an
# unstructured *part* made of *pieces*, each piece one element shape with
# its connectivity as a flat array of **1-based** node ids, as EnSight files
# store them. The plate is a regular grid of quads in the xy-plane. The
# arrays are created on the compute device, because a dataset lives where
# its arrays live and a recipe evaluated on another device cannot use it.

# %%
import warp_simdata as simdata
from warp_simdata.data_models.ensight_gold import ensight_shapes
from warp_simdata.data_models.ensight_gold.unstructured_part import create_dataset, create_piece_handle

N = 161  # nodes per side
xs, ys = np.meshgrid(np.linspace(-1.0, 1.0, N), np.linspace(-1.0, 1.0, N), indexing="ij")
points = np.stack([xs.ravel(), ys.ravel(), np.zeros(N * N)], axis=1).astype(np.float32)

i, j = np.meshgrid(np.arange(N - 1), np.arange(N - 1), indexing="ij")
corner = (i * N + j).ravel()  # 0-based id of each quad's first node
quads = np.stack([corner, corner + N, corner + N + 1, corner + 1], axis=1) + 1  # 1-based, counter-clockwise

plate = create_dataset(
    wp.array(points, dtype=wp.vec3f, device=device),
    [create_piece_handle(ensight_shapes.EN_quad4, wp.array(quads.ravel().astype(np.int32), device=device))],
)
print("Plate:", plate.get_num_nodes(), "nodes,", plate.get_num_elems(), "quads, on", plate.device)

# %% [markdown]
# ## The solver
#
# The stand-in for a simulation is one function. Given a time it returns the
# plate with a `height` field attached: a damped sine wave in the distance
# from the center, shifted by time so the rings travel outward. The dataset
# it returns is a **shallow copy** of the plate, sharing the points and
# connectivity and owning only the field, which is how a real coupling
# would avoid copying an unchanging mesh every step.

# %%
radius = np.hypot(points[:, 0], points[:, 1])


def solve(t: float):
    """Return the plate at time ``t`` with its ``height`` field."""
    height = (0.15 * np.sin(12.0 * radius - 6.0 * t) * np.exp(-2.0 * radius)).astype(np.float32)
    sample = plate.shallow_copy(pass_fields=False)
    sample.add_field("height", simdata.Field.from_array(wp.array(height, device=device), simdata.AssociationType.NODE))
    return sample


first = solve(0.0)
print(
    "Fields:",
    first.get_field_names(),
    " height range:",
    tuple(round(v, 3) for v in first.get_field("height").get_range()),
)

# %% [markdown]
# ## Author the recipe
#
# A recipe selects its dataset by a prim path, the **target** of its dataset
# selection. With a file, that path is the dataset prim the plug-in composed.
# With a Memory Data Source the path is a **key**: the source serves
# whatever was published under it, and nothing has to exist at that path in
# the scene. The plain prim defined here documents the key for anyone
# reading the file; OVCAE does not need it.
#
# The recipe itself is a Faces recipe from `create_faces`, exactly as in
# tutorial 3, colored by `height`. The field name is the one the dataset
# carries; the source attaches it under the recipe's role at evaluation.
# Camera, light, and render product are the application's, as before.

# %%
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdRender

import ovcae.authoring as author

KEY = "/World/Plate"
RECIPE = "/World/Surface"
RENDER_PRODUCT = "/Render/Camera"

scene_path = OUTPUT_DIR / "04_ripple_scene.usda"
scene_path.parent.mkdir(parents=True, exist_ok=True)
scene_path.unlink(missing_ok=True)

stage = Usd.Stage.CreateNew(str(scene_path))
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
stage.SetDefaultPrim(stage.DefinePrim("/World", "Xform"))
stage.DefinePrim(KEY)

author.create_faces(stage, RECIPE, source=KEY, colors="height", colormap="coolwarm", color_range=(-0.1, 0.1))

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
print("Saved", scene_path)

# %% [markdown]
# ## Populate, and create the source and session
#
# The stage and session are set up as in tutorial 1. The Data Source is
# the new part: `MemoryDataSource` starts empty, and the solver's results
# will be **published** into it inside the loop. It is a Data Source like
# the USD one, so `at_time` produces the Frame Data an update takes.

# %%
ov_stage = ovstage.Stage("tutorial-04")
ovstage.population.open_usd_with_desc(
    ov_stage,
    str(scene_path),
    ordinal=1,
    time_code=0.0,
    descs=ovcae.ovstage_population_desc(domains=ovstage.PopulationDomain.ALL),
)
ov_stage.advance_write_floor(1).wait()

source = ovcae.MemoryDataSource()
session = ovcae.Session(ov_stage, device=device)
if render:
    renderer.attach_ovstage(ov_stage)
print("Published keys:", source.published() or "none yet")

# %% [markdown]
# ## The coupling loop
#
# This is the shape of an in situ coupling. Each step the solver produces a
# result, the result is published under the key, an update evaluates the
# recipes against it, and the new version is committed and rendered.
# `publish` replaces whatever was under the key and marks it
# changed, so the update knows the Faces recipe has to run; recipes reading
# other data would have been reused.
#
# The Frame Data taken by `at_time` keeps the dataset it saw until it is
# released, so a solver publishing from another thread mid-update would
# not disturb the update. A dataset published without a time is served at
# every time, so the value passed to `at_time` is only a label here. To
# keep a history instead, publish each sample with `time=`; `at_time` then
# picks the sample at or before the requested time, as the
# [integration guide](../docs/integration-guide.md#in-memory-datasets)
# describes.

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
    source.publish(KEY, solve(float(t)))  # the solver's step, handed over
    ordinal, report = evaluate(float(t), ordinal)
    result = report.operators[0]
    colors = np.asarray(read_attribute(ov_stage, RECIPE, "primvars:colors", ordinal))
    print(f"t={t:.1f}  {result.status:<10} {len(colors):>6} colors, center {colors[len(colors) // 2]:+.3f}")
    if render:
        render_frame(ordinal)  # the first step after a change uploads; keep the second
        frames.append(save_png(render_frame(ordinal), OUTPUT_DIR / f"04_ripple_t{t:.1f}.png"))

# %% [markdown]
# The center value follows the sine wave through time, and each step the
# Faces operator ran because its dataset had been replaced. Update once more
# without publishing anything and it does not run: the source reports the
# same dataset, so the committed product is reused.

# %%
ordinal, report = evaluate(1.0, ordinal)
print("Without a new publish:", report.operators[0].status, "-", report.operators[0].reason)

# %% [markdown]
# ## Look at it
#
# One frame per step. The rings travel outward and the coloring follows the
# height; the geometry itself does not move, because Faces only extracts
# the surface. Tutorial 5 writes the operator that makes it move.

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
# A dataset built in memory with a Warp SimData data model was published to
# a Memory Data Source under a key, and a stock recipe selected it by that
# key exactly as it selects a prim in a file. Each publish marked the data
# changed, so the loop reran the recipe every step and reused it when
# nothing new arrived. The pipeline did not notice that no file was
# involved, which is what lets the same recipes and the same session serve
# a file viewer and an in situ coupling.
#
# The plate and its ripple live on in `ripple.py` beside this notebook, so
# tutorial 5 can pick them up and add the missing piece: an operator of
# your own that moves the points.
