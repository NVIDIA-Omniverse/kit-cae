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
# # 1. See it working
#
# This tutorial runs the whole OVCAE pipeline once, as a black box: open a
# scene that already describes a CAE dataset and visualization recipes,
# evaluate the recipes, render the result with OVRTX, and look at the picture.
# Nothing is authored here. The tutorials that follow open the boxes one at
# a time: how the dataset gets into the scene, how its arrays reach the GPU,
# how recipes are evaluated, and how to extend each step.
#
# This tutorial requires an RTX-capable NVIDIA GPU. The scene and dataset
# are in `tutorials/data`; run `git lfs pull` after cloning to fetch the data
# (see the README).

# %% [markdown]
# ## The scene
#
# The scene is an external-aerodynamics result: a car body in a fluid domain,
# with a set of recipes already authored on it. It is an ordinary OpenUSD
# layer, `aero_streamlines.usda` in the data directory. Three kinds of prims
# matter here:
#
# - **A dataset holder.** `/World/auto_aero_solver_result` payloads a CGNS
#   file. The CAE OpenUSD plug-ins compose that file into dataset prims
#   underneath it, one per zone and boundary, when usd-core opens the layer.
# - **Recipes.** The prims under `/World/CAE` are ordinary `Mesh` and
#   `BasisCurves` prims with OVCAE's operator API schemas applied. Five
#   `Faces` recipes extract the body, wheel, and seal surfaces, and one
#   `Streamlines` recipe traces velocity through the fluid domain from seeds
#   on a small sphere. Each recipe names the dataset it reads and carries its
#   settings as attributes; the body and the streamlines also bind materials
#   that color them by a field.
# - **A render rig.** Lights under `/Environment`, the camera at
#   `/World/Camera`, and a `RenderProduct` under `/Render` that tells OVRTX
#   which camera to render, at what resolution, and which output to keep.
#
# Open the file alongside this notebook to see them.

# %% [markdown]
# ## Initialize the runtimes
#
# Three runtimes take part, and the order in which they initialize matters
# because each one fixes some registry the moment it is first used:
#
# 1. `ovrtx.register_schema_paths()` first, so OVRTX's own schemas are known
#    before OVStage performs its first schema read.
# 2. `ovcae.setup()` next. Tutorial 0 explained what it registers: readers
#    and file formats with usd-core, CAE schemas with OVStage, and the material
#    library for the renderer. It has to run before anything in this process
#    opens a usd-core stage, which the data source below does.
# 3. Only then create the renderer, the stage, and the data source.
#
# The renderer is pinned to the GPU selected in tutorial 0, keeping
# evaluation and rendering on the same device.

# %%
import ovrtx

ovrtx.register_schema_paths()

import numpy as np
import ovstage
import warp as wp
from support import asset, cuda_gpu_index, tutorial_device

import ovcae
import ovcae.readers

ovcae.setup()
# Warp otherwise reports every kernel module it loads; tutorial 0 already showed the device banner.
wp.config.log_level = wp.LOG_WARNING

device = tutorial_device()
renderer = ovrtx.Renderer(ovrtx.RendererConfig(active_cuda_gpus=cuda_gpu_index(device)))
print("Compute device :", device)
print("Render GPU     :", cuda_gpu_index(device) or "OVRTX default")

# %% [markdown]
# ## Populate the stage
#
# The application owns an **OVStage**: the in-memory scene database that OVCAE
# writes products into and OVRTX renders from. Populating it from the layer
# copies every prim in, including the recipe prims with their settings, the
# materials, the lights, and the render rig. The dataset holder arrives as a
# plain prim: OVStage has no file-format plug-ins, so the CGNS payload is not
# composed here.
#
# Every write to an OVStage happens at an **ordinal**, a monotonically
# increasing version number. Population writes at ordinal 1. Advancing the
# **write floor** to 1 seals that version so readers can trust it will not
# change; nothing below the floor is ever modified again.

# %%
scene_path = asset("aero_streamlines.usda")
stage = ovstage.Stage("tutorial-01")
ovstage.population.open_usd_with_desc(
    stage,
    str(scene_path),
    ordinal=1,
    time_code=0.0,
    descs=ovcae.ovstage_population_desc(domains=ovstage.PopulationDomain.ALL),
)
stage.advance_write_floor(1).wait()
print("Populated", scene_path.name, "at ordinal 1")

# %% [markdown]
# ## Open the data source
#
# The dataset arrays have to come from somewhere, and that is the **Data
# Source**. `UsdDataSource` opens the same layer with usd-core, where the CAE
# plug-ins do compose the CGNS payload, and serves arrays from the composed
# dataset prims on request. It must see the same layers OVStage was populated
# from, because recipe prims refer to dataset prims by path and the two views
# have to agree on what those paths mean.

# %%
data_source = ovcae.readers.UsdDataSource.open(scene_path)
print("Data source resolves the holder:", data_source.resolves("/World/auto_aero_solver_result"))

# %% [markdown]
# ## Evaluate the recipes
#
# A **Session** binds operators to the recipe prims it finds on the stage and
# evaluates them through one worker. An **Update** reads the committed inputs
# at one ordinal and publishes products at the next: here the recipe settings
# committed at ordinal 1 become geometry at ordinal 2. The session never
# advances the write floor itself; that stays with the application, which
# seals ordinal 2 once the report says the update is committable.
#
# The **frame data** handed to the update is a lease on the data source at a
# point in scientific time. This dataset is a steady-state result, so time
# zero is the only time there is. The first update reads the whole CGNS file
# from disk, so expect this cell to take a few seconds.

# %%
session = ovcae.Session(stage, device=device)
with data_source.at_time(0.0) as frame_data:
    update = session.update(input_ordinal=1, output_ordinal=2, frame_data=frame_data)
    report = update.wait()

for result in report.operators:
    print(f"{result.operator_path:<44} {result.status:<10} {result.reason or ''}")
for diagnostic in report.diagnostics:
    print(f"[{diagnostic.severity}] {diagnostic.operator_path}: {diagnostic.message}")
assert report.committable, "the update produced nothing that can be committed"

stage.advance_write_floor(2).wait()
print("Committed ordinal 2")

# %% [markdown]
# The report has one line per recipe on the stage. Every one **succeeded**:
# its operator ran and published geometry into the recipe prim itself. On a
# first update everything is new, so everything runs. Later updates only rerun
# what changed, and recipes that are hidden are skipped outright; tutorial 2
# uses this report to show the first, editing one setting on a live scene.

# %% [markdown]
# ## Render
#
# Attaching the renderer to the stage lets OVRTX read the committed prims,
# including the surfaces and streamlines the update just published. Each
# `step` renders the render products you name at a given ordinal and returns
# their frames. The first step compiles shaders and uploads geometry, so this
# cell can take a minute on a cold cache; it steps twice and keeps the second
# frame, which is the one with everything in place.
#
# The frame's color output is mapped to host memory just long enough to copy
# it into a NumPy array.

# %%
RENDER_PRODUCT = "/Render/Camera"

renderer.attach_ovstage(stage)


def render_frame(ordinal: int) -> np.ndarray:
    products = renderer.step(render_products={RENDER_PRODUCT}, delta_time=0.0, ordinal=ordinal)
    frame = products[RENDER_PRODUCT].frames[0]
    mapped = frame.render_vars[f"{RENDER_PRODUCT}/LdrColor"].map(device=ovrtx.Device.CPU)
    try:
        return np.from_dlpack(mapped).copy()
    finally:
        mapped.unmap()


render_frame(2)
pixels = render_frame(2)
print("Rendered", pixels.shape[1], "x", pixels.shape[0], "RGBA pixels")

# %% [markdown]
# ## Look at it
#
# The pixels are an ordinary array, so save them as a PNG under `build/` and
# show that file. Nothing here is specific to OVCAE or OVRTX.

# %%
from support import OUTPUT_DIR, save_png

png_path = save_png(pixels, OUTPUT_DIR / "01_see_it_working.png")
print("Saved", png_path)

if "get_ipython" in globals():
    from IPython.display import Image, display

    display(Image(filename=str(png_path)))

# %% [markdown]
# The body is colored by surface pressure and the streamlines by velocity,
# through the colormap materials the recipes bind. Everything you see traces
# back to three things: a dataset in a file, recipe prims describing what to
# compute from it, and one update.

# %% [markdown]
# ## Shut down
#
# Close the session and data source, detach the renderer, then destroy the
# stage and renderer.

# %%
session.close()
data_source.close()
renderer.detach_ovstage()
stage.destroy()
renderer.destroy()
print("Done")

# %% [markdown]
# ## What just happened
#
# | Step | Who did it | Where to read more |
# | --- | --- | --- |
# | Recipes were evaluated and products published | OVCAE Session and operators | Tutorial 2, on a live scene |
# | The CGNS file became dataset prims | CAE OpenUSD plug-ins inside usd-core | Tutorial 3; appendix A1 without OVCAE |
# | Arrays were loaded onto the device | Warp SimData, through the Data Source | Tutorial 3; appendix A1 without OVCAE |
# | Somebody wrote the recipes | `ovcae.authoring` | Tutorial 3, which authors a scene like this one |
# | Data can come from memory, and new operators can be plugged in | You | Tutorials 4 and 5 |
