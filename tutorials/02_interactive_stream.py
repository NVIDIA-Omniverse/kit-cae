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
# # 2. Make it live
#
# Tutorial 1 ran the pipeline once and saved a picture. This tutorial keeps
# it running: OVRTX renders continuously, OVStream encodes each frame and
# sends it to this notebook page over WebRTC, and the reader orbits the
# camera with the mouse and moves an iso-surface with a slider. In doing so
# it introduces the loop every interactive OVCAE application is built
# around: one owner of the stage, writing edits at new ordinals, running an
# update for recipe edits, committing, and only then rendering. OVCAE ships
# that loop as `ovcae.app.UpdateLoop`, and the
# render thread and browser stream around it as `ovcae.stream.LiveView`;
# this tutorial uses both and explains what they do.
#
# The streaming half follows the `ovrtx_stream` example in the
# [ovstream repository](https://github.com/NVIDIA-Omniverse/ovstream)
# almost line for line; the browser client is that repository's stock
# `webrtc_client` page reduced to a cell. Nothing here is specific to CAE
# data until the slider.
#
# You need an RTX-capable NVIDIA GPU and a browser that can reach it. WebRTC
# carries video over UDP, so an SSH port forward is not enough when JupyterLab
# runs on a remote machine: the browser must reach the GPU host's signaling
# port (49100 below) and UDP media port directly, or through a TURN server.
# On the machine itself, or on a lab network, it just works. Set
# `OVCAE_TUTORIAL_STREAM_HOST` if the browser should connect to a different
# host name than the one the notebook page was loaded from.

# %% [markdown]
# ## The scene
#
# `aero_isosurface.usda` is tutorial 1's aerodynamics scene with the
# streamlines replaced by one `IsoSurface` recipe over the fluid domain. It
# contours Pressure at `cae:viz:isoSurface:isoValue`, and colors the
# resulting surface by the vertical velocity component through a rainbow
# colormap, so the surface shows where the air is climbing over the body
# and where it is falling back. The five body and wheel surfaces, camera,
# lights, and render product are unchanged.
#
# The iso-surface binds a material that is not in OVCAE's library:
# `aero_glass.mdl`, a file beside the scene, defines a colormapped thin
# glass with a glossy coat and partial cutout opacity, so the body stays
# visible through the surface. Its shader prim references the module by a
# path relative to the layer and reads the same `colors` primvar the
# library materials do. Nothing in OVCAE knows about it; the renderer
# resolves it like any other MDL module. A scene can bring its own look.

# %% [markdown]
# ## Initialize and evaluate once
#
# The same steps as tutorial 1, compressed: register OVRTX, set up OVCAE,
# create the renderer pinned to one GPU, populate the stage at ordinal 1,
# and open the data source. New is the last line: instead of creating a
# Session and calling `update` on it, the tutorial creates an
# `UpdateLoop`, which creates the Session, and asks it for the first
# update. `loop.update()` with no edits runs one update from the committed
# ordinal, waits for it, commits its products, and returns the report.

# %%
import ovrtx

ovrtx.register_schema_paths()

import ovstage
import warp as wp
from support import asset, cuda_gpu_index, tutorial_device

import ovcae
import ovcae.readers
from ovcae.app import UpdateLoop

ovcae.setup()
wp.config.log_level = wp.LOG_WARNING

device = tutorial_device()
gpu_index = cuda_gpu_index(device) or "0"
renderer = ovrtx.Renderer(ovrtx.RendererConfig(active_cuda_gpus=gpu_index))

scene_path = asset("aero_isosurface.usda")
stage = ovstage.Stage("tutorial-02")
ovstage.population.open_usd_with_desc(
    stage,
    str(scene_path),
    ordinal=1,
    time_code=0.0,
    descs=ovcae.ovstage_population_desc(domains=ovstage.PopulationDomain.ALL),
)
stage.advance_write_floor(1).wait()

data_source = ovcae.readers.UsdDataSource.open(scene_path)
loop = UpdateLoop(stage, data_source, ordinal=1, device=device)

report = loop.update()
for result in report.operators:
    print(f"{result.operator_path:<44} {result.status:<10} {result.reason or ''}")
print("Committed ordinal", loop.ordinal)

# %% [markdown]
# ## One owner for the stage
#
# From here on two things want to change the stage: the camera, driven by
# mouse events arriving on OVStream's callback thread, and the iso value,
# driven by a slider callback on the kernel's main thread. Neither may touch
# the stage directly, for two reasons that tutorial 1 only hinted at:
#
# - OVRTX cannot step while the stage holds writes above the write floor,
#   and a Session update writes its products above the floor until it
#   finishes. Rendering and updating must therefore take turns.
# - A Session allows one update in flight at a time, and the inputs it reads
#   must stay stable until it is done.
#
# The simplest arrangement that satisfies both is a single thread that owns
# every stage write and does the same things for each batch of edits: write
# them at a new ordinal, advance the write floor, run an update, commit it,
# and render at the committed ordinal. That is what `UpdateLoop` does.
# `request` buffers recipe edits as `{(prim path, property): value}`, latest
# value per key; `update` writes and evaluates a batch synchronously; `pump`,
# called once per frame by a render loop, submits the buffered batch and
# commits a finished update without blocking. The loop's `ordinal` is the
# one a renderer steps at.
#
# Not every edit needs an update. The camera's transform is presentation:
# no operator reads it, so recomputing would be wasted work, and the
# streamline recipes in particular re-integrate on every update because
# their seeds carry no identity to compare. The application knows which of
# its edits are presentation, so it says so: `write` puts them on the stage
# at a new ordinal and advances the floor, and nothing recomputes. The cell
# below does both, and the report of the recipe edit shows the point of the
# ordinal model: the five body and wheel recipes come back **unchanged**
# because none of their inputs moved, and only the iso-surface recomputes.

# %%
import numpy as np

CAMERA = "/World/Camera"
ISO_SURFACE = "/World/CAE/IsoSurface_Fluid_Domain"
ISO_VALUE = "cae:viz:isoSurface:isoValue"

# Preserve the camera rotation so it keeps looking at the scene.
nudged = np.array(loop.value(CAMERA, "omni:xform"), dtype=np.float64, copy=True).reshape(4, 4)
nudged[3, 2] += 0.1  # raise the saved eye slightly
loop.write({(CAMERA, "omni:xform"): nudged})
print("After the camera write: ordinal", loop.ordinal, "and no update ran")

report = loop.update({(ISO_SURFACE, ISO_VALUE): -300.0})
print(f"Update {report.input_ordinal} -> {report.output_ordinal}")
for result in report.operators:
    print(f"{result.operator_path:<44} {result.status:<10} {result.reason or ''}")

# %% [markdown]
# ## The render thread and the stream
#
# `LiveView` puts the loop on a background thread and streams what it
# renders. Each iteration of its thread empties two mailboxes, hands the
# presentation edits to `loop.write` and the recipe edits to `loop.update`
# as one batch each, renders at the committed ordinal, and passes the frame
# to OVStream. OVRTX writes each frame's `LdrColor` output
# as RGBA8 in GPU memory; OVStream wants BGRA8, so the view copies the
# frame into a long-lived Warp buffer, swaps red and blue in place with a
# tiny kernel, and hands the encoder a CUDA event to wait on instead of
# synchronizing the host every frame, exactly as `ovrtx_stream` does.
#
# The other half is input. OVStream delivers the browser's mouse events to
# the view, which drives an orbit camera: a left drag changes yaw and pitch
# around a fixed target, the wheel changes distance. The pose goes to the
# file's `/World/Camera` as `omni:xform` through the presentation mailbox,
# so no update follows a drag. The initial eye and target are the ones the
# file was saved with.
#
# Two OVStream settings matter on a machine with more than one GPU:
# `cuda_device` pins the video encoder to the GPU the frames live on, and
# `cuda_context` hands it the CUDA context those frames were produced in.
# The view sets both from `gpu_index`. The server has to be started with the
# render product's exact resolution, so the view renders one frame up front
# to learn it. Set `OVCAE_TUTORIAL_STREAM_HOST` if the browser should
# connect to a different host name than the notebook page's; the view
# passes it to the server and into the client page.

# %%
from support import stream_host

from ovcae.stream import LiveView

RENDER_PRODUCT = "/Render/Camera"
TARGET = (1.1769, 0.2626, 0.2201)
INITIAL_EYE = (-4.7114, -3.0067, 2.0345)

renderer.attach_ovstage(stage)
view = LiveView(
    loop,
    renderer,
    render_product=RENDER_PRODUCT,
    camera=CAMERA,
    eye=INITIAL_EYE,
    target=TARGET,
    gpu_index=gpu_index,
    public_host=stream_host(),
)
view.start()
print("Streaming", view.width, "x", view.height, "on signaling port", view.signal_port)

# %% [markdown]
# ## Play with it
#
# The cell below puts the two controls together: the live view and a
# slider for the iso value.
#
# The view is the ovstream browser client. It loads the client library from
# the ovstream repository at the pinned release, connects to the signaling
# port, and shows the video. Once the status line says connected, click the
# video to give it control, then left-drag to orbit and use the wheel to
# zoom. Every drag becomes mouse events on the server, a new camera matrix
# in the mailbox, and a new ordinal on the stage. While the video has
# control it also receives the keyboard, so the notebook's own shortcuts
# such as Shift+Enter do nothing; move the pointer off the video, or click
# anywhere else, to hand the keyboard back.
#
# The slider requests a new `isoValue` on the iso-surface recipe. Its
# callback runs on the kernel thread and only touches the mailbox; the
# render thread picks the value up, and the loop writes it and runs the
# update. The slider sends every value while you drag, and the mailbox keeps
# only the latest value per setting, skipping intermediate values while an
# update is running. Slide it and watch the surface follow in the video.

# %%
import ipywidgets as widgets
from IPython.display import HTML, display

iso_slider = widgets.FloatSlider(
    value=-300.0,
    min=-800.0,
    max=550.0,
    step=10.0,
    description="Pressure",
    continuous_update=True,  # every value during a drag; the mailbox keeps only the latest
    readout_format=".0f",
)
iso_slider.observe(lambda change: view.request(ISO_SURFACE, ISO_VALUE, change["new"]), names="value")

display(HTML(view.html()))
display(iso_slider)

# %% [markdown]
# ## What an edit costs
#
# Each slider move ends in an update report, which the view exposes as
# `last_report`. The cell sets the slider from code so it also works when
# nobody is dragging it; setting the value does exactly what a drag does.
# `wait_for_update` blocks until the render thread has written and
# evaluated every request made so far. The timing line shows how cheap a
# single-recipe update is against the several seconds the first one took.

# %%
import time

started = time.perf_counter()
iso_slider.value = -400.0  # the same thing dragging the slider does
report = view.wait_for_update()
print(f"Update {report.input_ordinal} -> {report.output_ordinal} after {time.perf_counter() - started:.2f} s")
for result in report.operators:
    print(f"{result.operator_path:<44} {result.status:<10} {result.reason or ''}")

# %% [markdown]
# Wait until the video shows the updated scene, then save the latest frame.
# The snapshot does not wait for rendering to finish.

# %%
from support import OUTPUT_DIR, save_png

png_path = save_png(view.snapshot(), OUTPUT_DIR / "02_interactive_stream.png")
print("Saved", png_path)
if "get_ipython" in globals():
    from IPython.display import Image

    display(Image(filename=str(png_path)))

# %% [markdown]
# ## Shut down
#
# Stop the view first, so no more writes or renders happen, then release
# in reverse order of creation as before. The view stops its thread and
# then the stream server; the loop closes the Session it created; the
# application still owns the data source, the renderer, and the stage.

# %%
view.stop()
loop.close()
data_source.close()
renderer.detach_ovstage()
stage.destroy()
renderer.destroy()
print("Done")

# %% [markdown]
# ## What just happened
#
# The pipeline from tutorial 1 became an application: an `UpdateLoop` owns
# the Session and the ordinal bookkeeping, edits arrive as batches, the
# application writes camera changes as presentation with no update, and
# requests recipe changes, whose update report shows exactly what
# recomputed. A
# `LiveView` put that loop on a render thread with a mailbox in front of
# it and OVStream behind it. Everything the CAE Viewer example does on the
# desktop is the same loop with omni.ui instead of a browser, and the
# appendices put a live view on scenes of their own with the same two
# classes.
