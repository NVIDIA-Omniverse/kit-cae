# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""The smallest OVUI desktop window on OVCAE: open data, move one slider, save, reopen.

Given a CGNS file, the window authors a scene under ``build/ovui-minimal``
(or ``--scene``), evaluates the recipes, and shows the result in an OVUI viewport rendered by OVRTX. A slider
moves the iso-surface, a checkbox shows the boundaries, and Save writes the
edits and the camera back into the scene. Given that saved scene instead, the
window reproduces the picture. Nothing here imports Kit.
"""

from __future__ import annotations

import argparse
import asyncio
import ctypes
import logging
import os
import time
from importlib.metadata import distribution
from pathlib import Path

try:
    from ovui_data_adapters.common import RendererAdapter
except ImportError:  # the tests import this module for its argument parsing without OVUI installed
    RendererAdapter = object

EVALUATION_BUDGET = 0.008  # seconds per frame the loop blocks so the OVCAE worker gets the GIL
SCENE_DIR = Path(__file__).resolve().parents[2] / "build" / "ovui-minimal"
EXAMPLE_DATA = Path(__file__).resolve().parents[2] / "tests" / "data" / "StaticMixer.cgns"
LOGGER = logging.getLogger("ovui_minimal")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "source",
        nargs="?",
        type=Path,
        default=EXAMPLE_DATA,
        help="a CGNS file to author a scene from, or a .usda scene this example saved (default: the static mixer)",
    )
    parser.add_argument("--scene", type=Path, help="where to write the authored scene (default: build/ovui-minimal/)")
    parser.add_argument("--dataset", help="dataset to contour when authoring (default: the first in the file)")
    parser.add_argument(
        "--field", help="field to contour when authoring (default: Temperature if the dataset has it, else its first)"
    )
    parser.add_argument(
        "--iso-value", type=float, help="starting iso value; for a reopened scene, applied as an edit on launch"
    )
    parser.add_argument("--device", default="cuda:0", help="Warp compute device; also pins OVRTX to that GPU")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--headless", action="store_true", help="use OVUI's offscreen Vulkan backend")
    parser.add_argument("--frames", type=int, default=0, help="exit after N rendered frames (0 runs until closed)")
    parser.add_argument("--save", action="store_true", help="save the scene on exit, as the Save button does")
    parser.add_argument("--screenshot", type=Path, help="save the whole window as PNG on exit; requires --frames")
    parser.add_argument("--frame-dump", type=Path, help="save the last rendered viewport frame as .npy on exit")
    args = parser.parse_args(argv)
    if min(args.width, args.height) <= 0 or args.frames < 0:
        parser.error("width and height must be positive; frames must be nonnegative")
    if (args.screenshot or args.headless) and not args.frames:
        parser.error("--screenshot and --headless require a positive --frames count; nothing else ends a headless run")
    if is_scene(args.source) and (args.scene or args.dataset or args.field):
        parser.error("--scene, --dataset, and --field apply when authoring from a data file, not to a saved scene")
    return args


def is_scene(path: Path) -> bool:
    return path.suffix.lower() in {".usda", ".usd", ".usdc"}


def scene_for(args) -> Path:
    """Author a scene from a data file, or accept a scene this example saved."""
    from workflow import author_scene

    source = args.source.expanduser().resolve()
    if is_scene(source):
        return source
    scene = args.scene or SCENE_DIR / (source.stem + ".usda")
    authored = author_scene(source, scene, dataset=args.dataset, field=args.field, iso_value=args.iso_value)
    low, high = authored.field_range
    LOGGER.info(
        "Authored %s: %s on %s in [%g, %g], iso value %g",
        scene,
        authored.field,
        authored.dataset,
        low,
        high,
        authored.iso_value,
    )
    return scene


class Window:
    """One viewport, one slider, one checkbox, one Save button, on one UI thread."""

    def __init__(self, workflow, args) -> None:
        self.workflow = workflow
        self.args = args
        self.rendered_frames = 0
        self.viewport = self.controls = self.status = self.slider = self.checkbox = None
        self.main_window = self.dockspace = self.adapter = None

    def run(self) -> None:
        import omni.ui as ui
        from ovui_widgets.common.settings import Settings
        from ovui_widgets.viewport import ViewportWidget

        ui.init("OVCAE minimal", width=self.args.width, height=self.args.height, max_fps=0)
        try:
            Settings.set_instance(Settings())
            self.main_window = ui.MainWindow()
            self.dockspace = ui.DockSpace(None)
            self.adapter = Adapter(self.workflow)
            self.viewport = ViewportWidget(
                renderer=self.adapter,
                viewport_id="cae",
                window_kwargs={"identifier": "cae_viewport"},
                chrome_options={
                    "show_toolbar": False,
                    "show_settings_button": False,
                    "show_livestream_overlay": False,
                    "show_anchored_panels": False,
                },
            )
            self.viewport.set_scene_name(self.workflow.scene_path.name)
            self.controls = ui.Window("Workflow", width=340, height=240)
            with self.controls.frame:
                self._build_controls()
            asyncio.run(self._run())
            if self.args.screenshot:
                from omni.ui.testing import capture_screenshot

                if not self.rendered_frames or not capture_screenshot(str(self.args.screenshot)):
                    raise RuntimeError(f"could not capture {self.args.screenshot}")
        finally:
            # Release every native window before OVUI shuts down; a DockSpace
            # outliving ui.shutdown() crashes at interpreter exit.
            if self.viewport is not None:
                self.viewport.destroy()
            if self.controls is not None:
                self.controls.destroy()
            self.dockspace = self.main_window = None
            Settings.set_instance(None)
            ui.shutdown()

    def _build_controls(self) -> None:
        import omni.ui as ui

        low, high = self.workflow.field_range
        with ui.VStack(spacing=6, style={"margin": 6}):
            ui.Label(self.workflow.scene_path.name, height=20, tooltip=str(self.workflow.scene_path))
            ui.Label(f"Iso value of {self.workflow.field}", height=20)
            self.slider = ui.FloatSlider(min=low, max=high, height=24)
            self.slider.model.set_value(self.workflow.iso_value)
            self.slider.model.add_value_changed_fn(lambda model: self.workflow.set_iso_value(model.as_float))
            with ui.HStack(height=24, spacing=6):
                self.checkbox = ui.CheckBox(width=24)
                self.checkbox.model.set_value(self.workflow.boundaries_visible)
                self.checkbox.model.add_value_changed_fn(lambda model: self.workflow.show_boundaries(model.as_bool))
                ui.Label("Show boundaries")
            ui.Button("Save", height=28, clicked_fn=self.save)
            ui.Label("Alt + drag: orbit · middle drag: pan · scroll: zoom", height=18)
            self.status = ui.Label("Ready", height=40, word_wrap=True)

    def save(self) -> None:
        try:
            path = self.workflow.save()
        except Exception as exc:  # the window must survive a failed save
            LOGGER.exception("Could not save")
            self.status.text = f"Could not save: {exc}"
        else:
            self.status.text = f"Saved {path.name}; reopen it with: app.py {path}"

    async def _run(self) -> None:
        import omni.ui as ui

        pump = asyncio.create_task(ui.run_async())
        frames = asyncio.create_task(self._frames())
        try:
            done, _ = await asyncio.wait((pump, frames), return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
        finally:
            for task in (frames, pump):
                task.cancel()
            await asyncio.gather(frames, pump, return_exceptions=True)

    async def _frames(self) -> None:
        import omni.ui as ui
        from ovui_data_adapters.common import BoundCameraPose

        await ui.next_frame()
        self.viewport.window.dock_in(ui.Workspace.get_window("DockSpace"), ui.DockPosition.SAME)
        self.controls.dock_in(self.viewport.window, ui.DockPosition.RIGHT, ratio=0.28)
        await ui.next_frame()
        pose = self.workflow.saved_pose() or self.workflow.overview_pose()
        if not self.viewport.apply_camera_pose(BoundCameraPose(**pose)):
            raise RuntimeError("OVUI rejected the scene camera")
        previous = time.perf_counter()
        while True:
            self._pump()
            now = time.perf_counter()
            self.viewport.update(now - previous)
            previous = now
            if self.viewport.render(0.0):
                self.rendered_frames += 1
            if self.workflow.evaluating:
                # The UI pump never idles; only a blocking wait cedes the GIL to the OVCAE worker.
                self.workflow.await_evaluation(EVALUATION_BUDGET)
            await ui.next_frame()
            if self.args.frames and self.rendered_frames >= self.args.frames:
                return

    def _pump(self) -> None:
        try:
            report = self.workflow.pump()
        except Exception as exc:
            LOGGER.exception("OVCAE update failed")
            self.status.text = str(exc)
            return
        if self.workflow.last_error:
            self.status.text, self.workflow.last_error = f"Update failed: {self.workflow.last_error}", None
        elif report is not None:
            changed = [
                f"{r.operator_path.rsplit('/', 1)[-1]}: {r.status}"
                for r in report.operators
                if r.status not in ("skipped", "unchanged")
            ]
            self.status.text = ("Updated " + ", ".join(changed)) if changed else "Updated; nothing to recompute"
        elif self.workflow.evaluating:
            self.status.text = "Evaluating…"


class Adapter(RendererAdapter):
    """Present the workflow's OVStage to OVUI's viewport; the workflow owns the renderer."""

    def __init__(self, workflow) -> None:
        self.workflow = workflow

    def load_stage(self, stage) -> None:
        if stage is not self.workflow.stage:
            raise ValueError("the viewport cannot replace the workflow's Stage")

    def render_frame(self, width, height, view_matrix, proj_matrix):
        return self.workflow.render(
            int(width or self.workflow.width), int(height or self.workflow.height), view_matrix, proj_matrix
        )

    def set_resolution(self, width, height) -> None:
        """The size is committed together with the camera on the next frame."""

    # This example has no selection, so the viewport's picking hooks do nothing.
    def pick(self, x, y, callback, query_name) -> None:
        pass

    def pick_rect(self, x0, y0, x1, y1, callback) -> None:
        pass

    def cancel_pick(self, query_name) -> None:
        pass

    def set_selection_highlight(self, paths) -> None:
        pass

    def shutdown(self) -> None:
        """The workflow is closed by the application, not the viewport."""


def main(argv=None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.headless:
        os.environ["OMNIUI_HEADLESS"] = "1"
        os.environ["OMNIUI_BACKEND"] = "vulkan"
    # Load the wheel's CUDA runtime by absolute path before OVUI's native extension.
    runtime = distribution("nvidia-cuda-runtime-cu12").locate_file("nvidia/cuda_runtime/lib/libcudart.so.12")
    cuda_runtime = ctypes.CDLL(str(runtime), mode=ctypes.RTLD_GLOBAL)
    from workflow import Workflow

    scene = scene_for(args)
    with Workflow(scene, device=args.device) as workflow:
        if args.iso_value is not None and is_scene(args.source):
            workflow.set_iso_value(args.iso_value)
        window = Window(workflow, args)
        window.run()
        if args.frame_dump:
            import numpy as np

            if workflow.last_frame is None:
                raise RuntimeError("no frame was rendered")
            np.save(args.frame_dump, workflow.last_frame)
        if args.save:
            workflow.save()
    del cuda_runtime


if __name__ == "__main__":
    main()
