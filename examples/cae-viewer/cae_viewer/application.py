# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Standalone CAE viewport with picking, gizmos, and in-view status."""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import ExitStack
from pathlib import Path

import omni.ui as ui
from ovui_data_adapters.common import BoundCameraPose
from ovui_data_adapters.services.undo import UndoManager
from ovui_widgets.common import scheduler
from ovui_widgets.common.selection import SelectionBus
from ovui_widgets.common.settings import Settings
from ovui_widgets.viewport import ViewportSurface

from .pipeline import SCENE, RenderPipeline
from .stage import ViewportStage
from .transforms import SceneTransforms
from .viewport import PipelineAdapter

TOOLS = (("Move", "translate"), ("Rotate", "rotate"), ("Scale", "scale"))
EVALUATION_BUDGET = 0.008  # seconds per frame the loop blocks on an in-flight OVCae Update
SUMMARY_PERIOD = 1.0  # seconds between frame-loop summaries in the trace log

LOGGER = logging.getLogger(__name__)


class WidgetCallback(scheduler.CallbackHandle):
    """Run OVUI's deferred callbacks after a real UI frame, with cancellable ownership."""

    def __init__(self, delay, callback):
        super().__init__(time.monotonic() + delay, callback)
        self.task = asyncio.create_task(self._run(delay))

    async def _run(self, delay):
        if delay > 0:
            await asyncio.sleep(delay)
        await ui.next_frame()
        if not self.is_cancelled:
            callback, self._callback = self._callback, None
            callback()

    def cancel(self):
        super().cancel()
        self.task.cancel()


class RenderClock:
    """Share one render budget between frame and native resize callbacks."""

    def __init__(self, fps):
        if fps < 0:
            raise ValueError("fps must be nonnegative")
        self.period, self.last = (1.0 / fps if fps else 0.0), None

    def should_render(self, now):
        if self.last is None:
            return 0.0
        elapsed = now - self.last
        return elapsed if elapsed >= self.period else None

    def commit(self, now):
        self.last = now


class Viewer:
    """Own widgets and native resources on one UI thread.

    Widget callbacks queue edits or a file-open request. The frame loop performs
    all Stage commits and evaluates CAE before rendering. A failed open leaves
    the previous pipeline and its unsaved edits intact. No source file is saved.
    """

    def __init__(self, args):
        self.args = args
        self.pending = {}
        self.pending_file = None
        self.running = True
        self.pipeline = self.viewport = self.dialog = None
        self.rendered_frames = 0
        self.clock = RenderClock(args.fps)
        self.selection = SelectionBus()
        self.selected_paths = ()
        self._widget_callbacks = set()
        self.error = None
        self._summary_state = None

    def _new_pipeline(self, filename):
        return RenderPipeline(
            scene=filename, width=self.args.width, height=self.args.height, fps=self.args.fps, device=self.args.device
        ).open()

    def run(self):
        """Release widgets/borrowers before the currently loaded pipeline and UI."""
        with ExitStack() as resources:
            resources.callback(ui.shutdown)
            self.pipeline = self._new_pipeline(self.args.scene)
            resources.callback(lambda: self.pipeline.close())
            ui.init("CAE Viewer", width=self.args.width, height=self.args.height, max_fps=self.args.fps)
            Settings.set_instance(Settings())
            resources.callback(Settings.set_instance, None)
            self.main_window = ui.MainWindow()
            resources.callback(setattr, self, "main_window", None)
            self.dockspace = ui.DockSpace(None)
            resources.callback(setattr, self, "dockspace", None)
            self.adapter = PipelineAdapter(self.pipeline)
            resources.callback(self.adapter.shutdown)
            self.stage_adapter = ViewportStage(self.pipeline)
            self.window = ui.Window("CAE Viewer", identifier="cae_viewport")
            resources.callback(self.window.destroy)
            self.viewport = ViewportSurface(
                renderer=self.adapter,
                bus=self.selection,
                stage_adapter_provider=lambda: self.stage_adapter,
                viewport_id="cae",
                chrome_options={
                    "show_toolbar": False,
                    "show_settings_button": False,
                    "show_text_hud": True,
                    "show_livestream_overlay": False,
                    "show_anchored_panels": False,
                },
            )
            resources.callback(self.viewport.destroy)
            self.window.frame.set_build_fn(self._build_view)
            self.viewport.set_shared_render_clock(self.clock)
            self.transforms = SceneTransforms(self.pipeline, self._request_edit, self._pending_value)
            self.undo = UndoManager()
            self._attach_transforms()
            resources.callback(self._destroy_dialog)
            selection_subscription = self.selection.subscribe(self._selection_changed)
            resources.callback(selection_subscription.cancel)
            asyncio.run(self._run())
            if self.args.screenshot:
                from omni.ui.testing import capture_screenshot

                if not self.rendered_frames or not capture_screenshot(str(self.args.screenshot)):
                    raise RuntimeError(f"could not capture {self.args.screenshot}")

    async def _run(self):
        """Drain both async tasks before any native ownership is unwound."""
        scheduler.set_call_later(self._call_later)
        pump, frames = asyncio.create_task(ui.run_async()), asyncio.create_task(self._prepare_frames())
        try:
            done, _ = await asyncio.wait((pump, frames), return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
        finally:
            scheduler.set_call_later(None)
            callbacks = tuple(self._widget_callbacks)
            for callback in callbacks:
                callback.cancel()
            for task in (frames, pump):
                task.cancel()
            await asyncio.gather(frames, pump, *(callback.task for callback in callbacks), return_exceptions=True)

    async def _prepare_frames(self):
        """Wait for native window construction before initializing its labels."""
        await ui.next_frame()
        self._scene_changed()
        await self._frames()

    def _attach_transforms(self):
        """Give OVUI's gizmo its transform, stage, and undo owners for the loaded scene."""
        self.viewport.attach_stage(self.transforms, self.stage_adapter, self.undo)

    def _request_edit(self, key, value):
        """The frame loop replaces the dict, so never bind a method of one instance."""
        self.pending[key] = value

    def _pending_value(self, key):
        return self.pending.get(key)

    def _call_later(self, delay, callback):
        handle = WidgetCallback(delay, callback)
        self._widget_callbacks.add(handle)
        handle.task.add_done_callback(lambda _: self._widget_callbacks.discard(handle))
        return handle

    async def _frames(self):
        await ui.next_frame()
        self.window.dock_in(ui.Workspace.get_window("DockSpace"), ui.DockPosition.SAME)
        await ui.next_frame()
        previous = summary_at = time.perf_counter()
        frames = awaited = 0.0
        while self.running:
            if self.pending_file is not None:
                filename, self.pending_file = self.pending_file, None
                self.status.text = f"Opening {filename.name}…"
                await ui.next_frame()
                self._open_scene(filename)
            self._pump_edits()
            now = time.perf_counter()
            self.viewport.update(now - previous)
            previous = now
            dt = self.clock.should_render(now)
            if dt is not None and self.viewport.render(dt):
                self.clock.commit(now)
                self.rendered_frames += 1
                self.adapter.complete_picks()
            if self.pipeline.evaluating:
                # The UI pump is a sibling task that never idles, so only a
                # blocking wait cedes the GIL to the OVCae worker.
                started = time.perf_counter()
                self.pipeline.await_evaluation(EVALUATION_BUDGET)
                awaited += time.perf_counter() - started
            await ui.next_frame()
            frames += 1
            if now - summary_at >= SUMMARY_PERIOD and LOGGER.isEnabledFor(logging.DEBUG):
                pipeline = self.pipeline
                LOGGER.debug(
                    "frames %.0f/s: rendered %d (%.1f ms each), cached %d, awaited %.0f ms, evaluating=%s",
                    frames / (now - summary_at),
                    pipeline.rendered_frames,
                    pipeline.render_time * 1000 / max(pipeline.rendered_frames, 1),
                    pipeline.cached_frames,
                    awaited * 1000,
                    pipeline.evaluating,
                )
                pipeline.rendered_frames = pipeline.cached_frames = 0
                pipeline.render_time = 0.0
                frames = awaited = 0.0
                summary_at = now
            if self.args.frames and self.rendered_frames >= self.args.frames:
                return

    def _pump_edits(self):
        """Buffer this frame's edits and advance CAE without ever blocking the frame.

        Gizmo previews land first so a commit in the same frame wins. The
        pipeline coalesces requests while an Update runs and submits them when
        it commits; widgets refresh whenever committed state moves.
        """
        previews = self.adapter.take_previews()
        if previews:
            LOGGER.debug("previews for %s", ", ".join(path.rsplit("/", 1)[-1] for path in previews))
        edits = {(path, "omni:xform"): matrix for path, matrix in previews.items()}
        if self.pending:
            pending, self.pending = self.pending, {}
            edits.update(pending)
        if edits:
            try:
                self.pipeline.request(edits)
            except Exception as exc:
                LOGGER.exception("Could not apply operator edits")
                self.error = str(exc)
                self.status.text = self.error
        try:
            report = self.pipeline.pump()
        except Exception as exc:
            LOGGER.exception("OVCae update failed")
            self.error = str(exc)
            self.status.text = self.error
            report = None
        if report is not None:
            diagnostics = [d.message for d in report.diagnostics if d.severity == "error"]
            self.error = " · ".join(diagnostics) or None
        self._refresh_reports()

    def _build_view(self):
        """Host the stock viewport surface; the overlay never owns scene reads."""
        self._summary_state = None
        with ui.VStack(spacing=0):
            with ui.HStack(height=34, spacing=6):
                ui.Button("Open USD…", clicked_fn=self.show_open_dialog, width=100)
                self.frame_button = ui.Button("Frame", clicked_fn=self.frame, width=120)
                ui.Button("Reset view", clicked_fn=self.reset_camera, width=95)
                ui.Spacer(width=12)
                for label, tool in TOOLS:
                    ui.Button(label, clicked_fn=lambda tool=tool: self.viewport.set_active_tool(tool), width=70)
                ui.Spacer()
                ui.Button("Undo", clicked_fn=self.undo.undo, width=65)
                ui.Button("Redo", clicked_fn=self.undo.redo, width=65)
            with ui.ZStack():
                self.viewport.build()
                with ui.VStack(opaque_for_mouse_events=False):
                    ui.Spacer(opaque_for_mouse_events=False)
                    with ui.ZStack(height=84, opaque_for_mouse_events=False):
                        ui.Rectangle(style={"background_color": 0xCC181818}, opaque_for_mouse_events=False)
                        with ui.HStack(height=84, opaque_for_mouse_events=False):
                            ui.Spacer(width=14)
                            with ui.VStack(spacing=3, opaque_for_mouse_events=False):
                                self.report_label = ui.Label("", height=20, opaque_for_mouse_events=False)
                                self.status = ui.Label("Ready", height=20, opaque_for_mouse_events=False)
                                ui.Label(
                                    "Click to select · Alt + drag to orbit · Middle drag to pan · Scroll to zoom",
                                    height=20,
                                    opaque_for_mouse_events=False,
                                )
                                ui.Spacer(height=10, opaque_for_mouse_events=False)

    def _refresh_reports(self):
        """Summarize selection and evaluation without inspecting native attributes."""
        state = (self.selected_paths, self.error, self.pipeline.evaluating)
        if state == self._summary_state:
            return
        self._summary_state = state
        if self.selected_paths:
            self.report_label.text = (
                self.selected_paths[0].rsplit("/", 1)[-1]
                if len(self.selected_paths) == 1
                else f"{len(self.selected_paths)} prims selected"
            )
        else:
            self.report_label.text = "Click an object to select it"
        self.report_label.tooltip = "\n".join(self.selected_paths)
        self.status.text = self.error or ("Updating…" if self.pipeline.evaluating else "Ready")
        self.status.tooltip = self.error or "Edits are in memory only"
        self.frame_button.text = "Frame selection" if self.selected_paths else "Frame scene"

    def _scene_changed(self):
        self.selected_paths = ()
        self.selection.clear()
        filename = self.pipeline.filename
        self.viewport.set_scene_name(filename.name if filename else "CAE Viewer")
        self.error = None
        self._refresh_reports()
        self.reset_camera()

    def request_open(self, filename):
        """Queue replacement; do not destroy resources inside native callbacks."""
        self.pending_file = Path(filename).expanduser().resolve()

    def _open_scene(self, filename):
        candidate = None
        try:
            candidate = self._new_pipeline(filename)
            # Resolve a held gesture against its original pipeline before rebinding.
            self.viewport.cancel_active_transform_drag(reason="stage_transition")
        except Exception as exc:
            if candidate is not None:
                candidate.close()
            LOGGER.exception("Could not open %s", filename)
            self.error = f"Could not open: {exc}"
            self.status.text = self.error
            return
        previous, self.pipeline = self.pipeline, candidate
        self.adapter.pipeline = candidate
        self.adapter.previews.clear()
        self.adapter.pending_picks.clear()
        self.stage_adapter.pipeline = candidate
        self.transforms.pipeline = candidate
        self.pending.clear()
        self.undo.clear()
        self._attach_transforms()
        previous.close()
        self._scene_changed()

    def _selection_changed(self, event):
        self.selected_paths = tuple(event.snapshot.paths())
        self._refresh_reports()

    def show_open_dialog(self):
        """Use OVUI's modal path-entry pattern; this build has no FilePickerDialog."""
        self._destroy_dialog()
        self.dialog = ui.Window("Open USD scene", width=620, height=190, flags=ui.WINDOW_FLAGS_MODAL)
        with self.dialog.frame, ui.VStack(spacing=10):
            ui.Label("USD file path (.usda, .usd, .usdc)", height=24)
            field = ui.StringField(height=28)
            field.model.set_value(str(self.pipeline.filename or SCENE))
            ui.Label("Opening another file discards unsaved in-memory edits.", height=30, word_wrap=True)
            with ui.HStack(height=30, spacing=8):

                def confirm():
                    value = field.model.as_string.strip()
                    if value:
                        self.request_open(value)
                        self.dialog.visible = False

                ui.Button("Open", clicked_fn=confirm)
                ui.Button("Cancel", clicked_fn=lambda: setattr(self.dialog, "visible", False))

    def _destroy_dialog(self):
        if self.dialog is not None:
            self.dialog.destroy()
            self.dialog = None

    def _framable_selection(self):
        """Selected file prims (never the viewer's rig) that framing should enclose."""
        return [path for path in self.selected_paths if self.pipeline.in_scene(path)]

    def frame(self):
        """Frame the selected prims' bounds, keeping the view direction; else the scene.

        A selected group frames its subtree. When nothing selected has visible
        point-based geometry, fall back to the scene overview.
        """
        paths = self._framable_selection()
        if paths and self.viewport.frame_paths(paths):
            return
        if not self.viewport.apply_camera_pose(BoundCameraPose(**self.pipeline.camera_pose())):
            raise RuntimeError("OVUI rejected the scene overview")

    def reset_camera(self):
        pose = self.stage_adapter.read_bound_camera()
        if pose is None:
            pose = BoundCameraPose(**self.pipeline.camera_pose())
        if not self.viewport.apply_camera_pose(pose):
            raise RuntimeError("OVUI rejected the scene camera")

    def stop(self):
        self.running = False
