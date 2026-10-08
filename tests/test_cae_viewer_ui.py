# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Exercise the real OVUI loop in a subprocess, without requiring RTX shaders.

Run with the example's environment on a Vulkan-capable host. The ordinary
library test environment skips this optional UI integration check.
"""

from __future__ import annotations

import importlib.metadata
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("fps", [0, 30])
def test_native_ui_controls_capture_and_shutdown(tmp_path, fps) -> None:
    try:
        importlib.metadata.version("ovui")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("OVUI is only installed in the viewer environment")
    example = Path(__file__).parents[1] / "examples" / "cae-viewer"
    screenshot = tmp_path / "viewer.png"
    script = r"""
import asyncio
import ctypes
import os
import sys
from importlib.metadata import distribution
from pathlib import Path
from types import SimpleNamespace

os.environ["OMNIUI_HEADLESS"] = "1"
os.environ["OMNIUI_BACKEND"] = "vulkan"
sys.path.insert(0, sys.argv[1])
runtime = ctypes.CDLL(str(distribution("nvidia-cuda-runtime-cu12").locate_file(
    "nvidia/cuda_runtime/lib/libcudart.so.12")), mode=ctypes.RTLD_GLOBAL)
import numpy as np
import omni.ui as ui
from cae_viewer import application
from cae_viewer.model import describe_operator

events = []

uncapped = application.RenderClock(0)
uncapped.commit(1.0)
assert uncapped.should_render(1.0) == 0.0
assert uncapped.should_render(1.001) is not None
capped = application.RenderClock(30)
capped.commit(1.0)
assert capped.should_render(1.001) is None
assert capped.should_render(1.04) is not None

class Pipeline:
    def __init__(self, **kwargs):
        self.width, self.height = kwargs["width"], kwargs["height"]
        self.renderer, self.stage = object(), object()
        self.filename = kwargs["scene"]
        self.last_report = None
        self.operators = tuple(describe_operator(path, {"CaeVizOperatorAPI", "CaeVizIsoSurfaceAPI"})
                               for path in ("/A/Contour", "/B/Contour"))
        self.current = {op.path: {p.name: p.default for p in op.properties} for op in self.operators}
    def open(self):
        if self.filename and "missing" in str(self.filename):
            raise FileNotFoundError(self.filename)
        return self
    def close(self):
        events.append("closed")
    def values(self, op):
        return self.current[op.path]
    def scene_type(self, path):
        return "Xform" if path in ("/A", "/B") else "Mesh"
    def hierarchy(self):
        raise AssertionError("Viewport interaction must not read the complete hierarchy")
    def hidden_by(self, op):
        return None
    def transformable(self, path):
        return path in ("/A", "/B")
    def local_transform(self, path):
        value = self.current.get(path, {}).get("omni:xform")
        return None if value is None else np.asarray(value)
    ordinal = 1
    evaluating = False
    def pump(self):
        return None
    def camera_pose(self):
        return dict(eye=(8,5,10), target=(0,0,0), up_axis="Y", fov_degrees=50, prim_path="")
    def bound_camera_pose(self):
        if "second" not in str(self.filename):
            return None
        return dict(eye=(0, 9, 0), target=(0, 0, 0), up_axis="Y", fov_degrees=40, prim_path="")
    def world_bounds(self, paths=None):
        return ((-1, -2, -3), (1, 2, 3)) if paths == ["/A"] else None
    def in_scene(self, path):
        return path in ("/A", "/B", "/A/Contour", "/B/Contour")
    def request(self, edits):
        for (path, name), value in edits.items():
            self.current.setdefault(path, {})[name] = value
        events.append(edits)
        return True
    def render_viewport(self, width, height, view, projection):
        pixels = np.full((height, width, 4), 127, dtype=np.uint8)
        pixels[..., 3] = 255
        return pixels
    picks, queue, highlighted = [], [], ()
    def request_pick(self, left, top, right, bottom):
        self.picks.append((left, top, right, bottom))
        self.queue.append((len(self.picks), ("/A/Contour",) if left < 0.5 else ()))
        return len(self.picks)
    def cancel_pick(self, ticket):
        self.queue = [item for item in self.queue if item[0] != ticket]
    def take_pick_hits(self):
        return self.queue.pop(0) if self.queue else None
    def highlight(self, paths):
        self.highlighted = tuple(paths)

class Viewer(application.Viewer):
    async def _frames(self):
        self.show_open_dialog()
        assert self.dialog.visible
        self.dialog.visible = False
        previous = self.pipeline
        self._open_scene(Path("missing.usda"))
        assert self.pipeline is previous
        self._open_scene(Path("second.usda"))
        assert self.pipeline is not previous
        assert not hasattr(self, "property_window") and not hasattr(self, "stage_window")
        from omni.ui.testing import wait_frames
        await wait_frames(2)
        self.window.dock_in(ui.Workspace.get_window("DockSpace"), ui.DockPosition.SAME)
        await wait_frames(2)
        # The second file carried Kit camera metadata; the first used the overview.
        camera = self.viewport._camera
        assert camera.state.target == [0.0, 0.0, 0.0] and camera.state.distance == 9.0 and camera.fov_degrees == 40.0
        assert self.frame_button.text == "Frame scene"
        self.selection.publish(["/A"], source="stage")
        assert self.report_label.text == "A"
        assert self.frame_button.text == "Frame selection"
        camera.orbit(0.7, 0.3)
        azimuth, elevation = camera.state.azimuth, camera.state.elevation
        self.frame()  # Frame the selection's bounds, keeping the view direction.
        assert camera.state.distance == 12.0 and (camera.state.azimuth, camera.state.elevation) == (azimuth, elevation)
        self.selection.publish(["/B"], source="stage")
        self.frame()  # Nothing framable under /B: back to the overview pose.
        assert abs(camera.state.distance - np.linalg.norm((8, 5, 10))) < 1e-3
        self.reset_camera()
        assert camera.state.distance == 9.0  # Reset restores the saved view.
        self.selection.publish(["/A"], source="stage")
        # Drive OVUI's real transform model through the viewer's adapters.
        model = self.viewport._transform_model
        assert model.has_adapters() and model.has_transformable_selection()
        assert self.viewport.set_active_tool("rotate") and self.viewport.set_active_tool("translate")
        model.on_drag_start()
        delta = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [2.0, 0, 0, 1]]
        model.on_drag_moved(delta)
        assert self.adapter.previews["/A"][3][:3] == [2.0, 0.0, 0.0]
        model.on_drag_ended()
        assert self.pending[("/A", "omni:xform")][3][:3] == [2.0, 0.0, 0.0]
        assert self.transforms.get_local_transform("/A")[3][:3] == [2.0, 0.0, 0.0]
        assert self.undo.can_undo()
        self.undo.undo()
        assert self.pending[("/A", "omni:xform")][3][:3] == [0.0, 0.0, 0.0]
        self.undo.redo()
        assert self.pending[("/A", "omni:xform")][3][:3] == [2.0, 0.0, 0.0]
        self.pending.clear()
        # A cancelled drag restores the drag-start matrix through the preview channel.
        model.on_drag_start()
        model.on_drag_moved(delta)
        model.on_drag_cancelled()
        assert self.adapter.previews["/A"][3][:3] == [0.0, 0.0, 0.0]
        self.adapter.previews.clear()
        self.selection.clear()
        # Runtime diagnostics stay visible without depending on selected properties.
        self.pipeline.last_report = SimpleNamespace(diagnostics=[SimpleNamespace(severity="error", message="bad data")])
        report = self.pipeline.last_report
        self.pipeline.pump = lambda: report
        self._pump_edits()
        assert self.status.text == "bad data"
        self.pipeline.pump = lambda: None
        self._pump_edits()
        assert self.status.text == "bad data"  # A cached frame must not clear errors.
        report.diagnostics = []
        self.pipeline.pump = lambda: report
        self._pump_edits()
        assert self.status.text == "Ready"
        self.pipeline.pump = lambda: None
        self.selection.publish(['/A/Contour'], source='viewport')
        assert self.report_label.text == "Contour"
        assert not self.viewport._transform_model.has_transformable_selection()
        # Viewport clicks arrive through OVUI's pick gesture and answer after the frame renders.
        self.selection.clear()
        self.viewport._on_pick(-0.5, 0.5)
        left, top, right, bottom = self.pipeline.picks[-1]
        assert 0 <= left < 0.5 and 0 <= top < 0.5 and right > left and bottom > top, self.pipeline.picks[-1]
        assert self.selected_paths == ()  # Answered only after the frame renders.
        self.adapter.complete_picks()
        assert self.selected_paths == ("/A/Contour",)
        assert self.pipeline.highlighted == ("/A/Contour",)
        self.viewport._on_pick(0.5, 0.5)  # Background clears the selection.
        self.adapter.complete_picks()
        assert self.selected_paths == () and self.pipeline.highlighted == ()
        # A marquee waiting for its frame survives the click that OVUI cancels and re-issues.
        self.viewport._on_pick_rect(-0.9, -0.9, 0.9, 0.9)
        self.viewport._on_pick(-0.5, 0.5)
        self.adapter.cancel_pick("viewport_click")
        assert len(self.adapter.pending_picks) == 1
        self.adapter.complete_picks()
        assert self.selected_paths == ("/A/Contour",)
        self.adapter.complete_picks()
        assert self.selected_paths == ("/A/Contour",) and not self.adapter.pending_picks
        # Source replacement must discard selection, previews, and undo history.
        self.selection.publish(["/A"], source="viewport")
        self.viewport._transform_model.on_drag_start()
        self.viewport._transform_model.on_drag_moved(delta)
        assert self.adapter.previews
        self._open_scene(Path('third.usda'))
        assert not self.selected_paths and not self.pending and not self.adapter.previews
        assert not self.undo.can_undo()
        self._open_scene(Path('second.usda'))
        self.pipeline.ordinal = 1
        self.selection.publish(['/A/Contour'], source='stage')
        assert self.viewport._transform_model.has_adapters()
        self.transforms.set_local_transform('/B', delta)
        self.adapter.set_live_local_transform('/B', delta)
        await super()._frames()
        assert self.window.docked
        assert self.viewport._fps_value_label.text  # Reuse OVUI's native FPS HUD.
        assert self.viewport._resolution_value_label.text

application.RenderPipeline = Pipeline
args = SimpleNamespace(width=1280, height=800, fps=int(sys.argv[3]), frames=3,
                       device="cuda:0", screenshot=Path(sys.argv[2]), scene=None)
viewer = Viewer(args)
viewer.run()
assert viewer.rendered_frames >= 3
assert viewer.pipeline.current["/B"]["omni:xform"][3][:3] == [2.0, 0.0, 0.0]
# The preview and the commit for /B coalesce into the same frame's request.
assert sum(("/B", "omni:xform") in e for e in events if isinstance(e, dict)) == 1
assert events[-1] == "closed"
assert viewer.viewport._destroyed
assert not viewer._widget_callbacks
print("UI controls, frame loop, capture, and shutdown passed")
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(example), str(screenshot), str(fps)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert screenshot.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
