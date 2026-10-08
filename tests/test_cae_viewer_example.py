# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

EXAMPLE = Path(__file__).parents[1] / "examples" / "cae-viewer"
sys.path.insert(0, str(EXAMPLE))

from cae_viewer.model import describe_operator  # noqa: E402
from cae_viewer.pipeline import RenderPipeline, render_gpus, viewport_camera  # noqa: E402
from cae_viewer.scene import compose_viewer_scene, read_bound_camera  # noqa: E402


def camera():
    view = np.eye(4)
    view[:3, 3] = (-2.0, -3.0, -12.0)
    projection = np.zeros((4, 4))
    projection[0, 0], projection[1, 1] = 1.5, 2.0
    projection[2, 2], projection[2, 3], projection[3, 2] = -1.00002, -0.200002, -1
    return view, projection


def operator(path="/Elsewhere/Contour", schema="CaeVizIsoSurfaceAPI"):
    return describe_operator(path, {"CaeVizOperatorAPI", schema})


def test_properties_use_schema_defaults_and_arbitrary_paths():
    contour = operator()
    assert contour.kind == "Iso surface"
    props = {p.name: p for p in contour.properties}
    assert props["cae:viz:isoSurface:isoValue"].default == 0
    props["cae:viz:isoSurface:isoValue"].validate(-7000)  # No mixer-specific range.
    stream = operator(schema="CaeVizStreamlinesAPI")
    assert not any("threshold" in p.name or "seed" in p.name.lower() for p in stream.properties)
    assert {"visibility", "cae:viz:operator:enabled", "cae:viz:streamlines:direction"} <= {
        p.name for p in stream.properties
    }


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 1e40, True])
def test_reject_invalid_numeric_value(value):
    prop = next(p for p in operator().properties if p.name.endswith("isoValue"))
    with pytest.raises(ValueError):
        prop.validate(value)


def test_unknown_operator_still_has_common_controls():
    op = describe_operator("/Custom", {"CaeVizOperatorAPI", "OtherAPI"})
    assert [p.name for p in op.properties] == ["visibility", "cae:viz:operator:enabled"]


def test_composition_keeps_relative_assets_and_avoids_rig_collision(tmp_path):
    from pxr import Sdf, Usd

    source = tmp_path / "other.usda"
    original = '#usda 1.0\n( upAxis = "Z" )\ndef Scope "_CaeViewer" {}\ndef Xform "Unrelated" {}\n'
    source.write_text(original)
    text, root, axis, bound_camera = compose_viewer_scene(source, 320, 240)
    assert root == "/_CaeViewer_" and axis == "Z" and bound_camera is None
    layer = Sdf.Layer.CreateAnonymous()
    assert layer.ImportFromString(text)
    stage = Usd.Stage.Open(layer)
    assert stage.GetPrimAtPath("/Unrelated")
    assert tuple(stage.GetPrimAtPath(root + "/Render").GetAttribute("resolution").Get()) == (320, 240)
    assert list(layer.subLayerPaths) == [str(source)]
    assert source.read_text() == original


KIT_SETTINGS = """    customLayerData = {
        dictionary cameraSettings = {
            dictionary Front = {
                double3 position = (50000, 0, 0)
                double radius = 500
            }
            dictionary Perspective = {
                double3 position = (0.5, 6, 0.5)
                double3 target = (0, 0, 0)
            }
            string boundCamera = "%s"
        }
    }
"""


def kit_stage(tmp_path, bound="/OmniverseKit_Persp", body="", up_axis="Z"):
    from pxr import Usd

    filename = tmp_path / "kit.usda"
    filename.write_text(f'#usda 1.0\n(\n    upAxis = "{up_axis}"\n{KIT_SETTINGS % bound})\n{body}')
    return Usd.Stage.Open(str(filename))


def test_kit_perspective_camera_metadata_becomes_the_initial_pose(tmp_path):
    pose = read_bound_camera(kit_stage(tmp_path), 1280, 720)
    assert pose["eye"] == (0.5, 6.0, 0.5) and pose["target"] == (0.0, 0.0, 0.0)
    assert pose["up_axis"] == "Z" and pose["prim_path"] == ""
    # Kit's implicit perspective camera has a 60 degree horizontal FOV; the
    # vertical FOV follows the launch aspect ratio.
    assert pose["fov_degrees"] == pytest.approx(35.98, abs=0.01)
    assert read_bound_camera(kit_stage(tmp_path, up_axis="Y"), 100, 100)["fov_degrees"] == pytest.approx(60.0)


def test_bound_camera_prim_is_read_from_its_composed_transform(tmp_path):
    body = """def Xform "World" {
    double3 xformOp:translate = (0, 0, 5)
    uniform token[] xformOpOrder = ["xformOp:translate"]
    def Camera "Cam" {
        float focalLength = 50
        float verticalAperture = 15.2908
        float focusDistance = 20
        double3 xformOp:rotateXYZ = (90, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:rotateXYZ"]
    }
}
"""
    pose = read_bound_camera(kit_stage(tmp_path, bound="/World/Cam", body=body), 1280, 720)
    assert pose["prim_path"] == "/World/Cam"
    np.testing.assert_allclose(pose["eye"], (0, 0, 5), atol=1e-9)
    np.testing.assert_allclose(pose["target"], (0, 20, 5), atol=1e-9)  # focusDistance along +Y
    assert pose["fov_degrees"] == pytest.approx(17.39, abs=0.01)


@pytest.mark.parametrize("bound", ["/OmniverseKit_Top", "/World/Missing", ""])
def test_unusable_bound_cameras_fall_back_to_the_overview(tmp_path, bound):
    pose = read_bound_camera(kit_stage(tmp_path, bound=bound), 1280, 720)
    # Orthographic Kit cameras cannot drive this perspective viewer, and the
    # Perspective settings describe Kit's implicit camera, not a missing prim.
    # An empty name means that implicit camera.
    assert (pose is None) is bool(bound)
    assert (pose or {}).get("eye", (0.5, 6.0, 0.5)) == (0.5, 6.0, 0.5)


def test_metadata_without_a_perspective_target_or_without_settings_is_ignored(tmp_path):
    from pxr import Usd

    plain = tmp_path / "plain.usda"
    plain.write_text('#usda 1.0\ndef Xform "World" {}\n')
    assert read_bound_camera(Usd.Stage.Open(str(plain)), 100, 100) is None
    partial = tmp_path / "partial.usda"
    partial.write_text(
        "#usda 1.0\n( customLayerData = { dictionary cameraSettings = {\n"
        "    dictionary Perspective = {\n        double3 position = (1, 2, 3)\n        double radius = 5\n    }\n"
        '    string boundCamera = "/OmniverseKit_Persp"\n} } )\n'
    )
    assert read_bound_camera(Usd.Stage.Open(str(partial)), 100, 100) is None


def test_composition_returns_the_bound_camera_without_authoring_it(tmp_path):
    from pxr import Sdf, Usd

    source = tmp_path / "kit.usda"
    settings = KIT_SETTINGS % "/OmniverseKit_Persp"
    source.write_text(f'#usda 1.0\n(\n    upAxis = "Y"\n{settings})\ndef Xform "World" {{}}\n')
    text, root, axis, bound_camera = compose_viewer_scene(source, 320, 240)
    assert axis == "Y" and bound_camera["eye"] == (0.5, 6.0, 0.5) and bound_camera["up_axis"] == "Y"
    layer = Sdf.Layer.CreateAnonymous()
    assert layer.ImportFromString(text)
    # The viewer still renders through its own rig; the saved pose only seeds OVUI's camera.
    stage = Usd.Stage.Open(layer)
    assert stage.GetPrimAtPath(root + "/Camera")
    assert "cameraSettings" not in (layer.customLayerData or {})


def mock_pipeline(monkeypatch, operators=None):
    """A RenderPipeline over a mocked stage, renderer, and update loop."""
    pipeline = RenderPipeline()
    pipeline.stage, pipeline.renderer = Mock(), Mock()
    loop = Mock(ordinal=4, pending={}, evaluating=False, last_report=None)
    loop.value.return_value = None
    loop.request.side_effect = lambda edits, types=None: loop.pending.update(edits)
    loop.submit.return_value = None
    loop.flush.return_value = None
    pipeline._loop = loop
    pipeline.operators = tuple(operators or [operator()])
    monkeypatch.setattr("cae_viewer.pipeline.read_attribute", lambda *args: None)
    return pipeline


def test_edits_go_to_the_loop_typed_and_are_submitted_synchronously(monkeypatch):
    first, second = operator("/A/Iso"), operator("/B/Iso")
    pipeline = mock_pipeline(monkeypatch, [first, second])
    pipeline._scene_paths = frozenset({second.path})
    report = Mock()
    pipeline._loop.submit.return_value = report
    edits = {(first.path, "cae:viz:isoSurface:isoValue"): -10.0, (second.path, "visibility"): "invisible"}
    assert pipeline.update(edits) is report
    pipeline._loop.request.assert_called_once_with(
        edits,
        types={(first.path, "cae:viz:isoSurface:isoValue"): "float", (second.path, "visibility"): "token"},
    )
    pipeline._loop.submit.assert_called_once_with(wait=True)
    pipeline.renderer.detach_ovstage.assert_not_called()


def test_invalid_batch_does_not_reach_the_loop(monkeypatch):
    pipeline = mock_pipeline(monkeypatch)
    pipeline._scene_paths = frozenset({"/Elsewhere/Contour"})
    with pytest.raises(ValueError):
        pipeline.update({("/Elsewhere/Contour", "visibility"): "invisible", ("/Missing", "visibility"): "inherited"})
    pipeline._loop.request.assert_not_called()
    pipeline._loop.submit.assert_not_called()
    # The schema default is the effective value, so requesting it changes nothing.
    assert pipeline.update({("/Elsewhere/Contour", "visibility"): "inherited"}) is None
    pipeline._loop.request.assert_not_called()


def test_showing_parent_prepares_initial_points_publication(monkeypatch):
    op = operator("/Group/Points", "CaeVizPointsAPI")
    pipeline = mock_pipeline(monkeypatch, [op])
    pipeline._scene_paths = frozenset({"/Group", op.path})
    pipeline._renderer_attached = True

    def read(_stage, path, name, _ordinal):
        if name == "visibility" and path == "/Group":
            return "invisible"
        if name == "cae:viz:operator:enabled":
            return True
        return None

    monkeypatch.setattr("cae_viewer.pipeline.read_attribute", read)
    pipeline._loop.value.side_effect = lambda path, name: read(None, path, name, None)
    events = []
    pipeline.renderer.detach_ovstage.side_effect = lambda: events.append("detach")
    pipeline.renderer.attach_ovstage.side_effect = lambda stage: events.append("attach")
    pipeline._loop.submit.side_effect = lambda wait: events.append(f"submit(wait={wait})")
    pipeline.request({("/Group", "visibility"): "inherited"})
    pipeline.pump()  # asynchronous by default, synchronous when the renderer must be detached
    assert events == ["detach", "submit(wait=True)", "attach"]


def test_integration_step_validation_precedes_writes(monkeypatch):
    op = operator(schema="CaeVizStreamlinesAPI")
    pipeline = mock_pipeline(monkeypatch, [op])
    with pytest.raises(ValueError, match="minimum <= initial <= maximum"):
        pipeline.update({(op.path, "cae:viz:streamlines:initialStepSize"): 20.0})
    pipeline._loop.request.assert_not_called()


def test_pump_defers_to_the_loop_while_evaluating(monkeypatch):
    pipeline = mock_pipeline(monkeypatch)
    pipeline._loop.evaluating = True
    pipeline._loop.pending = {("/Elsewhere/Contour", "visibility"): "invisible"}
    pipeline._loop.pump.return_value = None
    assert pipeline.pump() is None
    pipeline._loop.pump.assert_called_once_with()
    pipeline._loop.submit.assert_not_called()


def test_viewport_camera_preserves_pose_and_aspect():
    view, projection = camera()
    world, (focal, horizontal, vertical, near, far) = viewport_camera(view, projection)
    np.testing.assert_allclose(world[3, :3], (2, 3, 12))
    np.testing.assert_allclose(world.T @ view, np.eye(4))
    assert 2 * focal / horizontal == pytest.approx(projection[0, 0])
    assert 2 * focal / vertical == pytest.approx(projection[1, 1])
    assert near == pytest.approx(0.1)
    assert far == pytest.approx(10000.1)


@pytest.mark.parametrize("view,projection", [(np.eye(3), np.eye(4)), (np.eye(4), np.eye(4))])
def test_viewport_camera_rejects_unsupported_matrices(view, projection):
    with pytest.raises(ValueError):
        viewport_camera(view, projection)


def test_camera_and_resize_are_presentation_writes(monkeypatch):
    pipeline = mock_pipeline(monkeypatch)
    pipeline.ovrtx = Mock()
    pipeline.render = Mock()
    view, projection = camera()
    pipeline.render_viewport(400, 300, view, projection)
    edits, kwargs = pipeline._loop.write.call_args.args[0], pipeline._loop.write.call_args.kwargs
    assert edits[(pipeline.render_product, "resolution")] == (400, 300)
    assert kwargs["types"][(pipeline.camera_path, "omni:xform")] == "matrix4d"
    assert kwargs["types"][(pipeline.camera_path, "clippingRange")] == "float2"
    pipeline._loop.request.assert_not_called()  # no update for a camera move
    pipeline.render_viewport(400, 300, view, projection)
    pipeline._loop.write.assert_called_once()  # an unchanged viewport writes nothing


@pytest.mark.parametrize("fps", [0, 30])
def test_mapped_image_is_copied_before_unmap(monkeypatch, fps):
    pipeline = mock_pipeline(monkeypatch)
    pipeline.fps = fps
    timestamps = iter((10.0,) * 3 + (10.005,) * 3)  # each render reads the clock three times
    monkeypatch.setattr("cae_viewer.pipeline.time.perf_counter", lambda: next(timestamps))
    pipeline.ovrtx = Mock()
    view, projection = camera()
    world, intrinsics = viewport_camera(view, projection)
    pipeline._viewport_signature = (400, 300, world.tobytes(), intrinsics)
    pixels = np.ones((300, 400, 4), dtype=np.uint8)
    mapped = Mock()
    mapped.__dlpack__, mapped.__dlpack_device__ = pixels.__dlpack__, pixels.__dlpack_device__
    mapped.unmap.side_effect = lambda: pixels.fill(0)
    frame = Mock()
    frame.render_vars = {pipeline.render_product + "/LdrColor": Mock()}
    frame.render_vars[pipeline.render_product + "/LdrColor"].map.return_value = mapped
    pipeline.renderer.step.return_value = {pipeline.render_product: Mock(frames=[frame])}
    assert pipeline.render_viewport(400, 300, view, projection).min() == 1
    assert pixels.max() == 0
    mapped.unmap.assert_called_once()
    assert pipeline.renderer.step.call_args.kwargs["delta_time"] == pytest.approx(1 / fps if fps else 0.0)
    pipeline.render_viewport(400, 300, view, projection)
    assert pipeline.renderer.step.call_args.kwargs["delta_time"] == pytest.approx(1 / fps if fps else 0.005)


def test_viewport_borrows_pipeline():
    pytest.importorskip("ovui_data_adapters.common")
    from cae_viewer.viewport import PipelineAdapter

    pipeline = Mock(width=400, height=300)
    adapter = PipelineAdapter(pipeline)
    with pytest.raises(ValueError, match="cannot replace"):
        adapter.load_stage(object())
    adapter.shutdown()
    pipeline.close.assert_not_called()


def test_viewport_picks_translate_ndc_and_answer_after_the_rendered_frame():
    pytest.importorskip("ovui_data_adapters.common")
    from cae_viewer.viewport import PipelineAdapter

    pipeline = Mock(width=400, height=200)
    pipeline.request_pick.side_effect = range(1, 100)
    pipeline.take_pick_hits.return_value = None
    adapter = PipelineAdapter(pipeline)
    clicked, marquee = Mock(), Mock()
    # OVUI's render NDC is y-up in [-1, 1]; OVRTX wants [0, 1] from the top-left corner.
    adapter.pick(0.5, 0.5, clicked, "viewport_click")
    left, top, right, bottom = pipeline.request_pick.call_args.args
    assert (left, top) == pytest.approx((0.75, 0.25))
    assert (right - left, bottom - top) == pytest.approx((1 / 400, 1 / 200))
    adapter.complete_picks()  # Nothing rendered yet: no answer.
    clicked.assert_not_called()
    pipeline.take_pick_hits.return_value = (1, ("/World/Surface", "/World/Curves"))
    adapter.complete_picks()
    clicked.assert_called_once_with("/World/Surface", None)
    pipeline.take_pick_hits.return_value = (2, ())
    adapter.pick(-1.0, 1.0, clicked, "viewport_click")
    assert pipeline.request_pick.call_args.args[:2] == (0.0, 0.0)
    adapter.complete_picks()
    assert clicked.call_args.args == (None, None)
    adapter.pick(1.0, -1.0, clicked, "viewport_click")  # The far corner still covers one pixel.
    assert pipeline.request_pick.call_args.args == pytest.approx((1 - 1 / 400, 1 - 1 / 200, 1.0, 1.0))
    adapter.pick_rect(1.0, -1.0, 1.0, -1.0, marquee)
    assert pipeline.request_pick.call_args.args == pytest.approx((1 - 1 / 400, 1 - 1 / 200, 1.0, 1.0))
    # Cancelling the click leaves the marquee queued; neither answers a cancelled ticket.
    adapter.cancel_pick("viewport_click")
    pipeline.cancel_pick.assert_called_once_with(3)
    pipeline.take_pick_hits.return_value = (3, ("/World/Surface",))
    adapter.complete_picks()
    assert clicked.call_count == 2
    pipeline.take_pick_hits.return_value = (4, ("/World/Surface", "/World/Curves"))
    adapter.complete_picks()
    marquee.assert_called_once_with(["/World/Surface", "/World/Curves"])
    # Marquees normalize corner order.
    adapter.pick_rect(0.5, -0.5, -0.5, 0.5, marquee)
    assert pipeline.request_pick.call_args.args == pytest.approx((0.25, 0.25, 0.75, 0.75))
    adapter.set_selection_highlight(["/World/Surface"])
    pipeline.highlight.assert_called_once_with(["/World/Surface"])


def test_pick_query_rides_the_next_rendered_frame(monkeypatch):
    pipeline = mock_pipeline(monkeypatch)
    ovrtx = pipeline.ovrtx = Mock(
        OVRTX_RENDER_VAR_PICK_HIT="ovrtx_pick_hit", OVRTX_PICK_HIT_MAGIC=7, OVRTX_PICK_HIT_VERSION=1
    )
    monkeypatch.setattr("cae_viewer.pipeline.time.perf_counter", lambda: 10.0)
    frame = Mock()
    color = Mock()
    color.map.return_value.__dlpack__ = np.zeros((2, 2, 4), dtype=np.uint8).__dlpack__
    color.map.return_value.__dlpack_device__ = np.zeros(1).__dlpack_device__
    hits = Mock()
    ids = np.asarray([11, 12, 11], dtype=np.uint64)
    hits.map.return_value.__getitem__ = lambda self, name: {"primPath": ids}[name]
    hits.map.return_value.params = {"magic": np.asarray(7), "version": np.asarray(1)}
    frame.render_vars = {pipeline.render_product + "/LdrColor": color, "ovrtx_pick_hit": hits}
    pipeline.renderer.step.return_value = {pipeline.render_product: Mock(frames=[frame])}
    pipeline.renderer.resolve_prim_path_id.side_effect = {11: "/World/Surface", 12: "/World/Curves"}.__getitem__
    with pytest.raises(ValueError):
        pipeline.request_pick(0.5, 0.5, 0.5, 0.6)
    assert pipeline.take_pick_hits() is None
    first = pipeline.request_pick(0.25, 0.5, 0.26, 0.51)
    second = pipeline.request_pick(0.5, 0.5, 0.75, 1.5)  # Clamped, and answered by its own later frame.
    assert first != second
    pipeline.render(device=ovrtx.Device.CPU, consume=lambda mapped: None)
    pipeline.renderer.enqueue_pick_query.assert_called_once_with(pipeline.render_product, 0.25, 0.5, 0.26, 0.51)
    hits.map.return_value.unmap.assert_called_once()
    assert pipeline.take_pick_hits() == (first, ("/World/Surface", "/World/Curves"))
    assert pipeline.take_pick_hits() is None
    pipeline.render(device=ovrtx.Device.CPU, consume=lambda mapped: None)
    assert pipeline.renderer.enqueue_pick_query.call_args.args == (pipeline.render_product, 0.5, 0.5, 0.75, 1.0)
    assert pipeline.take_pick_hits()[0] == second
    # Frames without a query neither enqueue nor report anything.
    frame.render_vars.pop("ovrtx_pick_hit")
    pipeline.render(device=ovrtx.Device.CPU, consume=lambda mapped: None)
    assert pipeline.renderer.enqueue_pick_query.call_count == 2
    assert pipeline.take_pick_hits() is None
    # A schema mismatch reports no hits instead of misreading the tensors.
    ticket = pipeline.request_pick(0.0, 0.0, 1.0, 1.0)
    hits.map.return_value.params = {"magic": np.asarray(8), "version": np.asarray(1)}
    frame.render_vars["ovrtx_pick_hit"] = hits
    pipeline.render(device=ovrtx.Device.CPU, consume=lambda mapped: None)
    assert pipeline.take_pick_hits() == (ticket, ())
    pipeline.cancel_pick(pipeline.request_pick(0.0, 0.0, 1.0, 1.0))
    pipeline.render(device=ovrtx.Device.CPU, consume=lambda mapped: None)
    assert pipeline.renderer.enqueue_pick_query.call_count == 3
    assert pipeline.take_pick_hits() is None


def test_highlight_only_touches_the_renderer_when_the_selection_changes(monkeypatch):
    pipeline = mock_pipeline(monkeypatch)
    outline = pipeline.renderer.set_selection_outline_group_strings
    pipeline.highlight([])
    outline.assert_not_called()
    pipeline.highlight(["/A", "/B", "/A"])
    outline.assert_called_once_with(["/A", "/B"], 1)
    pipeline.highlight(("/A", "/B"))
    outline.assert_called_once()
    pipeline.highlight(["/B", "/C"])
    assert outline.call_args_list[1:] == [((["/A"], 0),), ((["/B", "/C"], 1),)]
    pipeline.highlight([])
    assert outline.call_args_list[-1] == ((["/B", "/C"], 0),)


def test_cli_optional_file_and_capture_validation():
    from app import parse_args

    assert parse_args([]).scene is None
    assert parse_args([]).fps == 0
    assert parse_args(["--fps", "0"]).fps == 0
    assert parse_args(["--fps", "60"]).fps == 60
    assert parse_args(["other.usda"]).scene == Path("other.usda")
    with pytest.raises(SystemExit):
        parse_args(["--screenshot", "x.png"])
    with pytest.raises(SystemExit):
        parse_args(["--fps", "-1"])


def test_pump_submits_pending_transforms_asynchronously(monkeypatch):
    op = operator("/A/Iso")
    pipeline = mock_pipeline(monkeypatch, [op])
    pipeline._transformable = frozenset({"/Seeds"})
    translated = np.eye(4)
    translated[3, 0] = 1.0
    assert pipeline.request({("/Seeds", "omni:xform"): translated.tolist()})
    translated[3, 0] = 2.0
    assert pipeline.request({("/Seeds", "omni:xform"): translated.tolist()})
    # The pending value is the transform the gizmo shows, and requesting it again is a no-op.
    assert pipeline.local_transform("/Seeds")[3, 0] == 2.0
    assert not pipeline.request({("/Seeds", "omni:xform"): translated.tolist()})
    key = ("/Seeds", "omni:xform")
    assert pipeline._loop.request.call_args.kwargs["types"] == {key: "matrix4d"}
    assert pipeline.pump() is None
    pipeline._loop.submit.assert_called_once_with(wait=False)


def test_render_presents_the_previous_frame_while_evaluating(monkeypatch):
    pipeline = mock_pipeline(monkeypatch)
    pipeline.width = pipeline.height = 8
    pipeline.ovrtx = Mock()
    pipeline.render = Mock(side_effect=["first", "second"])
    view, projection = camera()
    assert pipeline.render_viewport(8, 8, view, projection) == "first"
    assert pipeline._loop.write.call_count == 1
    # In flight: no camera write, no OVRTX step, the previous frame is shown.
    pipeline._loop.evaluating = True
    pipeline._loop.await_evaluation.return_value = False
    view[0, 3] = 1.0
    assert pipeline.render_viewport(8, 8, view, projection) == "first"
    assert pipeline.render.call_count == 1 and pipeline._loop.write.call_count == 1
    assert pipeline._loop.flush.call_count == 1  # only the first frame's routine flush
    # Finished: commit inline, then write the deferred camera and render fresh.
    pipeline._loop.await_evaluation.return_value = True
    pipeline._loop.flush.side_effect = lambda: setattr(pipeline._loop, "evaluating", False)
    assert pipeline.render_viewport(8, 8, view, projection) == "second"
    assert pipeline._loop.write.call_count == 2 and not pipeline.evaluating


@pytest.mark.parametrize(
    "device,expected", [("cuda:0", "0"), ("cuda:1", "1"), ("cuda", "0"), ("cpu", None), ("auto", None)]
)
def test_render_gpus_follow_the_compute_device(device, expected):
    assert render_gpus(device) == expected


def test_render_gpus_reject_malformed_cuda_device():
    with pytest.raises(ValueError):
        render_gpus("cuda:zero")


def test_glyph_runtime_settings_keep_typed_validation():
    op = describe_operator("/Glyphs", {"CaeVizOperatorAPI", "CaeVizGlyphsAPI"})
    assert op.kind == "Glyphs"
    settings = {p.name: p for p in op.properties}
    for suffix in ("maxCount", "scale", "useCellPoints", "orientationsMode"):
        assert f"cae:viz:glyphs:{suffix}" in settings
    count = settings["cae:viz:glyphs:maxCount"]
    assert count.validate(100) == 100
    with pytest.raises(ValueError):
        count.validate(-1)
