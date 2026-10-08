# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""The minimal OVUI example: author, reproduce after reopening, and the window itself.

The authoring and camera tests run in the library environment. The workflow
round trip needs OVRTX and a GPU. The window test runs the real OVUI loop in a
subprocess and only in the example's own environment, where OVUI is installed.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

EXAMPLE = Path(__file__).parents[1] / "examples" / "ovui-minimal"
DATA = Path(__file__).parent / "data" / "StaticMixer.cgns"


def load(name: str):
    """Import an example module by path; both examples have an ``app.py``."""
    spec = importlib.util.spec_from_file_location(f"ovui_minimal_{name}", EXAMPLE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(EXAMPLE))  # app.py imports its sibling as ``workflow``
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(EXAMPLE))
    return module


workflow = load("workflow")


def look_at(eye, target, up):
    """OVUI's column-vector view matrix and a 50-degree perspective projection for that pose."""
    eye, target, up = (np.asarray(v, dtype=np.float64) for v in (eye, target, up))
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)
    world = np.eye(4)
    world[0, :3], world[1, :3], world[2, :3], world[3, :3] = right, true_up, -forward, eye  # row vectors
    view = np.linalg.inv(world.T)
    near, far, aspect = 0.1, 1000.0, 4 / 3
    tangent = math.tan(math.radians(workflow.FOV_DEGREES / 2))
    projection = np.zeros((4, 4))
    projection[0, 0], projection[1, 1] = 1 / (aspect * tangent), 1 / tangent
    projection[2, 2], projection[2, 3], projection[3, 2] = (
        -(far + near) / (far - near),
        -2 * far * near / (far - near),
        -1,
    )
    return view, projection


def test_authored_scene_records_intent_only(tmp_path):
    from pxr import Usd, UsdGeom

    scene = tmp_path / "scenes" / "mixer.usda"
    authored = workflow.author_scene(DATA, scene)
    assert authored.dataset == "B1_P3" and authored.field == "Temperature"
    assert authored.field_range == pytest.approx((285.0, 315.0), abs=0.01)
    assert authored.iso_value == pytest.approx(300.0, abs=0.01)
    assert {path.rsplit("/", 1)[-1] for path in authored.boundaries} == {"StaticMixer_Default", "in1", "in2", "out"}

    stage = Usd.Stage.Open(str(scene))
    payloads = stage.GetPrimAtPath("/World/Data").GetMetadata("payload").GetAddedOrExplicitItems()
    assert not Path(payloads[0].assetPath).is_absolute(), "a relative payload keeps the scene relocatable"
    iso = stage.GetPrimAtPath(workflow.ISO_SURFACE)
    assert iso.GetAttribute(workflow.ISO_VALUE).Get() == pytest.approx(300.0, abs=0.01)
    assert len(UsdGeom.Mesh(iso).GetPointsAttr().Get()) == 0, "no derived arrays are saved"
    for path in authored.boundaries:
        assert UsdGeom.Imageable(stage.GetPrimAtPath(path)).GetVisibilityAttr().Get() == "invisible"
    camera = UsdGeom.Xformable(stage.GetPrimAtPath(workflow.CAMERA))
    assert np.allclose(np.array(camera.GetLocalTransformation()), workflow.PLACEHOLDER_CAMERA)
    assert stage.GetPrimAtPath(workflow.RENDER_PRODUCT).GetRelationship("camera").GetTargets() == [
        stage.GetPrimAtPath(workflow.CAMERA).GetPath()
    ]


def test_authoring_names_the_choices_when_a_selection_is_wrong(tmp_path):
    with pytest.raises(ValueError, match="B1_P3"):
        workflow.author_scene(DATA, tmp_path / "a.usda", dataset="nope")
    with pytest.raises(ValueError, match="Temperature"):
        workflow.author_scene(DATA, tmp_path / "b.usda", field="nope")
    with pytest.raises(FileNotFoundError):
        workflow.author_scene(tmp_path / "missing.cgns", tmp_path / "c.usda")


def test_saved_camera_pose_round_trips_through_the_matrix():
    eye, target = (3.0, 2.0, 6.0), (0.5, -0.25, 1.0)
    view, projection = look_at(eye, target, (0, 1, 0))
    world, intrinsics = workflow.viewport_camera(view, projection)
    assert np.allclose(world[3, :3], eye)
    assert intrinsics[3] == pytest.approx(0.1) and intrinsics[4] == pytest.approx(1000.0)
    pose = workflow.pose_from_matrix(world, target)
    assert pose["eye"] == pytest.approx(eye) and pose["target"] == pytest.approx(target)
    behind = workflow.pose_from_matrix(world, np.asarray(eye) - (np.asarray(target) - eye))
    assert math.dist(behind["eye"], behind["target"]) == pytest.approx(10.0), "a center behind the camera falls back"


def test_viewport_camera_rejects_what_the_scene_camera_cannot_express():
    view, projection = look_at((0, 0, 5), (0, 0, 0), (0, 1, 0))
    orthographic = projection.copy()
    orthographic[3, 2] = 0
    with pytest.raises(ValueError):
        workflow.viewport_camera(view, orthographic)
    with pytest.raises(ValueError):
        workflow.viewport_camera(np.eye(3), projection)


def test_render_gpus_pins_only_cuda_devices():
    assert workflow.render_gpus("cuda:1") == "1" and workflow.render_gpus("cuda") == "0"
    assert workflow.render_gpus("cpu") is None
    with pytest.raises(ValueError):
        workflow.render_gpus("cuda:x")


def test_arguments():
    app = load("app")
    args = app.parse_args([])
    assert args.source == app.EXAMPLE_DATA and args.device == "cuda:0" and not args.headless
    args = app.parse_args(["scene.usda", "--headless", "--frames", "3", "--iso-value", "295"])
    assert args.source == Path("scene.usda") and args.frames == 3 and args.iso_value == 295
    for bad in (["--screenshot", "out.png"], ["--headless"], ["scene.usda", "--scene", "copy.usda"]):
        with pytest.raises(SystemExit):
            app.parse_args(bad)


def differing_fraction(first, second, *, threshold=8) -> float:
    """Fraction of pixels whose RGB differs by more than ``threshold`` levels."""
    delta = np.abs(first[..., :3].astype(int) - second[..., :3].astype(int)).max(axis=-1)
    return float((delta > threshold).mean())


@pytest.mark.skipif(importlib.util.find_spec("ovrtx") is None, reason="OVRTX is not installed")
def test_workflow_reproduces_the_saved_scene(tmp_path):
    if not pytest.importorskip("warp").is_cuda_available():
        pytest.skip("CUDA is unavailable; OVRTX cannot initialize")
    scene = tmp_path / "mixer.usda"
    workflow.author_scene(DATA, scene)

    with workflow.Workflow(scene) as first:
        assert first.field == "Temperature" and first.field_range == pytest.approx((285.0, 315.0), abs=0.01)
        assert first.saved_pose() is None, "the authored scene carries the placeholder camera"
        pose = first.overview_pose()
        view, projection = look_at(pose["eye"], pose["target"], (0, 1, 0))
        before = [first.render(320, 240, view, projection) for _ in range(3)][-1]
        first.set_iso_value(305.0)
        first.show_boundaries(True)
        report = None
        while report is None:
            report = first.pump()
            first.await_evaluation(0.05)
        status = {r.operator_path.rsplit("/", 1)[-1]: r.status for r in report.operators}
        assert status["IsoSurface"] == "succeeded", status
        assert status["in1"] == "unchanged", "primed at open, so showing a boundary reuses its geometry"
        assert first.iso_value == 305.0 and first.boundaries_visible
        after = [first.render(320, 240, view, projection) for _ in range(3)][-1]
        assert differing_fraction(before, after) > 0.02, "the edit must change the picture"
        assert first.save() == scene

    with workflow.Workflow(scene) as second:
        assert second.iso_value == 305.0 and second.boundaries_visible
        saved = second.saved_pose()
        assert saved is not None and saved["eye"] == pytest.approx(pose["eye"], abs=1e-6)
        again = [second.render(320, 240, view, projection) for _ in range(3)][-1]
    assert differing_fraction(after, again) < 0.01, "reopening the saved scene reproduces the picture"


def test_window_opens_edits_saves_and_reproduces(tmp_path):
    try:
        importlib.metadata.version("ovui")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("OVUI is only installed in the example environment")
    if not pytest.importorskip("warp").is_cuda_available():
        pytest.skip("CUDA is unavailable; OVRTX cannot initialize")
    scene = tmp_path / "mixer.usda"

    def run(source, *options):
        command = [sys.executable, str(EXAMPLE / "app.py"), str(source), "--headless", "--frames", "12", *options]
        result = subprocess.run(command, capture_output=True, text=True, cwd=tmp_path, timeout=600)
        assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]

    run(DATA, "--scene", scene, "--iso-value", "305", "--save", "--frame-dump", tmp_path / "saved.npy")
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(scene))
    assert stage.GetPrimAtPath(workflow.ISO_SURFACE).GetAttribute(workflow.ISO_VALUE).Get() == 305.0
    camera = np.array(UsdGeom.Xformable(stage.GetPrimAtPath(workflow.CAMERA)).GetLocalTransformation())
    assert not np.allclose(camera, workflow.PLACEHOLDER_CAMERA), "Save records the camera"
    del stage

    run(scene, "--frame-dump", tmp_path / "reopened.npy", "--screenshot", tmp_path / "window.png")
    run(scene, "--iso-value", "295", "--frame-dump", tmp_path / "edited.npy")
    saved, reopened, edited = (np.load(tmp_path / f"{name}.npy") for name in ("saved", "reopened", "edited"))
    assert saved.shape == reopened.shape and saved[..., :3].std() > 1, "the frame shows something"
    assert differing_fraction(saved, reopened) < 0.01, "reopening reproduces the saved picture"
    assert differing_fraction(saved, edited) > 0.02, "a different iso value changes it"
    assert (tmp_path / "window.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
