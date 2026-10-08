# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Real GPU regressions for file loading, independent operators, and rendering."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

if importlib.util.find_spec("ovrtx") is None:
    pytest.skip("OVRTX is not installed", allow_module_level=True)
if not pytest.importorskip("warp").is_cuda_available():
    pytest.skip("CUDA is unavailable; OVRTX cannot initialize", allow_module_level=True)

EXAMPLE = Path(__file__).parents[1] / "examples" / "cae-viewer"
sys.path.insert(0, str(EXAMPLE))

from cae_viewer.pipeline import SCENE, RenderPipeline  # noqa: E402
from cae_viewer.scene import read_attribute  # noqa: E402


def render(pipeline):
    for _ in range(2):
        pixels = pipeline.render(device=pipeline.ovrtx.Device.CPU, consume=lambda mapped: np.from_dlpack(mapped).copy())
    assert pixels.shape == (pipeline.height, pipeline.width, 4)
    return pixels


def surface_pixels(pixels):
    rgb = pixels[..., :3].astype(float)
    return np.count_nonzero((rgb[..., 2] > rgb[..., 0] + 30) & (rgb[..., 1] > rgb[..., 0] + 15))


def outline_pixels(pixels):
    """Count pixels in the viewer's orange selection outline, absent from the blue-grey scene."""
    rgb = pixels[..., :3].astype(float)
    return np.count_nonzero((rgb[..., 0] > rgb[..., 2] + 80) & (rgb[..., 0] > rgb[..., 1] + 20))


def test_arbitrary_file_paths_and_independent_operator_edits(tmp_path, capfd):
    # Different root, different file location, and no app-required render paths.
    text = SCENE.read_text().replace('"World"', '"Experiment"').replace("/World/", "/Experiment/")
    text = text.replace("@./StaticMixer.cgns@", f"@{SCENE.parent / 'StaticMixer.cgns'}@")
    filename = tmp_path / "renamed.usda"
    filename.write_text(text)
    with RenderPipeline(scene=filename, width=320, height=240) as pipeline:
        assert len(pipeline.operators) == 4
        assert all(op.path.startswith("/Experiment/") for op in pipeline.operators)
        root = "/Experiment/Recipes/"
        surface, curves, contour, points = (root + name for name in ("Surface", "Streamlines", "IsoSurface", "Points"))
        assert not pipeline.last_report.diagnostics
        initial = surface_pixels(render(pipeline))
        assert initial > 100
        # Hidden iso edits must not periodically rerun the visible Surface as
        # its original product publication falls out of native history.
        from unittest.mock import patch

        from ovcae.operators._builtins.faces import Faces

        with patch.object(Faces, "execute", side_effect=AssertionError("Surface must remain unchanged")):
            for value in range(301, 313):
                report = pipeline.update({(contour, "cae:viz:isoSurface:isoValue"): float(value)})
                assert {r.operator_path: r.status for r in report.operators} == {
                    surface: "unchanged",
                    contour: "skipped",
                    curves: "skipped",
                    points: "skipped",
                }
            # Camera-only commits can age out the evidence between updates; the
            # rig camera is a presentation write and runs no update.
            for position in range(16):
                matrix = np.eye(4)
                matrix[3, 2] = 12 + position * 0.01
                before = pipeline.ordinal
                pipeline._loop.write({(pipeline.camera_path, "omni:xform"): matrix})
                assert pipeline.ordinal == before + 1
            report = pipeline.update({(contour, "cae:viz:isoSurface:isoValue"): 300.0})
            assert next(r for r in report.operators if r.operator_path == surface).status == "unchanged"
            # Restore the original presentation for the pixel regressions below.
            matrix = np.eye(4)
            matrix[3, 2] = 12
            pipeline._loop.write({(pipeline.camera_path, "omni:xform"): matrix})
        report = pipeline.update({(surface, "visibility"): "invisible", (curves, "visibility"): "inherited"})
        assert report.committable and not report.diagnostics
        assert surface_pixels(render(pipeline)) < initial * 0.05
        before = read_attribute(pipeline.stage, curves, "points", pipeline.ordinal)
        pipeline.update({(curves, "cae:viz:streamlines:maxSteps"): 8})
        after = read_attribute(pipeline.stage, curves, "points", pipeline.ordinal)
        assert not np.array_equal(before, after)
        # Showing an additional operator must not hide the already-visible one.
        pipeline.update({(contour, "visibility"): "inherited"})
        assert read_attribute(pipeline.stage, curves, "visibility", pipeline.ordinal) == "inherited"
        before = read_attribute(pipeline.stage, contour, "points", pipeline.ordinal)
        pipeline.update({(contour, "cae:viz:isoSurface:isoValue"): 305.0})
        assert not np.array_equal(before, read_attribute(pipeline.stage, contour, "points", pipeline.ordinal))
        render(pipeline)
        pipeline.update(
            {
                (curves, "visibility"): "invisible",
                (contour, "visibility"): "invisible",
                (points, "visibility"): "inherited",
            }
        )
        assert read_attribute(pipeline.stage, points, "points", pipeline.ordinal).size > 0
        pixels = render(pipeline)
        rgb = pixels[..., :3].astype(float)
        assert np.count_nonzero((rgb[..., 1] > rgb[..., 0] + 15) & (rgb[..., 1] > rgb[..., 2] + 15)) > 100
        pipeline.update({(points, "visibility"): "invisible", (surface, "visibility"): "inherited"})
        assert surface_pixels(render(pipeline)) > initial * 0.8
        pipeline.update({("/Experiment/Recipes", "visibility"): "invisible"})
        assert all(result.status == "skipped" for result in pipeline.last_report.operators)
        assert surface_pixels(render(pipeline)) < initial * 0.05
        pipeline.update({("/Experiment/Recipes", "visibility"): "inherited"})
        assert surface_pixels(render(pipeline)) > initial * 0.8
        assert filename.read_text() == text
    captured = capfd.readouterr()
    assert "Invalid points data" not in captured.out + captured.err


def test_empty_start_then_second_file_and_camera_only_updates(tmp_path):
    # Exercise the same overlapping pipeline lifetime as the UI's atomic open.
    with RenderPipeline(width=160, height=120) as empty:
        assert not empty.operators
        render(empty)
        with RenderPipeline(scene=SCENE, width=160, height=120) as loaded:
            assert not loaded.last_report.diagnostics
            render(loaded)
            empty.close()  # Same order as an accepted file-open in the UI.
            render(loaded)
    source = tmp_path / "plain.usda"
    source.write_text(
        '#usda 1.0\n( upAxis = "Z" )\ndef Mesh "Shape" {\n'
        "point3f[] points = [(10,0,0),(12,0,0),(10,2,0)]\n"
        "int[] faceVertexCounts = [3]\nint[] faceVertexIndices = [0,1,2]\n}\n"
    )
    with RenderPipeline(scene=source, width=160, height=120) as plain:
        assert not plain.operators and plain.up_axis == "Z"
        pose = plain.camera_pose()
        np.testing.assert_allclose(pose["target"], [11, 1, 0])
        view = np.eye(4)
        view[2, 3] = -12
        projection = np.zeros((4, 4))
        projection[0, 0], projection[1, 1] = 1.5, 2.0
        projection[2, 2], projection[2, 3], projection[3, 2] = -1.00002, -0.200002, -1
        report = plain.last_report
        assert plain.render_viewport(200, 150, view, projection).shape == (150, 200, 4)
        ordinal = plain.ordinal
        plain.render_viewport(200, 150, view, projection)
        assert plain.ordinal == ordinal and plain.last_report is report


def test_kit_camera_metadata_and_selection_bounds(tmp_path):
    # A Kit-saved layer over the example: its camera settings must win on open,
    # while framing reads visible geometry, including generated products.
    filename = tmp_path / "kit.usda"
    filename.write_text(
        f"""#usda 1.0
(
    upAxis = "Y"
    subLayers = [@{SCENE}@]
    customLayerData = {{
        dictionary cameraSettings = {{
            dictionary Perspective = {{
                double3 position = (0.5, 6, 0.5)
                double3 target = (0, 0, 0)
            }}
            string boundCamera = "/OmniverseKit_Persp"
        }}
    }}
)
"""
    )
    with RenderPipeline(scene=filename, width=320, height=240) as pipeline:
        pose = pipeline.bound_camera_pose()
        assert pose["eye"] == (0.5, 6.0, 0.5) and pose["target"] == (0.0, 0.0, 0.0) and pose["up_axis"] == "Y"
        pose["eye"] = None  # A copy: the pipeline's record is not shared with callers.
        assert pipeline.bound_camera_pose()["eye"] == (0.5, 6.0, 0.5)
        assert pipeline.camera_pose()["fov_degrees"] == 50.0  # The overview stays independent.
        low, high = pipeline.world_bounds()
        assert pipeline.world_bounds(["/World/Recipes/Surface"]) == (low, high)
        assert pipeline.world_bounds(["/World"]) == (low, high)
        # Invisible prims (the seeds) and groups without visible geometry give no bounds.
        assert pipeline.world_bounds(["/World/Controls"]) is None
        assert pipeline.world_bounds(["/World/Controls/Seeds"]) is None
        assert pipeline.world_bounds(["/Nowhere"]) is None
        assert pipeline.in_scene("/World/Recipes/Surface") and not pipeline.in_scene(pipeline.camera_path)
        # Showing a hidden operator's product makes it framable on its own.
        pipeline.update({("/World/Recipes/IsoSurface", "visibility"): "inherited"})
        iso = pipeline.world_bounds(["/World/Recipes/IsoSurface"])
        assert iso is not None and np.all(np.asarray(iso[0]) >= np.asarray(low) - 1e-6)
    with RenderPipeline(scene=SCENE, width=320, height=240) as plain:
        assert plain.bound_camera_pose() is None


def test_pick_resolves_products_and_highlight_outlines_them():
    with RenderPipeline(scene=SCENE, width=320, height=240) as pipeline:
        plain = render(pipeline)
        # The default pose looks at the mixer, so its centre pixel is the Surface product.
        centre = pipeline.request_pick(0.5, 0.5, 0.5 + 1 / 320, 0.5 + 1 / 240)
        assert pipeline.take_pick_hits() is None
        render(pipeline)
        assert pipeline.take_pick_hits() == (centre, ("/World/Recipes/Surface",))
        assert pipeline.take_pick_hits() is None
        background = pipeline.request_pick(0.0, 0.0, 1 / 320, 1 / 240)
        render(pipeline)
        assert pipeline.take_pick_hits() == (background, ())
        marquee = pipeline.request_pick(0.0, 0.0, 1.0, 1.0)
        render(pipeline)
        ticket, hits = pipeline.take_pick_hits()
        assert ticket == marquee and "/World/Recipes/Surface" in hits
        assert outline_pixels(plain) < 20
        pipeline.highlight(["/World/Recipes/Surface"])
        assert outline_pixels(render(pipeline)) > 200
        pipeline.highlight([])
        assert outline_pixels(render(pipeline)) < 20


def test_every_operator_occurrence_and_hidden_ancestor(tmp_path):
    filename = tmp_path / "operators.usda"
    filename.write_text("""#usda 1.0
def Xform "Group" {
    token visibility = "invisible"
    def Mesh "One" (prepend apiSchemas = ["CaeVizOperatorAPI", "CaeVizFacesAPI"]) {
        bool cae:viz:operator:enabled = true
    }
    def Mesh "Two" (prepend apiSchemas = ["CaeVizOperatorAPI", "CaeVizFacesAPI"]) {
        bool cae:viz:operator:enabled = true
    }
    def "Untyped" (prepend apiSchemas = ["CaeVizOperatorAPI"]) {}
}
""")
    with RenderPipeline(scene=filename, width=160, height=120) as pipeline:
        assert [op.path for op in pipeline.operators] == ["/Group/One", "/Group/Two", "/Group/Untyped"]
        assert all(pipeline.hidden_by(op) == "/Group" for op in pipeline.operators)
        assert not pipeline.last_report.diagnostics
        assert all(item.status == "skipped" for item in pipeline.last_report.operators)
        pipeline.update({("/Group/One", "cae:viz:faces:externalOnly"): False})
        assert read_attribute(pipeline.stage, "/Group/One", "cae:viz:faces:externalOnly", pipeline.ordinal) is False
        other = next(op for op in pipeline.operators if op.path == "/Group/Two")
        assert pipeline.values(other)["cae:viz:faces:externalOnly"] is True
