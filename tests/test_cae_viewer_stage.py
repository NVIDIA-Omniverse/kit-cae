# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Viewport metadata remains independent of native hierarchy and evaluation."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("ovui_data_adapters.common")
sys.path.insert(0, str(Path(__file__).parents[1] / "examples" / "cae-viewer"))

from cae_viewer.stage import ViewportStage


def test_selected_paths_need_no_hierarchy_or_native_reads():
    # Deliberately provide no Stage, hierarchy query, or change-stream interface.
    pipeline = SimpleNamespace(
        in_scene=lambda path: path == "/Seeds",
        scene_type=lambda path: "Mesh" if path == "/Seeds" else "",
        up_axis="Z",
        bound_camera_pose=lambda: None,
    )
    adapter = ViewportStage(pipeline)
    for _ in range(100):
        assert adapter.get_item_at_path("/Seeds") == "/Seeds"
        assert adapter.get_item_at_path("/_Viewer/Camera") is None
        assert adapter.get_type_category("/Seeds") == "Mesh"
        assert adapter.get_children("/Seeds") == ()
        assert adapter.compute_prim_world_aabb_with_extent_fallback("/Seeds") is None
    assert adapter.read_stage_up_axis() == "Z"
    assert adapter.read_bound_camera() is None
    # A new file replaces metadata; no stale snapshot survives the swap.
    adapter.pipeline = SimpleNamespace(in_scene=lambda path: path == "/Other")
    assert adapter.get_item_at_path("/Seeds") is None
    assert adapter.get_item_at_path("/Other") == "/Other"


def test_explicit_framing_uses_pipeline_serialization():
    calls = []
    bounds = ((-1, -2, -3), (1, 2, 3))

    def world_bounds(paths):
        calls.append(paths)
        return bounds

    adapter = ViewportStage(SimpleNamespace(world_bounds=world_bounds))
    assert adapter.compute_world_aabb(["/Seeds"]) == bounds
    assert calls == [["/Seeds"]]
