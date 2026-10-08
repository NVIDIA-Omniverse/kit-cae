# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Transform adapter semantics, without creating UI or rendering resources."""

import sys
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

pytest.importorskip("ovui_data_adapters.common")
sys.path.insert(0, str(Path(__file__).parents[1] / "examples" / "cae-viewer"))

from cae_viewer.transforms import SceneTransforms
from ovui_data_adapters.common import TransformEditMode


def translation(x, y, z):
    matrix = np.eye(4)
    matrix[3, :3] = (x, y, z)
    return matrix


def make_adapter(committed, transformable):
    pipeline = Mock()
    pipeline.local_transform.side_effect = committed.get
    pipeline.transformable.side_effect = lambda path: path in transformable
    pending = {}
    adapter = SceneTransforms(pipeline, pending.__setitem__, pending.get)
    return adapter, pending


def test_pending_edits_shadow_committed_transforms_and_compose_with_ancestors():
    committed = {"/World": translation(10, 0, 0), "/World/Seeds": translation(1, 2, 3)}
    adapter, pending = make_adapter(committed, {"/World/Seeds"})
    assert adapter.get_local_transform("/World/Seeds") == translation(1, 2, 3).tolist()
    assert adapter.get_local_transform("/World/Unauthored") == np.eye(4).tolist()
    # Row-vector convention: the world translation is local + parent.
    assert adapter.get_world_transform("/World/Seeds")[3][:3] == [11.0, 2.0, 3.0]
    adapter.set_local_transform("/World/Seeds", translation(5, 5, 5).tolist())
    assert pending == {("/World/Seeds", "omni:xform"): translation(5, 5, 5).tolist()}
    # A subsequent drag starts from the requested, not yet committed, matrix.
    assert adapter.get_local_transform("/World/Seeds") == translation(5, 5, 5).tolist()
    assert adapter.get_world_transform("/World/Seeds")[3][:3] == [15.0, 5.0, 5.0]


def test_policies_and_validation_reject_unsupported_edits():
    adapter, pending = make_adapter({}, {"/World/Seeds"})
    assert adapter.can_transform("/World/Seeds")
    assert adapter.get_transform_edit_policy("/World/Seeds").mode is TransformEditMode.DIRECT
    blocked = adapter.get_transform_edit_policy("/World/Recipes/Surface")
    assert blocked.mode is TransformEditMode.BLOCKED and blocked.reason
    with pytest.raises(ValueError):
        adapter.set_local_transform("/World/Recipes/Surface", np.eye(4).tolist())
    with pytest.raises(ValueError):
        adapter.set_local_transform("/World/Seeds", [[1, 2, 3]])
    with pytest.raises(ValueError):
        adapter.set_local_transform("/World/Seeds", np.full((4, 4), np.nan).tolist())
    assert not pending
