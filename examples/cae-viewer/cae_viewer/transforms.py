# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""OVUI transform-gizmo adapter over the committed viewer scene."""

from __future__ import annotations

import numpy as np
from ovui_data_adapters.common import TransformAdapter, TransformEditMode, TransformEditPolicy

from .pipeline import as_transform

IDENTITY = np.eye(4)


class SceneTransforms(TransformAdapter):
    """Read committed ``omni:xform`` matrices and enqueue edits, never author USD.

    OVStage and OVUI share USD's row-vector convention, so matrices pass through
    unchanged. A pending edit shadows the committed value until the frame loop
    applies it, so successive drags build on the latest requested transform.
    """

    def __init__(self, pipeline, request_edit, pending_value):
        self.pipeline = pipeline
        self._request_edit = request_edit
        self._pending_value = pending_value

    def local_matrix(self, path):
        pending = self._pending_value((path, "omni:xform"))
        if pending is not None:
            return as_transform(pending)
        value = self.pipeline.local_transform(path)
        return IDENTITY.copy() if value is None else as_transform(value)

    def world_matrix(self, path):
        matrix = self.local_matrix(path)
        parent = path.rpartition("/")[0]
        while parent:
            matrix = matrix @ self.local_matrix(parent)
            parent = parent.rpartition("/")[0]
        return matrix

    def get_local_transform(self, path):
        return self.local_matrix(path).tolist()

    def get_world_transform(self, path):
        return self.world_matrix(path).tolist()

    def set_local_transform(self, path, matrix):
        """Queue the commit; the frame loop writes it and re-evaluates CAE."""
        if not self.can_transform(path):
            raise ValueError(f"cannot transform {path}")
        self._request_edit((path, "omni:xform"), as_transform(matrix).tolist())

    def can_transform(self, path):
        return self.pipeline.transformable(path)

    def get_transform_edit_policy(self, path):
        if self.can_transform(path):
            return TransformEditPolicy(TransformEditMode.DIRECT)
        return TransformEditPolicy(
            TransformEditMode.BLOCKED, reason="Only Xform and point-based prims from the file can be moved"
        )
