# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Selected-prim services for OVUI's viewport, without a scene tree."""

from contextlib import nullcontext

from ovui_data_adapters.common import BoundCameraPose


class ViewportStage:
    """Borrow one pipeline; resolve picked paths from immutable file metadata.

    No subscriptions, native metadata queries, visibility editing, or hierarchy
    refresh. Bounds are read only on explicit framing, through pipeline.flush().
    Scene replacement swaps the pipeline after the new scene has opened.
    """

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def get_item_at_path(self, path):
        return path if self.pipeline.in_scene(path) else None

    def get_item_path(self, item):
        return item

    def get_children(self, item):
        # Viewport picking returns the actual hit prim; there is no tree/group UI.
        return ()

    def get_type_name(self, item):
        return self.pipeline.scene_type(item)

    def get_type_category(self, item):
        return self.get_type_name(item)

    def compute_world_aabb(self, paths):
        return self.pipeline.world_bounds(paths)

    def compute_prim_world_aabb_with_extent_fallback(self, path):
        # Gizmo placement uses transforms. Never read product bounds during a drag.
        return None

    def read_bound_camera(self):
        pose = self.pipeline.bound_camera_pose()
        return None if pose is None else BoundCameraPose(**pose)

    def read_stage_up_axis(self):
        return self.pipeline.up_axis

    def begin_undo_group(self, label):
        """OVUI's UndoManager owns commands; the pipeline owns deferred writes."""

    def end_undo_group(self):
        """There is no independent native authoring transaction to close."""

    def suppress_change_notifications(self):
        """No hierarchy subscribers exist; gizmo commits only queue edits."""
        return nullcontext()
