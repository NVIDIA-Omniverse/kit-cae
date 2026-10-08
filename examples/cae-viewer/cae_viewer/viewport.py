# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""OVUI renderer contract borrowing the application's existing pipeline."""

from __future__ import annotations

from ovui_data_adapters.common import RendererAdapter

from .pipeline import RenderPipeline


class PipelineAdapter(RendererAdapter):
    """Present one shared OVStage; never create or own another scene or renderer."""

    def __init__(self, pipeline: RenderPipeline) -> None:
        if pipeline.renderer is None or pipeline.stage is None:
            raise ValueError("the viewport requires an open pipeline")
        self.pipeline = pipeline
        self._resolution = (pipeline.width, pipeline.height)
        self._closed = False
        self.previews = {}
        self.pending_picks = {}

    def load_stage(self, stage) -> None:
        """Accept only the Stage already owned and attached by the application."""
        self._require_active()
        if stage is not self.pipeline.stage:
            raise ValueError("the viewport cannot replace the application's Stage")

    def render_frame(self, width, height, view_matrix, proj_matrix):
        """Return RGBA storage that remains valid after the OVRTX frame is unmapped."""
        self._require_active()
        width = int(width or self._resolution[0])
        height = int(height or self._resolution[1])
        return self.pipeline.render_viewport(width, height, view_matrix, proj_matrix)

    def set_resolution(self, width, height) -> None:
        """Defer resize publication until camera and extent can commit together."""
        self._require_active()
        self._resolution = (max(1, int(width)), max(1, int(height)))

    def pick(self, x, y, callback, query_name) -> None:
        """Queue a one-pixel pick; the frame loop answers it once a frame has rendered it.

        OVUI supplies render-image NDC in [-1, 1] with y up; OVRTX wants
        [0, 1] from the top-left corner. OVRTX answers one query per step,
        so picks queued while frames are cached wait their turn in order.
        """
        self._require_active()
        u, v = _to_render_coordinates(x, y)
        ticket = self.pipeline.request_pick(*self._pixel_rect(u, v, u, v))
        self.pending_picks[ticket] = (callback, query_name)

    def cancel_pick(self, query_name) -> None:
        """Forget the named outstanding picks; their callbacks are never invoked."""
        self._require_active()
        for ticket, (_, name) in tuple(self.pending_picks.items()):
            if name == query_name:
                del self.pending_picks[ticket]
                self.pipeline.cancel_pick(ticket)

    def pick_rect(self, x0, y0, x1, y1, callback) -> None:
        """Queue a marquee pick covering every prim inside the render-NDC rectangle."""
        self._require_active()
        (u0, v0), (u1, v1) = _to_render_coordinates(x0, y0), _to_render_coordinates(x1, y1)
        ticket = self.pipeline.request_pick(*self._pixel_rect(min(u0, u1), min(v0, v1), max(u0, u1), max(v0, v1)))
        self.pending_picks[ticket] = (callback, None)

    def _pixel_rect(self, left, top, right, bottom):
        """Widen a normalized rectangle to cover at least one pixel of the committed render size."""
        width, height = self.pipeline.width, self.pipeline.height
        left, top = min(left, 1.0 - 1.0 / width), min(top, 1.0 - 1.0 / height)
        return left, top, min(1.0, max(right, left + 1.0 / width)), min(1.0, max(bottom, top + 1.0 / height))

    def complete_picks(self) -> None:
        """Deliver the hits of a pick rendered this frame; the frame loop calls this once per frame.

        Callbacks publish selection and re-enter the viewport, so they run
        here, outside OVUI's render call, rather than inside render_frame.
        """
        answer = self.pipeline.take_pick_hits()
        if answer is None:
            return
        ticket, hits = answer
        if ticket not in self.pending_picks:
            return
        callback, query_name = self.pending_picks.pop(ticket)
        if query_name is None:
            callback(list(hits))
        else:
            callback(hits[0] if hits else None, None)

    def set_selection_highlight(self, paths) -> None:
        """Outline the selection in the shared renderer."""
        self._require_active()
        self.pipeline.highlight(paths)

    @property
    def supports_live_local_transform(self) -> bool:
        return True

    def set_live_local_transform(self, path, matrix) -> bool:
        """Stash the in-progress gizmo matrix; the frame loop publishes the latest one."""
        self._require_active()
        if not self.pipeline.transformable(path):
            return False
        self.previews[path] = [list(row) for row in matrix]
        return True

    def clear_live_local_transforms(self, paths) -> None:
        """Keep unpublished previews: a cancelled drag restores through this same channel."""
        self._require_active()

    def take_previews(self):
        """Hand the frame loop every preview requested since the previous frame."""
        previews, self.previews = self.previews, {}
        return previews

    def shutdown(self) -> None:
        """Invalidate the adapter; the application still owns the pipeline."""
        self._closed = True

    def _require_active(self) -> None:
        if self._closed:
            raise RuntimeError("the viewport adapter is closed")


def _to_render_coordinates(x, y) -> tuple[float, float]:
    """Map OVUI render NDC ([-1, 1], y up) to OVRTX's [0, 1] top-left-origin coordinates."""
    u = min(1.0, max(0.0, (float(x) + 1.0) / 2.0))
    v = min(1.0, max(0.0, (1.0 - float(y)) / 2.0))
    return u, v
