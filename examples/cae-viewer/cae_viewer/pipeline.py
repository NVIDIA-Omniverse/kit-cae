# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Application-owned OVStage, CAE evaluation, and OVRTX rendering lifecycle."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from ovcae.app import UpdateLoop

from .model import VISIBILITY, describe_operator
from .scene import compose_viewer_scene, hidden_ancestor, prim_paths, read_attribute, scene_bounds, world_aabb

# Prims whose composed local transform the viewer lets the gizmo edit.
TRANSFORMABLE_TYPES = frozenset(
    {"Xform", "Points", "Mesh", "BasisCurves", "NurbsCurves", "HermiteCurves", "NurbsPatch", "TetMesh"}
)
# OVRTX selection-outline group used for the viewport's selection; 0 clears.
SELECTION_GROUP = 1
SELECTION_OUTLINE = (1.0, 0.62, 0.1, 1.0)
SELECTION_FILL = (1.0, 0.62, 0.1, 0.15)

LOGGER = logging.getLogger(__name__)

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
SCENE = EXAMPLE_DIR / "assets" / "scene.usda"


class RenderPipeline:
    """Own the OVStage, the OVRTX renderer, and the OVCAE update loop behind one seam.

    Edits, ordinals, and updates go through :class:`ovcae.app.UpdateLoop`;
    this class adds what is OVRTX-specific: the render rig, picking, selection
    outlines, and the pinned build's need to be detached while an initially
    empty ``Points`` product grows.
    """

    def __init__(
        self,
        *,
        scene: Path | None = None,
        width: int = 1280,
        height: int = 720,
        fps: int = 0,
        device: str = "cuda:0",
    ) -> None:
        if fps < 0:
            raise ValueError("fps must be nonnegative")
        self.width = width
        self.height = height
        self.fps = fps
        self.device = device
        self.filename = Path(scene).expanduser().resolve() if scene is not None else None
        self.operators = ()
        self._scene_paths = frozenset()
        self._transformable = frozenset()
        self._scene_types = {}
        self._loop = None
        self._last_frame = None
        # Frame-loop counters; the application logs and resets them periodically.
        self.cached_frames = self.rendered_frames = 0
        self.render_time = 0.0
        self.render_product = "/_CaeViewer/Render"
        self.camera_path = "/_CaeViewer/Camera"
        self.up_axis = "Y"
        self.bound_camera = None
        self._viewport_signature = None
        self._last_render_time = None
        self._pick_requests = {}
        self._pick_ticket = 0
        self._pick_hits = None
        self._highlighted = ()

        self.renderer = None
        self.stage = None
        self.data_source = None
        self.ovrtx = None
        self._renderer_attached = False

    def open(self) -> RenderPipeline:
        """Create, populate, evaluate, and attach the complete native pipeline."""

        if self.renderer is not None:
            return self

        if self.filename is not None and not self.filename.is_file():
            raise FileNotFoundError(self.filename)

        import ovrtx

        # Register OVRTX before OVStage's first native schema read, matching
        # the fresh-process runtime-order check used by this repository.
        ovrtx.register_schema_paths()

        import ovstage

        import ovcae
        import ovcae.readers

        self.ovrtx = ovrtx
        try:
            # Every non-core schema family must register before either runtime
            # performs its first OpenUSD schema read.
            # Schemas before the first OVStage schema read; the material library
            # before the renderer, whose MDL SDK reads search paths only at start.
            ovcae.setup()
            # Pin rendering to the compute GPU. OVRTX otherwise activates every
            # CUDA device, and with two GPUs plus omni.ui's headless Vulkan
            # backend its scene-delegate copies of updated products fail with
            # cudaErrorInvalidValue, dropping the products from the render.
            self.renderer = ovrtx.Renderer(ovrtx.RendererConfig(active_cuda_gpus=render_gpus(self.device)))
            self.renderer.set_selection_group_styles(
                {SELECTION_GROUP: ovrtx.SelectionGroupStyle(outline_color=SELECTION_OUTLINE, fill_color=SELECTION_FILL)}
            )
            self.stage = ovstage.Stage("ovcae.example.cae-viewer")
            if self.filename:
                self.data_source = ovcae.readers.UsdDataSource.open(self.filename)
            else:
                self.data_source = ovcae.MemoryDataSource()
            text, root, self.up_axis, self.bound_camera = compose_viewer_scene(self.filename, self.width, self.height)
            self.render_product, self.camera_path = root + "/Render", root + "/Camera"
            ovstage.population.open_usd_from_string_with_desc(
                self.stage,
                text,
                ordinal=1,
                time_code=0.0,
                descs=ovcae.ovstage_population_desc(domains=ovstage.PopulationDomain.ALL),
            )
            self.stage.advance_write_floor(1).wait()
            self.operators = tuple(
                describe_operator(path, read_attribute(self.stage, path, "usd-schemas", 1))
                for path in prim_paths(self.stage, 1, schema="CaeVizOperatorAPI")
            )
            self._scene_paths = frozenset(
                path
                for path in set(prim_paths(self.stage, 1)) | {op.path for op in self.operators}
                if path != root and not path.startswith(root + "/")
            )
            self._scene_types = {
                path: read_attribute(self.stage, path, "usd-prim-type", 1) or "" for path in self._scene_paths
            }
            self._transformable = frozenset(
                path for path, kind in self._scene_types.items() if kind in TRANSFORMABLE_TYPES
            )
            self._loop = UpdateLoop(self.stage, self.data_source, ordinal=1, device=self.device)
            self._loop.update()
            LOGGER.info("Attaching OVRTX; the first launch can take several minutes to compile shaders")
            self.renderer.attach_ovstage(self.stage)
            self._renderer_attached = True
            return self
        except BaseException:
            self.close()
            raise

    @property
    def ordinal(self) -> int:
        """The committed ordinal the renderer steps at."""
        return self._loop.ordinal if self._loop is not None else 0

    @property
    def last_report(self):
        """The report of the most recent committed update, or None before the first."""
        return self._loop.last_report if self._loop is not None else None

    def values(self, operator):
        """Read committed control values, falling back to packaged schema defaults."""
        result = {}
        for prop in operator.properties:
            value = read_attribute(self.stage, operator.path, prop.name, self.ordinal)
            result[prop.name] = prop.default if value is None else value
        return result

    def hidden_by(self, operator):
        """Explain why a locally visible operator remains hidden by a parent."""
        return hidden_ancestor(self.stage, operator.path, self.ordinal)

    def scene_type(self, path):
        """File type captured before evaluation; selection never reads native metadata."""
        return self._scene_types.get(path, "")

    def transformable(self, path):
        """Only the file's own Xform and point-based prims accept gizmo edits."""
        return path in self._transformable

    def local_transform(self, path):
        """Return the requested or committed composed local matrix; None when unauthored."""
        requested = self._loop.pending.get((path, "omni:xform"))
        if requested is not None:
            return np.array(requested, dtype=np.float64).reshape(4, 4)
        value = read_attribute(self.stage, path, "omni:xform", self.ordinal)
        return None if value is None else np.asarray(value, dtype=np.float64).reshape(4, 4)

    @property
    def evaluating(self) -> bool:
        """True while a submitted Update has not been committed by pump()."""
        return self._loop is not None and self._loop.evaluating

    def await_evaluation(self, budget: float) -> bool:
        """Block up to ``budget`` seconds for the in-flight Update; True once it is done.

        OVUI's uncapped loop never idles, so the frame loop must give the OVCae
        worker the GIL explicitly or evaluations only finish between drags.
        """
        return self._loop.await_evaluation(budget)

    def camera_pose(self):
        """Return an overview scaled to current visible geometry and stage up axis."""
        # Products are being written while an Update runs; reading them raises.
        self.flush()
        center, radius = scene_bounds(self.stage, self.ordinal)
        direction = np.array((1.0, 0.65, 1.25) if self.up_axis == "Y" else (1.0, -1.25, 0.65))
        eye = center + direction / np.linalg.norm(direction) * radius * 3.2
        return dict(eye=tuple(eye), target=tuple(center), up_axis=self.up_axis, fov_degrees=50.0, prim_path="")

    def bound_camera_pose(self):
        """Return the pose Kit saved with the file, or None to use the overview."""
        return None if self.bound_camera is None else dict(self.bound_camera)

    def world_bounds(self, paths=None):
        """World AABB of visible geometry under ``paths`` (default: everything), or None.

        Framing a selected group covers its subtree; a selection without any
        visible point-based geometry yields None so callers can frame the scene.
        """
        self._require_open()
        self.flush()
        bounds = world_aabb(self.stage, self.ordinal, None if paths is None else tuple(paths))
        if bounds is None:
            return None
        low, high = bounds
        return tuple(float(v) for v in low), tuple(float(v) for v in high)

    def in_scene(self, path):
        """True for prims that came from the file, as opposed to the viewer's own rig."""
        return path in self._scene_paths

    def request(self, edits) -> bool:
        """Validate a batch of (prim path, property name) edits and buffer it.

        Nothing is written here. The viewer checks the batch against its
        controls (types, ranges, the streamline step-size ordering), drops
        values equal to the requested or committed state, and hands the rest
        to the update loop as recipe edits, which coalesce latest-value-per-key
        until pump() submits them. An invalid batch is rejected whole.
        """
        self._require_open()
        properties = {(op.path, prop.name): prop for op in self.operators for prop in op.properties}
        properties.update({(path, "visibility"): VISIBILITY for path in self._scene_paths})
        changes = {}
        for key, value in edits.items():
            if key[1] == "omni:xform" and self.transformable(key[0]):
                matrix = as_transform(value)
                current = self.local_transform(key[0])
                if current is None or not np.array_equal(matrix, current):
                    changes[key] = matrix
                continue
            if key not in properties:
                raise ValueError(f"not an editable operator property: {key}")
            prop = properties[key]
            prop.validate(value)
            if value != self._effective(key, prop.default):
                changes[key] = value
        if not changes:
            return False
        step_properties = {
            "cae:viz:streamlines:minStepSize",
            "cae:viz:streamlines:initialStepSize",
            "cae:viz:streamlines:maxStepSize",
        }
        for op in self.operators:
            if "CaeVizStreamlinesAPI" in op.schemas and any(
                path == op.path and name in step_properties for path, name in changes
            ):
                values = self.values(op)
                values.update({name: value for (path, name), value in changes.items() if path == op.path})
                prefix = "cae:viz:streamlines:"
                if (
                    not values[prefix + "minStepSize"]
                    <= values[prefix + "initialStepSize"]
                    <= values[prefix + "maxStepSize"]
                ):
                    raise ValueError("Step sizes must satisfy minimum <= initial <= maximum")
        types = {key: "matrix4d" if key[1] == "omni:xform" else properties[key].type_name for key in changes}
        self._loop.request(changes, types=types)
        return True

    def pump(self):
        """Advance evaluation without blocking; return a report when one commits.

        The loop commits a finished Update before submitting anything else, so
        the caller renders the committed products first. When a batch would
        make an initially empty ``Points`` product visible, it is submitted
        synchronously with the renderer detached; see :meth:`_submit`.
        """
        self._require_open()
        if self._loop.evaluating or not self._loop.pending:
            return self._loop.pump()
        return self._submit(wait=False)

    def update(self, edits):
        """Synchronously request, submit, and commit one batch; None if nothing changed or no update ran."""
        self._require_open()
        self._loop.flush()
        if not self.request(edits):
            return None
        return self._submit(wait=True)

    def flush(self):
        """Wait for the in-flight Update, if any, and commit it."""
        return self._loop.flush() if self._loop is not None else None

    def _effective(self, key, default):
        """The value an evaluation would see once buffered requests are written."""
        current = self._loop.value(*key)
        return default if current is None else current

    def _submit(self, *, wait: bool):
        """Submit the loop's pending batch, detaching OVRTX around it when a Points product may first appear.

        The pinned OVRTX build cannot grow an initially empty Points buffer, so
        the renderer is detached before the update that fills it and reattached
        once the products are committed; such a batch is always synchronous.
        """
        pending = self._loop.pending

        def will_be_visible(path):
            while path:
                visibility = pending.get((path, "visibility"))
                if visibility is None:
                    visibility = read_attribute(self.stage, path, "visibility", self.ordinal)
                if visibility == "invisible":
                    return False
                path = path.rpartition("/")[0]
            return True

        refresh = False
        for op in self.operators:
            if "CaeVizPointsAPI" not in op.schemas:
                continue
            values = self.values(op)
            values.update({name: value for (path, name), value in pending.items() if path == op.path})
            if values["cae:viz:operator:enabled"] and will_be_visible(op.path):
                points = read_attribute(self.stage, op.path, "points", self.ordinal)
                refresh |= points is None or np.asarray(points).size == 0
        if refresh and self._renderer_attached:
            self.renderer.detach_ovstage()
            self._renderer_attached = False
        try:
            # The renderer must stay detached until the refreshed buffers commit.
            return self._loop.submit(wait=wait or refresh)
        finally:
            if refresh and not self._renderer_attached:
                self.renderer.attach_ovstage(self.stage)
                self._renderer_attached = True

    def render_viewport(self, width: int, height: int, view, projection) -> np.ndarray:
        """Commit OVUI's camera and extent, then return an owned CPU RGBA image.

        OVUI supplies column-vector matrices; USD cameras use row vectors.
        Camera navigation only changes presentation attributes on the shared
        Stage, so it does not need a CAE update. Copy the rendered image before
        unmapping OVRTX's storage: OVUI retains it beyond this call.

        OVRTX cannot step while the Stage holds unsealed writes, and an in-flight
        Update writes its products above the floor. Commit a finished Update
        first; otherwise present the previous frame and defer the camera write.
        """
        self._require_open()
        if width <= 0 or height <= 0:
            raise ValueError("viewport dimensions must be positive")
        if self._loop.evaluating and self._loop.await_evaluation(0.0):
            self._loop.flush()
        if self._loop.evaluating and self._last_frame is not None:
            self.cached_frames += 1
            return self._last_frame
        self._loop.flush()
        world, intrinsics = viewport_camera(view, projection)
        signature = (width, height, world.tobytes(), intrinsics)
        if signature != self._viewport_signature:
            focal, horizontal, vertical, near, far = intrinsics
            edits = {
                (self.render_product, "resolution"): (width, height),
                (self.camera_path, "omni:xform"): world,
                (self.camera_path, "focalLength"): focal,
                (self.camera_path, "horizontalAperture"): horizontal,
                (self.camera_path, "verticalAperture"): vertical,
                (self.camera_path, "clippingRange"): (near, far),
            }
            types = {
                (self.render_product, "resolution"): "int2",
                (self.camera_path, "omni:xform"): "matrix4d",
                (self.camera_path, "focalLength"): "float",
                (self.camera_path, "horizontalAperture"): "float",
                (self.camera_path, "verticalAperture"): "float",
                (self.camera_path, "clippingRange"): "float2",
            }
            self._loop.write(edits, types=types)  # presentation: the floor advances, no update
            self.width, self.height = width, height
            self._viewport_signature = signature
        started = time.perf_counter()
        self._last_frame = self.render(
            device=self.ovrtx.Device.CPU, consume=lambda mapped: np.from_dlpack(mapped).copy()
        )
        self.render_time += time.perf_counter() - started
        self.rendered_frames += 1
        return self._last_frame

    def request_pick(self, left: float, top: float, right: float, bottom: float) -> int:
        """Queue one pick rectangle and return its ticket; rendered frames answer them in order.

        Coordinates are OVRTX's normalized RenderProduct coordinates: [0, 1]
        with the origin at the top-left corner, ``right``/``bottom`` marking
        the edge just past the last included pixel. OVRTX answers one query
        per step, on the step that renders it, so each request waits for its
        own rendered frame and is collected by :meth:`take_pick_hits`.
        """
        self._require_open()
        rect = tuple(min(1.0, max(0.0, float(value))) for value in (left, top, right, bottom))
        if not (rect[0] < rect[2] and rect[1] < rect[3]):
            raise ValueError("a pick rectangle must include at least one pixel")
        self._pick_ticket += 1
        self._pick_requests[self._pick_ticket] = rect
        return self._pick_ticket

    def cancel_pick(self, ticket: int) -> None:
        """Drop a queued pick that has not been rendered yet."""
        self._pick_requests.pop(ticket, None)

    def take_pick_hits(self):
        """Return ``(ticket, prim paths)`` for the query the last frame answered, once; else None."""
        hits, self._pick_hits = self._pick_hits, None
        return hits

    def highlight(self, paths) -> None:
        """Outline exactly ``paths`` in the renderer; unchanged selections cost nothing."""
        self._require_open()
        paths = tuple(dict.fromkeys(str(path) for path in paths if path))
        if paths == self._highlighted:
            return
        stale = [path for path in self._highlighted if path not in paths]
        if stale:
            self.renderer.set_selection_outline_group_strings(stale, 0)
        if paths:
            self.renderer.set_selection_outline_group_strings(list(paths), SELECTION_GROUP)
        self._highlighted = paths

    def render(self, *, device, consume: Callable[[Any], Any]):
        """Render one frame and consume its mapped RGBA array while it is valid."""

        self._require_open()
        now = time.perf_counter()
        # Uncapped rendering has no nominal frame period. Report elapsed wall
        # time to OVRTX; CAE scientific time remains independently fixed at zero.
        elapsed = 0.0 if self._last_render_time is None else max(0.0, now - self._last_render_time)
        pick = next(iter(self._pick_requests), None)
        if pick is not None:
            self.renderer.enqueue_pick_query(self.render_product, *self._pick_requests.pop(pick))
        products = self.renderer.step(
            render_products={self.render_product},
            delta_time=1.0 / self.fps if self.fps else elapsed,
            ordinal=self.ordinal,
        )
        self._last_render_time = now
        frame = products[self.render_product].frames[0]
        if pick is not None:
            self._pick_hits = (pick, self._read_pick_hits(frame))
        mapped = frame.render_vars[f"{self.render_product}/LdrColor"].map(device=device)
        try:
            return consume(mapped)
        finally:
            mapped.unmap()
            del mapped, frame, products

    def _read_pick_hits(self, frame) -> tuple[str, ...]:
        """Resolve the frame's pick-hit records to unique prim paths in OVRTX's order."""
        output = frame.render_vars.get(self.ovrtx.OVRTX_RENDER_VAR_PICK_HIT)
        if output is None:
            LOGGER.warning("OVRTX rendered the pick query without a %s output", self.ovrtx.OVRTX_RENDER_VAR_PICK_HIT)
            return ()
        mapped = output.map(device=self.ovrtx.Device.CPU)
        try:
            params = mapped.params
            magic, version = (int(np.from_dlpack(params[name])) for name in ("magic", "version"))
            if (magic, version) != (self.ovrtx.OVRTX_PICK_HIT_MAGIC, self.ovrtx.OVRTX_PICK_HIT_VERSION):
                LOGGER.warning("Unsupported OVRTX pick-hit schema %#x v%d", magic, version)
                return ()
            ids = np.from_dlpack(mapped["primPath"]).astype(np.uint64, copy=True)
        finally:
            mapped.unmap()
        return tuple(dict.fromkeys(self.renderer.resolve_prim_path_id(int(prim)) for prim in ids))

    def close(self) -> None:
        """Release consumers before the borrowed Stage and renderer."""

        self._last_render_time = None
        self._last_frame = None
        self._pick_requests, self._pick_hits = {}, None
        self._highlighted = ()
        if self._loop is not None:
            # Session.close drains the worker; an in-flight output is deliberately left uncommitted.
            self._loop.close()
            self._loop = None
        if self.data_source is not None:
            self.data_source.close()
            self.data_source = None
        if self.renderer is not None and self._renderer_attached:
            self.renderer.detach_ovstage()
            self._renderer_attached = False
        if self.stage is not None:
            self.stage.destroy()
            self.stage = None
        if self.renderer is not None:
            self.renderer.destroy()
            self.renderer = None
        self.ovrtx = None

    def __enter__(self) -> RenderPipeline:
        return self.open()

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.close()

    def _require_open(self) -> None:
        if self.renderer is None:
            raise RuntimeError("RenderPipeline is not open")


def render_gpus(device: str) -> str | None:
    """Map a Warp device string to OVRTX's ``active_cuda_gpus`` list.

    ``cuda:N`` pins the renderer to that GPU and bare ``cuda`` to the first one;
    anything else (``cpu``, ``auto``) leaves OVRTX's own device selection alone.
    """
    name, _, index = str(device).partition(":")
    if name != "cuda":
        return None
    if index and not index.isdigit():
        raise ValueError(f"unsupported CUDA device: {device!r}")
    return index or "0"


def as_transform(value) -> np.ndarray:
    """Validate one requested local transform as a finite 4x4 row-vector matrix."""
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.size != 16:
        raise ValueError("a transform must have sixteen components")
    matrix = matrix.reshape(4, 4)
    if not np.isfinite(matrix).all():
        raise ValueError("a transform must be finite")
    return matrix


def viewport_camera(view, projection) -> tuple[np.ndarray, tuple[float, ...]]:
    """Translate OVUI's symmetric perspective camera into USD camera values."""
    view = np.asarray(view, dtype=np.float64)
    projection = np.asarray(projection, dtype=np.float64)
    if view.shape != (4, 4) or projection.shape != (4, 4):
        raise ValueError("view and projection matrices must be 4x4")
    if not np.isfinite(view).all() or not np.isfinite(projection).all():
        raise ValueError("camera matrices must be finite")
    if projection[0, 0] <= 0 or projection[1, 1] <= 0 or not np.isclose(projection[3, 2], -1):
        raise ValueError("the viewer requires a perspective camera")
    if projection[2, 2] >= -1 or projection[2, 3] >= 0:
        raise ValueError("the viewer requires finite perspective clipping planes")
    near = projection[2, 3] / (projection[2, 2] - 1)
    far = projection[2, 3] / (projection[2, 2] + 1)
    focal_length = 28.0
    intrinsics = (focal_length, 2 * focal_length / projection[0, 0], 2 * focal_length / projection[1, 1], near, far)
    return np.ascontiguousarray(np.linalg.inv(view).T), intrinsics
