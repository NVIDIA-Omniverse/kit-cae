# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
Automatic dynamic LUT publishing for USD ``Colormap`` prims.

Use case
--------
CAE scenes already use ``Colormap`` prims as the canonical description of a
scientific color ramp. IndeX volume rendering can consume those prims directly,
while MDL shaders typically expect a texture asset for the ``lut`` input. This
module bridges that gap by turning every ``Colormap`` prim on the active stage
into a dynamic texture automatically.

Only prims with ``CaeVizColormapTextureAPI`` applied are managed. The API
carries a ``cae:viz:colormapTexture:identifier`` attribute — a unique string
set once at prim creation — that the manager uses to name the texture:

``dynamic://cae_colormap_<identifier>``

Because the URL is derived from the identifier rather than the prim path, it
remains stable if the prim is relocated via reference/payload composition.

Design
------
The manager is a process-wide singleton owned by ``omni.cae.viz.Extension``.
Call ``ColormapTextureManager.get_instance()`` to obtain the active instance.
Only one instance may exist at a time; constructing a second raises
``RuntimeError``.

It subscribes to stage attach/detach events and listens for USD object changes
on the active stage. Whenever the stage becomes dirty, it rescans all
``Colormap`` prims via the usdrt Fabric stage, samples their
``rgbaPoints``/``xPoints`` into a 1D LUT, and uploads the result to an
``omni.ui.DynamicTextureProvider``.

The manager intentionally has a narrow contract:

- Texture names are derived from the prim's ``identifier`` attribute.
- LUT contents are generated only from ``rgbaPoints`` and ``xPoints``.
- Domain and boundary-mode semantics are left to the consuming shader/system.

This keeps the service reusable for any renderer or shader path that wants a
normalized color ramp texture.
"""

from __future__ import annotations

__all__ = [
    "ColormapTextureManager",
    "build_colormap_lut",
    "get_dynamic_url_for_identifier",
]

import hashlib
from dataclasses import dataclass
from logging import getLogger

import numpy as np
import omni.ui as ui
from omni.cae.schema import viz as cae_viz
from omni.stageupdate import get_stage_update_interface
from omni.usd import get_context
from pxr import Sdf, Tf, Usd, UsdUtils
from usdrt import Usd as UsdRT

logger = getLogger(__name__)


def get_dynamic_url_for_identifier(identifier: str) -> str:
    """Return the ``dynamic://`` URL for a Colormap prim with the given ``CaeVizColormapTextureAPI`` identifier."""
    return f"dynamic://cae_colormap_{identifier}"


def _normalize_colormap_points(rgba_points, x_points) -> tuple[np.ndarray, np.ndarray]:
    """Validate and sort colormap control points by x, returning (rgba Nx4, xs N) float32 arrays."""
    rgba = np.asarray(rgba_points if rgba_points is not None else [], dtype=np.float32)
    xs = np.asarray(x_points if x_points is not None else [], dtype=np.float32)

    if rgba.ndim != 2 or rgba.shape[1] != 4:
        raise ValueError("rgbaPoints must be an Nx4 array")
    if xs.ndim != 1:
        raise ValueError("xPoints must be a 1D array")
    if rgba.shape[0] == 0 or rgba.shape[0] != xs.shape[0]:
        raise ValueError("rgbaPoints and xPoints must have the same non-zero length")

    order = np.argsort(xs, kind="stable")
    rgba = rgba[order]
    xs = xs[order]
    xs = np.clip(xs, 0.0, 1.0)
    return rgba, xs


def build_colormap_lut(rgba_points, x_points, resolution: int) -> np.ndarray:
    """Sample the colormap control points into a uniform ``(resolution, 4)`` float32 LUT in [0, 1]."""
    rgba, xs = _normalize_colormap_points(rgba_points, x_points)
    if resolution <= 0:
        raise ValueError("resolution must be positive")

    if resolution == 1:
        sample_x = np.array([0.0], dtype=np.float32)
    else:
        sample_x = np.linspace(0.0, 1.0, resolution, dtype=np.float32)

    out = np.empty((resolution, 4), dtype=np.float32)
    out[:, 0] = np.interp(sample_x, xs, rgba[:, 0])
    out[:, 1] = np.interp(sample_x, xs, rgba[:, 1])
    out[:, 2] = np.interp(sample_x, xs, rgba[:, 2])
    out[:, 3] = np.interp(sample_x, xs, rgba[:, 3])
    return np.ascontiguousarray(np.clip(out, 0.0, 1.0))


@dataclass
class _ColormapTextureEntry:
    prim_path: str
    texture_name: str
    dynamic_url: str
    provider: ui.DynamicTextureProvider
    fingerprint: str | None = None


class ColormapTextureManager:
    _instance: "ColormapTextureManager | None" = None

    @classmethod
    def get_instance(cls) -> "ColormapTextureManager | None":
        """Return the active singleton instance, or ``None`` if none has been created yet."""
        return cls._instance

    def __init__(self, resolution: int = 256):
        """Create the manager and subscribe to stage attach/detach/update events."""
        if ColormapTextureManager._instance is not None:
            raise RuntimeError("ColormapTextureManager is a singleton — call get_instance() to access the existing one")
        ColormapTextureManager._instance = self
        self._resolution = resolution
        self._stage: Usd.Stage | None = None
        self._stage_rt: UsdRT.Stage | None = None
        self._listener = None
        self._dirty = False
        self._entries: dict[str, _ColormapTextureEntry] = {}

        stage_update_iface = get_stage_update_interface()
        self._stage_subscription = stage_update_iface.create_stage_update_node(
            "cae.viz.colormap_textures",
            on_attach_fn=self.on_attach,
            on_detach_fn=self.on_detach,
            on_update_fn=self.on_update,
        )
        logger.info("ColormapTextureManager initialized (resolution=%d)", resolution)
        self.attach_stage(get_context().get_stage())

    def __del__(self):
        self.finalize()

    def finalize(self):
        """Release all subscriptions, listeners, and texture entries."""
        if self._stage_subscription is not None:
            del self._stage_subscription
            self._stage_subscription = None

        self._listener = None
        self._stage = None
        self._dirty = False
        self._entries.clear()
        if ColormapTextureManager._instance is self:
            ColormapTextureManager._instance = None
        logger.info("ColormapTextureManager finalized")

    def attach_stage(self, stage: Usd.Stage | None):
        """Switch to a new stage (or detach if ``None``), resetting all entries and listeners."""
        self._listener = None
        self._stage = stage if self._is_stage_valid(stage) else None
        self._stage_rt = None
        self._dirty = self._stage is not None
        if self._stage is not None:
            cache = UsdUtils.StageCache.Get()
            stage_id = cache.GetId(self._stage).ToLongInt()
            self._stage_rt = UsdRT.Stage.Attach(stage_id)
            self._listener = Tf.Notice.Register(Usd.Notice.ObjectsChanged, self.on_objects_changed, self._stage)
            logger.info("Attached to stage %d", stage_id)
            self.refresh_stage(self._stage)
        else:
            self._entries.clear()

    def on_attach(self, stageId: int, metersPerUnit: float):
        """Stage-update callback: look up the stage by ID and attach to it."""
        del metersPerUnit
        cache = UsdUtils.StageCache.Get()
        stage = cache.Find(Usd.StageCache.Id.FromLongInt(stageId))
        self.attach_stage(stage)

    def on_detach(self):
        """Stage-update callback: detach from the current stage and clear all entries."""
        logger.info("Detaching from stage")
        self.attach_stage(None)

    def on_update(self, _0, _1):
        """Stage-update callback: flush pending LUT rebuilds if the stage was marked dirty."""
        if self._dirty and self._stage is not None:
            self.refresh_stage(self._stage)

    def on_objects_changed(self, notice: Usd.Notice.ObjectsChanged, stage):
        """USD notice callback: mark the stage dirty if any Colormap prim or its points changed."""
        if stage != self._stage or not isinstance(stage, Usd.Stage):
            return

        changed_prim_paths = {path.GetPrimPath() for path in notice.GetChangedInfoOnlyPaths() if path.IsPropertyPath()}
        changed_prim_paths.update(path for path in notice.GetResyncedPaths() if path.IsPrimPath())
        changed_prim_paths.update(path.GetPrimPath() for path in notice.GetResyncedPaths() if path.IsPropertyPath())

        for path in changed_prim_paths:
            if self._is_relevant_colormap_path(path):
                self._dirty = True
                return

    def has_colormap(self, prim_or_path: Usd.Prim | str | Sdf.Path) -> bool:
        """Return True if a texture entry exists for the given Colormap prim or path."""
        path = self._get_path_string(prim_or_path)
        return path in self._entries

    def get_dynamic_url(self, prim_or_path: Usd.Prim | str | Sdf.Path) -> str:
        """Return the ``dynamic://`` texture URL for the given Colormap prim or path."""
        path = self._get_path_string(prim_or_path)
        entry = self._entries.get(path)
        if entry is None:
            raise KeyError(f"No colormap texture entry for {path!r} — is CaeVizColormapTextureAPI applied?")
        return entry.dynamic_url

    def get_entry(self, prim_or_path: Usd.Prim | str | Sdf.Path) -> _ColormapTextureEntry | None:
        """Return the internal texture entry for the given Colormap prim or path, or ``None`` if not tracked."""
        path = self._get_path_string(prim_or_path)
        return self._entries.get(path)

    def refresh_stage(self, stage: Usd.Stage | None = None):
        """Scan all Colormap prims, publish updated LUTs, and remove entries for deleted prims."""
        if stage is not None:
            self._stage = stage
        if not self._is_stage_valid(self._stage):
            self._entries.clear()
            self._dirty = False
            return

        seen_paths: set[str] = set()
        for prim_path in self._stage_rt.GetPrimsWithTypeName("Colormap"):
            prim = self._stage.GetPrimAtPath(str(prim_path))
            if not prim:
                continue
            if not prim.HasAPI(cae_viz.ColormapTextureAPI):
                continue
            path = prim.GetPath().pathString
            seen_paths.add(path)
            self._create_or_update_entry(prim)

        stale_paths = set(self._entries) - seen_paths
        for path in stale_paths:
            logger.info("Removing stale colormap entry: %s", path)
            del self._entries[path]

        self._dirty = False

    def _create_or_update_entry(self, prim: Usd.Prim):
        """Create a texture entry for ``prim`` if one doesn't exist, then upload the LUT if the content changed."""
        rgba_attr = prim.GetAttribute("rgbaPoints")
        x_attr = prim.GetAttribute("xPoints")
        rgba_points = rgba_attr.Get() if rgba_attr else None
        x_points = x_attr.Get() if x_attr else None
        h = hashlib.blake2b(digest_size=16)
        h.update(np.asarray(rgba_points if rgba_points is not None else [], dtype=np.float32).tobytes())
        h.update(np.asarray(x_points if x_points is not None else [], dtype=np.float32).tobytes())
        h.update(self._resolution.to_bytes(4, "little"))
        fingerprint = h.hexdigest()

        path = prim.GetPath().pathString
        entry = self._entries.get(path)
        if entry is None:
            identifier = cae_viz.ColormapTextureAPI(prim).GetIdentifierAttr().Get()
            texture_name = f"cae_colormap_{identifier}"
            dynamic_url = f"dynamic://{texture_name}"
            logger.info("Registering colormap texture: %s -> %s", path, texture_name)
            entry = _ColormapTextureEntry(
                prim_path=path,
                texture_name=texture_name,
                dynamic_url=dynamic_url,
                provider=ui.DynamicTextureProvider(texture_name),
            )
            self._entries[path] = entry

        if entry.fingerprint == fingerprint:
            return

        try:
            lut = build_colormap_lut(rgba_points, x_points, self._resolution)
        except Exception as exc:
            logger.warning("Skipping Colormap LUT publish for %s: %s", prim.GetPath(), exc)
            return

        tex_data = np.ascontiguousarray((lut.reshape(1, self._resolution, 4) * 255.0).round().astype(np.uint8))
        entry.provider.set_data_array(tex_data, [self._resolution, 1])
        entry.fingerprint = fingerprint
        logger.info("Published LUT for %s", path)

    def _is_relevant_colormap_path(self, path: Sdf.Path) -> bool:
        """Return True if ``path`` refers to a Colormap prim or one of its tracked point/identifier attributes."""
        if path.IsPrimPath():
            prim = self._stage.GetPrimAtPath(path) if self._stage is not None else None
            # conservative: if the prim no longer exists, trigger a refresh to clean up stale entries
            return not prim or prim.GetTypeName() == "Colormap"

        if path.IsPropertyPath():
            prim = self._stage.GetPrimAtPath(path.GetPrimPath()) if self._stage is not None else None
            if not prim or prim.GetTypeName() != "Colormap":
                return False
            attr_name = str(path.GetNameToken())
            if attr_name in {"rgbaPoints", "xPoints"}:
                return prim.HasAPI(cae_viz.ColormapTextureAPI)
            if prim.HasAPI(cae_viz.ColormapTextureAPI):
                return attr_name == cae_viz.ColormapTextureAPI(prim).GetIdentifierAttr().GetName()

        return False

    @staticmethod
    def _get_path_string(prim_or_path: Usd.Prim | str | Sdf.Path) -> str:
        """Normalize a prim, ``Sdf.Path``, or string to a plain path string."""
        if isinstance(prim_or_path, Usd.Prim):
            return prim_or_path.GetPath().pathString
        if isinstance(prim_or_path, Sdf.Path):
            return prim_or_path.pathString
        return str(prim_or_path)

    @staticmethod
    def _is_stage_valid(stage: Usd.Stage | None) -> bool:
        """Return True if ``stage`` is non-None and its pseudo-root is accessible."""
        if stage is None:
            return False
        try:
            stage.GetPseudoRoot()
        except Exception:
            return False
        return True
