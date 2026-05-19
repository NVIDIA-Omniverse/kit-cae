"""Minimal example: load a USDA file as a USD reference onto the live stage.

The caller registers a scene whose entry points at this script (or supplies
`usda_path` directly). On load, this script defines an Xform under
`/World/<scene_id>`, adds the USDA as a reference, polls bounds until the
referenced layer composes (USD references resolve asynchronously), and
frames the viewport.

This is the simplest possible script-based scene loader — useful as a copy
target for any "drop a static USD asset onto the stage" workflow. For
multi-format assets (STL/OBJ/PLY/etc. or VTK), see `asset_reference.py`,
which handles asset_converter and the VTK importer paths.
"""

from __future__ import annotations

import os
from pathlib import Path

from omni.cae.testing import wait_for_update
from omni.kit.viewport.utility import frame_viewport_prims
from omni.usd import get_context
from pxr import Usd, UsdGeom


_DEFAULT_USDA_REL = "skills/cae-streaming/scenes/_artifacts/agent_sphere.usda"


def _resolve_usda(entry: dict) -> Path:
    raw = entry.get("usda_path") or _DEFAULT_USDA_REL
    p = Path(os.path.expandvars(raw))
    if not p.is_absolute():
        kit_cae_dir = Path(os.environ.get("KIT_CAE_DIR", ".")).resolve()
        p = (kit_cae_dir / p).resolve()
    return p


async def load(scene_id: str, entry: dict) -> dict:
    usda_path = _resolve_usda(entry)
    if not usda_path.exists():
        return {"ok": False, "error": f"usda not found: {usda_path}"}

    prim_path = f"/World/{scene_id}"
    stage = get_context().get_stage()

    xform = UsdGeom.Xform.Define(stage, prim_path)
    xform.GetPrim().GetReferences().AddReference(str(usda_path))

    # USD references resolve asynchronously; ComputeWorldBound returns an
    # empty range until the referenced layer is composed. Poll bounds before
    # framing so the camera doesn't fit to an empty box.
    prim = stage.GetPrimAtPath(prim_path)
    imageable = UsdGeom.Imageable(prim)
    for _ in range(120):
        await wait_for_update()
        bbox = imageable.ComputeWorldBound(Usd.TimeCode.Default(), UsdGeom.Tokens.default_)
        if not bbox.GetRange().IsEmpty():
            break

    # Use the same path the viewport's "F" hotkey takes (frame_viewport_selection
    # → frame_viewport_prims). omni.cae.testing.frame_prims goes through the
    # FramePrimsCommand pipeline, which doesn't fit the camera reliably for
    # geometry that arrives via a USD reference rather than import_to_stage.
    frame_viewport_prims(prims=[prim_path])
    await wait_for_update()

    return {
        "ok": True,
        "scene_id": scene_id,
        "prim_path": prim_path,
        "viz_prim_path": prim_path,
        "source": str(usda_path),
    }
