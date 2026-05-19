#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# Reference long-lived listener for cae-streaming.
#
# Pairs with `omniverse_api.py` (next to this file), which provides the
# `@api.request` and `@api.signal` decorators. Authoring a new request type
# is one decorator; the framework handles payload extraction, sync/async
# dispatch, response correlation, and event-stream wiring.
#
# Usage (from the kit-cae repo root):
#
#     bash skills/cae-streaming/scripts/launch_streaming.sh

import asyncio
import importlib
import json
import os
import string
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import carb
import omni.kit.app

# Generic Kit-CAE imports — small, safe to require at module load.
from omni.cae.data.commands import execute_command
from omni.cae.testing import frame_prims, wait_for_update
from omni.usd import get_context

# `omniverse_api.py` lives next to this script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from omniverse_api import OmniverseAPI, exclusive  # noqa: E402

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None  # noqa: N816

# ---------------------------------------------------------------------------
# 1. boot

LOG_INFO = carb.log_info
LOG_WARN = carb.log_warn
LOG_ERROR = carb.log_error
SKILL_DIR = Path(__file__).resolve().parent.parent  # .../skills/cae-streaming
KIT_CAE_DIR = Path(os.environ.get("KIT_CAE_DIR", os.getcwd())).resolve()


def _expand(value: str) -> str:
    return string.Template(value).safe_substitute(os.environ, KIT_CAE_DIR=str(KIT_CAE_DIR))


# Embedded default registry — runnable without external config. `color_field`
# is optional; omit it for a flat-coloured viz.
DEFAULT_SCENES: dict = {
    "static_mixer": {
        "path": "${KIT_CAE_DIR}/data/StaticMixer.cgns",
        "format": "cgns",
        "prim_path": "/World/StaticMixer",
        "dataset_path": "/World/StaticMixer/Base/StaticMixer/StaticMixer_Default",
        "default_viz": "Faces",
        "default_viz_command": "CreateCaeVizFaces",
        "color_field": "/World/StaticMixer/Base/StaticMixer/Flow_Solution/Temperature",
    },
    "disk_out_ref": {
        "path": "${KIT_CAE_DIR}/data/disk_out_ref.npz",
        "format": "npz",
        "schema_type": "Point Cloud",
        "prim_path": "/World/disk_out_ref_npz",
        "dataset_path": "/World/disk_out_ref_npz/NumPyDataSet",
        "default_viz": "Volume",
        "default_viz_command": "CreateCaeVizVolume",
        "default_viz_kwargs": {"type": "vdb"},
        "color_field": "/World/disk_out_ref_npz/NumPyArrays/Temp",
    },
}


def _expand_paths(scenes: dict) -> dict:
    out: dict = {}
    for sid, entry in scenes.items():
        copy = dict(entry)
        if "path" in copy:
            copy["path"] = _expand(copy["path"])
        if "script" in copy:
            copy["script"] = _expand(copy["script"])
        out[sid] = copy
    return out


def _load_registry() -> dict:
    """Resolution order: scenes.json → scenes.yaml (PyYAML) → DEFAULT_SCENES."""
    json_path = SKILL_DIR / "scenes.json"
    if json_path.exists():
        try:
            raw = json.loads(json_path.read_text())
            return _expand_paths(raw.get("scenes", raw))
        except Exception as err:
            LOG_WARN(f"[cae-streaming] failed to parse {json_path}: {err}; falling through")

    yaml_path = SKILL_DIR / "scenes.yaml"
    if yaml_path.exists() and yaml is not None:
        try:
            raw = yaml.safe_load(yaml_path.read_text()) or {}
            return _expand_paths(raw.get("scenes", {}))
        except Exception as err:
            LOG_WARN(f"[cae-streaming] failed to parse {yaml_path}: {err}; falling through")
    elif yaml_path.exists():
        LOG_INFO("[cae-streaming] PyYAML missing; using embedded defaults (drop scenes.json to override)")

    return _expand_paths(DEFAULT_SCENES)


REGISTRY: dict = _load_registry()


def _scene_summary(entry: dict, scene_id: str) -> dict:
    """Client-facing summary of a scene — small, schema-agnostic, safe to push."""
    return {
        "id": scene_id,
        "label": entry.get("label", scene_id.replace("_", " ").title()),
        "description": entry.get("description", ""),
        "format": entry.get("format", ""),
        "thumbnail_path": entry.get("thumbnail_path", ""),
        "asset_path": entry.get("asset_path", entry.get("usda_path", "")),
        "asset_kind": entry.get("asset_kind", ""),
    }


def _emit_scenes_changed() -> None:
    """Push a scenes_changed_signal so connected clients refresh their dropdown."""
    try:
        api.dispatch_signal(
            "scenes_changed",
            {"scenes": [_scene_summary(e, sid) for sid, e in REGISTRY.items()]},
        )
    except Exception as err:
        LOG_WARN(f"[cae-streaming] could not dispatch scenes_changed_signal: {err}")


def _emit_scene_loaded(scene_id: str, result: dict) -> None:
    """Push a scene_loaded_signal on successful load. Distinct from
    scenes_changed (which fires on register), this fires once a scene's prims
    are on the stage — exactly when any thumbnail produced by the loader is
    on disk.

    Clients that track per-scene assets (e.g., showing a thumbnail) listen
    for this signal. The thumbnail_path is whatever the registry entry
    carried; it's the consumer's responsibility to expose that path via its
    own asset-serving endpoint. The `result` arg is currently unused but
    kept on the signature so future custom loaders can pass through extra
    metadata.
    """
    del result  # currently unused; see docstring
    try:
        api.dispatch_signal(
            "scene_loaded",
            {
                "scene_id": scene_id,
                "thumbnail_path": REGISTRY.get(scene_id, {}).get("thumbnail_path", ""),
                "ts": time.time(),
            },
        )
    except Exception as err:
        LOG_WARN(f"[cae-streaming] could not dispatch scene_loaded_signal: {err}")

# Lazy importer resolution — separate Kit extensions ship per format.
_IMPORTER_MODULE = {
    "cgns": "omni.cae.importer.cgns",
    "vtk": "omni.cae.importer.vtk",
    "npz": "omni.cae.importer.npz",
    "ensight": "omni.cae.importer.ensight",
}
_importer_cache: dict[str, Callable[..., Any] | None] = {}


def _resolve_importer(fmt: str) -> Callable[..., Any] | None:
    if fmt in _importer_cache:
        return _importer_cache[fmt]
    module_name = _IMPORTER_MODULE.get(fmt)
    if not module_name:
        _importer_cache[fmt] = None
        return None
    try:
        module = importlib.import_module(module_name)
        fn = getattr(module, "import_to_stage", None)
        _importer_cache[fmt] = fn
        return fn
    except ImportError as err:
        LOG_WARN(f"[cae-streaming] {module_name} not available: {err}")
        _importer_cache[fmt] = None
        return None


api = OmniverseAPI()


# Optional outbound probe — useful when first wiring a new handler set, but
# noisy in production (logs every outbound message snippet). Opt in with
# CAE_STREAMING_DEBUG_PROBE=1.
def _install_outbound_probe() -> None:
    try:
        import carb.eventdispatcher
        from carb.settings import get_settings as _get_settings
        send_evt_name = _get_settings().get_as_string(
            "exts/omni.kit.livestream.messaging/send_message_event"
        )
        if not send_evt_name:
            LOG_WARN("[cae-streaming] cannot install outbound probe: send_message_event setting empty")
            return
        ed = carb.eventdispatcher.get_eventdispatcher()
        def _on_outbound(e):
            try:
                msg = e.payload.get("message")
                LOG_INFO(f"[cae-streaming] outbound probe FIRED: {msg[:200] if msg else '(no message)'}")
            except Exception as err:
                LOG_WARN(f"[cae-streaming] outbound probe formatting err: {err}")
        global _OUTBOUND_PROBE_SUB
        _OUTBOUND_PROBE_SUB = ed.observe_event(event_name=send_evt_name, on_event=_on_outbound)
        LOG_INFO(f"[cae-streaming] outbound probe installed on {send_evt_name!r}")
    except Exception as err:
        LOG_WARN(f"[cae-streaming] failed to install outbound probe: {err}")


_OUTBOUND_PROBE_SUB = None
if os.environ.get("CAE_STREAMING_DEBUG_PROBE") == "1":
    _install_outbound_probe()


# ---------------------------------------------------------------------------
# 2. handlers
#
# Authoring a handler is one decorator. Sync or async — the framework handles
# both. The function name is the operation name (request: load_scene →
# `load_scene_request` / `load_scene_response`).


# Tracks what we currently have on the stage so each load_scene cleans up
# before the next imports — otherwise scenes accumulate.
_LOADED: dict = {"scene_id": None, "prim_path": None, "viz_prim_path": None}


def _clear_previous_load() -> None:
    """Remove the previously loaded scene's prims from the stage (if any)."""
    if _LOADED["scene_id"] is None:
        return
    stage = get_context().get_stage()
    if stage is None:
        return
    paths = [_LOADED.get("prim_path"), _LOADED.get("viz_prim_path")]
    for p in paths:
        if not p:
            continue
        prim = stage.GetPrimAtPath(p)
        if prim and prim.IsValid():
            stage.RemovePrim(p)
            LOG_INFO(f"[cae-streaming] removed prev prim {p}")
    _LOADED.update({"scene_id": None, "prim_path": None, "viz_prim_path": None})


async def _load_scene_via_script(scene_id: str, entry: dict) -> dict:
    """Run a custom Python script for this scene instead of the bundled flow.

    Contract: the script defines `async def load(scene_id: str, entry: dict) -> dict`
    and returns `{"ok": bool, "prim_path": str, "viz_prim_path": str, ...}`.
    The script owns its full setup (import + viz + field binding + framing); we
    only track the returned paths so the next load can clean them up.
    """
    script_path = entry.get("script", "")
    if not script_path:
        return {"ok": False, "error": "entry.script is empty", "scene_id": scene_id}
    full_path = Path(script_path)
    if not full_path.is_absolute():
        full_path = (KIT_CAE_DIR / script_path).resolve()
    if not full_path.exists():
        return {
            "ok": False,
            "error": f"script not found: {full_path}",
            "scene_id": scene_id,
        }

    _clear_previous_load()
    await wait_for_update()

    LOG_INFO(f"[cae-streaming] running scene script {full_path} for {scene_id!r}")
    # Read source + compile + exec directly. Going through importlib's
    # SourceFileLoader honours `__pycache__/*.pyc` — and Kit's bytecompile step
    # writes hash-mode .pyc files that ignore source mtime, so iterative edits
    # to a scene script silently shadow themselves. Bypass the cache entirely.
    import types  # local import: only needed on the script-scene path

    module_name = f"_cae_scene_{scene_id}"
    try:
        source = full_path.read_text(encoding="utf-8")
        code = compile(source, str(full_path), "exec")
    except (OSError, SyntaxError) as err:
        LOG_ERROR(
            f"[cae-streaming] script {full_path.name} failed to read/compile:\n{traceback.format_exc()}"
        )
        return {"ok": False, "error": f"compile_error: {err}", "scene_id": scene_id}

    module = types.ModuleType(module_name)
    module.__file__ = str(full_path)
    sys.modules[module_name] = module  # so any internal `import` of itself works
    try:
        exec(code, module.__dict__)
    except Exception as err:
        LOG_ERROR(
            f"[cae-streaming] script {full_path.name} failed at import:\n{traceback.format_exc()}"
        )
        return {"ok": False, "error": f"import_error: {err}", "scene_id": scene_id}

    load_fn = getattr(module, "load", None)
    if not callable(load_fn):
        return {
            "ok": False,
            "error": f"script {full_path.name} missing load(scene_id, entry) function",
            "scene_id": scene_id,
        }

    try:
        result = await load_fn(scene_id, entry)
    except Exception as err:
        LOG_ERROR(
            f"[cae-streaming] script {full_path.name} load() raised:\n{traceback.format_exc()}"
        )
        return {"ok": False, "error": str(err), "scene_id": scene_id}

    if not isinstance(result, dict):
        return {
            "ok": False,
            "error": f"script returned {type(result).__name__}, expected dict",
            "scene_id": scene_id,
        }
    if not result.get("ok"):
        return {**{"scene_id": scene_id}, **result}

    _LOADED.update({
        "scene_id": scene_id,
        "prim_path": result.get("prim_path"),
        "viz_prim_path": result.get("viz_prim_path"),
    })
    _emit_scene_loaded(scene_id, result)
    # Echo scene_id for caller convenience; let script-supplied keys override.
    return {**{"scene_id": scene_id}, **result}


@api.request
@exclusive
async def load_scene(scene_id: str = "", **_kwargs) -> dict:
    """Bundled example: import a scene, create a default viz, bind a color field.

    Replace with your own handlers as needed. New handler = new decorated
    function. The framework auto-registers `<name>_request` /
    `<name>_response` over the data channel.

    Each load REPLACES the previous one — tracked prims are removed before
    importing so the viewport shows a single dataset, not an accumulation.

    Scenes with a `script` field bypass the bundled importer/CreateCaeViz flow
    and instead defer to that script's `load()` function (see
    `_load_scene_via_script`).
    """
    try:
        if not scene_id:
            return {"ok": False, "error": "scene_id required"}
        entry = REGISTRY.get(scene_id)
        if not entry:
            return {"ok": False, "error": f"unknown scene_id {scene_id!r}", "scene_id": scene_id}

        if entry.get("script"):
            return await _load_scene_via_script(scene_id, entry)

        fmt = entry.get("format", "")
        importer = _resolve_importer(fmt)
        if importer is None:
            return {
                "ok": False,
                "error": f"format {fmt!r} not available in this Kit app (try omni.cae_vtk.kit for VTK)",
                "scene_id": scene_id,
            }

        prim_path = entry["prim_path"]
        import_kwargs = {}
        if fmt == "npz" and "schema_type" in entry:
            import_kwargs["schema_type"] = entry["schema_type"]

        # Replace any previous load before importing.
        _clear_previous_load()
        await wait_for_update()

        LOG_INFO(f"[cae-streaming] importing {scene_id} from {entry['path']}")
        await importer(entry["path"], prim_path, **import_kwargs)
        await wait_for_update()
        _log_imported_prims(prim_path)

        # Resolve dataset_path: configured first, then discover under the import root.
        dataset_path = _discover_dataset_path(prim_path, entry.get("dataset_path"))
        if dataset_path is None:
            return {
                "ok": False,
                "error": f"could not locate a CAE dataset prim under {prim_path!r}; check entry['dataset_path']",
                "scene_id": scene_id,
            }

        viz_prim_path = ""
        color_bound = False
        cmd = entry.get("default_viz_command")
        if cmd:
            label = entry.get("default_viz", "Viz")
            viz_prim_path = f"/World/CAE/{label}_{scene_id}"
            viz_kwargs = dict(entry.get("default_viz_kwargs", {}))
            LOG_INFO(f"[cae-streaming] creating {cmd} at {viz_prim_path} on dataset {dataset_path}")
            await execute_command(
                cmd,
                dataset_path=dataset_path,
                prim_path=viz_prim_path,
                **viz_kwargs,
            )
            await wait_for_update()

            color_field = entry.get("color_field")
            if color_field:
                try:
                    color_bound = _bind_color_field(viz_prim_path, color_field)
                    if color_bound:
                        await wait_for_update()
                except Exception as color_err:
                    LOG_WARN(f"[cae-streaming] color binding failed for {scene_id}: {color_err}")

            # Best-effort camera framing — hard-timed so a slow/missing
            # viewport widget can't hang the response dispatch.
            await _frame_camera([viz_prim_path], zoom=0.05, timeout=15.0)

        # Track this load so the next call can clean it up.
        _LOADED.update({
            "scene_id": scene_id,
            "prim_path": prim_path,
            "viz_prim_path": viz_prim_path or None,
        })

        result = {
            "ok": True,
            "scene_id": scene_id,
            "prim_path": prim_path,
            "viz_prim_path": viz_prim_path,
            "color_field_bound": color_bound,
        }
        _emit_scene_loaded(scene_id, result)
        return result
    except Exception as err:
        LOG_ERROR(f"[cae-streaming] load_scene {scene_id!r} failed:\n{traceback.format_exc()}")
        return {"ok": False, "error": str(err), "scene_id": scene_id}


@api.request
def list_scenes(**_kwargs) -> dict:
    """Return the current scene registry as a flat list. The frontend calls this
    on connect (and after a `scenes_changed_signal`) to populate its dropdown.
    """
    return {
        "ok": True,
        "scenes": [_scene_summary(e, sid) for sid, e in REGISTRY.items()],
    }


@api.request
def register_scene(scene_id: str = "", entry: dict | None = None, **_kwargs) -> dict:
    """Add (or replace) a scene at runtime. Emits `scenes_changed_signal` so all
    connected clients pick up the new entry without a reload.

    Use case: an agent that programmatically generates a new visualization
    POSTs the result here so users can switch to it.

    Required entry keys: `path`, `format`, `prim_path`, `dataset_path`,
    `default_viz_command`. Optional: `label`, `description`, `default_viz`,
    `default_viz_kwargs`, `color_field`, `schema_type`.
    """
    if not scene_id:
        return {"ok": False, "error": "scene_id required"}
    if not isinstance(entry, dict):
        return {"ok": False, "error": "entry must be a dict"}
    if entry.get("script"):
        # Script-based scenes own their own setup; only the script field is required.
        required: tuple[str, ...] = ("script",)
    else:
        required = ("path", "format", "prim_path", "dataset_path", "default_viz_command")
    missing = [k for k in required if k not in entry]
    if missing:
        return {"ok": False, "error": f"missing required keys: {missing}"}
    REGISTRY[scene_id] = _expand_paths({scene_id: entry})[scene_id]
    LOG_INFO(f"[cae-streaming] registered scene: {scene_id}")
    _emit_scenes_changed()
    return {"ok": True, "scene_id": scene_id, "scenes": [_scene_summary(e, sid) for sid, e in REGISTRY.items()]}


@api.request
def deregister_scene(scene_id: str = "", **_kwargs) -> dict:
    """Remove a scene from the registry. Idempotent — already-absent
    scene_id returns ok=true with `removed: false`. Emits
    `scenes_changed_signal` so connected clients refresh their dropdown.

    Use case: a register_scene + load_scene cycle failed (bad loader,
    missing field, malformed asset) and the entry needs to come out of
    the dropdown without restarting the streaming kit.
    """
    if not scene_id:
        return {"ok": False, "error": "scene_id required"}
    removed = REGISTRY.pop(scene_id, None) is not None
    if removed:
        LOG_INFO(f"[cae-streaming] deregistered scene: {scene_id}")
        _emit_scenes_changed()
    else:
        LOG_INFO(f"[cae-streaming] deregister_scene {scene_id!r}: already absent")
    return {
        "ok": True,
        "scene_id": scene_id,
        "removed": removed,
        "scenes": [_scene_summary(e, sid) for sid, e in REGISTRY.items()],
    }


def _log_imported_prims(prim_path: str, max_depth: int = 6) -> None:
    """Walk the imported prim subtree and log it. Helps diagnose 'load reported
    success but nothing visible' — usually means the configured dataset_path
    didn't match any prim the importer created."""
    try:
        from pxr import Usd
    except Exception:
        return
    stage = get_context().get_stage()
    root = stage.GetPrimAtPath(prim_path) if stage else None
    if root is None or not root.IsValid():
        LOG_WARN(f"[cae-streaming] imported root {prim_path} is not valid")
        return
    base_depth = root.GetPath().pathElementCount
    for prim in Usd.PrimRange(root):
        depth = prim.GetPath().pathElementCount - base_depth
        if depth > max_depth:
            continue
        indent = "  " * depth
        LOG_INFO(f"[cae-streaming] tree: {indent}{prim.GetPath()} ({prim.GetTypeName() or '<no type>'})")


def _discover_dataset_path(prim_path: str, configured: str | None = None) -> str | None:
    """Resolve a CAE dataset prim under `prim_path`. Tries the configured path
    first, then walks the subtree looking for a prim with any cae:* attribute
    (the importer applies CAE schemas with that namespace). Returns the first
    valid path or None if nothing matches.
    """
    try:
        from pxr import Usd
    except Exception:
        return configured
    stage = get_context().get_stage()
    if stage is None:
        return configured
    if configured:
        prim = stage.GetPrimAtPath(configured)
        if prim and prim.IsValid():
            return configured
        LOG_WARN(f"[cae-streaming] configured dataset_path {configured!r} not in stage; falling back to discovery")
    root = stage.GetPrimAtPath(prim_path)
    if not root or not root.IsValid():
        return None
    # First pass: any prim with a cae:* property — these are CAE datasets/fields.
    for prim in Usd.PrimRange(root):
        if prim.GetPath() == root.GetPath():
            continue
        for attr in prim.GetPropertyNames():
            if attr.startswith("cae:") or "cae:" in attr.lower():
                LOG_INFO(f"[cae-streaming] discovered dataset prim: {prim.GetPath()}")
                return str(prim.GetPath())
    return None


async def _frame_camera(prim_paths: list[str], zoom: float = 0.05, timeout: float = 15.0) -> bool:
    """Best-effort camera framing with a hard timeout so we never hang the
    response dispatch on a slow / missing viewport widget. Returns True on
    success."""
    try:
        from omni.cae.testing import frame_prims as _frame_prims
        await asyncio.wait_for(_frame_prims(prim_paths, zoom=zoom), timeout=timeout)
        return True
    except asyncio.TimeoutError:
        LOG_WARN(f"[cae-streaming] frame_prims timed out after {timeout}s for {prim_paths}")
        return False
    except Exception as err:
        LOG_WARN(f"[cae-streaming] frame_prims failed for {prim_paths}: {err}")
        return False


def _bind_color_field(viz_prim_path: str, field_path: str) -> bool:
    """Bind `field_path` as the `colors` target on a CAE viz prim. Returns True
    on success. Mirrors the pattern from cae-visualization/SKILL.md and
    example_faces.py."""
    try:
        from omni.cae.schema import viz as cae_viz
    except ImportError as err:
        LOG_WARN(f"[cae-streaming] omni.cae.schema.viz not available: {err}")
        return False
    stage = get_context().get_stage()
    if stage is None:
        return False
    viz_prim = stage.GetPrimAtPath(viz_prim_path)
    if not viz_prim or not viz_prim.IsValid():
        LOG_WARN(f"[cae-streaming] viz prim {viz_prim_path} not valid; cannot bind color")
        return False
    cae_viz.FieldSelectionAPI(viz_prim, "colors").CreateTargetRel().SetTargets([field_path])
    LOG_INFO(f"[cae-streaming] bound colors → {field_path} on {viz_prim_path}")
    return True


def _hide_all_ui() -> None:
    """Hide editor chrome and viewport gizmos. The `.kit` template sets the
    same hideUi setting at startup; this is the defensive re-apply.
    `toggle_ui` action would flip-flop, so we drive the setting directly.
    """
    try:
        carb.settings.get_settings().set("/app/window/hideUi", True)
    except Exception as err:
        LOG_WARN(f"[cae-streaming] could not set /app/window/hideUi: {err}")
        return

    # Hide viewport overlay gizmos. These ARE per-action toggles to specific state.
    try:
        from omni.kit.actions.core import get_action_registry
    except ImportError:
        get_action_registry = None  # type: ignore

    if get_action_registry is not None:
        registry = get_action_registry()
        for action_name in (
            "toggle_camera_visibility",
            "toggle_light_visibility",
            "toggle_hud_visibility",
            "toggle_grid_visibility",
            "toggle_axis_visibility",
        ):
            a = registry.get_action("omni.kit.viewport.actions", action_name)
            if a is not None:
                try:
                    a.execute(visible=False)
                except Exception:
                    pass

    LOG_INFO("[cae-streaming] viewport-only: hideUi=True via carb.settings")


# ---------------------------------------------------------------------------
# 3. main loop


async def _run() -> None:
    LOG_INFO(f"[cae-streaming] starting; scenes={list(REGISTRY)}")
    if get_context().get_stage() is None:
        await get_context().new_stage_async()
        await wait_for_update()

    # Wait for the editor to fully construct its windows before toggling
    # the UI off. Increase if you see the menubar flicker back on.
    app = omni.kit.app.get_app()
    for _ in range(120):
        await app.next_update_async()
    _hide_all_ui()

    while app.is_running():
        await app.next_update_async()


def main() -> None:
    LOG_INFO(f"[cae-streaming] startup; KIT_CAE_DIR={KIT_CAE_DIR}")
    asyncio.ensure_future(_run())


# Kit's --exec imports this module rather than running it as __main__,
# so dispatch unconditionally.
main()
