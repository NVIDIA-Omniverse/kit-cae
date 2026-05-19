"""Generic asset loader: drop USD/USDA/STL/OBJ/PLY/FBX/GLB or VTK files
onto the live cae-streaming stage.

Invoked when a registry entry's `script` field points at this file (see
`register_scene` in `serve.py`). Lets a caller drop an asset onto the
live stage by registering its path — no need to author a bespoke
per-scene loader for simple preview / visualization use.

Three load paths, dispatched by file extension:

  USD / USDA / USDC / USDZ
      Reference the layer directly onto a fresh Xform at
      `/World/<scene_id>`. A local `xformOp:scale` is applied via
      `XformCommonAPI.SetScale(ref_mpu / stage_mpu)` so the geometry
      lands at correct physical size on the first composed frame —
      USD's local opinion overrides any same-name opinion the
      referenced layer authored. With honest metadata, no snap-reframe.

  STL / OBJ / PLY / FBX / GLB / GLTF / 3MF / DAE / STEP
      Run through `omni.kit.asset_converter` with two non-default
      options:
        - `use_meter_as_world_unit=True`  — the converter rescales the
          asset and writes the output USD with `metersPerUnit=1.0`.
          This is the magic that fixes the long-standing "labeled cm,
          values in m" bug in the default converter output.
        - `convert_stage_up_z=True`       — output Z-up, matching the
          cae-streaming stage. No 90° rotation correction needed.
      Cache the output next to the source as `<name>.<ext>.cache.usda`,
      then take the USD path above.

  VTU / VTI / VTP / VTS / VTK
      Hand the file to `omni.cae.importer.vtk.import_to_stage(path,
      prim_path)`. The importer is async, doesn't write a sidecar USD,
      and authors `cae.DataSet` (with format-specific schema APIs and
      every point/cell field array preserved as `cae_vtk.FieldArray`
      children) directly at `prim_path` on the live stage. The
      `omni.cae.delegate.vtk` extension picks up the schema at
      composition time and exposes the volume to viz pipelines —
      transfer functions, glyphs, slices, streamlines.

In all three branches the post-load steps are identical: a brief Fabric
settle, `frame_viewport_prims([prim_path])`, then a background thumbnail
capture so the load response and `scene_loaded_signal` fire promptly
without waiting on encoder settle.

Cross-format USDA wrapper handling: if the source is a thin USDA whose
first inline reference points at a non-USD layer (the
`wrapper.usda → @./model.stl@` pattern), follow the reference and
process the inner file. asset_converter doesn't recursively resolve
cross-format refs and the VTK importer doesn't read USD wrappers
either, so unwrapping here is the only path that produces a usable
result for that emit pattern.
"""

from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path

import carb
from omni.cae.testing import wait_for_update
from omni.kit.viewport.utility import (
    capture_viewport_to_file,
    frame_viewport_prims,
    get_active_viewport,
)
from omni.usd import get_context
from pxr import Gf, Usd, UsdGeom

_USD_EXTS = frozenset({".usd", ".usda", ".usdc", ".usdz"})
_VTK_EXTS = frozenset({".vtu", ".vti", ".vtp", ".vts", ".vtk"})

# USD asset paths in text USDA look like `@./foo.stl@` or `@/abs/foo.obj@`.
# Producers sometimes emit a thin USDA wrapper that `references` an
# external mesh — asset_converter on the wrapper produces an empty shell
# (it doesn't recursively resolve cross-format refs), so we unwrap and
# convert the referenced file directly.
_NON_USD_REF_RE = re.compile(
    r"@([^@\n]+\.(?:stl|obj|ply|fbx|gltf|glb|3mf|step|stp|dae|vtu|vti|vtp|vts|vtk))\b[^@\n]*@",
    re.IGNORECASE,
)


def _resolve_source(entry: dict) -> Path | None:
    """Resolve the source asset path from a registry entry.

    Canonical field is `asset_path`; older entries may carry `usda_path`
    or `usd_ref` from before the rename — accepted for transition
    compatibility. Relative paths are tried under `$CAE_FS_ROOT` first
    (an optional shared root for caller-provided assets), then
    `$KIT_CAE_DIR` (where bundled scene assets live). Returns the first
    existing match, or the `$CAE_FS_ROOT`-resolved path if neither
    hits — so the load() error message points at the more useful root.
    """
    raw = (
        entry.get("asset_path")
        or entry.get("usda_path")
        or entry.get("usd_ref")
    )
    if not raw:
        return None
    p = Path(os.path.expandvars(raw))
    if p.is_absolute():
        return p
    cae_fs_root = os.environ.get("CAE_FS_ROOT")
    candidates: list[Path] = []
    if cae_fs_root:
        candidates.append((Path(cae_fs_root).expanduser().resolve() / p).resolve())
    candidates.append(
        (Path(os.environ.get("KIT_CAE_DIR", ".")).resolve() / p).resolve()
    )
    for cand in candidates:
        if cand.exists():
            return cand
    # Fall back to the first candidate so the "source not found" error
    # message references the path the caller likely intended.
    return candidates[0]


def _extract_first_non_usd_ref(usda: Path) -> Path | None:
    """If `usda` is a text USDA whose first inline reference points at a
    non-USD layer, return the resolved host path to that layer. Returns
    None otherwise (USDC is binary, so we don't try).
    """
    if usda.suffix.lower() != ".usda":
        return None
    try:
        with usda.open("r", encoding="utf-8", errors="ignore") as f:
            head = f.read(64 * 1024)
    except OSError:
        return None
    m = _NON_USD_REF_RE.search(head)
    if not m:
        return None
    raw = m.group(1)
    p = Path(raw)
    if not p.is_absolute():
        p = (usda.parent / p).resolve()
    return p if p.exists() else None


async def _convert_to_usd(source: Path) -> Path | None:
    """Convert a non-USD mesh file to USDA via omni.kit.asset_converter.

    Cached next to the source as `<name>.<ext>.cache.usda` and reused
    unless the source mtime is newer.

    `use_meter_as_world_unit=True` is what makes the output well-behaved:
    the converter rescales the asset and tags the output with
    `metersPerUnit=1.0` — honest metadata that downstream code can rely
    on for unit conversion.
    """
    out_path = source.with_suffix(source.suffix + ".cache.usda")
    if out_path.exists() and out_path.stat().st_mtime >= source.stat().st_mtime:
        return out_path

    try:
        import omni.kit.asset_converter as converter  # type: ignore[import-not-found]
    except ImportError as exc:
        carb.log_warn(
            f"[asset_reference] omni.kit.asset_converter not available: {exc}"
        )
        return None

    options = converter.AssetConverterContext()
    options.use_meter_as_world_unit = True
    options.convert_stage_up_z = True
    options.ignore_animations = True
    options.ignore_camera = True
    options.ignore_light = True

    task_manager = converter.get_instance()
    task = task_manager.create_converter_task(
        str(source), str(out_path), None, options
    )
    try:
        success = await task.wait_until_finished()
    except Exception as exc:  # noqa: BLE001
        carb.log_warn(f"[asset_reference] asset_converter raised: {exc}")
        return None
    if not success:
        msg = ""
        if hasattr(task, "get_error_message"):
            try:
                msg = task.get_error_message()
            except Exception:  # noqa: BLE001
                pass
        carb.log_warn(f"[asset_reference] asset_converter failed: {msg}")
        return None
    if not out_path.exists():
        carb.log_warn(
            f"[asset_reference] asset_converter reported success but {out_path} is missing"
        )
        return None
    return out_path


async def _import_vtk(source: Path, prim_path: str) -> str | None:
    """Hand a VTK file to `omni.cae.importer.vtk.import_to_stage`.

    The importer authors a `cae.DataSet` prim (with format-specific
    schema APIs and all field arrays preserved) directly at
    `prim_path` on the live stage. No sidecar USD is written.

    Returns None on success, or an error string describing the failure.
    """
    try:
        from omni.cae.importer.vtk import import_to_stage  # type: ignore[import-not-found]
    except ImportError as exc:
        return f"omni.cae.importer.vtk unavailable: {exc}"
    try:
        await import_to_stage(str(source), prim_path)
    except Exception as exc:  # noqa: BLE001
        return f"VTK import failed: {exc}"
    return None


# Field-name heuristic for default color binding. The first match wins;
# non-matches fall through to "first scalar PointData field." Names are
# the slash-suffix of the prim path (`<dataset>/PointData/<name>`). The
# list covers common solver outputs (velocity magnitudes, pressure,
# density, temperature) and classic VTK demo datasets (which often use
# `Scalars_`).
_PREFERRED_COLOR_FIELDS = (
    "u_magnitude", "u_mag", "velocity_magnitude", "velocity",
    "speed",
    "pressure", "p", "rho", "density",
    "temperature", "T", "T_out",
    "Scalars_", "scalars",
)


def _find_dataset_prim(stage, root_prim_path: str) -> str | None:
    """Return the path of the first `cae.DataSet`-typed prim under
    (or at) `root_prim_path`, or None if none exists."""
    try:
        from omni.cae.schema import cae as _cae  # type: ignore[import-not-found]
    except ImportError as exc:
        carb.log_warn(f"[asset_reference] omni.cae.schema not importable: {exc}")
        return None
    root = stage.GetPrimAtPath(root_prim_path)
    if not root or not root.IsValid():
        return None
    for prim in Usd.PrimRange(root):
        if prim.IsA(_cae.DataSet):
            return prim.GetPath().pathString
    return None


def _pick_color_field(stage, root_prim_path: str) -> str | None:
    """Find a sensible field prim under `<root>/PointData/` or
    `<root>/CellData/` to bind to the volume's `colors` channel.

    The VTK importer puts field arrays at `<root>/{PointData|CellData}/<name>`
    — siblings of the dataset prim, not children — so we walk those
    scopes directly rather than the dataset.
    """
    candidates: list[str] = []
    for container in ("PointData", "CellData"):
        scope = stage.GetPrimAtPath(f"{root_prim_path}/{container}")
        if not scope or not scope.IsValid():
            continue
        for child in scope.GetChildren():
            candidates.append(child.GetPath().pathString)
    if not candidates:
        return None
    for preferred in _PREFERRED_COLOR_FIELDS:
        for path in candidates:
            if path.rsplit("/", 1)[-1] == preferred:
                return path
    return candidates[0]


def _pick_viz_kind(stage, dataset_path: str) -> str:
    """Return one of `vdb`, `irregular`, or `faces` based on the
    format-specific schema applied to the dataset prim. Default is
    `vdb` (DenseVolume, StructuredGrid, ImageData)."""
    prim = stage.GetPrimAtPath(dataset_path)
    if not prim or not prim.IsValid():
        return "vdb"
    schemas = list(prim.GetAppliedSchemas())
    joined = " ".join(schemas)
    if "PolyData" in joined:
        return "faces"
    if "UnstructuredGrid" in joined or "PointCloud" in joined:
        return "irregular"
    return "vdb"


async def _visualize_vtk(stage, scene_id: str, root_prim_path: str) -> str:
    """Wire a default volume / faces viz operator to a freshly imported
    VTK dataset and bind a field for color. Returns the viz prim path
    (or `root_prim_path` on any failure — the caller can still frame on
    the import root, the user just won't see anything until manual viz
    setup).
    """
    try:
        from omni.cae.data.commands import execute_command  # type: ignore[import-not-found]
        from omni.cae.schema import viz as cae_viz  # type: ignore[import-not-found]
    except ImportError as exc:
        carb.log_warn(
            f"[asset_reference] cae viz commands unavailable; default viz "
            f"setup skipped: {exc}"
        )
        return root_prim_path

    dataset_path = _find_dataset_prim(stage, root_prim_path)
    if not dataset_path:
        carb.log_warn(
            f"[asset_reference] no cae.DataSet prim under {root_prim_path}; "
            "default viz setup skipped"
        )
        return root_prim_path

    viz_kind = _pick_viz_kind(stage, dataset_path)
    field_path = _pick_color_field(stage, root_prim_path)

    try:
        if viz_kind == "faces":
            viz_path = f"{root_prim_path}/Faces"
            await execute_command(
                "CreateCaeVizFaces",
                dataset_path=dataset_path,
                prim_path=viz_path,
            )
        else:
            viz_path = f"{root_prim_path}/Volume"
            await execute_command(
                "CreateCaeVizVolume",
                dataset_path=dataset_path,
                prim_path=viz_path,
                type=viz_kind,
            )
    except Exception as exc:  # noqa: BLE001
        carb.log_warn(f"[asset_reference] CreateCaeViz{viz_kind} failed: {exc}")
        return root_prim_path

    if field_path:
        try:
            viz_prim = stage.GetPrimAtPath(viz_path)
            cae_viz.FieldSelectionAPI(viz_prim, "colors").CreateTargetRel().SetTargets([field_path])
            carb.log_info(
                f"[asset_reference] {scene_id}: {viz_kind} bound to color "
                f"field {field_path}"
            )
        except Exception as exc:  # noqa: BLE001
            carb.log_warn(f"[asset_reference] color binding failed: {exc}")
    else:
        carb.log_warn(
            f"[asset_reference] {scene_id}: no PointData/CellData fields "
            "found; volume will render flat"
        )
    return viz_path


def _apply_unit_correction(stage, xform, ref_path: Path) -> None:
    """Set a uniform `xformOp:scale` on the local Xform that compensates
    for the reference's `metersPerUnit` vs. the stage's. With honest
    metadata from the converter, this is the entire fix — geometry
    lands at correct physical size on the very first composed frame.
    """
    try:
        ref_stage = Usd.Stage.Open(str(ref_path))
        if ref_stage is None:
            return
        ref_mpu = UsdGeom.GetStageMetersPerUnit(ref_stage) or 1.0
        stage_mpu = UsdGeom.GetStageMetersPerUnit(stage) or 1.0
        if abs(ref_mpu - stage_mpu) > 1e-9 and stage_mpu > 0:
            scale = ref_mpu / stage_mpu
            UsdGeom.XformCommonAPI(xform).SetScale(
                Gf.Vec3f(scale, scale, scale)
            )
    except Exception as exc:  # noqa: BLE001
        carb.log_warn(f"[asset_reference] unit correction failed: {exc}")


async def _capture_thumbnail(asset_path: Path) -> None:
    """Background viewport capture so the load response isn't gated on
    encoder settle. Best-effort: failures are logged but never raise.
    Thumbnail path matches the asset's path with `.png` appended, so
    consumers can probe `<asset>.png` without knowing the load branch.
    """
    try:
        png_path = asset_path.with_suffix(asset_path.suffix + ".png")
        viewport = get_active_viewport()
        if viewport is None:
            return
        for _ in range(4):
            await wait_for_update()
        capture_viewport_to_file(viewport, file_path=str(png_path))
        for _ in range(20):
            await wait_for_update()
    except Exception as exc:  # noqa: BLE001
        carb.log_warn(f"[asset_reference] thumbnail capture failed: {exc}")


async def load(scene_id: str, entry: dict) -> dict:
    source_path = _resolve_source(entry)
    if source_path is None:
        return {
            "ok": False,
            "error": (
                "registry entry missing 'asset_path'; set it on the "
                "entry passed to register_scene."
            ),
        }
    if not source_path.exists():
        return {"ok": False, "error": f"source not found: {source_path}"}

    # Unwrap thin USDA wrappers that point at a non-USD layer (the
    # `wrapper.usda → @./model.stl@` pattern). Neither asset_converter
    # nor the VTK importer follow cross-format USD references, so we
    # process the inner file directly.
    convert_input = source_path
    ext = source_path.suffix.lower()
    if ext == ".usda":
        inner = _extract_first_non_usd_ref(source_path)
        if inner is not None:
            carb.log_info(
                f"[asset_reference] {source_path.name} wraps {inner.name}; "
                "using the referenced file directly"
            )
            convert_input = inner
            ext = convert_input.suffix.lower()

    prim_path = f"/World/{scene_id}"
    stage = get_context().get_stage()

    viz_prim_path = prim_path
    if ext in _VTK_EXTS:
        # VTK path: importer authors `cae.DataSet` + format-specific
        # schema APIs directly at `prim_path` with all field arrays
        # exposed. No Xform / AddReference / unit correction needed.
        # Importing alone isn't enough — without a viz operator the
        # dataset prim is invisible. _visualize_vtk wires a default
        # CreateCaeVizVolume (or CreateCaeVizFaces for PolyData) and
        # binds a sensible color field so the user actually sees
        # something on first composed frame.
        carb.log_info(
            f"[asset_reference] importing {convert_input.name} via "
            "omni.cae.importer.vtk"
        )
        err = await _import_vtk(convert_input, prim_path)
        if err is not None:
            return {"ok": False, "error": err, "scene_id": scene_id}
        # Brief settle so the imported prim hierarchy + field prims are
        # discoverable by USD before we author viz ops on top of them.
        for _ in range(4):
            await wait_for_update()
        viz_prim_path = await _visualize_vtk(stage, scene_id, prim_path)
    else:
        # USD / asset_converter path: resolve to a USD layer, single
        # Xform with AddReference, local `xformOp:scale` for honest
        # unit correction.
        if ext in _USD_EXTS:
            ref_path = convert_input
        else:
            carb.log_info(
                f"[asset_reference] converting {convert_input.name} via "
                "asset_converter (use_meter_as_world_unit=True, "
                "convert_stage_up_z=True)"
            )
            converted = await _convert_to_usd(convert_input)
            if converted is None:
                return {
                    "ok": False,
                    "error": (
                        f"could not convert {convert_input.name} to USD via "
                        "omni.kit.asset_converter; see kit log for details."
                    ),
                    "scene_id": scene_id,
                }
            ref_path = converted

        # USD's local opinion on the prim overrides any same-name opinion
        # the referenced layer authored (USD value resolution: local
        # layer wins). Apply the scale BEFORE any wait_for_update so the
        # geometry is at correct size on the first rendered frame — no
        # visible snap-to-frame.
        xform = UsdGeom.Xform.Define(stage, prim_path)
        xform.GetPrim().GetReferences().AddReference(str(ref_path))
        _apply_unit_correction(stage, xform, ref_path)

    # Brief settle for Fabric to pick up the new prim hierarchy.
    for _ in range(8):
        await wait_for_update()

    frame_viewport_prims(prims=[prim_path])
    await wait_for_update()

    # Background thumbnail — keeps the load response prompt.
    asyncio.ensure_future(_capture_thumbnail(source_path))

    return {
        "ok": True,
        "scene_id": scene_id,
        "prim_path": prim_path,
        "viz_prim_path": viz_prim_path,
        "source": str(source_path),
    }
