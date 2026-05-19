---
name: cae-core
description: >
  Shared foundation for Kit-CAE agent skills. Contains import APIs, visualization commands,
  stage discovery, format references, and utility scripts used by cae-visualization
  and cae-capture. Not invoked directly by users - loaded as a dependency by other
  CAE skills.
version: "2.1.0"
metadata:
  author: "NVIDIA"
  tags:
    - kit-cae
    - cae
    - foundation
    - shared-dependency
    - import
    - z-up
---

# CAE Core - Shared Foundation

Reference library for other CAE skills (`cae-visualization`, `cae-capture`).
When formats or APIs change, update files here once.

## Purpose

Shared foundation for Kit-CAE agent skills (`cae-visualization`, `cae-capture`, `cae-streaming`). Centralizes preflight, launch flags, import APIs, stage discovery, and the Z-up coordinate convention so the other skills do not re-document the same primitives.

## Prerequisites

A built Kit-CAE checkout. Validate with the steps in `## Preflight Checklist` below.

## Instructions

1. **Run preflight** — verify each check in `## Preflight Checklist` before any other action.
2. **Launch Kit-CAE** — pick the right `.kit` file and pass the launch flags from `## Launching Kit-CAE`.
3. **Run a script** — invoke any helper in `scripts/` with the `run_script` pattern shown in `## Running Scripts` (`./repo.sh launch -n <kit> -- --exec <path/to/script.py> --no-window`).
4. **Apply the critical rules** — Z-up convention, USDRT-only stage queries, top-level imports — see `## Critical Rules` and `## Z-Up Coordinate System`.
5. **Reference the API** — detailed call patterns live in `references/kit-cae-api.md`; format-specific stage layout lives in `references/formats.md`.

## Examples

Bundled scripts (invoke via the `run_script` pattern in `## Running Scripts`):

| Script | Purpose |
|--------|---------|
| `scripts/inspect_cgns.py` | Inspect a CGNS file: bases, zones, datasets, flow solutions, fields. |
| `scripts/inspect_vtk.py` | Inspect a VTK file and list its fields. |
| `scripts/query_stats.py` | Query field min/max/mean for a stage path. |
| `scripts/warmup.py` | Compile the Kit shader cache by launching briefly and exiting. |

See `references/kit-cae-api.md` for additional code patterns.

## Limitations

Kit-CAE skills inspect and visualize CAE datasets — they do not modify source files. Workflows require an interactive Kit-CAE session with a renderer; headless operation requires a virtual display. Time-varying data is supported only for formats that expose a time index.

## Preflight Checklist

1. *Kit-CAE repo present?* — look for `repo.sh` at the root
2. *Built?* — check `_build/linux-x86_64/release/` (or `_build\windows-x86_64\release\`)
   - If missing: `./repo.sh build -r`
3. *Pip deps installed?* — check `_build/target-deps/pip_prebundle/vtk/`
   - If missing: `./repo.sh pip_download` (VTK, h5py, lz4)
4. *RTX GPU?* — `nvidia-smi`, need ≥8GB VRAM (16GB+ recommended)
5. *No other Kit running?* — `pgrep -f "kit.sh\|kit " | head`, kill stale instances first
6. *Shader cache warm?* — first launch compiles shaders (~2–3 min). Front-load:
   ```bash
   ./repo.sh launch -n omni.cae.kit -- --no-window --exec skills/cae-core/scripts/warmup.py
   ```

## Launching Kit-CAE

| Kit file | Use for |
|----------|---------|
| `omni.cae_vtk.kit` | VTK formats (.vti/.vtu/.vts/.vtp/.vtk) |
| `omni.cae.kit` | Everything else (CGNS, EnSight, OpenFOAM, NPZ, EDEM) |

> **For browser-streamed sessions** (long-lived Kit-CAE that a remote client
> watches and drives), see `skills/cae-streaming/SKILL.md`. That skill ships
> a streaming-enabled `.kit` template, a launcher, and a handler-registration
> reference. Do not use the "Script shutdown (MANDATORY)" template below for
> streaming listeners — they must run until the app is signaled.

### Basic (headless)

```bash
cd <kit-cae-dir>
./repo.sh launch -n <kit_file> -- --exec scripts/<script>.py --no-window
```

### Render-Product Capture (MANDATORY for clean output)

Settings that affect the renderer *MUST* be launch args, never `carb.settings`
mid-render (`ERROR_DEVICE_LOST`):

```bash
./repo.sh launch -n <kit_file> -- --exec scripts/<script>.py --no-window \
    --/app/asyncRendering=false \
    --/rtx/materialDb/syncLoads=true \
    --/omni.kit.plugin/syncUsdLoads=true \
    --/rtx/hydra/materialSyncLoads=true \
    --/rtx-transient/resourcemanager/texturestreaming/async=false \
    --/rtx-transient/resourcemanager/enableTextureStreaming=false \
    --/exts/omni.kit.window.viewport/blockingGetViewportDrawable=true \
    --/rtx-transient/dlssg/enabled=false \
    --/persistent/app/viewport/defaults/fillViewport=false
```

For volume rendering, also add:

```bash
    --/renderer/enabled="rtx" \
    --/rtx/rendermode="RaytracedLighting" \
    --/rtx/directLighting/sampledLighting/enabled=true
```

## Running Scripts

> **Hard rule**: Every Python script runs inside Kit-CAE via `./repo.sh launch ... --exec`.
> Never call `python3` or system Python directly — the host Python environment is
> unknown and unsupported. Kit's `--exec` provides the only guaranteed Python
> environment (numpy, pxr, omni.*, carb, etc.). This applies to eval validators,
> data generators, inspection scripts, and any new script you write.

### Inspect scripts

```bash
CAE_INSPECT_FILE=<file> ./repo.sh launch -n omni.cae_vtk.kit -- \
    --exec skills/cae-core/scripts/inspect_vtk.py --no-window

CAE_INSPECT_FILE=<file> ./repo.sh launch -n omni.cae.kit -- \
    --exec skills/cae-core/scripts/inspect_cgns.py --no-window
```

### Statistics query

```bash
CAE_STATS_FILE=<file> ./repo.sh launch -n omni.cae.kit -- \
    --exec skills/cae-core/scripts/query_stats.py --no-window
```

Outputs JSON between `STATS_BEGIN`/`STATS_END` markers. Optional env vars:
`CAE_STATS_FORMAT` (force format), `CAE_STATS_FIELDS` (comma-separated prim paths).

### Script shutdown (MANDATORY)

```python
app.post_quit()
for _ in range(10):
    await app.next_update_async()
import os
os._exit(0)
```

`os._exit(0)` is required — `post_quit()` alone doesn't terminate headless Kit.

## Reference Files

- `references/kit-cae-api.md` — Import, stage discovery, viz commands, statistics, script template
- `references/capture-api.md` — Render-product capture, camera animation, H.264 encoding
- `references/formats.md` — Per-format import signatures, stage hierarchies, quirks
- `references/extensibility.md` — Custom format onboarding architecture

## Critical Rules

- All `omni.*` imports at *top level* — never inside functions (`UnboundLocalError`)
- Wait *≥600 frames* before first capture; 120+ extra for first capture in session
- `--no-window` for headless; `--exec` for script execution
- *Always create a bounding box* before `frame_prims`
- VTK files → `omni.cae_vtk.kit`; everything else → `omni.cae.kit`
- CGNS: paths under import root directly; dots/spaces → underscores
- NPZ `"SIDS Unstructured"` requires field association fix (see `kit-cae-api.md`)
- Velocity as separate scalars → pass all components to `SetTargets()`
- `stage.GetPrimAtPath()` never returns None — use `prim.IsValid()`
- `Gf.Vec2f`/`Gf.Vec3f` require Python `float`, not numpy types
- *Kit-CAE is Z-up* — orbit in X-Y plane, elevate along Z
- Viewport evidence is authoritative for visualization work. USD attributes,
  transforms, and logs can be correct while runtime-rendered CAE output is
  stale, hidden, or unchanged. Capture and inspect frames before declaring success.
- CAE viz operators may render through IndeX, Fabric, or USDRT. Time-sampled USD
  attributes are not guaranteed to drive rendered output. If playback looks
  static, prove a static pose first, then drive current/default values from a
  Kit timeline or update callback.
- Debug visualizations incrementally: import → static operator visible → color
  domain correct → hiding/composition → camera → animation → capture.

## Z-Up Coordinate System

Kit-CAE sets `upAxis = 'Z'`. Camera math must account for this.

### look_at_matrix (Z-up)

```python
from pxr import Gf

def look_at_matrix(eye, target, up=Gf.Vec3d(0, 0, 1)):
    forward = (target - eye).GetNormalized()
    right = Gf.Cross(forward, up).GetNormalized()
    actual_up = Gf.Cross(right, forward).GetNormalized()
    m = Gf.Matrix4d(1)
    m.SetRow(0, Gf.Vec4d(right[0], right[1], right[2], 0))
    m.SetRow(1, Gf.Vec4d(actual_up[0], actual_up[1], actual_up[2], 0))
    m.SetRow(2, Gf.Vec4d(-forward[0], -forward[1], -forward[2], 0))
    m.SetRow(3, Gf.Vec4d(eye[0], eye[1], eye[2], 1))
    return m
```

### Z-Up Orbit

```python
import math
from pxr import Usd

r_xy = orbit_radius * math.cos(math.radians(elevation_deg))
z_offset = orbit_radius * math.sin(math.radians(elevation_deg))

for frame in range(total_frames + 1):
    azimuth = start_az + (frame / total_frames) * 2 * math.pi
    eye = Gf.Vec3d(center[0] + r_xy * math.sin(azimuth),
                   center[1] + r_xy * math.cos(azimuth),
                   center[2] + z_offset)
    xform_op.Set(look_at_matrix(eye, center), Usd.TimeCode(frame))
```

Use `AddTransformOp()` with full matrix — Euler angles break in Z-up.
Camera must be a NEW prim (not `/OmniverseKit_Persp`).

### Axis Alignment Rotation

If geometry's primary axis isn't Z, rotate the import prim:

```python
xformable = UsdGeom.Xformable(stage.GetPrimAtPath("/World/<import_root>"))
xformable.AddRotateYOp().Set(-90.0)  # RotY(-90°): world_x=-local_z, world_z=local_x
# Recompute orbit center in world space after rotation
```
