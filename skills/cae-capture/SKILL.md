---
name: cae-capture
description: >
  Capture clean render-only images and movies from Kit-CAE visualizations. Produces
  output without application UI - just the rendered scene. Supports high-resolution
  PNG/EXR screenshots, MP4 video with NVENC hardware encoding, and animated camera
  orbits. Triggers on requests for "capture image", "render screenshot", "export movie",
  "high-res render", "save visualization", "record animation", "camera pan", or any
  request to produce clean output from a Kit-CAE scene.
depends:
  - cae-core
version: "2.1.0"
metadata:
  author: "NVIDIA"
  tags:
    - kit-cae
    - cae
    - rendering
    - screenshot
    - movie
    - png
    - exr
    - mp4
    - nvenc
---

# CAE Capture - Clean Render Output

Capture production-quality images and movies from Kit-CAE scenes — rendered
content only, no application UI.

## Purpose

Capture clean render-only output from a Kit-CAE scene — high-resolution PNG/EXR screenshots and MP4 video with NVENC hardware encoding, with no application UI in the frame.

## Prerequisites

`cae-core` (loaded automatically). Kit must be launched with the Farm-Style args from `cae-core/references/capture-api.md` for any capture work; mid-session `carb.settings` changes will crash the renderer.

## Instructions

1. **Select an approach** — choose single image, MP4 video, or animated orbit per `## Approach Selection`.
2. **Launch Kit with Farm-Style args** — required for any capture work; see `## Launch (MANDATORY)` and `cae-core/references/capture-api.md`. Do not change renderer settings mid-session.
3. **Frame the camera** — apply the settle-frame, target-prim, and FOV tips in `## Camera Tips`.
4. **Capture the output** — for animation follow `## Time-Varying Data Capture`; for single-image use the patterns linked in `## Approach Selection`.
5. **Verify delivery** — confirm output paths, formats, and resolution against `## Delivery`.

## Examples

The `## Time-Varying Data Capture` and `## Camera Tips` sections include runnable Kit launch commands and Python snippets covering single-image, EXR, MP4-via-NVENC, and orbit-animation workflows.

## Limitations

NVENC encoding requires a supported NVIDIA GPU. Headless capture requires a virtual display. Capture is renderer-only — UI elements (gizmos, viewport overlays) are deliberately excluded by the Farm-Style launch path.

## Dependencies

- `cae-core/SKILL.md` — Preflight, Z-up, launch args, clean exit
- `cae-core/references/capture-api.md` — **Full capture API reference** (all details)
- `cae-core/references/kit-cae-api.md` — Import, viz setup, framing

*Always run the preflight checklist from `cae-core/SKILL.md` first.*

## Approach Selection

| Goal | Approach | Reference |
|------|----------|-----------|
| Single screenshot | `capture_viewport_to_file` | capture-api.md § Single-Frame |
| MP4 video (turntable, animation) | CaptureExtension | capture-api.md § MP4 Video |
| Per-frame control (fallback) | Manual capture + H.264 encoder | capture-api.md § Manual Frame Capture |

*Always try CaptureExtension first for video.* Fall back to manual capture only
if CaptureExtension produces incorrect frames after tuning settle latency.

## Launch (MANDATORY)

Use farm-style launch args from `cae-core/SKILL.md` § Render-Product Capture.
For volumes, add the RTX renderer args. **Settings that affect the renderer
MUST be launch args, never `carb.settings` mid-render** (`ERROR_DEVICE_LOST`).

## Time-Varying Data Capture

### Pre-Cache Time Steps (ESSENTIAL for smooth video)

```python
for frame in range(0, TOTAL_FRAMES + 1, 2):
    timeline.set_current_time(frame / FPS)
    for _ in range(20):
        await app.next_update_async()
for frame in range(1, TOTAL_FRAMES, 2):
    timeline.set_current_time(frame / FPS)
    for _ in range(10):
        await app.next_update_async()
```

### Fixed Color Range

Lock scalar color range to prevent per-frame auto-rescaling. See
`cae-visualization/SKILL.md` § Fixed Color Range.

## Camera Tips

- *Always `AddTransformOp()` with full matrix* — Euler angles break in Z-up
- *Create NEW camera prim* — `/OmniverseKit_Persp` ignores keyframes
- *Orbit in X-Y plane, elevate along Z* (Z-up coordinate system)
- *N+1 keyframes for seamless loop* — frame N == frame 0, capture `[0, N)`
- *`end_frame` is exclusive* — set `end_frame = N` for N frames, not `N-1`

Full camera setup, orbit patterns, and non-constant paths: capture-api.md § Camera Animation.

## Delivery

### macOS / Apple Devices

Kit's NVENC encoder writes fragmented MP4 that macOS reads as 2× duration.
Always remux before delivery:

```bash
ffmpeg -y -i raw.mp4 -c copy -movflags +faststart -f mp4 final.mp4
```

Details: capture-api.md § macOS AVFoundation Duration Quirk.

### H.264 Required for Web/Chat

MP4v works locally but won't play inline on Slack/Discord/web. Use Kit's
NVENC encoder (H.264 by default) or ffmpeg with `-c:v libx264`.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ERROR_DEVICE_LOST` | Use farm-style launch args (not `carb.settings`) |
| Black/empty output | Wait ≥600 frames for convergence |
| UI in output | Use `capture_viewport_to_file`, not swapchain capture |
| Camera won't animate | Create new camera prim (not `/OmniverseKit_Persp`) |
| Resolution ignored (headless) | Set `fillViewport=false` (launch arg + carb setting) |
| Color shifts per frame | Disable RescaleRange, set explicit domain |
| Uniform color in video | Increase settle latency or use manual capture |
| `0/N` frames found, but PNGs appear after exit | Scan frame files after a final 120-300 update flush; don't rely on immediate per-frame existence checks |
| Frame count off by one | `end_frame` is exclusive — use `N` not `N-1` |
| Kit process survives | Add `os._exit(0)` after `post_quit()` |
| macOS 2× duration | Remux with `ffmpeg -c copy -movflags +faststart` |
