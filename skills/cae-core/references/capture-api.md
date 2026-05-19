# Capture API Reference

Render-product capture: clean images/movies without application UI.

## ⚠ Farm-Style Launch Args (MANDATORY)

**Renderer settings MUST be launch args, never `carb.settings` mid-render** →
`ERROR_DEVICE_LOST` GPU crash.

```bash
--/app/asyncRendering=false
--/rtx/materialDb/syncLoads=true
--/omni.kit.plugin/syncUsdLoads=true
--/rtx/hydra/materialSyncLoads=true
--/rtx-transient/resourcemanager/texturestreaming/async=false
--/rtx-transient/resourcemanager/enableTextureStreaming=false
--/exts/omni.kit.window.viewport/blockingGetViewportDrawable=true
--/rtx-transient/dlssg/enabled=false
--/persistent/app/viewport/defaults/fillViewport=false
```

Volume rendering adds: `--/renderer/enabled="rtx" --/rtx/rendermode="RaytracedLighting" --/rtx/directLighting/sampledLighting/enabled=true`

### Convergence Frame Counts

| Phase | Frames | Notes |
|-------|--------|-------|
| Initial convergence | 600-800 | After import + viz creation |
| First capture in session | 120+ extra | Shader compilation (one-time) |
| Post-camera-switch | 300-400 | After `vp.camera_path =` change |
| Post-timeCode change (volumes) | 60-100 | After `tl.set_current_time()` when rendering time-varying dense volumes through IndeX. The voxelization cache is keyed by timeCode; fewer ticks can leave the previous timestep on screen. |
| Per-frame settle (video) | 6-10 | Between state changes in capture loop |
| Post-capture wait | 3-10 | After `capture_viewport_to_file` |
| Frame file flush | 120-300 | After the capture loop, before scanning PNGs for encoding |

## Single-Frame Capture

```python
from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file

vp = get_active_viewport()
for _ in range(600):
    await app.next_update_async()
capture_viewport_to_file(vp, file_path="/tmp/output.png")
for _ in range(30):
    await app.next_update_async()
```

### Headless Resolution

```python
import carb
settings = carb.settings.get_settings()
settings.set(f"/persistent/app/viewport/{vp.id}/fillViewport", False)
vp.resolution = (1280, 720)
```

Also pass `--/persistent/app/viewport/defaults/fillViewport=false` as launch arg.

## MP4 Video (CaptureExtension)

### Access

```python
from omni.kit.capture.viewport.extension import capture_instance as capture_ext
from omni.kit.capture.viewport.capture_options import (
    CaptureRangeType, CaptureRenderPreset, CaptureMovieType
)
```

### Configuration

```python
options = capture_ext.options
options.camera = "/World/AnimCamera"
options.range_type = CaptureRangeType.FRAMES
options.fps = 24
options.start_frame = 0
options.end_frame = 240           # EXCLUSIVE — produces 240 frames (0..239)
options.res_width = 1920
options.res_height = 1080
options.render_preset = CaptureRenderPreset.RAY_TRACE
options.output_folder = "/tmp/output"
options.file_name = "movie"
options.file_type = ".mp4"
options.save_alpha = False
options.hdr_output = False
options.overwrite_existing_frames = True
options.movie_type = CaptureMovieType.SEQUENCE
options.animation_fps = 24
```

### Settle & Wait

```python
options.real_time_settle_latency_frames = 10  # default 0=auto (5 for RT)
options.rt_wait_for_render_resolve_in_seconds = 5  # default -1 (disabled)
options.preroll_frames = 30
```

Global: `--/app/captureSequence/waitFrames=10`

CaptureExtension auto-enables `syncLoads` and disables texture streaming.

### Execution

```python
capture_ext.start()
while not capture_ext.progress.done:
    await app.next_update_async()
```

### Render Presets

| Preset | Enum | Speed |
|--------|------|-------|
| Ray Trace | `RAY_TRACE` | Fast (default for CAE) |
| Path Trace | `PATH_TRACE` | Slow (publication) |
| Real-Time PT | `REAL_TIME_PATHTRACING` | Medium |
| IRay | `IRAY` | Slowest (reference) |

### NVENC Encoding Options

| Setting | Default |
|---------|---------|
| `mp4_encoding_preset` | PRESET_DEFAULT |
| `mp4_encoding_profile` | H264_PROFILE_HIGH |
| `mp4_encoding_rc_mode` | RC_VBR |
| `mp4_encoding_bitrate` | 16777216 (16 Mbps) |
| `mp4_encoding_iframe_interval` | 60 |

## Manual Frame Capture + H.264 Encoder (Fallback)

Use when CaptureExtension produces incorrect frames after tuning settle latency.

```python
from pathlib import Path

import carb
from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
try:
    from video_encoding import get_video_encoding_interface  # NOT omni.videoencoding
except ImportError:
    get_video_encoding_interface = lambda: None

vp = get_active_viewport()
FPS = 24; NF = 240
frames_dir = Path("/tmp/frames")
frames_dir.mkdir(parents=True, exist_ok=True)

def collect_frame_files():
    files = []
    for f in range(NF):
        frame_path = frames_dir / f"frame_{f:04d}.png"
        if frame_path.is_file() and frame_path.stat().st_size > 1000:
            files.append(frame_path)
    return files

for f in range(NF):
    timeline.set_current_time(f / FPS)
    xform_op.Set(look_at_matrix(eye, center))
    for _ in range(8):
        await app.next_update_async()
    frame_path = frames_dir / f"frame_{f:04d}.png"
    future = capture_viewport_to_file(vp, file_path=str(frame_path))
    try:
        await future.wait_for_result()
    except Exception:
        # Some Kit builds still flush the PNG after wait_for_result reports.
        pass
    for _ in range(3):
        await app.next_update_async()

for _ in range(120):
    await app.next_update_async()

frame_files = collect_frame_files()
if len(frame_files) != NF:
    for _ in range(180):
        await app.next_update_async()
    frame_files = collect_frame_files()
if len(frame_files) != NF:
    raise RuntimeError(f"captured {len(frame_files)}/{NF} frames")

# Encode
encoder = get_video_encoding_interface()
settings = carb.settings.get_settings()
settings.set("/exts/omni.videoencoding/bitrate", 8_000_000)
import time; time.sleep(0.5)

encoder.start_encoding("/tmp/output.mp4", FPS, len(frame_files), True)  # True = overwrite
for png in frame_files:
    encoder.encode_next_frame_from_file(str(png))
encoder.finalize_encoding()
```

### Encoder API

| Method | Args |
|--------|------|
| `start_encoding` | `(path, fps, frame_count, overwrite)` |
| `encode_next_frame_from_file` | `(png_path)` |
| `finalize_encoding` | `()` |

### ffmpeg Fallback

If Kit encoder unavailable, encode PNGs externally:
```bash
ffmpeg -y -framerate 24 -i /tmp/frames/frame_%04d.png \
    -c:v libx264 -pix_fmt yuv420p -crf 18 -preset slow /tmp/output.mp4
```

## Camera Animation

### Setup

Default camera (`/OmniverseKit_Persp`) ignores keyframes. *Always create a new prim:*

```python
from pxr import UsdGeom, Gf, Sdf, Usd
import math

anim_cam = UsdGeom.Camera.Define(stage, "/World/AnimCamera")
xformable = UsdGeom.Xformable(anim_cam.GetPrim())

orig = UsdGeom.Camera(stage.GetPrimAtPath(get_active_viewport().camera_path))
anim_cam.GetFocalLengthAttr().Set(orig.GetFocalLengthAttr().Get())
anim_cam.GetClippingRangeAttr().Set(Gf.Vec2f(1.0, orbit_radius * 4.0))

xform_op = xformable.AddTransformOp()

for frame in range(TOTAL_FRAMES + 1):  # +1 for seamless loop
    t = frame / TOTAL_FRAMES
    azimuth = start_az + t * 2 * math.pi
    eye = Gf.Vec3d(center[0] + r_xy * math.sin(azimuth),
                   center[1] + r_xy * math.cos(azimuth),
                   center[2] + z_offset)
    xform_op.Set(look_at_matrix(eye, center), Usd.TimeCode(frame))

get_active_viewport().camera_path = Sdf.Path("/World/AnimCamera")
```

### Rules

- `AddTransformOp()` with full matrix (not Euler angles)
- N+1 keyframes for loop (frame N == frame 0), capture `[0, N)` via `end_frame = N`
- Orbit radius ≈ 1.5–2.0× max horizontal span

### Non-Constant Paths

Vary radius, azimuth, elevation per frame:
```python
r_xy = radius * math.cos(elevation)
eye = Gf.Vec3d(center[0] + r_xy * math.sin(azimuth),
               center[1] + r_xy * math.cos(azimuth),
               center[2] + radius * math.sin(elevation))
```

Interpolation: `smoothstep(t) = t*t*(3-2*t)`, `sin(pi*t)` for elevation.

### Narrative Camera vs Turntable

A *turntable* (constant-radius orbit) is a solid default for showcasing geometry
and is often the right choice when the user wants a clean 360° view. Use it
freely when it fits the task.

When a task calls for more creative or data-driven camera work, consider:

- **Track features**: If something grows/propagates, move the camera to follow it
- **Reveal structure**: Start close on interesting detail, pull back to show context
- **Vary distance**: Closer during detail moments, farther during overview
- **Change elevation**: Show the data from above, side, below
- **Use smoothstep**: Ease in/out with `t*t*(3-2*t)` or `0.5 - 0.5*cos(pi*t)`

The choice depends on what the visualization is trying to communicate. Turntable
for spatial overview; narrative paths when the camera should guide attention to
specific features or temporal evolution.

### Timeline

```python
import omni.timeline
tl = omni.timeline.get_timeline_interface()
tl.set_time_codes_per_second(FPS)
tl.set_start_time(0)
tl.set_end_time(TOTAL_FRAMES / FPS)
```

## Clean Exit

```python
app.post_quit()
for _ in range(30):
    await app.next_update_async()
import os
os._exit(0)
```

Between jobs: `pkill -f "kit/kit"; pkill -f "omni.telemetry.transmitter"`

## macOS MP4 Duration Quirk

Kit's NVENC writes fragmented MP4; macOS AVFoundation reports 2× duration and
freezes on last frame's phantom second half. Always remux:

```bash
ffmpeg -y -i raw.mp4 -c copy -movflags +faststart -f mp4 final.mp4
```

No re-encode, sub-second. Verify: `ffprobe` duration should match `mdls -name kMDItemDurationSeconds`.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ERROR_DEVICE_LOST` ~24s | Use launch args, not `carb.settings` mid-render |
| UI in output | Use `capture_viewport_to_file`, not swapchain |
| Camera won't animate | Create new camera prim |
| Black output | Wait ≥600 frames |
| Captured N-1 frames | `end_frame` is exclusive — set N, not N-1 |
| Kit survives post_quit | Add `os._exit(0)` |
| macOS 2× duration | Remux: `ffmpeg -c copy -movflags +faststart` |
