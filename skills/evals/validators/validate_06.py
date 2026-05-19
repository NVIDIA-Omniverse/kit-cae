"""
Eval 06 Validator: Time-Varying Video
Runs inside Kit-CAE via --exec. Imports hex_timesteps.cgns, creates volume,
sets up animated camera with keyframes, captures frames across time steps,
and produces an H.264 MP4.
"""
import asyncio
import json
import math
import os

import omni.kit.app
import omni.timeline
import omni.usd
from omni.cae.data.commands import execute_command
from omni.cae.data import array_utils, usd_utils
from omni.cae.schema import cae, viz as cae_viz
from omni.cae.testing import frame_prims, wait_for_update
from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
from omni.usd import get_context
from pxr import Gf, Sdf, Tf, Usd, UsdGeom

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data")
BASE_RENDER_DIR = os.environ.get("KIT_CAE_EVAL_RENDER_DIR") or os.path.join(DATA_DIR, "renders")
os.makedirs(BASE_RENDER_DIR, exist_ok=True)
RENDER_DIR = os.path.join(BASE_RENDER_DIR, "eval_06_frames")
OUTPUT_PATH = os.path.join(BASE_RENDER_DIR, "eval_06_timevary.mp4")
NUM_FRAMES = 48
FPS = 24
WIDTH, HEIGHT = 1280, 720


async def main():
    app = omni.kit.app.get_app()
    checks = []

    # 1. Import
    try:
        from omni.cae.importer.cgns import import_to_stage
        await import_to_stage(os.path.join(DATA_DIR, "hex_timesteps.cgns"), "/World/hex")
        await wait_for_update(30)
        checks.append({"name": "import_success", "pass": True, "detail": "Imported hex_timesteps.cgns"})
    except Exception as e:
        checks.append({"name": "import_success", "pass": False, "detail": str(e)})
        _emit_result("06_time_varying_video", checks)
        _shutdown(app)
        return

    stage = get_context().get_stage()

    # 2. Discover datasets and fields
    import usdrt
    fabric_stage = usdrt.Usd.Stage.Attach(omni.usd.get_context().get_stage_id())

    dataset_type = Tf.Type.Find(cae.DataSet)
    datasets = [p.GetString() for p in fabric_stage.GetPrimsWithTypeName(dataset_type.typeName)]
    mesh_datasets = [d for d in datasets if "GridCoordinates" not in d]

    field_type = Tf.Type.Find(cae.FieldArray)
    all_fields = [p.GetString() for p in fabric_stage.GetPrimsWithTypeName(field_type.typeName)]
    sinusoid_fields = [f for f in all_fields if "PointSinusoid" in f or "CellSinusoid" in f]

    has_sinusoid = len(sinusoid_fields) > 0
    checks.append({"name": "sinusoid_field_found", "pass": has_sinusoid,
                    "detail": sinusoid_fields[0] if has_sinusoid else "No sinusoid field"})
    checks.append({"name": "datasets_found", "pass": len(mesh_datasets) > 0,
                    "detail": f"{len(mesh_datasets)} mesh datasets"})

    dataset_path = mesh_datasets[0] if mesh_datasets else None
    field_path = sinusoid_fields[0] if sinusoid_fields else None

    if not dataset_path or not field_path:
        _emit_result("06_time_varying_video", checks)
        _shutdown(app)
        return

    # 3. Create volume + bounding box
    vol_path = "/World/CAE/Volume"
    bbox_path = "/World/CAE/BoundingBox"

    try:
        await execute_command("CreateCaeVizBoundingBox", dataset_paths=[dataset_path], prim_path=bbox_path)
        await execute_command("CreateCaeVizVolume", dataset_path=dataset_path, prim_path=vol_path, type="vdb")
        
        # Bind field IMMEDIATELY before controller runs
        vol_prim = stage.GetPrimAtPath(vol_path)
        cae_viz.FieldSelectionAPI(vol_prim, "colors").CreateTargetRel().SetTargets([field_path])
        await wait_for_update(60)

        # Set volume colormap domain
        try:
            field_prim = stage.GetPrimAtPath(field_path)
            farray = await usd_utils.get_array(field_prim, Usd.TimeCode.EarliestTime())
            ranges = array_utils.get_componentwise_ranges(farray)
            if ranges:
                fmin, fmax = float(ranges[0][0]), float(ranges[0][1])
                colormap_prim = stage.GetPrimAtPath(f"{vol_path}/Material/Colormap")
                if colormap_prim.IsValid():
                    colormap_prim.GetAttribute("domain").Set(Gf.Vec2f(fmin, fmax))
                    colormap_prim.GetAttribute("domainBoundaryMode").Set("clampToEdge")
                    # Disable auto-rescale
                    if vol_prim.HasAPI(cae_viz.RescaleRangeAPI, "colors"):
                        cae_viz.RescaleRangeAPI(vol_prim, "colors").CreateRescaleModeAttr().Set("disable")
                    await wait_for_update(5)
        except Exception as e:
            print(f"WARNING: Could not set volume colormap: {e}")
        await wait_for_update(10)
        checks.append({"name": "volume_created", "pass": True, "detail": vol_path})
    except Exception as e:
        checks.append({"name": "volume_created", "pass": False, "detail": str(e)})

    # 4. Create animated camera — 360° orbit using look_at_matrix (Z-up safe)
    cam_path = "/World/EvalCamera"
    try:
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

        # Compute orbit from bounding box
        await frame_prims([bbox_path], zoom=0.9)
        for _ in range(30):
            await app.next_update_async()

        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render'])
        bbox = bbox_cache.ComputeWorldBound(stage.GetPrimAtPath(bbox_path))
        bbox_range = bbox.ComputeAlignedRange()
        center = Gf.Vec3d(bbox_range.GetMidpoint())
        size = bbox_range.GetSize()
        max_span = max(size[0], size[1])
        orbit_radius = max_span * 5.0 if max_span > 0.1 else 15.0
        print(f"Camera orbit: center={center}, radius={orbit_radius}")

        cam_prim = UsdGeom.Camera.Define(stage, cam_path)
        xformable = UsdGeom.Xformable(cam_prim.GetPrim())
        xformable.ClearXformOpOrder()
        xform_op = xformable.AddTransformOp()
        cam_prim.GetClippingRangeAttr().Set(Gf.Vec2f(1.0, float(orbit_radius * 4.0)))

        elevation_deg = 25.0
        r_xy = orbit_radius * math.cos(math.radians(elevation_deg))
        z_offset = orbit_radius * math.sin(math.radians(elevation_deg))

        for frame in range(NUM_FRAMES + 1):
            t = frame / NUM_FRAMES
            azimuth = t * 2.0 * math.pi
            eye = Gf.Vec3d(
                center[0] + r_xy * math.sin(azimuth),
                center[1] + r_xy * math.cos(azimuth),
                center[2] + z_offset,
            )
            xform_op.Set(look_at_matrix(eye, center), Usd.TimeCode(frame))

        vp = get_active_viewport()
        vp.camera_path = Sdf.Path(cam_path)

        checks.append({"name": "camera_created", "pass": True, "detail": f"{cam_path} with {NUM_FRAMES} keyframes"})

        # Verify keyframes
        ops = xformable.GetOrderedXformOps()
        total_samples = sum(op.GetNumTimeSamples() for op in ops)
        checks.append({"name": "camera_animated", "pass": total_samples >= NUM_FRAMES,
                        "detail": f"{total_samples} time samples across {len(ops)} ops"})
    except Exception as e:
        checks.append({"name": "camera_created", "pass": False, "detail": str(e)})

    # 5. Set resolution
    try:
        vp = get_active_viewport()
        vp.resolution = (WIDTH, HEIGHT)
        await wait_for_update(5)
    except Exception:
        pass

    # 6. Capture frames
    os.makedirs(RENDER_DIR, exist_ok=True)
    captured_count = 0

    try:
        timeline = omni.timeline.get_timeline_interface()
        timeline.set_start_time(0.0)
        timeline.set_end_time(NUM_FRAMES / FPS)
        timeline.set_time_codes_per_second(FPS)

        # Settle renderer after camera switch
        for _ in range(600):
            await app.next_update_async()

        for frame in range(NUM_FRAMES):
            timeline.set_current_time(frame / FPS)
            # More settle frames for first frame (shader compilation)
            settle = 120 if frame == 0 else 15
            for _ in range(settle):
                await app.next_update_async()

            frame_path = os.path.join(RENDER_DIR, f"frame_{frame:04d}.png")
            capture_viewport_to_file(vp, file_path=frame_path)
            for _ in range(15):
                await app.next_update_async()

        # Wait for all async captures to flush
        for _ in range(120):
            await app.next_update_async()

        # Count successfully captured frames
        for frame in range(NUM_FRAMES):
            frame_path = os.path.join(RENDER_DIR, f"frame_{frame:04d}.png")
            if os.path.isfile(frame_path) and os.path.getsize(frame_path) > 1000:
                captured_count += 1

        checks.append({"name": "frames_captured", "pass": captured_count >= NUM_FRAMES * 0.9,
                        "detail": f"{captured_count}/{NUM_FRAMES} frames"})
    except Exception as e:
        checks.append({"name": "frames_captured", "pass": False, "detail": str(e)})

    # 7. Encode video with Kit-native H.264 encoder
    try:
        from video_encoding import get_video_encoding_interface  # NOT omni.videoencoding
        encoder = get_video_encoding_interface()

        import carb.settings
        settings = carb.settings.get_settings()
        settings.set("/exts/omni.videoencoding/bitrate", 8_000_000)
        settings.set("/exts/omni.videoencoding/gopSize", 12)

        encoder.start_encoding(OUTPUT_PATH, FPS, captured_count, True)

        for frame in range(NUM_FRAMES):
            frame_path = os.path.join(RENDER_DIR, f"frame_{frame:04d}.png")
            if os.path.isfile(frame_path):
                encoder.encode_next_frame_from_file(frame_path)

        encoder.finalize_encoding()

        mp4_exists = os.path.isfile(OUTPUT_PATH)
        mp4_size = os.path.getsize(OUTPUT_PATH) if mp4_exists else 0
        checks.append({"name": "mp4_exists", "pass": mp4_exists and mp4_size > 50000,
                        "detail": f"{OUTPUT_PATH} ({mp4_size} bytes)"})

        # Check H.264 codec
        if mp4_exists and mp4_size > 1000:
            with open(OUTPUT_PATH, "rb") as f:
                content = f.read(min(mp4_size, 8192))
                is_h264 = b"avc1" in content or b"h264" in content.lower()
                checks.append({"name": "mp4_h264", "pass": is_h264,
                                "detail": "H.264 codec detected" if is_h264 else "Codec not detected"})
        else:
            checks.append({"name": "mp4_h264", "pass": False, "detail": f"MP4 too small: {mp4_size}"})

    except Exception as e:
        checks.append({"name": "mp4_exists", "pass": False, "detail": f"Encoding failed: {str(e)}"})

    _emit_result("06_time_varying_video", checks)
    _shutdown(app)


def _emit_result(eval_name, checks):
    passed = sum(1 for c in checks if c["pass"])
    total = len(checks)
    result = {
        "eval": eval_name,
        "pass": all(c["pass"] for c in checks),
        "score": round(passed / total * 100) if total > 0 else 0,
        "checks": checks,
    }
    print(f"\nEVAL_RESULT_BEGIN\n{json.dumps(result, indent=2)}\nEVAL_RESULT_END")


def _shutdown(app):
    async def _do_shutdown():
        app.post_quit()
        for _ in range(10):
            await app.next_update_async()
        os._exit(0)
    asyncio.ensure_future(_do_shutdown())


if __name__ == "__main__":
    asyncio.ensure_future(main())
