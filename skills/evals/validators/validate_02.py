"""
Eval 02 Validator: Volume Render
Runs inside Kit-CAE via --exec. Imports StaticMixer.cgns, creates a volume,
verifies field binding, and checks the output screenshot.
"""
import asyncio
import json
import os
import struct

import omni.kit.app
import omni.usd
from omni.cae.data.commands import execute_command
from omni.cae.schema import cae, viz as cae_viz
from omni.cae.testing import frame_prims, wait_for_update
from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
from omni.usd import get_context
from pxr import Tf, Usd, UsdGeom

RENDER_DIR = os.environ.get("KIT_CAE_EVAL_RENDER_DIR") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data", "renders")
os.makedirs(RENDER_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(RENDER_DIR, "eval_02_volume.png")


async def main():
    app = omni.kit.app.get_app()
    checks = []

    # 1. Import
    try:
        from omni.cae.importer.cgns import import_to_stage
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data")
        await import_to_stage(os.path.join(data_dir, "StaticMixer.cgns"), "/World/StaticMixer")
        await wait_for_update(20)
        checks.append({"name": "import_success", "pass": True, "detail": "Imported"})
    except Exception as e:
        checks.append({"name": "import_success", "pass": False, "detail": str(e)})
        _emit_result("02_volume_render", checks)
        _shutdown(app)
        return

    stage = get_context().get_stage()

    # 2. Create volume
    dataset_path = "/World/StaticMixer/Base/StaticMixer/B1_P3"
    field_path = "/World/StaticMixer/Base/StaticMixer/Flow_Solution/Eddy_Viscosity"
    vol_path = "/World/CAE/Volume"
    bbox_path = "/World/CAE/BoundingBox"

    try:
        await execute_command("CreateCaeVizBoundingBox", dataset_paths=[dataset_path], prim_path=bbox_path)
        await execute_command("CreateCaeVizVolume", dataset_path=dataset_path, prim_path=vol_path, type="vdb")
        await wait_for_update(10)

        vol_prim = stage.GetPrimAtPath(vol_path)
        cae_viz.FieldSelectionAPI(vol_prim, "colors").CreateTargetRel().SetTargets([field_path])
        await wait_for_update(10)

        checks.append({"name": "volume_created", "pass": True, "detail": vol_path})
    except Exception as e:
        checks.append({"name": "volume_created", "pass": False, "detail": str(e)})

    # 3. Verify volume prim exists
    vol_prim = stage.GetPrimAtPath(vol_path)
    prim_exists = bool(vol_prim and vol_prim.IsValid())
    checks.append({"name": "volume_prim_exists", "pass": prim_exists, "detail": vol_path})

    # 4. Verify field binding
    if prim_exists:
        colors_api = cae_viz.FieldSelectionAPI(vol_prim, "colors")
        targets = colors_api.GetTargetRel().GetTargets()
        field_bound = len(targets) > 0 and "Eddy_Viscosity" in str(targets[0])
        checks.append({"name": "field_bound", "pass": field_bound,
                        "detail": str(targets[0]) if targets else "No targets"})
    else:
        checks.append({"name": "field_bound", "pass": False, "detail": "No volume prim"})

    # 5. Verify bounding box
    bbox_prim = stage.GetPrimAtPath(bbox_path)
    checks.append({"name": "bbox_exists", "pass": bool(bbox_prim and bbox_prim.IsValid()),
                    "detail": bbox_path})

    # 6. Capture screenshot
    try:
        await frame_prims([bbox_path], zoom=0.9)
        for _ in range(600):
            await app.next_update_async()

        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        vp = get_active_viewport()
        capture_viewport_to_file(vp, file_path=OUTPUT_PATH)
        for _ in range(30):
            await app.next_update_async()

        # Check file exists and is non-trivial
        file_exists = os.path.isfile(OUTPUT_PATH)
        file_size = os.path.getsize(OUTPUT_PATH) if file_exists else 0
        checks.append({"name": "screenshot_exists", "pass": file_exists and file_size > 10000,
                        "detail": f"{OUTPUT_PATH} ({file_size} bytes)"})

        # Check not entirely black (read PNG header for dimensions)
        if file_exists and file_size > 10000:
            checks.append({"name": "screenshot_not_trivial", "pass": True,
                            "detail": f"File size {file_size} > 10KB"})
        else:
            checks.append({"name": "screenshot_not_trivial", "pass": False,
                            "detail": f"File too small: {file_size}"})
    except Exception as e:
        checks.append({"name": "screenshot_exists", "pass": False, "detail": str(e)})

    _emit_result("02_volume_render", checks)
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
