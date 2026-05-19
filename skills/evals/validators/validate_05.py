"""
Eval 05 Validator: Streamlines
Runs inside Kit-CAE via --exec. Imports NPZ, applies field association fix,
creates streamlines with seed sphere, verifies velocity/color bindings, captures screenshot.
"""
import asyncio
import json
import os

import omni.kit.app
import omni.usd
from omni.cae.data.commands import execute_command
from omni.cae.data import array_utils, usd_utils
from omni.cae.schema import cae, viz as cae_viz
from omni.cae.testing import frame_prims, wait_for_update
from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
from omni.usd import get_context
from pxr import Gf, Tf, Usd, UsdGeom, UsdShade

RENDER_DIR = os.environ.get("KIT_CAE_EVAL_RENDER_DIR") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data", "renders")
os.makedirs(RENDER_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(RENDER_DIR, "eval_05_streamlines.png")


async def main():
    app = omni.kit.app.get_app()
    checks = []

    # 1. Import NPZ as SIDS Unstructured
    try:
        from omni.cae.importer.npz import import_to_stage
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data")
        await import_to_stage(os.path.join(data_dir, "disk_out_ref.npz"), "/World/disk",
                              schema_type="SIDS Unstructured")
        await wait_for_update(20)
        checks.append({"name": "import_success", "pass": True, "detail": "Imported"})
    except Exception as e:
        checks.append({"name": "import_success", "pass": False, "detail": str(e)})
        _emit_result("05_streamlines", checks)
        _shutdown(app)
        return

    stage = get_context().get_stage()

    # 2. Field association fix
    try:
        array_base = stage.GetPrimAtPath("/World/disk/NumPyArrays")
        fix_count = 0
        if array_base and array_base.IsValid():
            for child in array_base.GetAllChildren():
                cae.FieldArray(child).CreateFieldAssociationAttr().Set(cae.Tokens.vertex)
                fix_count += 1
        await wait_for_update(10)
        checks.append({"name": "field_assoc_fix", "pass": fix_count > 0,
                        "detail": f"Fixed {fix_count} fields"})
    except Exception as e:
        checks.append({"name": "field_assoc_fix", "pass": False, "detail": str(e)})

    # 3. Discover field paths dynamically
    import usdrt
    fabric_stage = usdrt.Usd.Stage.Attach(omni.usd.get_context().get_stage_id())
    field_type = Tf.Type.Find(cae.FieldArray)
    all_fields = [p.GetString() for p in fabric_stage.GetPrimsWithTypeName(field_type.typeName)]
    disk_fields = [f for f in all_fields if f.startswith("/World/disk/NumPyArrays/")]

    # Find velocity field
    v_field = None
    for f in disk_fields:
        name = f.split("/")[-1]
        if name.lower() in ("v", "velocity", "vel"):
            v_field = f
            break

    # Find temperature field (for coloring)
    t_field = None
    for f in disk_fields:
        name = f.split("/")[-1]
        if name.lower() in ("temp", "t", "temperature"):
            t_field = f
            break

    if not t_field:
        t_field = disk_fields[0] if disk_fields else None
    if not v_field:
        v_field = disk_fields[1] if len(disk_fields) > 1 else t_field

    print(f"Fields: t_field={t_field}, v_field={v_field}")

    # 4. Create streamlines + seed
    dataset_path = "/World/disk/NumPyDataSet"
    bbox_path = "/World/CAE/BoundingBox"
    stream_path = "/World/CAE/Streamlines"
    seed_path = "/World/CAE/Seed"

    try:
        await execute_command("CreateCaeVizBoundingBox", dataset_paths=[dataset_path], prim_path=bbox_path)

        await execute_command("CreateCaeVizStreamlines", dataset_path=dataset_path,
                              prim_path=stream_path, type="standard")
        await execute_command("CreateCaeVizMeshPrim", prim_type="UnitSphere", prim_path=seed_path)
        await execute_command("TransformPrimSRT", path=seed_path, new_scale=[0.2, 0.2, 0.2])

        # Bind fields IMMEDIATELY — before any wait
        stream_prim = stage.GetPrimAtPath(stream_path)
        cae_viz.StreamlinesAPI(stream_prim).GetDirectionAttr().Set(cae_viz.Tokens.forward)
        cae_viz.DatasetSelectionAPI(stream_prim, "seeds").GetTargetRel().SetTargets([seed_path])
        if v_field:
            cae_viz.FieldSelectionAPI(stream_prim, "velocities").GetTargetRel().SetTargets([v_field])
        if t_field:
            cae_viz.FieldSelectionAPI(stream_prim, "colors").CreateTargetRel().SetTargets([t_field])

        # Wait for controller to process field bindings and auto-rescale
        await wait_for_update(120)

        # Ensure coloring is enabled (fallback if controller hasn't processed)
        shader_path = f"{stream_path}/Materials/ScalarColor/Shader"
        shader = UsdShade.Shader(stage.GetPrimAtPath(shader_path))
        if shader.GetPrim().IsValid():
            ec = shader.GetInput("enable_coloring")
            dm = shader.GetInput("domain")

            if ec and not ec.Get():
                ec.Set(True)
            if dm and t_field:
                dval = dm.Get()
                if dval and dval[1] < dval[0]:
                    t_prim = stage.GetPrimAtPath(t_field)
                    farray = await usd_utils.get_array(t_prim, Usd.TimeCode.EarliestTime())
                    ranges = array_utils.get_componentwise_ranges(farray)
                    if ranges:
                        fmin, fmax = float(ranges[0][0]), float(ranges[0][1])
                        dm.Set(Gf.Vec2f(fmin, fmax))
                        ec.Set(True)

        await wait_for_update(30)
        checks.append({"name": "streamlines_created", "pass": True, "detail": stream_path})
    except Exception as e:
        checks.append({"name": "streamlines_created", "pass": False, "detail": str(e)})

    # 5. Verify streamlines prim
    stream_prim = stage.GetPrimAtPath(stream_path)
    prim_exists = bool(stream_prim and stream_prim.IsValid())
    checks.append({"name": "streamlines_prim_exists", "pass": prim_exists, "detail": stream_path})

    # 6. Verify seed prim
    seed_prim = stage.GetPrimAtPath(seed_path)
    seed_exists = bool(seed_prim and seed_prim.IsValid())
    checks.append({"name": "seed_prim_exists", "pass": seed_exists, "detail": seed_path})

    # 7. Verify velocity binding
    if prim_exists:
        vel_targets = cae_viz.FieldSelectionAPI(stream_prim, "velocities").GetTargetRel().GetTargets()
        vel_bound = len(vel_targets) > 0
        checks.append({"name": "velocity_bound", "pass": vel_bound,
                        "detail": str(vel_targets[0]) if vel_targets else "No targets"})

    # 8. Verify color binding
    if prim_exists:
        color_targets = cae_viz.FieldSelectionAPI(stream_prim, "colors").GetTargetRel().GetTargets()
        color_bound = len(color_targets) > 0
        checks.append({"name": "color_bound", "pass": color_bound,
                        "detail": str(color_targets[0]) if color_targets else "No targets"})

    # 9. Verify seed binding
    if prim_exists:
        seed_targets = cae_viz.DatasetSelectionAPI(stream_prim, "seeds").GetTargetRel().GetTargets()
        seed_bound = len(seed_targets) > 0
        checks.append({"name": "seed_bound", "pass": seed_bound,
                        "detail": str(seed_targets[0]) if seed_targets else "No targets"})

    # 10. Capture screenshot
    try:
        await frame_prims([bbox_path], zoom=0.9)
        for _ in range(600):
            await app.next_update_async()

        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        vp = get_active_viewport()
        capture_viewport_to_file(vp, file_path=OUTPUT_PATH)
        for _ in range(30):
            await app.next_update_async()

        file_exists = os.path.isfile(OUTPUT_PATH)
        file_size = os.path.getsize(OUTPUT_PATH) if file_exists else 0
        checks.append({"name": "screenshot_exists", "pass": file_exists and file_size > 10000,
                        "detail": f"{OUTPUT_PATH} ({file_size} bytes)"})
    except Exception as e:
        checks.append({"name": "screenshot_exists", "pass": False, "detail": str(e)})

    _emit_result("05_streamlines", checks)
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
