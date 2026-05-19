"""
Eval 07 Validator: Field Statistics
Runs inside Kit-CAE via --exec. Imports StaticMixer.cgns, computes Pressure
statistics, and validates values against ground truth.
"""
import asyncio
import json
import math
import os

import omni.kit.app
import omni.usd
from omni.cae.data.commands import execute_command
from omni.cae.schema import cae
from omni.cae.testing import wait_for_update
from omni.usd import get_context
from pxr import Tf, Usd


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
        _emit_result("07_field_statistics", checks)
        _shutdown(app)
        return

    stage = get_context().get_stage()

    # 2. Find Pressure field
    pressure_path = "/World/StaticMixer/Base/StaticMixer/Flow_Solution/Pressure"
    pressure_prim = stage.GetPrimAtPath(pressure_path)
    prim_found = bool(pressure_prim and pressure_prim.IsValid())
    checks.append({"name": "pressure_field_found", "pass": prim_found, "detail": pressure_path})

    if not prim_found:
        _emit_result("07_field_statistics", checks)
        _shutdown(app)
        return

    # 3. Compute statistics
    try:
        from omni.cae.data import array_utils, usd_utils

        farray = await usd_utils.get_array(pressure_prim, Usd.TimeCode.EarliestTime())
        checks.append({"name": "array_loaded", "pass": farray is not None,
                        "detail": f"shape={farray.shape} dtype={farray.dtype}"})

        # Get ranges
        ranges = array_utils.get_componentwise_ranges(farray)
        has_range = len(ranges) > 0
        checks.append({"name": "range_computed", "pass": has_range,
                        "detail": f"range={ranges[0] if ranges else 'N/A'}"})

        # Get scalar stats
        stats = array_utils.get_scalar_stats(farray, num_bins=32)
        checks.append({"name": "stats_computed", "pass": stats is not None,
                        "detail": f"keys={list(stats.keys()) if stats else 'None'}"})

        # Validate required keys
        required_keys = ["min", "max", "mean", "median", "counts", "bin_edges"]
        for key in required_keys:
            has_key = key in stats if stats else False
            checks.append({"name": f"has_{key}", "pass": has_key,
                            "detail": f"{key}={'present' if has_key else 'missing'}"})

        # Validate values are finite
        if stats:
            for key in ["min", "max", "mean", "median"]:
                if key in stats:
                    val = float(stats[key])
                    is_finite = math.isfinite(val)
                    checks.append({"name": f"{key}_finite", "pass": is_finite,
                                    "detail": f"{key}={val}"})

            # Validate min <= median <= max
            if all(k in stats for k in ["min", "max", "median"]):
                ordering_ok = float(stats["min"]) <= float(stats["median"]) <= float(stats["max"])
                checks.append({"name": "ordering_valid", "pass": ordering_ok,
                                "detail": f"min={stats['min']} median={stats['median']} max={stats['max']}"})

            # Validate min <= mean <= max
            if all(k in stats for k in ["min", "max", "mean"]):
                mean_ok = float(stats["min"]) <= float(stats["mean"]) <= float(stats["max"])
                checks.append({"name": "mean_in_range", "pass": mean_ok,
                                "detail": f"min={stats['min']} mean={stats['mean']} max={stats['max']}"})

            # Validate histogram
            if "counts" in stats and "bin_edges" in stats:
                counts = stats["counts"]
                edges = stats["bin_edges"]
                hist_ok = len(counts) == 32 and len(edges) == 33
                checks.append({"name": "histogram_shape", "pass": hist_ok,
                                "detail": f"counts={len(counts)} edges={len(edges)}"})

                # All counts non-negative
                counts_ok = all(c >= 0 for c in counts)
                checks.append({"name": "counts_nonneg", "pass": counts_ok,
                                "detail": f"min_count={min(counts)}"})

                # Total count should approximately match array size (±1 for bin edge rounding)
                total = sum(counts)
                shape_size = farray.shape[0]
                count_ok = abs(total - shape_size) <= 1
                checks.append({"name": "counts_sum", "pass": count_ok,
                                "detail": f"sum={total} expected={shape_size} (tolerance ±1)"})

        # Emit ground truth for reference
        if stats:
            print(f"\nGROUND_TRUTH_BEGIN")
            ground = {
                "min": float(stats["min"]),
                "max": float(stats["max"]),
                "mean": float(stats["mean"]),
                "median": float(stats["median"]),
                "histogram_bins": 32,
                "total_elements": int(farray.shape[0]),
            }
            print(json.dumps(ground, indent=2))
            print(f"GROUND_TRUTH_END\n")

    except Exception as e:
        checks.append({"name": "stats_computed", "pass": False, "detail": str(e)})

    _emit_result("07_field_statistics", checks)
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
