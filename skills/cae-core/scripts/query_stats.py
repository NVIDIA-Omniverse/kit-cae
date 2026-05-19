#!/usr/bin/env python3
"""
Query field statistics from a Kit-CAE dataset.

Imports a file, discovers all data fields, and prints per-field statistics
(min, max, mean, median, quartiles, histogram) in a structured format that
an agent can parse and report to users.

Environment variables:
    CAE_STATS_FILE      (required)  Path to the data file
    CAE_STATS_FORMAT    (optional)  Force format: cgns | vtk | ensight | npz
                                    Auto-detected from extension if omitted
    CAE_STATS_FIELDS    (optional)  Comma-separated field prim paths to query.
                                    If omitted, queries all discovered data fields.

Runs inside Kit-CAE via --exec:
    CAE_STATS_FILE=/path/to/data.cgns ./repo.sh launch -n omni.cae.kit -- \
        --exec skills/cae-core/scripts/query_stats.py --no-window
"""
import asyncio
import json
import os
import sys

import omni.kit.app
import omni.usd
from omni.cae.data import array_utils, usd_utils
from omni.cae.schema import cae
from omni.usd import get_context
from pxr import Tf, Usd

import usdrt


async def main():
    app = omni.kit.app.get_app()

    file_path = os.environ.get("CAE_STATS_FILE", "")
    if not file_path or not os.path.isfile(file_path):
        print(f"ERROR: CAE_STATS_FILE not set or file not found: '{file_path}'")
        app.post_quit()
        for _ in range(10):
            await app.next_update_async()
        os._exit(1)

    forced_format = os.environ.get("CAE_STATS_FORMAT", "").lower()
    requested_fields = os.environ.get("CAE_STATS_FIELDS", "")

    # Detect format from extension
    ext = os.path.splitext(file_path)[1].lower()
    fmt = forced_format or {
        ".cgns": "cgns", ".vti": "vtk", ".vtu": "vtk", ".vts": "vtk",
        ".vtp": "vtk", ".vtk": "vtk", ".case": "ensight", ".encas": "ensight",
        ".npz": "npz", ".npy": "npz",
    }.get(ext, "")

    if not fmt:
        print(f"ERROR: Cannot detect format for extension '{ext}'. Set CAE_STATS_FORMAT.")
        app.post_quit()
        for _ in range(10):
            await app.next_update_async()
        os._exit(1)

    # Import
    prim_path = "/World/StatsTarget"
    print(f"Importing {file_path} as {fmt}...")
    if fmt == "cgns":
        from omni.cae.importer.cgns import import_to_stage
        await import_to_stage(file_path, prim_path)
    elif fmt == "vtk":
        from omni.cae.importer.vtk import import_to_stage
        await import_to_stage(file_path, prim_path)
    elif fmt == "ensight":
        from omni.cae.importer.ensight import import_to_stage
        await import_to_stage(file_path, prim_path)
    elif fmt == "npz":
        from omni.cae.importer.npz import import_to_stage
        await import_to_stage(file_path, prim_path, schema_type="SIDS Unstructured")

    stage = get_context().get_stage()

    # Wait for data delegates to load
    for _ in range(30):
        await app.next_update_async()

    # Discover datasets and fields via USDRT
    fabric_stage = usdrt.Usd.Stage.Attach(omni.usd.get_context().get_stage_id())

    dataset_type = Tf.Type.Find(cae.DataSet)
    datasets = [p.GetString() for p in fabric_stage.GetPrimsWithTypeName(dataset_type.typeName)]

    field_base_type = Tf.Type.Find(cae.FieldArray)
    all_fields = [p.GetString() for p in fabric_stage.GetPrimsWithTypeName(field_base_type.typeName)]

    # Filter to data fields (skip mesh/coordinate arrays)
    containers = [
        "PointData", "CellData", "FlowSolution", "Flow_Solution",
        "SolutionCellCenter", "SolutionVertex",
        "Variables", "NumPyArrays",
    ]
    data_fields = [f for f in all_fields
                   if any(f"/{c}/" in f or f.endswith(f"/{c}") for c in containers)]
    if not data_fields:
        data_fields = all_fields

    # If specific fields requested, filter to those
    if requested_fields:
        target_fields = [f.strip() for f in requested_fields.split(",") if f.strip()]
        data_fields = [f for f in data_fields if f in target_fields]

    # Print dataset summary
    result = {
        "file": file_path,
        "format": fmt,
        "datasets": datasets,
        "fields": {},
    }

    for field_path in sorted(data_fields):
        prim = stage.GetPrimAtPath(field_path)
        if not prim.IsValid():
            continue

        field_info = {"path": field_path}

        try:
            farray = await usd_utils.get_array(prim, Usd.TimeCode.EarliestTime())
            if farray is None:
                field_info["error"] = "Could not read array data"
                result["fields"][field_path] = field_info
                continue

            field_info["dtype"] = str(farray.dtype)
            field_info["shape"] = list(farray.shape)
            field_info["device"] = str(array_utils.get_device(farray))

            # Component-wise ranges
            ranges = array_utils.get_componentwise_ranges(farray)
            field_info["ranges"] = [{"component": i, "min": r[0], "max": r[1]}
                                    for i, r in enumerate(ranges)]

            # Scalar stats (histogram, mean, median, quartiles) for scalar fields
            is_scalar = farray.ndim == 1 or farray.shape[-1] == 1
            if is_scalar:
                stats = array_utils.get_scalar_stats(farray, num_bins=32)
                field_info["statistics"] = {
                    "min": stats["min"],
                    "max": stats["max"],
                    "mean": stats["mean"],
                    "median": stats["median"],
                    "q1": list(stats["q1"]),
                    "q2": list(stats["q2"]),
                    "q3": list(stats["q3"]),
                    "q4": list(stats["q4"]),
                    "histogram": {
                        "counts": stats["counts"],
                        "bin_edges": stats["bin_edges"],
                    },
                }
            else:
                field_info["statistics"] = "Vector field — per-component ranges above. " \
                    "Use magnitude or individual components for scalar statistics."

        except Exception as e:
            field_info["error"] = str(e)

        result["fields"][field_path] = field_info

    # Output as JSON for easy parsing
    print("STATS_BEGIN")
    print(json.dumps(result, indent=2, default=str))
    print("STATS_END")

    app.post_quit()
    for _ in range(10):
        await app.next_update_async()
    os._exit(0)


if __name__ == "__main__":
    asyncio.ensure_future(main())
