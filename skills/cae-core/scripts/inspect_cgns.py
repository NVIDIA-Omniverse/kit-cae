#!/usr/bin/env python3
"""Inspect a CGNS file structure using h5py.

Prints bases, zones, datasets, flow solutions, and field arrays with
shapes and dtypes.

Runs inside Kit-CAE via --exec. Reads the target file path from the
environment variable ``CAE_INSPECT_FILE``.

Usage (from the kit-cae repo root):
    CAE_INSPECT_FILE=/path/to/file.cgns ./repo.sh launch -n omni.cae.kit -- \
        --exec skills/cae-core/scripts/inspect_cgns.py --no-window
"""

import asyncio
import os
import sys

import omni.kit.app

# h5py is available inside Kit via pip_prebundle
import h5py


def inspect_cgns(filepath):
    print(f"\n=== CGNS File: {filepath} ===\n")

    with h5py.File(filepath, "r") as f:
        for base_name in sorted(f.keys()):
            base = f[base_name]
            if not isinstance(base, h5py.Group):
                continue

            label = ""
            if " label" in base.attrs:
                raw = base.attrs[" label"]
                label = raw.decode() if isinstance(raw, bytes) else str(raw)

            print(f"Base: {base_name}  ({label})")

            for zone_name in sorted(base.keys()):
                zone = base[zone_name]
                if not isinstance(zone, h5py.Group):
                    continue

                zone_label = ""
                if " label" in zone.attrs:
                    raw = zone.attrs[" label"]
                    zone_label = raw.decode() if isinstance(raw, bytes) else str(raw)

                print(f"  Zone: {zone_name}  ({zone_label})")

                for child_name in sorted(zone.keys()):
                    child = zone[child_name]
                    if not isinstance(child, h5py.Group):
                        continue

                    child_label = ""
                    if " label" in child.attrs:
                        raw = child.attrs[" label"]
                        child_label = raw.decode() if isinstance(raw, bytes) else str(raw)

                    print(f"    {child_name}  ({child_label})")

                    for field_name in sorted(child.keys()):
                        field = child[field_name]
                        if isinstance(field, h5py.Group):
                            if " data" in field:
                                data = field[" data"]
                                print(f"      {field_name}  [{data.dtype}, shape={data.shape}]")
                            else:
                                fl = ""
                                if " label" in field.attrs:
                                    raw = field.attrs[" label"]
                                    fl = raw.decode() if isinstance(raw, bytes) else str(raw)
                                print(f"      {field_name}  ({fl})")
                        elif isinstance(field, h5py.Dataset):
                            print(f"      {field_name}  [{field.dtype}, shape={field.shape}]")

    print("\n--- Stage path mapping (after import to /World/<root>) ---")
    print("Datasets:  /World/<root>/Base/<ZoneName>/<DatasetName>")
    print("Fields:    /World/<root>/Base/<ZoneName>/<FlowSolution>/<FieldName>")
    print("")
    print("NOTE: Dots and spaces in CGNS names become underscores in USD paths.")
    print("  e.g., 'B1.P3' -> 'B1_P3', 'Flow Solution' -> 'Flow_Solution'")


async def main():
    app = omni.kit.app.get_app()

    file_path = os.environ.get("CAE_INSPECT_FILE", "")
    if not file_path:
        print("ERROR: Set CAE_INSPECT_FILE=/path/to/file.cgns before launching.")
    elif not os.path.isfile(file_path):
        print(f"ERROR: File not found: {file_path}")
    else:
        inspect_cgns(file_path)

    print("INSPECT_COMPLETE")
    app.post_quit()
    for _ in range(10):
        await app.next_update_async()
    os._exit(0)


if __name__ == "__main__":
    asyncio.ensure_future(main())
