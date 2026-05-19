#!/usr/bin/env python3
"""Inspect a VTK file and list its fields.

Runs inside Kit-CAE via --exec. Reads the target file path from the
environment variable ``CAE_INSPECT_FILE``.

Usage (from the kit-cae repo root):
    CAE_INSPECT_FILE=/path/to/file.vti ./repo.sh launch -n omni.cae_vtk.kit -- \
        --exec skills/cae-core/scripts/inspect_vtk.py --no-window
"""

import asyncio
import os
import sys

import omni.kit.app

# vtk is available inside Kit via pip_prebundle
import vtk


def inspect_vtk(path):
    ext = os.path.splitext(path)[1].lower()
    readers = {
        ".vti": vtk.vtkXMLImageDataReader,
        ".vtu": vtk.vtkXMLUnstructuredGridReader,
        ".vts": vtk.vtkXMLStructuredGridReader,
        ".vtp": vtk.vtkXMLPolyDataReader,
        ".vtk": vtk.vtkDataSetReader,
    }
    if ext not in readers:
        print(f"ERROR: Unsupported extension: {ext}")
        return

    reader = readers[ext]()
    reader.SetFileName(path)
    reader.Update()
    data = reader.GetOutput()

    print(f"\nFile:       {path}")
    print(f"Type:       {data.GetClassName()}")
    if hasattr(data, "GetNumberOfPoints"):
        print(f"Points:     {data.GetNumberOfPoints()}")
    if hasattr(data, "GetNumberOfCells"):
        print(f"Cells:      {data.GetNumberOfCells()}")

    pd = data.GetPointData()
    print(f"\nPoint data fields ({pd.GetNumberOfArrays()}):")
    for i in range(pd.GetNumberOfArrays()):
        arr = pd.GetArray(i)
        comps = arr.GetNumberOfComponents()
        name = arr.GetName() or f"(unnamed-{i})"
        print(f"  - {name}  (components: {comps}, tuples: {arr.GetNumberOfTuples()})")

    cd = data.GetCellData()
    print(f"\nCell data fields ({cd.GetNumberOfArrays()}):")
    for i in range(cd.GetNumberOfArrays()):
        arr = cd.GetArray(i)
        comps = arr.GetNumberOfComponents()
        name = arr.GetName() or f"(unnamed-{i})"
        print(f"  - {name}  (components: {comps}, tuples: {arr.GetNumberOfTuples()})")


async def main():
    app = omni.kit.app.get_app()

    file_path = os.environ.get("CAE_INSPECT_FILE", "")
    if not file_path:
        print("ERROR: Set CAE_INSPECT_FILE=/path/to/file before launching.")
    elif not os.path.isfile(file_path):
        print(f"ERROR: File not found: {file_path}")
    else:
        inspect_vtk(file_path)

    print("INSPECT_COMPLETE")
    app.post_quit()
    for _ in range(10):
        await app.next_update_async()
    os._exit(0)


if __name__ == "__main__":
    asyncio.ensure_future(main())
