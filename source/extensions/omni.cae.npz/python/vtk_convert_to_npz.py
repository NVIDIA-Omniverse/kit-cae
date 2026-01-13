# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import argparse
from enum import Enum
from pathlib import Path

import numpy as np
import vtk
from vtkmodules import vtkCommonDataModel
from vtkmodules.numpy_interface import dataset_adapter as dsa


class ElementType(Enum):
    # only listing linear types
    ElementTypeNull = 0
    NODE = 2
    BAR_2 = 3
    TRI_3 = 5
    QUAD_4 = 7
    TETRA_4 = 10
    PYRA_5 = 12
    PENTA_6 = 16
    HEXA_8 = 17
    MIXED = 20


vtk_cell_types_to_sids_cell_types = {
    vtkCommonDataModel.VTK_VERTEX: ElementType.NODE,
    vtkCommonDataModel.VTK_POLY_VERTEX: ElementType.NODE,
    vtkCommonDataModel.VTK_LINE: ElementType.BAR_2,
    vtkCommonDataModel.VTK_POLY_LINE: ElementType.BAR_2,
    vtkCommonDataModel.VTK_TRIANGLE: ElementType.TRI_3,
    vtkCommonDataModel.VTK_QUAD: ElementType.QUAD_4,
    vtkCommonDataModel.VTK_TETRA: ElementType.TETRA_4,
    vtkCommonDataModel.VTK_PYRAMID: ElementType.PYRA_5,
    vtkCommonDataModel.VTK_WEDGE: ElementType.PENTA_6,
    vtkCommonDataModel.VTK_HEXAHEDRON: ElementType.HEXA_8,
}


def map_types(cell_types: np.ndarray) -> np.ndarray:
    lookup_array = np.empty(shape=vtkCommonDataModel.VTK_NUMBER_OF_CELL_TYPES, dtype=np.uint8)
    for t in range(lookup_array.shape[0]):
        # ltype = get_linear_element_type(t)
        lookup_array[t] = vtk_cell_types_to_sids_cell_types.get(t, ElementType.ElementTypeNull).value
    result = lookup_array[cell_types]
    if ElementType.ElementTypeNull.value in result:
        # TODO: print which type is the one that's unsupported.
        unique_types = np.unique(cell_types).tolist()
        for k in vtk_cell_types_to_sids_cell_types.keys():
            if k in unique_types:
                unique_types.remove(k)
        unique_type_names = [
            vtkCommonDataModel.vtkCellTypes.GetClassNameFromTypeId(k) + f" ({k})" for k in unique_types
        ]
        raise RuntimeError(
            "Currently unsupported cell type encountered! Use -p/--points-only to skip connectivity. "
            "Or -t/--tetrahedralize to use tets instead. Following are the unsupported cells types "
            "in the dataset: %s" % unique_type_names
        )
    return result


def read_data(fname):
    p = Path(fname)
    if p.suffix.lower() == ".vtk":
        reader = vtk.vtkDataSetReader()
        reader.SetFileName(fname)
        reader.Update()
        return reader.GetOutputDataObject(0)
    raise RuntimeError("Unsupported file %s", fname)


def get_arrays(a: np.ndarray, name: str, split: bool) -> dict[str, np.ndarray]:
    if split and a.ndim == 2 and a.shape[1] > 1:
        r = {}
        for comp in range(a.shape[1]):
            r[f"{name}_{comp}"] = a[:, comp]
        return r
    else:
        return {name: a}


def reorder_wedges(ids: np.ndarray) -> np.ndarray:
    ids = ids.reshape((-1, 6))  # wedges have 6 points.
    # swap every 2nd and 3rd id
    ids[:, [1, 2]] = ids[:, [2, 1]]
    return ids.ravel()


def reorder_wedges_mixed(ids: np.ndarray, types: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    offsets = offsets[types == vtkCommonDataModel.VTK_WEDGE]
    # expand offsets to include all offsets for the ids
    offsets = offsets[:, None] + np.arange(6)
    offsets = offsets.ravel()
    # now, reorder just the chosen subset
    ids[offsets] = reorder_wedges(ids[offsets])
    return ids


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="VTK dataset to load. Supported extensions are .vtk", required=True)
parser.add_argument("-o", "--output", help="Name of the output file", required=True)
parser.add_argument("-t", "--tetrahedralize", action="store_true", help="Tetrahedralize dataset", default=False)
parser.add_argument(
    "-m", "--force-mixed", action="store_true", help="Saved MIXED connecitivity even when not", default=False
)
parser.add_argument("-s", "--split", action="store_true", help="Split multicomponent arrays", default=False)
parser.add_argument("-c", "--compress", action="store_true", help="Compress output npz file.", default=False)
parser.add_argument(
    "-p", "--points-only", action="store_true", help="Only process points (ignoring topology)", default=False
)
args = parser.parse_args()

dataset = read_data(args.input)
if args.tetrahedralize:
    print("tetrahedralizing ...")
    f = vtk.vtkDataSetTriangleFilter()
    f.SetInputDataObject(dataset)
    f.Update()
    dataset = f.GetOutputDataObject(0)
    print("                 ... done")

if not dataset.IsA("vtkUnstructuredGrid"):
    raise RuntimeError("Currenrly only unstructured grids are supported.")

dataset = dsa.WrapDataObject(dataset)

arrays = {}
arrays.update(get_arrays(dataset.Points, "coords", args.split))
for name in dataset.PointData.keys():
    arrays.update(get_arrays(dataset.PointData[name], name, args.split))


numCells = dataset.GetNumberOfCells()
if numCells > 0:
    arrays.update({"element_range": np.array([1, numCells], dtype=np.int64)})

    cells = dataset.VTKObject.GetCells()
    if args.points_only:
        arrays.update({"element_type": np.array(ElementType.NODE.value, dtype=np.uint8)})

    # SIDS, element connectivity is 1-based
    conn = dsa.vtkDataArrayToVTKArray(cells.GetConnectivityArray()) + 1

    types = dsa.vtkDataArrayToVTKArray(dataset.GetCellTypesArray())
    if not args.force_mixed and np.amin(types) == np.amax(types):
        # dataset has uniform cells!
        stype = vtk_cell_types_to_sids_cell_types.get(types[0])
        arrays.update({"element_type": np.array([stype.value], dtype=np.uint8)})
        if stype == ElementType.PENTA_6.value:
            conn = reorder_wedges(conn)
    else:
        # non uniform cells.
        arrays.update({"element_type": np.array([ElementType.MIXED.value], dtype=np.uint8)})
        offsets = dsa.vtkDataArrayToVTKArray(cells.GetOffsetsArray())
        types = map_types(types)
        if ElementType.PENTA_6.value in types:
            conn = reorder_wedges(conn, types, offsets)

        # now we need to combine types and conn and recompute offsets.
        conn = np.insert(conn, offsets[:-1], types)
        offsets = offsets + np.arange(len(offsets))
        arrays.update({"start_offsets": offsets})

    arrays.update({"element_connectivity": conn})

if args.compress:
    print("saving (compressed)", args.output)
    np.savez_compressed(args.output, **arrays)
else:
    print("saving (uncompressed)", args.output)
    np.savez(args.output, **arrays)
