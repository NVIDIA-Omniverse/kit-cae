# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
This module handles data conversion from various API schema types to VTK dataset.
"""

__all__ = ["get_vtk_dataset_from_sids_unstructured"]

from logging import getLogger

import numpy as np
from omni.cae.data import progress, usd_utils
from omni.cae.schema import cae, sids
from omni.cae.vtk.commands import ConvertToVTKDataSet
from pxr import Usd
from vtkmodules import vtkCommonDataModel
from vtkmodules.numpy_interface import dataset_adapter as dsa

from .. import sids_unstructured, types
from ..types import ElementType

logger = getLogger(__name__)


# we're only filling this up with linear cells.
sids_cell_types_to_vtk_cell_types = {
    ElementType.NODE: vtkCommonDataModel.VTK_POLY_VERTEX,
    ElementType.BAR_2: vtkCommonDataModel.VTK_LINE,
    ElementType.TRI_3: vtkCommonDataModel.VTK_TRIANGLE,
    ElementType.QUAD_4: vtkCommonDataModel.VTK_QUAD,
    ElementType.TETRA_4: vtkCommonDataModel.VTK_TETRA,
    ElementType.PYRA_5: vtkCommonDataModel.VTK_PYRAMID,
    ElementType.PENTA_6: vtkCommonDataModel.VTK_WEDGE,
    ElementType.HEXA_8: vtkCommonDataModel.VTK_HEXAHEDRON,
}


def _get_unstructured_grid(section: sids_unstructured.Elements_t) -> dsa.DataSet:
    ug = dsa.WrapDataObject(vtkCommonDataModel.vtkUnstructuredGrid())
    elementType = ElementType(section.elementType)
    if elementType == ElementType.NFACE_n.value or elementType == ElementType.NGON_n.value:
        raise usd_utils.QuietableException("Polyhedral elements are currently not supported")
    elif elementType == ElementType.MIXED:
        elementConnectivity = section.elementConnectivity.numpy()
        mask = np.ones_like(elementConnectivity, dtype=bool)
        offsets = section.elementStartOffset.numpy() - section.elementStartOffsetShift
        mask[offsets[:-1]] = False

        vtk_vertex_ids = elementConnectivity[mask] - 1
        vtk_offsets = (offsets - np.arange(offsets.shape[0])).astype(vtk_vertex_ids.dtype, copy=False)
        vtk_cell_types = map_to_vtk_celltypes(section.elementConnectivity.numpy()[~mask])

        if vtkCommonDataModel.VTK_WEDGE in vtk_cell_types:
            # reorder mixed wedge
            vtk_vertex_ids = reorder_wedges_mixed(vtk_vertex_ids, vtk_cell_types, vtk_offsets)

        cells = vtkCommonDataModel.vtkCellArray()
        cells.SetData(dsa.numpyTovtkDataArray(vtk_offsets), dsa.numpyTovtkDataArray(vtk_vertex_ids))
        ug.VTKObject.SetCells(dsa.numpyTovtkDataArray(vtk_cell_types), cells)
    else:
        # uniform
        if elementType not in sids_cell_types_to_vtk_cell_types:
            raise RuntimeError(f"Unsupported cell type {section.elementType}")

        vtk_cell_type = sids_cell_types_to_vtk_cell_types[elementType]
        vtk_vertex_ids = section.elementConnectivity.numpy() - 1
        vtk_offsets = np.arange(
            0, vtk_vertex_ids.shape[0] + 1, types.get_vertex_count(elementType), dtype=vtk_vertex_ids.dtype
        )
        vtk_offsets.astype(vtk_vertex_ids.dtype, copy=False)

        if vtk_cell_type == vtkCommonDataModel.VTK_WEDGE:
            # wedges are the only linear cell type that differs in order between VTK and SIDS
            # so we need to adjust the vtk_vertex_ids.
            vtk_vertex_ids = reorder_wedges(vtk_vertex_ids)

        cells = vtkCommonDataModel.vtkCellArray()
        cells.SetData(dsa.numpyTovtkDataArray(vtk_offsets), dsa.numpyTovtkDataArray(vtk_vertex_ids))
        ug.VTKObject.SetCells(vtk_cell_type, cells)
        vtk_cell_types = None
    ug._py_offsets = vtk_offsets
    ug._py_vertex_ids = vtk_vertex_ids
    ug._py_cell_types = vtk_cell_types
    return ug


def map_to_vtk_celltypes(cell_types: np.ndarray) -> np.ndarray:
    lookup_array = np.empty(shape=ElementType.HEXA_125.value + 1, dtype=np.uint8)
    for t in range(lookup_array.shape[0]):
        # ltype = get_linear_element_type(t)
        lookup_array[t] = sids_cell_types_to_vtk_cell_types.get(ElementType(t), vtkCommonDataModel.VTK_EMPTY_CELL)

    result = lookup_array[cell_types]
    if vtkCommonDataModel.VTK_EMPTY_CELL in result:
        # TODO: print which type is the one that's unsupported.
        raise RuntimeError("Currently unsupported cell type encountered!")
    return result


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


async def get_vtk_dataset_from_sids_unstructured(
    prim: Usd.Prim, fieldNames: list[str], timeCode: Usd.TimeCode
) -> dsa.DataSet:

    elementType = sids_unstructured.get_element_type(prim)
    if elementType == ElementType.NFACE_n or elementType == ElementType.NGON_n:
        raise NotImplementedError("Polyhedral elements are currently not supported")

    nb_work_units = 3 + 1 + len(fieldNames)

    with progress.ProgressContext(scale=3.0 / nb_work_units):
        coords = (
            await usd_utils.get_vecN_from_relationship(prim, sids.Tokens.caeSidsGridCoordinates, 3, timeCode)
        ).numpy()
    # bug in VTK may cause the pts array to be garbage collected.
    # so we explicitly track it. this is only a hack and only works till dataset
    # python object is alive.
    # dataset.Points = coords
    # dataset._py_points = coords

    with progress.ProgressContext(shift=3.0 / nb_work_units, scale=1.0 / nb_work_units):
        sections = await sids_unstructured.get_sections(prim, timeCode)

    if len(sections) > 1:
        raise usd_utils.QuietableException("Multiple sections not supported")

    dataset = _get_unstructured_grid(sections[0])
    # bug in VTK may cause the pts array to be garbage collected.
    # so we explicitly track it. this is only a hack and only works till dataset
    # python object is alive.
    dataset.Points = coords
    dataset._py_points = coords

    # handle fields
    dataset._py_fields = []
    for idx, fname in enumerate(fieldNames):
        with progress.ProgressContext(shift=(4.0 + idx) / nb_work_units, scale=1.0 / nb_work_units):
            fprim = usd_utils.get_target_prim(prim, f"field:{fname}")
            assc = cae.FieldArray(fprim).GetFieldAssociationAttr().Get(timeCode)
            farray = (await usd_utils.get_array(fprim, timeCode)).numpy()
            if assc == cae.Tokens.vertex:
                # vertex arrays are passed on untransformed
                dataset.PointData.append(farray, fname)
                dataset._py_fields.append(farray)
            elif assc == cae.Tokens.cell:
                # FIXME: use section.elementRange
                dataset.CellData.append(farray, fname)
                dataset._py_fields.append(farray)
            else:
                raise usd_utils.QuietableException(f"Unhandled association {assc}")
    return dataset


class CaeSidsUnstructuredConvertToVTKDataSet(ConvertToVTKDataSet):

    async def do(self):
        logger.info("executing %s.do()", self.__class__.__name__)
        params = self.params
        if not params.dataset.HasAPI(sids.UnstructuredAPI):
            raise usd_utils.QuietableException("Dataset (%s) does not support sids.UnstructuredAPI!" % params.dataset)

        return await get_vtk_dataset_from_sids_unstructured(params.dataset, params.fields, self.timeCode)
