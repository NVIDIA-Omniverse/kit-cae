# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

r"""
This module includes implementation for CreateIrregularVolumeSubset to enable
CaeDataSetImporter to import a dataset with VTK APIs.
"""

from logging import getLogger
from typing import Any

import numpy as np
import warp as wp
from omni.cae.data import usd_utils
from omni.cae.index.bindings import (
    Attribute_affiliation,
    Attribute_parameters,
    Attribute_storage,
    Attribute_type,
    IIrregular_volume_subset,
    Mesh_parameters,
    Mesh_storage,
)
from omni.cae.index.commands import CreateIrregularVolumeSubset
from vtkmodules.numpy_interface import dataset_adapter as dsa
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkCommonCore import vtkObjectFactory
from vtkmodules.vtkCommonDataModel import VTK_VOXEL, vtkCell, vtkCell3D, vtkCellTypes, vtkUnstructuredGrid

from ..commands import ConvertToVTKDataSet

logger = getLogger(__name__)


@wp.struct
class Cell:
    nb_vertices: int
    nb_faces: int
    nb_face_vertices: int
    face_vertices: wp.array(ndim=2, dtype=wp.int32)


@wp.kernel
def fill_faces_uniform(
    faces: wp.array(ndim=2, dtype=wp.uint32),
    cells: wp.array(ndim=2, dtype=wp.uint32),
    face_vtx_count: wp.array(dtype=wp.uint32),
):
    cellIdx = wp.tid()
    faceIdx = wp.int32(cells[cellIdx][1])
    for i in range(face_vtx_count.shape[0]):
        faces[faceIdx + i][0] = face_vtx_count[i]


@wp.kernel
def fill_start_index(array: wp.array(ndim=2, dtype=wp.uint32)):
    # faces[:,0] is nb_vertices
    # faces[:,1] is start_vertex_index
    array[0][1] = wp.uint32(0)
    for i in range(1, array.shape[0]):
        array[i][1] = array[i - 1][0] + array[i - 1][1]


@wp.kernel
def fill_tid(array: wp.array(dtype=wp.uint32)):
    t = wp.tid()
    array[t] = wp.uint32(t)


@wp.kernel
def fill_face_vtx_indices_uniform(
    face_vtx_indices: wp.array(ndim=1, dtype=wp.uint32),
    faces: wp.array(ndim=2, dtype=wp.uint32),
    cells: wp.array(ndim=2, dtype=wp.uint32),
    vtk_connectivity: wp.array(dtype=Any),
    cell: Cell,
):
    cell_id = wp.tid()
    nb_faces = wp.int32(cells[cell_id][0])
    face_offset = wp.int32(cells[cell_id][1])

    vtk_offset = cell.nb_vertices * cell_id  # since this is for uniform cells

    for f in range(nb_faces):
        nb_face_vertices = wp.int32(faces[face_offset + f][0])
        face_vtx_offset = wp.int32(faces[face_offset + f][1])
        for v in range(nb_face_vertices):
            vtx = cell.face_vertices[f][v]
            face_vtx_indices[face_vtx_offset + v] = wp.uint32(vtk_connectivity[vtk_offset + vtx])


def get_vtk_cell(cell_type: vtkCell3D) -> Cell:
    cell = Cell()
    cell.nb_vertices = cell_type.GetNumberOfPoints()
    cell.nb_faces = cell_type.GetNumberOfFaces()

    max_face_size = np.max([cell_type.GetFace(i).GetNumberOfPoints() for i in range(cell.nb_faces)])
    cell.nb_face_vertices = 0
    face_vertices = np.full(shape=(cell_type.GetNumberOfFaces(), max_face_size), dtype=np.int32, fill_value=-1)
    for i in range(cell.nb_faces):
        nb_vertices = cell_type.GetFace(i).GetNumberOfPoints()
        face_vertices[i, :nb_vertices] = np.array(cell_type.GetFaceArray(i))[:nb_vertices]
        if cell_type.GetCellType() == VTK_VOXEL:
            # vtkVoxel returns faces in "triangle-strip" order, we need to reorder them to match the expected quad order
            face_vertices[i] = face_vertices[i][[0, 1, 3, 2]]
        cell.nb_face_vertices += nb_vertices

    cell.face_vertices = wp.array(face_vertices, dtype=wp.int32, device="cpu")
    logger.info(
        f"Cell ({cell_type.GetClassName()}): \n"
        f"  nb_vertices: {cell.nb_vertices:,}\n"
        f"  nb_faces: {cell.nb_faces:,}\n"
        f"  nb_face_vertices: {cell.nb_face_vertices:,}\n"
        f"  face_vertices: {cell.face_vertices}"
    )
    return cell


class CaeVtkUnstructuredGridCreateIrregularVolumeSubset(CreateIrregularVolumeSubset):

    async def do(self):
        vtk_dataset = await ConvertToVTKDataSet.invoke(
            self.dataset, self.fields, forcePointData=False, timeCode=self.timeCode
        )

        assert vtk_dataset.IsA("vtkUnstructuredGrid"), (
            "Expected vtkUnstructuredGrid, got %s" % vtk_dataset.GetClassName()
        )
        assert vtk_dataset.IsHomogeneous(), "Currently non-homogeneous unstructured grids are not supported"

        vtk_connectivity = vtk_to_numpy(vtk_dataset.VTKObject.GetCells().GetConnectivityArray())

        cell = get_vtk_cell(vtk_dataset.GetCell(0))
        params = Mesh_parameters()
        params.nb_vertices = vtk_dataset.Points.shape[0]
        params.nb_cells = vtk_dataset.VTKObject.GetNumberOfCells()
        params.nb_faces = params.nb_cells * cell.nb_faces
        params.nb_face_vtx_indices = params.nb_cells * cell.nb_face_vertices
        params.nb_cell_face_indices = params.nb_faces  # since we don't share faces between cells

        logger.info("Mesh parameters: %s", params)

        subset: IIrregular_volume_subset = self.subset
        storage: Mesh_storage = subset.generate_mesh_storage(params)

        s_verts = storage.get_vertices(params)
        np.copyto(s_verts, vtk_dataset.Points, casting="same_kind")

        s_cells = storage.get_cells(params)
        assert s_cells.ndim == 2
        s_cells[:, 0] = cell.nb_faces
        with wp.ScopedDevice("cpu"):
            wp.launch(fill_start_index, dim=1, inputs=[wp.array(s_cells, copy=False)])

        s_cell_face_indices = storage.get_cell_face_indices(params)
        with wp.ScopedDevice("cpu"):
            wp.launch(fill_tid, dim=s_cell_face_indices.shape[0], inputs=[wp.array(s_cell_face_indices, copy=False)])

        s_faces: np.ndarray = storage.get_faces(params)
        assert s_faces.shape[0] == params.nb_faces

        face_vtx_count = np.count_nonzero(cell.face_vertices.numpy() != -1, axis=1)
        assert face_vtx_count.shape[0] == cell.nb_faces
        assert cell.nb_faces * params.nb_cells == s_faces.shape[0]
        with wp.ScopedDevice("cpu"):
            wp.launch(
                fill_faces_uniform,
                dim=params.nb_cells,
                inputs=[
                    wp.array(s_faces, dtype=wp.uint32, copy=False),
                    wp.array(s_cells, dtype=wp.uint32, copy=False),
                    wp.array(face_vtx_count, dtype=wp.uint32, copy=False),
                ],
            )
            wp.launch(fill_start_index, dim=1, inputs=[wp.array(s_faces, copy=False)])

        # finally populate the face_vtx_indices
        s_face_vtx_indices = storage.get_face_vtx_indices(params)
        with wp.ScopedDevice("cpu"):
            wp.launch(
                fill_face_vtx_indices_uniform,
                dim=params.nb_cells,
                inputs=[
                    wp.array(s_face_vtx_indices, copy=False),
                    wp.array(s_faces, copy=False),
                    wp.array(s_cells, copy=False),
                    wp.array(vtk_connectivity, copy=False),
                    cell,
                ],
            )

        # logger.warning("Face vertex indices %s", s_face_vtx_indices)
        # logger.warning("Cells %s", s_cells)
        # logger.warning("Faces %s", s_faces)
        # logger.warning("vtk_connectivity: %s", vtk_connectivity)
        # from omni.cae.index.impl.helpers import save_vtk
        # save_vtk(params, storage, {}, "/tmp/irregular_volume_subset.vtk")
        self.do_arrays(vtk_dataset)

    def do_arrays(self, vtk_dataset):
        attrib_types = [
            Attribute_type.ATTRIB_TYPE_FLOAT32,
            Attribute_type.ATTRIB_TYPE_FLOAT32_2,
            Attribute_type.ATTRIB_TYPE_FLOAT32_3,
            Attribute_type.ATTRIB_TYPE_FLOAT32_4,
        ]
        for idx, field in enumerate(self.fields):
            if field in vtk_dataset.PointData.keys():
                array = vtk_dataset.PointData[field]
                affiliation = Attribute_affiliation.ATTRIB_AFFIL_PER_VERTEX
            elif field in vtk_dataset.CellData.keys():
                array = vtk_dataset.CellData[field]
                affiliation = Attribute_affiliation.ATTRIB_AFFIL_PER_CELL
            else:
                logger.error("Failed to find field %s in PointData or CellData", field)
            if array.ndim == 1 or (array.ndim == 2 and array.shape[1] <= 4):
                params = Attribute_parameters()
                params.type = attrib_types[0] if array.ndim == 1 else attrib_types[array.shape[1] - 1]
                params.affiliation = affiliation
                params.nb_attrib_values = array.shape[0]

                storage = self.subset.generate_attribute_storage(idx, params)
                assert storage is not None

                s_attrib_values = storage.get_attrib_values(params)
                np.copyto(s_attrib_values, array, casting="same_kind")
            else:
                logger.error("Failed to process field %s: unsupported array shape %s", field, array.shape)
