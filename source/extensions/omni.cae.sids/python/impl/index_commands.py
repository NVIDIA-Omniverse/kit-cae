# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
#  its affiliates is strictly prohibited.

r"""
This module includes implementation for CreateIrregularVolumeSubset to enable
CaeDataSetImporter to import a dataset with SIDS API.
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
from omni.cae.schema import cae, sids

from .. import sids_unstructured, types

logger = getLogger(__name__)


@wp.kernel
def copy_indexed(
    in_array: wp.array(dtype=Any), idx_array: wp.array(dtype=wp.int32), out_array: wp.array(dtype=Any), offset: wp.int32
):
    t = wp.tid()
    out_array[t] = out_array.dtype(in_array[idx_array[t] + offset])


@wp.kernel
def copy_offsetted(in_array: wp.array(dtype=Any), out_array: wp.array(dtype=Any), offset: wp.int32):
    t = wp.tid()
    out_array[t] = out_array.dtype(in_array[t + offset])


@wp.struct
class Face:
    nb_vertices: wp.uint32
    start_vertex_index: wp.uint32


@wp.struct
class Cell:
    nb_faces: wp.uint32
    start_face_index: wp.uint32


@wp.kernel
def fill_counts_mixed(
    out: wp.array(ndim=2, dtype=wp.uint32), table: wp.array(dtype=wp.int32), section: sids_unstructured.Elements_t
):
    tid = wp.tid()
    offset = wp.int32(section.elementStartOffset[tid] - section.elementStartOffsetShift)
    e_type = section.elementConnectivity[offset]
    out[tid][0] = wp.uint32(table[wp.int32(e_type)])


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
def fill_faces_mixed(
    faces: wp.array(ndim=2, dtype=wp.uint32),
    cells: wp.array(ndim=2, dtype=wp.uint32),
    face_vtx_counts: wp.array(ndim=2, dtype=wp.uint32),
    section: sids_unstructured.Elements_t,
):
    cellIdx = wp.tid()
    nb_faces = wp.int32(cells[cellIdx][0])
    faceIdx = wp.int32(cells[cellIdx][1])

    offset = wp.int32(section.elementStartOffset[cellIdx] - section.elementStartOffsetShift)
    cellType = section.elementConnectivity[offset]
    face_vtx_count = face_vtx_counts[cellType]
    for i in range(nb_faces):
        faces[faceIdx + i][0] = face_vtx_count[i]


@wp.func
def fill_face_vtx_indices(
    cid: wp.int32,
    face_vtx_indices: wp.array(dtype=wp.uint32),
    faces: wp.array(ndim=2, dtype=wp.uint32),
    cells: wp.array(ndim=2, dtype=wp.uint32),
    cell_faces: wp.array(ndim=2, dtype=wp.int32),
    sids_cell_offset: wp.int32,
    section: sids_unstructured.Elements_t,
):

    nb_faces = wp.int32(cells[cid][0])
    start_face_idx = wp.int32(cells[cid][1])

    for f in range(nb_faces):
        o_fid = start_face_idx + f
        cell_face = cell_faces[f]

        nb_vtxs = wp.int32(faces[o_fid][0])
        start_vtx_idx = wp.int32(faces[o_fid][1])

        for v in range(nb_vtxs):
            sids_face_id = wp.int32(cell_face[v])
            face_vtx_indices[start_vtx_idx + v] = wp.uint32(
                section.elementConnectivity[sids_cell_offset + (sids_face_id - 1)] - 1
            )


@wp.kernel
def fill_face_vtx_indices_mixed(
    face_vtx_indices: wp.array(dtype=wp.uint32),
    faces: wp.array(ndim=2, dtype=wp.uint32),
    cells: wp.array(ndim=2, dtype=wp.uint32),
    faces_map: wp.array(ndim=3, dtype=wp.int32),
    section: sids_unstructured.Elements_t,
):
    cid = wp.tid()
    offset = wp.int32(section.elementStartOffset[cid] - section.elementStartOffsetShift)
    cellType = section.elementConnectivity[offset]
    cell_faces = faces_map[cellType]

    fill_face_vtx_indices(cid, face_vtx_indices, faces, cells, cell_faces, offset + 1, section)


@wp.kernel
def fill_face_vtx_indices_uniform(
    face_vtx_indices: wp.array(dtype=wp.uint32),
    faces: wp.array(ndim=2, dtype=wp.uint32),
    cells: wp.array(ndim=2, dtype=wp.uint32),
    cell_faces: wp.array(ndim=2, dtype=wp.int32),
    cell_vtx_count: wp.int32,
    section: sids_unstructured.Elements_t,
):
    cid = wp.tid()
    offset = cid * cell_vtx_count
    fill_face_vtx_indices(cid, face_vtx_indices, faces, cells, cell_faces, offset, section)


class CaeSidsUnstructuredCreateIrregularVolumeSubset(CreateIrregularVolumeSubset):

    async def do(self):
        dataset = self.dataset
        eType = sids_unstructured.get_element_type(dataset)
        if eType == types.ElementType.NGON_n:
            raise RuntimeError(
                "NGON_n elements cannot be volume rendered. Did you mean to select NFACE_n element block instead?"
            )

        if not types.get_is_volumetric(eType):
            raise RuntimeError(f"{eType} is not a volumetric element type!")

        if eType == types.ElementType.NFACE_n:
            return await self.do_polyhedra()

        # handle non-polyhedral grids.
        timeCode = self.timeCode
        cellFieldArrays = []
        pointFieldArrays = []
        for field in self.fields:
            field = usd_utils.get_target_prim(dataset, f"field:{field}")
            fieldAssociation = usd_utils.get_attribute(field, cae.Tokens.fieldAssociation)
            if fieldAssociation == cae.Tokens.cell:
                cellFieldArrays.append(await usd_utils.get_array(field, timeCode))
            elif fieldAssociation == cae.Tokens.vertex:
                pointFieldArrays.append(await usd_utils.get_array(field, timeCode))
            else:
                raise RuntimeError(f"Unsupported field array association {fieldAssociation}")

        with wp.ScopedDevice("cpu"):
            section = await sids_unstructured.get_section(dataset, timeCode)
            wp_coords = await sids_unstructured.get_original_grid_coordinates(dataset, timeCode)
            nb_cells = section.elementRange[1] - section.elementRange[0] + 1
            # assert nb_cells == section.elementStartOffset.shape[0] - 1

            params = Mesh_parameters()
            params.nb_vertices = wp_coords.shape[0]
            params.nb_cells = nb_cells

            # nb_faces is the sum of faces for each cell which depends on each cells type
            # nb_face_vtx_indices is sum of vertex count for all faces.
            if types.get_is_uniform(eType):
                params.nb_faces = params.nb_cells * types.get_face_count(eType)
                params.nb_face_vtx_indices = params.nb_cells * types.get_face_vertex_count(eType)
            else:
                assert eType == types.ElementType.MIXED
                wp_types = sids_unstructured.get_mixed_types(section)
                params.nb_faces = np.sum(types.get_face_counts()[wp_types.numpy()])
                params.nb_face_vtx_indices = np.sum(types.get_face_vertex_counts()[wp_types.numpy()])

            # since we won't be sharing faces, nb_cell_face_indices == nb_faces
            params.nb_cell_face_indices = params.nb_faces

            logger.info("\n%s", params)

            subset: IIrregular_volume_subset = self.subset
            storage: Mesh_storage = subset.generate_mesh_storage(params)
            assert storage is not None

            s_verts: np.ndarray = storage.get_vertices(params)
            assert (
                s_verts.dtype == np.float32
                and s_verts.ndim == 2
                and s_verts.shape[1] == 3
                and s_verts.shape[0] == params.nb_vertices
            )
            np.copyto(s_verts, wp_coords, casting="same_kind")
            # print(s_verts)

            s_cells: np.ndarray = storage.get_cells(params)
            assert s_cells.shape[0] == params.nb_cells
            if types.get_is_uniform(eType):
                s_cells[:, 0] = types.get_face_count(eType)
            else:
                wp.launch(
                    fill_counts_mixed,
                    dim=s_cells.shape[0],
                    inputs=[wp.array(s_cells, copy=False), wp.array(types.get_face_counts(), copy=False), section],
                )
            wp.launch(fill_start_index, dim=1, inputs=[wp.array(s_cells, copy=False)])
            # print("s_cells: ", s_cells)

            s_cell_face_indices: np.ndarray = storage.get_cell_face_indices(params)
            wp.launch(fill_tid, dim=s_cell_face_indices.shape[0], inputs=[wp.array(s_cell_face_indices, copy=False)])
            # print("s_cell_face_indices: ", s_cell_face_indices)

            s_faces: np.ndarray = storage.get_faces(params)
            assert s_faces.shape[0] == params.nb_faces

            if types.get_is_uniform(eType):
                # get vtx count for each face in each cell
                face_vtx_count = np.count_nonzero(types.get_faces(eType), axis=1)
                assert face_vtx_count.shape[0] == types.get_face_count(eType)
                assert types.get_face_count(eType) * params.nb_cells == s_faces.shape[0]
                wp.launch(
                    fill_faces_uniform,
                    dim=params.nb_cells,
                    inputs=[
                        wp.array(s_faces, copy=False),
                        wp.array(s_cells, copy=False),
                        wp.array(face_vtx_count, dtype=wp.uint32),
                    ],
                )
            else:
                face_map: np.ndarray = types.get_face_map()
                face_vtx_counts = np.zeros(shape=face_map.shape[:2], dtype=np.uint32)
                for i in range(face_map.shape[0]):
                    face_vtx_counts[i, :] = np.count_nonzero(face_map[i], axis=1)
                wp.launch(
                    fill_faces_mixed,
                    dim=params.nb_cells,
                    inputs=[
                        wp.array(s_faces, copy=False),
                        wp.array(s_cells, copy=False),
                        wp.array(face_vtx_counts, copy=False),
                        section,
                    ],
                )

            wp.launch(fill_start_index, dim=1, inputs=[wp.array(s_faces, copy=False)])
            # print(s_faces, s_faces.shape)

            # finally, need to fill the face vtk indices.
            # we'll populate with original indices and then remap

            s_face_vtx_indices: np.ndarray = storage.get_face_vtx_indices(params)
            if types.get_is_uniform(eType):
                wp.launch(
                    fill_face_vtx_indices_uniform,
                    dim=nb_cells,
                    inputs=[
                        wp.array(s_face_vtx_indices, copy=False),
                        wp.array(s_faces, copy=False),
                        wp.array(s_cells, copy=False),
                        wp.array(types.get_faces(eType), dtype=wp.int32),
                        types.get_vertex_count(eType),
                        section,
                    ],
                )
            else:
                wp.launch(
                    fill_face_vtx_indices_mixed,
                    dim=nb_cells,
                    inputs=[
                        wp.array(s_face_vtx_indices, copy=False),
                        wp.array(s_faces, copy=False),
                        wp.array(s_cells, copy=False),
                        wp.array(types.get_face_map(), dtype=wp.int32),
                        section,
                    ],
                )
            # print(s_face_vtx_indices)
            self.do_arrays(pointFieldArrays, cellFieldArrays, params, subset, section)
            wp.synchronize()

    def do_arrays(self, pointFieldArrays, cellFieldArrays, params, subset, section):
        arrays = [
            (Attribute_affiliation.ATTRIB_AFFIL_PER_VERTEX, params.nb_vertices, array) for array in pointFieldArrays
        ] + [(Attribute_affiliation.ATTRIB_AFFIL_PER_CELL, params.nb_cells, array) for array in cellFieldArrays]

        for idx, (affiliation, nb_attrib_values, fieldArray) in enumerate(arrays):
            attrib_params = Attribute_parameters()
            attrib_params.type = Attribute_type.ATTRIB_TYPE_FLOAT32  # FIXME
            attrib_params.affiliation = affiliation
            attrib_params.nb_attrib_values = nb_attrib_values

            attrib_storage: Attribute_storage = subset.generate_attribute_storage(idx, attrib_params)
            assert attrib_storage is not None

            s_attrib_values: np.ndarray = attrib_storage.get_attrib_values(attrib_params)
            if attrib_params.affiliation == Attribute_affiliation.ATTRIB_AFFIL_PER_VERTEX:
                np.copyto(s_attrib_values, fieldArray, casting="same_kind")
            else:
                np.copyto(
                    s_attrib_values,
                    fieldArray.numpy()[section.elementRange[0] - 1 : section.elementRange[1]],
                    casting="same_kind",
                )

    async def do_polyhedra(self):
        dataset = self.dataset
        eType = sids_unstructured.get_element_type(dataset)
        assert eType == types.ElementType.NFACE_n

        timeCode = self.timeCode
        cellFieldArrays = []
        pointFieldArrays = []
        for field in self.fields:
            field = usd_utils.get_target_prim(dataset, f"field:{field}")
            fieldAssociation = usd_utils.get_attribute(field, cae.Tokens.fieldAssociation)
            if fieldAssociation == cae.Tokens.cell:
                cellFieldArrays.append(await usd_utils.get_array(field, timeCode))
            elif fieldAssociation == cae.Tokens.vertex:
                pointFieldArrays.append(await usd_utils.get_array(field, timeCode))
            else:
                raise RuntimeError(f"Unsupported field array association {fieldAssociation}")

        with wp.ScopedDevice("cpu"):
            nface_section = await sids_unstructured.get_section(dataset, timeCode)
            ngon_sections = [
                await sids_unstructured.get_section(ngon, timeCode)
                for ngon in usd_utils.get_target_prims(dataset, sids.Tokens.caeSidsNgons)
            ]

            # sort by start element, we're assuming they are contiguous and non-overlapping
            ngon_sections = sorted(ngon_sections, key=lambda x: x.elementRange[0])

            # Currently, IndeX only supports triangular and quad faces. So give up if we have any faces > 4 verts.
            for ngon_section in ngon_sections:
                elementStartOffset = ngon_section.elementStartOffset.numpy()
                counts = elementStartOffset[1:] - elementStartOffset[:-1]
                indices = np.where(counts > 4)[0]
                if indices.shape[0] > 0:
                    raise RuntimeError(
                        "Polyhedra faces must be triangles or quads. Other polygonal shapes are not yet supported."
                    )

            # get the grid coordinates
            wp_coords = await sids_unstructured.get_original_grid_coordinates(dataset, timeCode)

            params = Mesh_parameters()
            params.nb_vertices = wp_coords.shape[0]
            params.nb_cells = nface_section.elementRange[1] - nface_section.elementRange[0] + 1
            # sum of all faces in all NGON_n blocks associated with this NFACE_n
            params.nb_faces = np.sum(
                [ngon_section.elementRange[1] - ngon_section.elementRange[0] + 1 for ngon_section in ngon_sections]
            )
            # sum of lengths of connectivity arrays for all related ngons
            params.nb_face_vtx_indices = np.sum(
                [ngon_section.elementConnectivity.shape[0] for ngon_section in ngon_sections]
            )
            # size of nface_n connectivity array
            params.nb_cell_face_indices = nface_section.elementConnectivity.shape[0]

            logger.info("Mesh parameters: %s", params)
            subset: IIrregular_volume_subset = self.subset
            storage: Mesh_storage = subset.generate_mesh_storage(params)
            assert storage is not None

            # now, copy over data
            s_verts: np.ndarray = storage.get_vertices(params)
            np.copyto(s_verts, wp_coords, casting="same_kind")

            # populate cells; this is simply the SIDS conn/offset - 1
            s_cells: np.ndarray = storage.get_cells(params)
            # 0: count aka, number of faces for that cell
            s_cells[:, 0] = nface_section.elementStartOffset.numpy()[1:] - nface_section.elementStartOffset.numpy()[:-1]
            # 1: 0-based start offset.
            s_cells[:, 1] = nface_section.elementStartOffset.numpy()[:-1] - nface_section.elementStartOffsetShift

            s_cell_face_indices: np.ndarray = storage.get_cell_face_indices(params)
            s_cell_face_indices[:] = nface_section.elementConnectivity.numpy() - 1

            # populate faces;
            s_faces: np.ndarray = storage.get_faces(params)
            s_face_vtx_indices: np.ndarray = storage.get_face_vtx_indices(params)
            face_vtx_index_offset = 0
            for ngon_section in ngon_sections:
                e_start = ngon_section.elementRange[0]
                e_end = ngon_section.elementRange[1]
                s_faces[e_start - 1 : e_end, 0] = (
                    ngon_section.elementStartOffset.numpy()[1:] - ngon_section.elementStartOffset.numpy()[:-1]
                )
                s_faces[e_start - 1 : e_end, 1] = (
                    ngon_section.elementStartOffset.numpy()[:-1]
                    - ngon_section.elementStartOffsetShift
                    + face_vtx_index_offset
                )
                s_face_vtx_indices[
                    face_vtx_index_offset : face_vtx_index_offset + ngon_section.elementConnectivity.shape[0]
                ] = (ngon_section.elementConnectivity.numpy() - 1)
                face_vtx_index_offset += ngon_section.elementConnectivity.shape[0]
            # print("s_faces", s_faces[s_faces[:, 0] > 4], np.amin(s_faces[:, 0]), np.amax(s_faces[:, 0]))

            # now, handle attribute arrays
            self.do_arrays(pointFieldArrays, cellFieldArrays, params, subset, nface_section)
            wp.synchronize()
