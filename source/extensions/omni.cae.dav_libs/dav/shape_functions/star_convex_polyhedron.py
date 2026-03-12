# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import warp as wp

import dav

from .typing import ShapeAPI, UniformShapesLibraryAPI


@dav.func
def get_standard_barycentric_coords(p: wp.vec3f, a: wp.vec3f, b: wp.vec3f, c: wp.vec3f, d: wp.vec3f) -> wp.vec4f:
    # Computes barycentric coords (u, v, w, x) for point p in tet abcd
    # Returns vec4(-1.0) if point is outside

    # Vectors
    vap = p - a
    # vbp = p - b

    # Compute sub-volumes (scalar triple products)
    # Vol_total
    vab = b - a
    vac = c - a
    vad = d - a
    vol_total = wp.dot(vab, wp.cross(vac, vad))

    if wp.abs(vol_total) < 1e-9:
        return wp.vec4f(-1.0, -1.0, -1.0, -1.0)  # Degenerate tet (skip; use 1e-9 to avoid skipping thin-but-valid tets)

    inv_vol = 1.0 / vol_total

    # Calculate volume of sub-tets formed by P and faces
    # W_d (Weight for d) is proportional to Vol(a,b,c,p)
    w_d = wp.dot(vab, wp.cross(vac, vap)) * inv_vol

    # W_c (Weight for c)
    w_c = wp.dot(vab, wp.cross(vap, vad)) * inv_vol

    # W_b (Weight for b)
    w_b = wp.dot(vap, wp.cross(vac, vad)) * inv_vol

    # W_a (Weight for a) = 1.0 - sum(others)
    w_a = 1.0 - (w_b + w_c + w_d)

    # Check if inside (all weights between 0 and 1, allowing for slight float error)
    eps = -1e-7
    if w_a > eps and w_b > eps and w_c > eps and w_d > eps:
        return wp.vec4f(w_a, w_b, w_c, w_d)

    return wp.vec4f(-1.0, -1.0, -1.0, -1.0)


def get_shape(data_model: dav.DataModel, shapes_library: UniformShapesLibraryAPI = None) -> ShapeAPI:
    """
    Get the API class for Polyhedron shape functions.
    This is intended to be used internally by a data model and hence accepts arguments
    that are part of the data model protocol but not a complete data model itself.
    """

    @dav.func
    def distribute_weights(ds: data_model.DatasetHandle, pos: wp.vec3f, cell: data_model.CellHandle, face_idx: int, pt_id_a: int, pt_id_b: int, tet_weights: wp.vec4f) -> wp.vec(
        length=dav.config.max_points_per_cell, dtype=wp.float32
    ):
        # --- DISTRIBUTE WEIGHTS ---

        # 1. Direct assignment
        # Add w_a to output for idx_a
        # Add w_b to output for idx_b

        # 2. Face Distribution
        # For every node k in this face:
        #   add (w_face / count_node) to output for node k

        # 3. Cell Distribution
        # For every node m in this cell:
        #   add (w_cell / num_cell_nodes) to output for node m

        weights = wp.vec(length=dav.config.max_points_per_cell, dtype=wp.float32)
        cell_point_ids = wp.vec(length=dav.config.max_points_per_cell, dtype=wp.int32)

        nb_cell_points = data_model.CellAPI.get_num_points(cell, ds)
        temp_id = wp.int32(0)
        cell_weight = tet_weights[0] / wp.float32(nb_cell_points)  # w_a distributed to all cell nodes
        for i in range(nb_cell_points):
            temp_id = data_model.CellAPI.get_point_id(cell, i, ds)
            if temp_id == pt_id_a:
                weights[i] += tet_weights[2]
            if temp_id == pt_id_b:
                weights[i] += tet_weights[3]
            weights[i] += cell_weight
            cell_point_ids[i] = temp_id

        nb_face_points = data_model.CellAPI.get_face_num_points(cell, face_idx, ds)
        face_weight = tet_weights[1] / wp.float32(nb_face_points)
        for i in range(nb_face_points):
            temp_id = data_model.CellAPI.get_face_point_id(cell, face_idx, i, ds)
            for j in range(nb_cell_points):
                if cell_point_ids[j] == temp_id:
                    weights[j] += face_weight
                    break

        return weights

    @dav.func
    def compute_shape_functions_from_position(ds: data_model.DatasetHandle, pos: wp.vec3f, cell: data_model.CellHandle) -> wp.vec(
        length=dav.config.max_points_per_cell, dtype=wp.float32
    ):
        assert data_model.CellAPI.is_valid(cell), "Invalid cell handle"

        found = wp.bool(False)
        cell_center = data_model.PolyhedralCellAPI.get_cell_center(cell, ds)

        nb_faces = data_model.CellAPI.get_num_faces(cell, ds)
        for face_idx in range(nb_faces):
            if found:
                break

            nb_face_points = data_model.CellAPI.get_face_num_points(cell, face_idx, ds)
            face_center = data_model.PolyhedralCellAPI.get_face_center(cell, face_idx, ds)

            for face_pt_idx in range(nb_face_points):
                face_pt_idx_next = (face_pt_idx + 1) % nb_face_points
                pt_id_a = data_model.CellAPI.get_face_point_id(cell, face_idx, face_pt_idx, ds)
                pt_id_b = data_model.CellAPI.get_face_point_id(cell, face_idx, face_pt_idx_next, ds)
                p_a = data_model.DatasetAPI.get_point(ds, pt_id_a)
                p_b = data_model.DatasetAPI.get_point(ds, pt_id_b)

                # Check Tet: (C_cell, C_face, A, B)
                # orientation matters. We want to ensure that the tetrahedron formed by (C_cell, C_face, A, B) is consistently oriented with an outward normal.
                tet_weights = get_standard_barycentric_coords(pos, cell_center, face_center, p_a, p_b)

                if tet_weights[0] == -1.0:
                    continue

                # FOUND IT!
                return distribute_weights(ds, pos, cell, face_idx, pt_id_a, pt_id_b, tet_weights)

        empty = wp.vec(length=dav.config.max_points_per_cell, dtype=wp.float32)
        return empty

    @dav.func
    def point_in_cell(ds: data_model.DatasetHandle, pos: wp.vec3f, cell: data_model.CellHandle) -> wp.bool:
        assert data_model.CellAPI.is_valid(cell), "Invalid cell handle"

        cell_center = data_model.PolyhedralCellAPI.get_cell_center(cell, ds)
        nb_faces = data_model.CellAPI.get_num_faces(cell, ds)
        for face_idx in range(nb_faces):
            nb_face_points = data_model.CellAPI.get_face_num_points(cell, face_idx, ds)
            face_center = data_model.PolyhedralCellAPI.get_face_center(cell, face_idx, ds)
            for face_pt_idx in range(nb_face_points):
                face_pt_idx_next = (face_pt_idx + 1) % nb_face_points
                pt_id_a = data_model.CellAPI.get_face_point_id(cell, face_idx, face_pt_idx, ds)
                pt_id_b = data_model.CellAPI.get_face_point_id(cell, face_idx, face_pt_idx_next, ds)
                p_a = data_model.DatasetAPI.get_point(ds, pt_id_a)
                p_b = data_model.DatasetAPI.get_point(ds, pt_id_b)

                # Check Tet: (C_cell, C_face, A, B)
                tet_weights = get_standard_barycentric_coords(pos, cell_center, face_center, p_a, p_b)

                if tet_weights[0] > -1.0:
                    return True

        return False

    class PolyhedronShape:
        """Implements the ShapeAPI protocol for Polyhedron shape functions."""

        @staticmethod
        @dav.func
        def is_point_in_cell(point: wp.vec3f, cell: data_model.CellHandle, dataset: data_model.DatasetHandle, cell_type: wp.int32) -> bool:
            return point_in_cell(dataset, point, cell)

        @staticmethod
        @dav.func
        def get_weights(point: wp.vec3f, cell: data_model.CellHandle, dataset: data_model.DatasetHandle, cell_type: wp.int32) -> wp.vec(
            length=dav.config.max_points_per_cell, dtype=wp.float32
        ):
            return compute_shape_functions_from_position(dataset, point, cell)

    return PolyhedronShape
