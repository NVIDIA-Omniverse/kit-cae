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

from .typing import UniformShapesLibraryAPI


@dav.cached
def get_compute_face_centers_kernel(data_model: dav.DataModel):
    """
    Factory function to create a function that computes face centers for polyhedral cells.
    """

    @dav.kernel(module="unique")
    @dav.utils.set_qualname("dav_sfu_compute_face_centers_kernel")
    def compute_face_centers_kernel(ds: data_model.DatasetHandle, cell_face_offsets: wp.array(dtype=wp.int32), out_face_centers: wp.array(dtype=wp.vec3f)):
        """
        Compute face centers for a given polyhedral cell.

        For each face of the cell, compute the center as the average of its vertices.
        """

        cell_idx = wp.tid()
        cell_id = data_model.DatasetAPI.get_cell_id_from_idx(ds, cell_idx)
        cell = data_model.DatasetAPI.get_cell(ds, cell_id)
        assert data_model.CellAPI.is_valid(cell), "Invalid cell handle"
        assert (cell_idx + 1) < cell_face_offsets.shape[0], "Offsets array must have at least num_cells + 1 entries"

        # we use cell_face_offsets to get the face count rather than using CellAPI. This
        # enables us to skip non-polyhedral cells for use-cases like VTK's.
        oid_offset = cell_face_offsets[cell_idx]
        nb_faces = cell_face_offsets[cell_idx + 1] - oid_offset
        if nb_faces == 0:
            return

        for face_idx in range(nb_faces):
            nb_face_points = data_model.CellAPI.get_face_num_points(cell, face_idx, ds)
            assert nb_face_points > 0, "Face must have at least one point"

            # Compute face center.
            face_center = wp.vec3f(0.0)
            for idx in range(nb_face_points):
                pid = data_model.CellAPI.get_face_point_id(cell, face_idx, idx, ds)
                pos = data_model.DatasetAPI.get_point(ds, pid)

                # accumulate center.
                face_center += pos

            out_face_centers[oid_offset + face_idx] = face_center / wp.float32(nb_face_points)

    return compute_face_centers_kernel


@dav.cached
def get_compute_cell_centers_kernel(data_model: dav.DataModel):
    """
    Factory function to create a kernel that computes cell centers for any cell.

    Args:
        data_model: Data model containing DatasetAPI, CellAPI, and DatasetHandle
    """

    @dav.kernel(module="unique")
    @dav.utils.set_qualname("dav_sfu_compute_cell_centers_kernel")
    def compute_cell_centers_kernel(ds: data_model.DatasetHandle, out_cell_centers: wp.array(dtype=wp.vec3f)):
        cell_idx = wp.tid()
        cell_id = data_model.DatasetAPI.get_cell_id_from_idx(ds, cell_idx)
        cell = data_model.DatasetAPI.get_cell(ds, cell_id)
        assert data_model.CellAPI.is_valid(cell), "Invalid cell handle"
        assert cell_idx < out_cell_centers.shape[0], "Output array must have at least num_cells entries"

        num_points = data_model.CellAPI.get_num_points(cell, ds)
        assert num_points > 0, "Cell must have at least one point"

        cell_center = wp.vec3f(0.0)
        for idx in range(num_points):
            point_id = data_model.CellAPI.get_point_id(cell, idx, ds)
            point_pos = data_model.DatasetAPI.get_point(ds, point_id)
            cell_center += point_pos
        out_cell_centers[cell_idx] = cell_center / wp.float32(num_points)

    return compute_cell_centers_kernel


@dav.cached
def get_compute_face_orientations_kernel(data_model: dav.DataModel):
    """
    Factory function to create a kernel that populates face orientations for polyhedral cells.

    For each face of a polyhedral cell, determine if the order of the face connectivity
    is such that the normal points outward from the cell. If so, set orientation to 1,
    otherwise set it to -1.
    """

    @dav.kernel(module="unique")
    @dav.utils.set_qualname("dav_sfu_compute_face_orientations_kernel")
    def compute_face_orientations_kernel(
        ds: data_model.DatasetHandle,
        cell_centers: wp.array(dtype=wp.vec3f),
        cell_face_offsets: wp.array(dtype=wp.int32),
        cell_face_centers: wp.array(dtype=wp.vec3f),
        cell_face_orientations: wp.array(dtype=wp.int8),
    ):
        cell_idx = wp.tid()
        cell_id = data_model.DatasetAPI.get_cell_id_from_idx(ds, cell_idx)
        cell = data_model.DatasetAPI.get_cell(ds, cell_id)
        assert data_model.CellAPI.is_valid(cell), "Invalid cell handle"
        assert (cell_idx + 1) < cell_face_offsets.shape[0], "Offsets array must have at least num_cells + 1 entries"

        oid_offset = cell_face_offsets[cell_idx]
        nb_faces = cell_face_offsets[cell_idx + 1] - oid_offset
        if nb_faces == 0:
            return

        cell_center = cell_centers[cell_idx]
        for face_idx in range(nb_faces):
            nb_face_points = data_model.CellAPI.get_face_num_points(cell, face_idx, ds)
            assert nb_face_points >= 3, "Face must have at least 3 points to define a plane"

            # Get first three points to compute face normal
            pt_id = data_model.CellAPI.get_face_point_id(cell, face_idx, 0, ds)
            p0 = data_model.DatasetAPI.get_point(ds, pt_id)
            pt_id = data_model.CellAPI.get_face_point_id(cell, face_idx, 1, ds)
            p1 = data_model.DatasetAPI.get_point(ds, pt_id)
            pt_id = data_model.CellAPI.get_face_point_id(cell, face_idx, 2, ds)
            p2 = data_model.DatasetAPI.get_point(ds, pt_id)

            # Compute outward-facing normal
            v1 = p1 - p0
            v2 = p2 - p0
            face_normal = wp.cross(v1, v2)

            # Determine orientation by checking the dot product of the face normal and the vector from the cell center to the face center
            to_face_center = cell_face_centers[oid_offset + face_idx] - cell_center
            cell_face_orientations[oid_offset + face_idx] = wp.int8(1) if wp.dot(face_normal, to_face_center) > 0 else wp.int8(-1)

    return compute_face_orientations_kernel


def build_shape_functions_library(shapes: list[dict]) -> UniformShapesLibraryAPI:
    """
    Build a shape functions API from a list of shape definitions.

    Constructs topology information for the specified cell types,
    including node connectivity, face topology, and shape function mappings.
    Returns a class with static methods for querying shape information.

    The first cell type in the shapes list is used as the default/fallback cell type.

    Args:
        shapes: List of shape dictionaries, each containing:
            - "cell_type" (int): Data model specific identifier for the cell type.
              This is used to map from cell types to shape information.
            - "shape_function_type" (int): Shape function element type constant
            - "element_node_ids" (list[int]): List of corner node indices (0-based ordering)
            - "element_node_ids_vtk" (optional, list[int]): List of corner node indices in VTK ordering (0-based).
                   If not provided, "element_node_ids" will be used for both.
            - "element_mid_ids" (list[int]): List of midpoint node indices (empty for linear elements)
            - "element_face_node_ids" (list[list[int]]): List of faces, each face is a list of
              node indices. All faces MUST be specified with outward facing normals (right-hand rule).
              The ordering should match the shape function ordering expected by the data model.
            The first shape in the list will be used as the default cell type.

    Returns:
        A class following the UniformShapesLibraryAPI protocol, with static methods for querying
        shape information based on cell type.

    Example:
        >>> shapes = [
        ...     {
        ...         "cell_type": 0,  # Empty cell type identifier (will be used as default)
        ...         "shape_function_type": 0,
        ...         "element_node_ids": [],
        ...         "element_mid_ids": [],
        ...         "element_face_node_ids": [],
        ...     },
        ...     {
        ...         "cell_type": 12,  # Hexahedron cell type identifier
        ...         "shape_function_type": 1,
        ...         "element_node_ids": [0, 1, 2, 3, 4, 5, 6, 7],
        ...         "element_mid_ids": [],
        ...         "element_face_node_ids": [[0, 4, 7, 3], [1, 2, 6, 5], [0, 1, 5, 4], [3, 7, 6, 2], [0, 3, 2, 1], [4, 5, 6, 7]],
        ...     },
        ... ]
        >>> api = get_shape_functions_api(shapes)
    """

    # Prepare flattened arrays
    shape_function_types = []
    node_indices = []
    node_indices_vtk = []
    face_node_indices = []
    midpoint_counts = []
    node_indices_counts = []
    face_node_counts = []
    element_face_counts = []

    for i, shape in enumerate(shapes):
        shape_function_types.append(shape["shape_function_type"])
        midpoint_counts.append(len(shape["element_mid_ids"]))

        # Store corner nodes only (0-based)
        if "element_node_ids_vtk" in shape:
            node_indices_vtk.extend(shape["element_node_ids_vtk"])
        else:
            node_indices_vtk.extend(shape["element_node_ids"])

        node_indices.extend(shape["element_node_ids"])
        node_indices_counts.append(len(shape["element_node_ids"]))

        # Faces
        for face in shape["element_face_node_ids"]:
            face_node_indices.extend(face)
            face_node_counts.append(len(face))

        element_face_counts.append(len(shape["element_face_node_ids"]))

    # Convert counts to offsets
    node_indices_offsets = [0]
    for count in node_indices_counts:
        node_indices_offsets.append(node_indices_offsets[-1] + count)

    face_node_offsets = [0]
    for count in face_node_counts:
        face_node_offsets.append(face_node_offsets[-1] + count)

    element_face_offsets = [0]
    for count in element_face_counts:
        element_face_offsets.append(element_face_offsets[-1] + count)

    # Define constants
    nb_shapes = wp.constant(wp.int32(len(shapes)))
    vtk_cell_types = wp.constant(wp.vec(length=len(shapes), dtype=wp.int32)(*[shape["cell_type"] for shape in shapes]))
    shape_function_types_const = wp.constant(wp.vec(length=len(shapes), dtype=wp.int32)(*shape_function_types))
    node_indices_offsets_const = wp.constant(wp.vec(length=len(node_indices_offsets), dtype=wp.int32)(*node_indices_offsets))
    node_indices_vtk_const = wp.constant(wp.vec(length=len(node_indices_vtk), dtype=wp.int32)(*node_indices_vtk))
    node_indices_const = wp.constant(wp.vec(length=len(node_indices), dtype=wp.int32)(*node_indices))
    face_node_offsets_const = wp.constant(wp.vec(length=len(face_node_offsets), dtype=wp.int32)(*face_node_offsets))
    face_node_indices_const = wp.constant(wp.vec(length=len(face_node_indices), dtype=wp.int32)(*face_node_indices))
    element_face_offsets_const = wp.constant(wp.vec(length=len(element_face_offsets), dtype=wp.int32)(*element_face_offsets))
    midpoint_counts_const = wp.constant(wp.vec(length=len(midpoint_counts), dtype=wp.int32)(*midpoint_counts))

    # Standalone helper defined before ShapesLibrary so that methods can call it without
    # referencing ShapesLibrary mid-class-body (which would leave an empty closure cell and
    # cause Warp to set has_unresolved_static_expressions=True).
    @dav.func
    def _get_index_from_cell_type(cell_type: wp.int32) -> wp.int32:
        for i in range(nb_shapes):
            if vtk_cell_types[i] == cell_type:
                return i
        return 0  # always first cell type

    class ShapesLibrary:
        @staticmethod
        @dav.func
        def get_num_shapes() -> wp.int32:
            return nb_shapes

        @staticmethod
        @dav.func
        def get_index_from_cell_type(cell_type: wp.int32) -> wp.int32:
            return _get_index_from_cell_type(cell_type)

        @staticmethod
        @dav.func
        def get_shape_function_type(cell_type: wp.int32) -> wp.int32:
            index = _get_index_from_cell_type(cell_type)
            return shape_function_types_const[index]

        @staticmethod
        @dav.func
        def get_num_all_nodes(cell_type: wp.int32) -> wp.int32:
            index = _get_index_from_cell_type(cell_type)
            return node_indices_offsets_const[index + 1] - node_indices_offsets_const[index] + midpoint_counts_const[index]

        @staticmethod
        @dav.func
        def get_num_corner_nodes(cell_type: wp.int32) -> wp.int32:
            index = _get_index_from_cell_type(cell_type)
            return node_indices_offsets_const[index + 1] - node_indices_offsets_const[index]

        @staticmethod
        @dav.func
        def get_vtk_corner_node_index(cell_type: wp.int32, node_idx: wp.int32) -> wp.int32:
            index = _get_index_from_cell_type(cell_type)
            return node_indices_vtk_const[node_indices_offsets_const[index] + node_idx]

        @staticmethod
        @dav.func
        def get_corner_node_index(cell_type: wp.int32, node_idx: wp.int32) -> wp.int32:
            index = _get_index_from_cell_type(cell_type)
            return node_indices_const[node_indices_offsets_const[index] + node_idx]

        @staticmethod
        @dav.func
        def get_num_faces(cell_type: wp.int32) -> wp.int32:
            index = _get_index_from_cell_type(cell_type)
            return element_face_offsets_const[index + 1] - element_face_offsets_const[index]

        @staticmethod
        @dav.func
        def get_num_face_corner_nodes(cell_type: wp.int32, face_idx: wp.int32) -> wp.int32:
            index = _get_index_from_cell_type(cell_type)
            face_offset_base = element_face_offsets_const[index]
            return face_node_offsets_const[face_offset_base + face_idx + 1] - face_node_offsets_const[face_offset_base + face_idx]

        @staticmethod
        @dav.func
        def get_face_corner_node_index(cell_type: wp.int32, face_idx: wp.int32, node_idx: wp.int32) -> wp.int32:
            index = _get_index_from_cell_type(cell_type)
            face_offset_base = element_face_offsets_const[index]
            start_idx = face_node_offsets_const[face_offset_base + face_idx]
            return face_node_indices_const[start_idx + node_idx]

    return ShapesLibrary
