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
VTK Unstructured Grid Data Model
==================================

This module provides a data model implementation for VTK Unstructured Grid (vtkUnstructuredGrid).

Unstructured grids represent the most general mesh type with:
- Explicit point coordinates
- Arbitrary cell types (tetrahedra, hexahedra, pyramids, wedges, etc.)
- Explicit cell-to-point connectivity stored in arrays

Type System
-----------
- **Point IDs**: wp.int32 (0-based, contiguous)
- **Cell IDs**: wp.int32 (0-based, contiguous)
- **Indices**: wp.int32 (same as IDs for unstructured grids)

Key Features
------------
- Supports multiple cell types in same mesh
- Explicit topology stored in connectivity arrays
- BVH-based locators for efficient cell location
- Explicit cell links for point-to-cell queries
- Maximum 8 points per cell (configurable via MAX_CELL_POINTS)
"""

from typing import Any

import warp as wp
from dav import locators
from dav.shape_functions import hexa_8 as hex_shape
from dav.shape_functions import penta_6 as penta_shape
from dav.shape_functions import pyra_5 as pyramid_shape
from dav.shape_functions import tetra_4 as tet_shape
from dav.shape_functions import voxel_8 as voxel_shape

MAX_CELL_POINTS = 8  # maximum number of points per cell we will support

# VTK cell type constants
VTK_EMPTY_CELL = wp.constant(0)
VTK_VERTEX = wp.constant(1)
VTK_LINE = wp.constant(3)
VTK_POLY_LINE = wp.constant(4)
VTK_TRIANGLE = wp.constant(5)
VTK_QUAD = wp.constant(9)
VTK_TETRA = wp.constant(10)
VTK_HEXAHEDRON = wp.constant(12)
VTK_WEDGE = wp.constant(13)  # Pentahedron/Prism
VTK_PYRAMID = wp.constant(14)


@wp.struct
class DatasetHandle:
    points: wp.array(dtype=wp.vec3f)
    cell_types: wp.array(dtype=wp.int32)
    cell_offsets: wp.array(dtype=wp.int32)
    cell_connectivity: wp.array(dtype=wp.int32)
    cell_bvh_id: wp.uint64
    cell_links: locators.CellLinks
    hex_is_axis_aligned: bool


@wp.struct
class CellHandle:
    cell_id: wp.int32
    cell_type: wp.int32


@wp.struct
class InterpolatedCellHandle:
    cell_id: wp.int32
    cell_type: wp.int32
    inside: bool
    parametric_coords: wp.vec3f
    weights: wp.vec(length=MAX_CELL_POINTS, dtype=wp.float32)


@wp.struct
class CellLinkHandle:
    """Cell link for a given point.

    Simple structure that just stores the point_id.
    All operations work together with DatasetHandle.cell_links to query cells.
    """

    point_id: wp.int32


@wp.func
def _get_vertex_count_from_cell_type(cell_type: wp.int32) -> wp.int32:
    """Get number of vertices for a VTK cell type."""
    if cell_type == VTK_TETRA:
        return 4
    elif cell_type == VTK_PYRAMID:
        return 5
    elif cell_type == VTK_WEDGE:
        return 6
    elif cell_type == VTK_HEXAHEDRON:
        return 8
    elif cell_type == VTK_TRIANGLE:
        return 3
    elif cell_type == VTK_QUAD:
        return 4
    elif cell_type == VTK_LINE:
        return 2
    elif cell_type == VTK_VERTEX:
        return 1
    else:
        return 0


@wp.func
def _convert_weights_to_fixed_size(
    weights: Any, num_points: wp.int32
) -> wp.vec(length=MAX_CELL_POINTS, dtype=wp.float32):
    """Convert weights to fixed-size vector."""
    result = wp.vec(length=MAX_CELL_POINTS, dtype=wp.float32)
    for i in range(num_points):
        result[i] = weights[i]
    for i in range(num_points, MAX_CELL_POINTS):
        result[i] = 0.0
    return result


# Face connectivity helper functions for each cell type
# Using VTK ordering conventions


@wp.func
def _get_num_faces_for_cell_type(cell_type: wp.int32) -> wp.int32:
    """Get number of faces for a cell type."""
    if cell_type == VTK_TETRA:
        return 4
    elif cell_type == VTK_PYRAMID:
        return 5
    elif cell_type == VTK_WEDGE:
        return 5
    elif cell_type == VTK_HEXAHEDRON:
        return 6
    else:
        return 0


@wp.func
def _get_face_num_points_for_cell(cell_type: wp.int32, face_idx: wp.int32) -> wp.int32:
    """Get number of points in a specific face."""
    if cell_type == VTK_TETRA:
        return 3  # All faces are triangles
    elif cell_type == VTK_PYRAMID:
        if face_idx == 0:
            return 4  # Base is quad
        else:
            return 3  # Sides are triangles
    elif cell_type == VTK_WEDGE:
        if face_idx < 3:
            return 4  # Three quad faces
        else:
            return 3  # Two triangle faces
    elif cell_type == VTK_HEXAHEDRON:
        return 4  # All faces are quads
    else:
        return 0


@wp.func
def _get_face_point_local_idx_vtk(cell_type: wp.int32, face_idx: wp.int32, local_idx: wp.int32) -> wp.int32:
    """Get the local point index for a face point (VTK ordering)."""
    # Hexahedron faces (VTK ordering)
    if cell_type == VTK_HEXAHEDRON:
        if face_idx == 0:  # -X face
            if local_idx == 0:
                return 0
            elif local_idx == 1:
                return 4
            elif local_idx == 2:
                return 7
            else:
                return 3
        elif face_idx == 1:  # +X face
            if local_idx == 0:
                return 1
            elif local_idx == 1:
                return 2
            elif local_idx == 2:
                return 6
            else:
                return 5
        elif face_idx == 2:  # -Y face
            if local_idx == 0:
                return 0
            elif local_idx == 1:
                return 1
            elif local_idx == 2:
                return 5
            else:
                return 4
        elif face_idx == 3:  # +Y face
            if local_idx == 0:
                return 3
            elif local_idx == 1:
                return 7
            elif local_idx == 2:
                return 6
            else:
                return 2
        elif face_idx == 4:  # -Z face
            if local_idx == 0:
                return 0
            elif local_idx == 1:
                return 3
            elif local_idx == 2:
                return 2
            else:
                return 1
        else:  # face_idx == 5, +Z face
            if local_idx == 0:
                return 4
            elif local_idx == 1:
                return 5
            elif local_idx == 2:
                return 6
            else:
                return 7

    # Tetrahedron faces (all triangles)
    elif cell_type == VTK_TETRA:
        if face_idx == 0:
            if local_idx == 0:
                return 0
            elif local_idx == 1:
                return 1
            else:
                return 3
        elif face_idx == 1:
            if local_idx == 0:
                return 1
            elif local_idx == 1:
                return 2
            else:
                return 3
        elif face_idx == 2:
            if local_idx == 0:
                return 2
            elif local_idx == 1:
                return 0
            else:
                return 3
        else:  # face_idx == 3, base
            if local_idx == 0:
                return 0
            elif local_idx == 1:
                return 2
            else:
                return 1

    # Pyramid faces (1 quad base + 4 triangles)
    elif cell_type == VTK_PYRAMID:
        if face_idx == 0:  # Quad base
            if local_idx == 0:
                return 0
            elif local_idx == 1:
                return 3
            elif local_idx == 2:
                return 2
            else:
                return 1
        elif face_idx == 1:  # Triangle
            if local_idx == 0:
                return 0
            elif local_idx == 1:
                return 1
            else:
                return 4
        elif face_idx == 2:  # Triangle
            if local_idx == 0:
                return 1
            elif local_idx == 1:
                return 2
            else:
                return 4
        elif face_idx == 3:  # Triangle
            if local_idx == 0:
                return 2
            elif local_idx == 1:
                return 3
            else:
                return 4
        else:  # face_idx == 4, Triangle
            if local_idx == 0:
                return 3
            elif local_idx == 1:
                return 0
            else:
                return 4

    # Wedge faces (3 quads + 2 triangles) - VTK ordering
    elif cell_type == VTK_WEDGE:
        if face_idx == 0:  # Quad
            if local_idx == 0:
                return 0
            elif local_idx == 1:
                return 1
            elif local_idx == 2:
                return 4
            else:
                return 3
        elif face_idx == 1:  # Quad
            if local_idx == 0:
                return 1
            elif local_idx == 1:
                return 2
            elif local_idx == 2:
                return 5
            else:
                return 4
        elif face_idx == 2:  # Quad
            if local_idx == 0:
                return 2
            elif local_idx == 1:
                return 0
            elif local_idx == 2:
                return 3
            else:
                return 5
        elif face_idx == 3:  # Triangle (bottom)
            if local_idx == 0:
                return 0
            elif local_idx == 1:
                return 2
            else:
                return 1
        else:  # face_idx == 4, Triangle (top)
            if local_idx == 0:
                return 3
            elif local_idx == 1:
                return 4
            else:
                return 5

    return -1


@wp.func
def _evaluate_tetra_cell(ds: DatasetHandle, cell: CellHandle, position: wp.vec3f) -> InterpolatedCellHandle:
    """Evaluate position for tetrahedral cell."""
    i_cell = InterpolatedCellHandle()
    i_cell.cell_id = cell.cell_id
    i_cell.cell_type = cell.cell_type

    start_offset = ds.cell_offsets[cell.cell_id]

    # Extract 4 points
    p0 = ds.points[ds.cell_connectivity[start_offset + 0]]
    p1 = ds.points[ds.cell_connectivity[start_offset + 1]]
    p2 = ds.points[ds.cell_connectivity[start_offset + 2]]
    p3 = ds.points[ds.cell_connectivity[start_offset + 3]]

    # Compute parametric coordinates
    i_cell.parametric_coords = tet_shape.compute_parametric_coordinates(p0, p1, p2, p3, position)

    # Check if inside
    i_cell.inside = tet_shape.is_inside(i_cell.parametric_coords)

    # Compute weights
    i_cell.weights = _convert_weights_to_fixed_size(tet_shape.compute_shape_functions(i_cell.parametric_coords), 4)

    return i_cell


@wp.func
def _evaluate_hexahedron_cell(ds: DatasetHandle, cell: CellHandle, position: wp.vec3f) -> InterpolatedCellHandle:
    """Evaluate position for hexahedral cell."""
    i_cell = InterpolatedCellHandle()
    i_cell.cell_id = cell.cell_id
    i_cell.cell_type = cell.cell_type

    start_offset = ds.cell_offsets[cell.cell_id]

    # Extract 8 points
    p0 = ds.points[ds.cell_connectivity[start_offset + 0]]
    p1 = ds.points[ds.cell_connectivity[start_offset + 1]]
    p2 = ds.points[ds.cell_connectivity[start_offset + 2]]
    p3 = ds.points[ds.cell_connectivity[start_offset + 3]]
    p4 = ds.points[ds.cell_connectivity[start_offset + 4]]
    p5 = ds.points[ds.cell_connectivity[start_offset + 5]]
    p6 = ds.points[ds.cell_connectivity[start_offset + 6]]
    p7 = ds.points[ds.cell_connectivity[start_offset + 7]]

    # Compute parametric coordinates
    i_cell.parametric_coords = hex_shape.compute_parametric_coordinates(p0, p1, p2, p3, p4, p5, p6, p7, position)

    # Check if inside
    i_cell.inside = hex_shape.is_inside(i_cell.parametric_coords)

    # Compute weights
    i_cell.weights = _convert_weights_to_fixed_size(hex_shape.compute_shape_functions(i_cell.parametric_coords), 8)

    return i_cell


@wp.func
def _evaluate_voxel_cell(ds: DatasetHandle, cell: CellHandle, position: wp.vec3f) -> InterpolatedCellHandle:
    """Evaluate position for voxel cell (axis-aligned hexahedron)."""
    i_cell = InterpolatedCellHandle()
    i_cell.cell_id = cell.cell_id
    i_cell.cell_type = cell.cell_type

    start_offset = ds.cell_offsets[cell.cell_id]

    # Extract 8 points
    p0 = ds.points[ds.cell_connectivity[start_offset + 0]]
    p1 = ds.points[ds.cell_connectivity[start_offset + 1]]
    p2 = ds.points[ds.cell_connectivity[start_offset + 2]]
    p3 = ds.points[ds.cell_connectivity[start_offset + 3]]
    p4 = ds.points[ds.cell_connectivity[start_offset + 4]]
    p5 = ds.points[ds.cell_connectivity[start_offset + 5]]
    p6 = ds.points[ds.cell_connectivity[start_offset + 6]]
    p7 = ds.points[ds.cell_connectivity[start_offset + 7]]

    # Compute parametric coordinates using axis-aligned optimization
    i_cell.parametric_coords = voxel_shape.compute_parametric_coordinates(p0, p1, p2, p3, p4, p5, p6, p7, position)

    # Check if inside
    i_cell.inside = voxel_shape.is_inside(i_cell.parametric_coords)

    # Compute weights
    i_cell.weights = _convert_weights_to_fixed_size(voxel_shape.compute_shape_functions(i_cell.parametric_coords), 8)

    return i_cell


@wp.func
def _evaluate_pyramid_cell(ds: DatasetHandle, cell: CellHandle, position: wp.vec3f) -> InterpolatedCellHandle:
    """Evaluate position for pyramid cell."""
    i_cell = InterpolatedCellHandle()
    i_cell.cell_id = cell.cell_id
    i_cell.cell_type = cell.cell_type

    start_offset = ds.cell_offsets[cell.cell_id]

    # Extract 5 points
    p0 = ds.points[ds.cell_connectivity[start_offset + 0]]
    p1 = ds.points[ds.cell_connectivity[start_offset + 1]]
    p2 = ds.points[ds.cell_connectivity[start_offset + 2]]
    p3 = ds.points[ds.cell_connectivity[start_offset + 3]]
    p4 = ds.points[ds.cell_connectivity[start_offset + 4]]

    # Compute parametric coordinates
    i_cell.parametric_coords = pyramid_shape.compute_parametric_coordinates(p0, p1, p2, p3, p4, position)

    # Check if inside
    i_cell.inside = pyramid_shape.is_inside(i_cell.parametric_coords)

    # Compute weights
    i_cell.weights = _convert_weights_to_fixed_size(pyramid_shape.compute_shape_functions(i_cell.parametric_coords), 5)

    return i_cell


@wp.func
def _evaluate_wedge_cell(ds: DatasetHandle, cell: CellHandle, position: wp.vec3f) -> InterpolatedCellHandle:
    """Evaluate position for wedge (pentahedron/prism) cell."""
    i_cell = InterpolatedCellHandle()
    i_cell.cell_id = cell.cell_id
    i_cell.cell_type = cell.cell_type

    start_offset = ds.cell_offsets[cell.cell_id]

    # Extract 6 points
    p0 = ds.points[ds.cell_connectivity[start_offset + 0]]
    p1 = ds.points[ds.cell_connectivity[start_offset + 1]]
    p2 = ds.points[ds.cell_connectivity[start_offset + 2]]
    p3 = ds.points[ds.cell_connectivity[start_offset + 3]]
    p4 = ds.points[ds.cell_connectivity[start_offset + 4]]
    p5 = ds.points[ds.cell_connectivity[start_offset + 5]]

    # Compute parametric coordinates
    i_cell.parametric_coords = penta_shape.compute_parametric_coordinates(p0, p1, p2, p3, p4, p5, position)

    # Check if inside
    i_cell.inside = penta_shape.is_inside(i_cell.parametric_coords)

    # Compute weights
    i_cell.weights = _convert_weights_to_fixed_size(penta_shape.compute_shape_functions(i_cell.parametric_coords), 6)

    return i_cell


class CellAPI:
    @staticmethod
    @wp.func
    def is_valid(cell: CellHandle) -> wp.bool:
        return cell.cell_id >= 0

    @staticmethod
    @wp.func
    def empty() -> CellHandle:
        cell = CellHandle()
        cell.cell_id = -1
        cell.cell_type = VTK_EMPTY_CELL
        return cell

    @staticmethod
    @wp.func
    def get_cell_id(cell: CellHandle) -> wp.int32:
        return cell.cell_id

    @staticmethod
    @wp.func
    def get_num_points(cell: CellHandle, ds: DatasetHandle) -> wp.int32:
        # VTK uses cell_offsets to determine the start and end of connectivity for a cell
        start_offset = ds.cell_offsets[cell.cell_id]
        end_offset = ds.cell_offsets[cell.cell_id + 1]
        return end_offset - start_offset

    @staticmethod
    @wp.func
    def get_point_id(cell: CellHandle, local_idx: wp.int32, ds: DatasetHandle) -> wp.int32:
        start_offset = ds.cell_offsets[cell.cell_id]
        return ds.cell_connectivity[start_offset + local_idx]

    @staticmethod
    @wp.func
    def get_num_faces(cell: CellHandle, ds: DatasetHandle) -> wp.int32:
        """Get the number of faces in a cell."""
        return _get_num_faces_for_cell_type(cell.cell_type)

    @staticmethod
    @wp.func
    def get_face_num_points(cell: CellHandle, face_idx: wp.int32, ds: DatasetHandle) -> wp.int32:
        """Get the number of points in a face."""
        return _get_face_num_points_for_cell(cell.cell_type, face_idx)

    @staticmethod
    @wp.func
    def get_face_point_id(cell: CellHandle, face_idx: wp.int32, local_idx: wp.int32, ds: DatasetHandle) -> wp.int32:
        """Get a point ID from a face."""
        # Get the local point index within the cell
        cell_local_idx = _get_face_point_local_idx_vtk(cell.cell_type, face_idx, local_idx)
        if cell_local_idx < 0:
            return -1
        # Convert to global point ID
        return CellAPI.get_point_id(cell, cell_local_idx, ds)


class InterpolatedCellAPI:
    @staticmethod
    @wp.func
    def empty() -> InterpolatedCellHandle:
        i_cell = InterpolatedCellHandle()
        i_cell.cell_id = -1
        i_cell.cell_type = VTK_EMPTY_CELL
        i_cell.inside = False
        i_cell.parametric_coords = wp.vec3f(0.0, 0.0, 0.0)
        return i_cell

    @staticmethod
    @wp.func
    def is_valid(i_cell: InterpolatedCellHandle) -> wp.bool:
        return i_cell.cell_id >= 0

    @staticmethod
    @wp.func
    def get_cell_id(i_cell: InterpolatedCellHandle) -> wp.int32:
        return i_cell.cell_id

    @staticmethod
    @wp.func
    def is_inside(i_cell: InterpolatedCellHandle) -> wp.bool:
        return i_cell.inside

    @staticmethod
    @wp.func
    def get_weight(i_cell: InterpolatedCellHandle, local_idx: wp.int32) -> wp.float32:
        return i_cell.weights[local_idx]


class CellLinksAPI:
    """Operations on cell links.

    Works together with DatasetHandle.cell_links to query which cells use a given point.
    """

    @staticmethod
    @wp.func
    def empty() -> CellLinkHandle:
        """Create an empty cell link."""
        cell_link = CellLinkHandle()
        cell_link.point_id = -1
        return cell_link

    @staticmethod
    @wp.func
    def is_valid(cell_link: CellLinkHandle) -> wp.bool:
        """Check if cell link is valid.

        Valid if point_id is non-negative (VTK uses 0-based indexing).
        """
        return cell_link.point_id >= 0

    @staticmethod
    @wp.func
    def get_point_id(cell_link: CellLinkHandle) -> wp.int32:
        """Get the point id for a given cell link."""
        return cell_link.point_id

    @staticmethod
    @wp.func
    def get_num_cells(cell_link: CellLinkHandle, ds: DatasetHandle) -> wp.int32:
        """Get the number of cells that use this point.

        Uses DatasetHandle.cell_links to look up the cell count.
        """
        if not CellLinksAPI.is_valid(cell_link):
            return 0

        point_idx = DatasetAPI.get_point_idx_from_id(ds, cell_link.point_id)

        # Get the range from offsets
        start = ds.cell_links.offsets[point_idx]
        end = ds.cell_links.offsets[point_idx + 1]

        return end - start

    @staticmethod
    @wp.func
    def get_cell_id(cell_link: CellLinkHandle, cell_idx: wp.int32, ds: DatasetHandle) -> wp.int32:
        """Get the cell id for a given cell index in the cell link.

        Args:
            cell_link: The cell link to query
            cell_idx: Local index within the cells using this point (0-based)
            ds: The dataset containing cell links

        Returns:
            The cell id at the given local index, or -1 if invalid
        """
        if not CellLinksAPI.is_valid(cell_link):
            return -1

        point_idx = DatasetAPI.get_point_idx_from_id(ds, cell_link.point_id)

        # Get the range from offsets
        start = ds.cell_links.offsets[point_idx]
        end = ds.cell_links.offsets[point_idx + 1]

        # Check bounds
        if cell_idx < 0 or cell_idx >= (end - start):
            return -1

        # Return the cell id from the flat array
        return ds.cell_links.cell_ids[start + cell_idx]


class DatasetAPI:
    @staticmethod
    @wp.func
    def get_cell_id_from_idx(ds: DatasetHandle, cell_idx: wp.int32) -> wp.int32:
        return cell_idx

    @staticmethod
    @wp.func
    def get_cell_idx_from_id(ds: DatasetHandle, cell_id: wp.int32) -> wp.int32:
        return cell_id

    @staticmethod
    @wp.func
    def get_cell(ds: DatasetHandle, cell_id: wp.int32) -> CellHandle:
        cell = CellHandle()
        cell.cell_id = cell_id
        cell.cell_type = ds.cell_types[cell_id]
        return cell

    @staticmethod
    @wp.func
    def get_num_cells(ds: DatasetHandle) -> wp.int32:
        return ds.cell_types.shape[0]

    @staticmethod
    @wp.func
    def get_num_points(ds: DatasetHandle) -> wp.int32:
        return ds.points.shape[0]

    @staticmethod
    @wp.func
    def get_point_id_from_idx(ds: DatasetHandle, point_idx: wp.int32) -> wp.int32:
        return point_idx

    @staticmethod
    @wp.func
    def get_point_idx_from_id(ds: DatasetHandle, point_id: wp.int32) -> wp.int32:
        return point_id

    @staticmethod
    @wp.func
    def get_point(ds: DatasetHandle, point_id: wp.int32) -> wp.vec3f:
        return ds.points[point_id]

    @staticmethod
    @wp.func
    def get_cell_link(ds: DatasetHandle, point_id: wp.int32) -> CellLinkHandle:
        """Get the cell link for a given point id.

        Returns a CellLinkHandle that can be queried with CellLinksAPI
        to find which cells use this point.

        Args:
            ds: The dataset to query
            point_id: The point id to get cells for (0-based for VTK)

        Returns:
            CellLinkHandle containing the point_id
        """
        cell_link = CellLinkHandle()
        cell_link.point_id = point_id
        return cell_link

    @staticmethod
    @wp.func
    def get_field_id_from_idx(dataset: DatasetHandle, local_idx: wp.int32) -> wp.int32:
        return local_idx

    @staticmethod
    @wp.func
    def get_field_idx_from_id(dataset: DatasetHandle, id: wp.int32) -> wp.int32:
        return id

    @staticmethod
    @wp.func
    def evaluate_position(ds: DatasetHandle, position: wp.vec3f, cell: CellHandle) -> InterpolatedCellHandle:
        # Check if cell is valid
        if not CellAPI.is_valid(cell):
            return InterpolatedCellAPI.empty()

        # Dispatch to appropriate shape function based on VTK cell type
        if cell.cell_type == VTK_TETRA:
            return _evaluate_tetra_cell(ds, cell, position)
        elif cell.cell_type == VTK_HEXAHEDRON:
            return (
                _evaluate_voxel_cell(ds, cell, position)
                if ds.hex_is_axis_aligned
                else _evaluate_hexahedron_cell(ds, cell, position)
            )
        elif cell.cell_type == VTK_PYRAMID:
            return _evaluate_pyramid_cell(ds, cell, position)
        elif cell.cell_type == VTK_WEDGE:
            return _evaluate_wedge_cell(ds, cell, position)
        else:
            # Unsupported cell type
            return InterpolatedCellAPI.empty()

    @staticmethod
    @wp.func
    def find_cell_containing_point(ds: DatasetHandle, position: wp.vec3f, hint: CellHandle) -> CellHandle:
        if ds.cell_bvh_id == 0:
            wp.printf("ERROR: Cell locator not built for dataset\n")
            return CellAPI.empty()

        if CellAPI.is_valid(hint):
            i_cell = DatasetAPI.evaluate_position(ds, position, hint)
            if i_cell.inside:
                return hint

        radius = wp.vec3f(1.0e-6, 1.0e-6, 1.0e-6)
        query = wp.bvh_query_aabb(ds.cell_bvh_id, position - radius, position + radius)
        cell_idx = wp.int32(-1)
        while wp.bvh_query_next(query, cell_idx):
            cell_id = DatasetAPI.get_cell_id_from_idx(ds, cell_idx)
            cell = DatasetAPI.get_cell(ds, cell_id)
            i_cell = DatasetAPI.evaluate_position(ds, position, cell)
            if i_cell.inside:
                return cell
        return CellAPI.empty()

    @staticmethod
    def build_cell_locator(data_model, ds: DatasetHandle, device=None) -> tuple[bool, Any]:
        """Build a spatial acceleration structure for cell location queries.

        Args:
            data_model: The data model module (should be 'unstructured_grid')
            ds: The dataset
            device: Device to build the locator on

        Returns:
            tuple: (success, locator) - Success flag and locator instance
        """
        locator = locators.build_cell_locator(data_model, ds, device)
        if locator is not None:
            ds.cell_bvh_id = locator.get_bvh_id()
            return (True, locator)
        else:
            ds.cell_bvh_id = 0
            return (False, None)

    @staticmethod
    def build_cell_links(data_model, ds: DatasetHandle, device=None) -> tuple[bool, Any]:
        """Build the cell links for the dataset.

        Args:
            data_model: The data model module
            ds: The dataset
            device: Device to build the links on

        Returns:
            tuple: (success, links) - Success flag and CellLinks instance
        """
        cell_links = locators.build_cell_links(data_model, ds, device)
        if cell_links is not None:
            ds.cell_links = cell_links
            return (True, cell_links)
        else:
            ds.cell_links = None
            return (False, None)


# DataModel protocol implementation
class DataModel:
    """VTK Unstructured Grid data model implementation."""

    # Handle types
    DatasetHandle = DatasetHandle
    CellHandle = CellHandle
    InterpolatedCellHandle = InterpolatedCellHandle
    CellLinkHandle = CellLinkHandle
    PointIdHandle = wp.int32
    CellIdHandle = wp.int32

    # API types
    DatasetAPI = DatasetAPI
    CellAPI = CellAPI
    InterpolatedCellAPI = InterpolatedCellAPI
    CellLinksAPI = CellLinksAPI
