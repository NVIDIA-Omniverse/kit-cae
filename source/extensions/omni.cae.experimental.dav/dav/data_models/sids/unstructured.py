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
SIDS Unstructured Grid Data Model
===================================

This module provides a data model implementation for CGNS/SIDS Unstructured Grid.

CGNS (CFD General Notation System) SIDS (Standard Interface Data Structures) represents
computational meshes with:
- Explicit point coordinates
- Element sections with connectivity arrays
- Support for multiple element types (TETRA, HEXA, PYRA, PENTA, etc.)

Type System
-----------
- **Point IDs**: wp.int32 (0-based internally, CGNS convention)
- **Cell IDs**: wp.int32 (0-based internally)
- **Indices**: wp.int32 (0-based)

IMPORTANT - Node Ordering Conventions:
======================================
CGNS/SIDS and VTK use different node ordering conventions for some element types.
Our shape functions follow VTK conventions (as documented in each shape function module).
This module handles the necessary node reordering from CGNS to VTK convention.

Element Ordering Differences:
-----------------------------
1. HEXA_8 (Hexahedron):
   CGNS: 0-1-2-3 (bottom face), 4-5-6-7 (top face)
   VTK:  0-1-2-3 (bottom face), 4-5-6-7 (top face)
   → SAME ordering, no reordering needed

2. PENTA_6 (Wedge/Prism): **DIFFERENT ORDERING**
   CGNS: Bottom triangle nodes 0-2-1, top triangle nodes 3-5-4
   VTK:  Bottom triangle nodes 0-1-2, top triangle nodes 3-4-5
   → Reordering needed: CGNS[0,1,2,3,4,5] → VTK[0,2,1,3,5,4]

3. PYRA_5 (Pyramid):
   CGNS: 0-1-2-3 (quad base), 4 (apex)
   VTK:  0-1-2-3 (quad base), 4 (apex)
   → SAME ordering, no reordering needed

4. TETRA_4 (Tetrahedron):
   CGNS: 0-1-2-3
   VTK:  0-1-2-3
   → SAME ordering, no reordering needed

Key Features
------------
- Supports multiple element types in same mesh
- Explicit topology stored in element connectivity arrays
- BVH-based locators for efficient cell location
- Explicit cell links for point-to-cell queries
- Maximum 8 points per cell (configurable via MAX_CELL_POINTS)
- Automatic node reordering from CGNS to VTK conventions
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

ET_ElementTypeNull = wp.constant(0)
ET_ElementTypeUserDefined = wp.constant(1)
ET_NODE = wp.constant(2)
ET_BAR_2 = wp.constant(3)
ET_TRI_3 = wp.constant(5)
ET_QUAD_4 = wp.constant(7)
ET_TETRA_4 = wp.constant(10)
ET_PYRA_5 = wp.constant(12)
ET_PENTA_6 = wp.constant(14)
ET_HEXA_8 = wp.constant(17)
ET_MIXED = wp.constant(20)
ET_NGON_n = wp.constant(22)
ET_NFACE_n = wp.constant(23)


# Face connectivity helper functions for each element type
# Following VTK face ordering conventions (after CGNS-to-VTK node reordering)


@wp.func
def _get_num_faces_for_element_type(element_type: wp.int32) -> wp.int32:
    """Get number of faces for an element type."""
    if element_type == ET_TETRA_4:
        return 4
    elif element_type == ET_PYRA_5:
        return 5
    elif element_type == ET_PENTA_6:
        return 5
    elif element_type == ET_HEXA_8:
        return 6
    else:
        return 0


@wp.func
def _get_face_num_points_for_element(element_type: wp.int32, face_idx: wp.int32) -> wp.int32:
    """Get number of points in a specific face."""
    if element_type == ET_TETRA_4:
        return 3  # All faces are triangles
    elif element_type == ET_PYRA_5:
        if face_idx == 0:
            return 4  # Base is quad
        else:
            return 3  # Sides are triangles
    elif element_type == ET_PENTA_6:
        if face_idx < 3:
            return 4  # Three quad faces
        else:
            return 3  # Two triangle faces
    elif element_type == ET_HEXA_8:
        return 4  # All faces are quads
    else:
        return 0


@wp.func
def _get_face_point_local_idx(element_type: wp.int32, face_idx: wp.int32, local_idx: wp.int32) -> wp.int32:
    """Get the local point index for a face point.

    Note: For PENTA_6, this returns indices in VTK order (after CGNS-to-VTK conversion).
    """
    # Hexahedron faces (VTK ordering)
    if element_type == ET_HEXA_8:
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
    elif element_type == ET_TETRA_4:
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
    elif element_type == ET_PYRA_5:
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

    # Pentahedron/Wedge faces (3 quads + 2 triangles)
    # Note: Returns VTK ordering (after reordering from CGNS)
    elif element_type == ET_PENTA_6:
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


@wp.struct
class DatasetHandle:
    grid_coords: wp.array(dtype=wp.vec3f)
    element_type: wp.int32
    hex_is_axis_aligned: wp.bool  # True if all hexahedral elements are axis-aligned, False otherwise
    element_range: wp.vec2i  # NOTE: this is inclusive range [start, end]
    element_connectivity: wp.array(dtype=wp.int32)
    element_start_offset: wp.array(dtype=wp.int32)
    cell_bvh_id: wp.uint64
    cell_links: locators.CellLinks


@wp.struct
class CellHandle:
    cell_id: wp.int32
    element_type: wp.int32
    offset_range: wp.vec2i  # NOTE: this is [start, end) range in element_connectivity array


@wp.struct
class InterpolatedCellHandle:
    cell_id: wp.int32
    element_type: wp.int32
    inside: wp.bool
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
def _get_vertex_count_from_element_type(element_type: wp.int32) -> wp.int32:
    if element_type == ET_BAR_2:
        return 2
    elif element_type == ET_TRI_3:
        return 3
    elif element_type == ET_QUAD_4:
        return 4
    elif element_type == ET_TETRA_4:
        return 4
    elif element_type == ET_PYRA_5:
        return 5
    elif element_type == ET_PENTA_6:
        return 6
    elif element_type == ET_HEXA_8:
        return 8
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


@wp.func
def _reorder_nodes_cgns_to_vtk_penta(
    p0: wp.vec3f, p1: wp.vec3f, p2: wp.vec3f, p3: wp.vec3f, p4: wp.vec3f, p5: wp.vec3f
) -> tuple[wp.vec3f, wp.vec3f, wp.vec3f, wp.vec3f, wp.vec3f, wp.vec3f]:
    """
    Reorder pentahedron/wedge nodes from CGNS to VTK convention.

    CGNS PENTA_6 uses ordering: 0-2-1 (bottom triangle), 3-5-4 (top triangle)
    VTK WEDGE uses ordering: 0-1-2 (bottom triangle), 3-4-5 (top triangle)

    Mapping: CGNS[0,1,2,3,4,5] → VTK[0,2,1,3,5,4]

    Returns: (p0, p1, p2, p3, p4, p5) in VTK order
    """
    # Reorder: swap nodes 1↔2 and 4↔5
    return (p0, p2, p1, p3, p5, p4)


@wp.func
def _evaluate_tetra_cell(ds: DatasetHandle, cell: CellHandle, position: wp.vec3f) -> InterpolatedCellHandle:
    """Evaluate position for tetrahedral cell."""
    i_cell = InterpolatedCellHandle()
    i_cell.cell_id = cell.cell_id
    i_cell.element_type = cell.element_type

    # Extract 4 points
    p0 = DatasetAPI.get_point(ds, ds.element_connectivity[cell.offset_range.x + 0])
    p1 = DatasetAPI.get_point(ds, ds.element_connectivity[cell.offset_range.x + 1])
    p2 = DatasetAPI.get_point(ds, ds.element_connectivity[cell.offset_range.x + 2])
    p3 = DatasetAPI.get_point(ds, ds.element_connectivity[cell.offset_range.x + 3])

    # Compute parametric coordinates
    i_cell.parametric_coords = tet_shape.compute_parametric_coordinates(p0, p1, p2, p3, position)

    # Check if inside
    i_cell.inside = tet_shape.is_inside(i_cell.parametric_coords)

    # Compute weights
    i_cell.weights = _convert_weights_to_fixed_size(tet_shape.compute_shape_functions(i_cell.parametric_coords), 4)

    return i_cell


@wp.func
def _evaluate_hexa_cell(ds: DatasetHandle, cell: CellHandle, position: wp.vec3f) -> InterpolatedCellHandle:
    """Evaluate position for hexahedral cell (CGNS and VTK ordering are the same for HEXA_8)."""
    i_cell = InterpolatedCellHandle()
    i_cell.cell_id = cell.cell_id
    i_cell.element_type = cell.element_type

    # Extract 8 points
    p0 = DatasetAPI.get_point(ds, ds.element_connectivity[cell.offset_range.x + 0])
    p1 = DatasetAPI.get_point(ds, ds.element_connectivity[cell.offset_range.x + 1])
    p2 = DatasetAPI.get_point(ds, ds.element_connectivity[cell.offset_range.x + 2])
    p3 = DatasetAPI.get_point(ds, ds.element_connectivity[cell.offset_range.x + 3])
    p4 = DatasetAPI.get_point(ds, ds.element_connectivity[cell.offset_range.x + 4])
    p5 = DatasetAPI.get_point(ds, ds.element_connectivity[cell.offset_range.x + 5])
    p6 = DatasetAPI.get_point(ds, ds.element_connectivity[cell.offset_range.x + 6])
    p7 = DatasetAPI.get_point(ds, ds.element_connectivity[cell.offset_range.x + 7])

    # Compute parametric coordinates
    i_cell.parametric_coords = hex_shape.compute_parametric_coordinates(p0, p1, p2, p3, p4, p5, p6, p7, position)

    # Check if inside
    i_cell.inside = hex_shape.is_inside(i_cell.parametric_coords)

    # Compute weights
    i_cell.weights = _convert_weights_to_fixed_size(hex_shape.compute_shape_functions(i_cell.parametric_coords), 8)

    return i_cell


@wp.func
def _evaluate_voxel_cell(ds: DatasetHandle, cell: CellHandle, position: wp.vec3f) -> InterpolatedCellHandle:
    """Evaluate position for voxel cell."""
    i_cell = InterpolatedCellHandle()
    i_cell.cell_id = cell.cell_id
    i_cell.element_type = cell.element_type

    # Extract 8 points
    p0 = DatasetAPI.get_point(ds, ds.element_connectivity[cell.offset_range.x + 0])
    p1 = DatasetAPI.get_point(ds, ds.element_connectivity[cell.offset_range.x + 1])
    p2 = DatasetAPI.get_point(ds, ds.element_connectivity[cell.offset_range.x + 2])
    p3 = DatasetAPI.get_point(ds, ds.element_connectivity[cell.offset_range.x + 3])
    p4 = DatasetAPI.get_point(ds, ds.element_connectivity[cell.offset_range.x + 4])
    p5 = DatasetAPI.get_point(ds, ds.element_connectivity[cell.offset_range.x + 5])
    p6 = DatasetAPI.get_point(ds, ds.element_connectivity[cell.offset_range.x + 6])
    p7 = DatasetAPI.get_point(ds, ds.element_connectivity[cell.offset_range.x + 7])

    # Compute parametric coordinates
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
    i_cell.element_type = cell.element_type

    # Extract 5 points
    p0 = DatasetAPI.get_point(ds, ds.element_connectivity[cell.offset_range.x + 0])
    p1 = DatasetAPI.get_point(ds, ds.element_connectivity[cell.offset_range.x + 1])
    p2 = DatasetAPI.get_point(ds, ds.element_connectivity[cell.offset_range.x + 2])
    p3 = DatasetAPI.get_point(ds, ds.element_connectivity[cell.offset_range.x + 3])
    p4 = DatasetAPI.get_point(ds, ds.element_connectivity[cell.offset_range.x + 4])

    # Compute parametric coordinates
    i_cell.parametric_coords = pyramid_shape.compute_parametric_coordinates(p0, p1, p2, p3, p4, position)

    # Check if inside
    i_cell.inside = pyramid_shape.is_inside(i_cell.parametric_coords)

    # Compute weights
    i_cell.weights = _convert_weights_to_fixed_size(pyramid_shape.compute_shape_functions(i_cell.parametric_coords), 5)

    return i_cell


@wp.func
def _evaluate_penta_cell(ds: DatasetHandle, cell: CellHandle, position: wp.vec3f) -> InterpolatedCellHandle:
    """Evaluate position for pentahedron/wedge cell with CGNS to VTK reordering."""
    i_cell = InterpolatedCellHandle()
    i_cell.cell_id = cell.cell_id
    i_cell.element_type = cell.element_type

    # Extract 6 points in CGNS order
    p0 = DatasetAPI.get_point(ds, ds.element_connectivity[cell.offset_range.x + 0])
    p1 = DatasetAPI.get_point(ds, ds.element_connectivity[cell.offset_range.x + 1])
    p2 = DatasetAPI.get_point(ds, ds.element_connectivity[cell.offset_range.x + 2])
    p3 = DatasetAPI.get_point(ds, ds.element_connectivity[cell.offset_range.x + 3])
    p4 = DatasetAPI.get_point(ds, ds.element_connectivity[cell.offset_range.x + 4])
    p5 = DatasetAPI.get_point(ds, ds.element_connectivity[cell.offset_range.x + 5])

    # Reorder from CGNS to VTK convention
    p0, p1, p2, p3, p4, p5 = _reorder_nodes_cgns_to_vtk_penta(p0, p1, p2, p3, p4, p5)

    # Compute parametric coordinates using VTK-ordered nodes
    i_cell.parametric_coords = penta_shape.compute_parametric_coordinates(p0, p1, p2, p3, p4, p5, position)

    # Check if inside
    i_cell.inside = penta_shape.is_inside(i_cell.parametric_coords)

    # Compute weights
    weights_penta = penta_shape.compute_shape_functions(i_cell.parametric_coords)
    i_cell.weights = _convert_weights_to_fixed_size(weights_penta, 6)

    return i_cell


class CellAPI:
    @staticmethod
    @wp.func
    def is_valid(cell: CellHandle) -> wp.bool:
        # SIDS cell ids are 1-based; invalid cell has id <= 0
        return cell.cell_id > 0

    @staticmethod
    @wp.func
    def empty() -> CellHandle:
        cell = CellHandle()
        cell.cell_id = -1
        cell.element_type = ET_ElementTypeNull
        cell.offset_range = wp.vec2i(-1, -1)
        return cell

    @staticmethod
    @wp.func
    def get_cell_id(cell: CellHandle) -> wp.int32:
        return cell.cell_id

    @staticmethod
    @wp.func
    def get_num_points(cell: CellHandle, ds: DatasetHandle) -> wp.int32:
        return cell.offset_range.y - cell.offset_range.x

    @staticmethod
    @wp.func
    def get_point_id(cell: CellHandle, local_idx: wp.int32, ds: DatasetHandle) -> wp.int32:
        # Get point ID from connectivity array
        # offset_range is [start, end) in element_connectivity
        if local_idx >= 0 and local_idx < (cell.offset_range.y - cell.offset_range.x):
            return ds.element_connectivity[cell.offset_range.x + local_idx]
        return -1

    @staticmethod
    @wp.func
    def get_num_faces(cell: CellHandle, ds: DatasetHandle) -> wp.int32:
        """Get the number of faces in a cell."""
        return _get_num_faces_for_element_type(cell.element_type)

    @staticmethod
    @wp.func
    def get_face_num_points(cell: CellHandle, face_idx: wp.int32, ds: DatasetHandle) -> wp.int32:
        """Get the number of points in a face."""
        return _get_face_num_points_for_element(cell.element_type, face_idx)

    @staticmethod
    @wp.func
    def get_face_point_id(cell: CellHandle, face_idx: wp.int32, local_idx: wp.int32, ds: DatasetHandle) -> wp.int32:
        """Get a point ID from a face."""
        # Get the local point index within the cell
        cell_local_idx = _get_face_point_local_idx(cell.element_type, face_idx, local_idx)
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
        i_cell.element_type = ET_ElementTypeNull
        i_cell.inside = False
        i_cell.parametric_coords = wp.vec3f(0.0, 0.0, 0.0)
        return i_cell

    @staticmethod
    @wp.func
    def is_valid(i_cell: InterpolatedCellHandle) -> wp.bool:
        # SIDS cell ids are 1-based; invalid cell has id <= 0
        return i_cell.cell_id > 0

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

        Valid if point_id is positive (SIDS uses 1-based indexing).
        """
        return cell_link.point_id > 0

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

        # Convert point_id to point_idx (SIDS uses 1-based indexing)
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

        # Convert point_id to point_idx
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
    def get_cell_id_from_idx(ds: DatasetHandle, idx: wp.int32) -> wp.int32:
        return ds.element_range.x + idx

    @staticmethod
    @wp.func
    def get_cell_idx_from_id(ds: DatasetHandle, id: wp.int32) -> wp.int32:
        return id - ds.element_range.x

    @staticmethod
    @wp.func
    def get_cell(dataset: DatasetHandle, id: wp.int32) -> CellHandle:
        cell_idx = DatasetAPI.get_cell_idx_from_id(dataset, id)

        if dataset.element_type == ET_NFACE_n or dataset.element_type == ET_NGON_n:
            e_start = dataset.element_start_offset[cell_idx]
            e_end = dataset.element_start_offset[cell_idx + 1]
        elif dataset.element_type == ET_MIXED:
            e_start = dataset.element_start_offset[cell_idx]
            e_end = dataset.element_start_offset[cell_idx + 1]

            # for MIXED, first entry is element type
            e_type = dataset.element_connectivity[e_start]
            e_start += 1
        else:
            vtx_count = _get_vertex_count_from_element_type(dataset.element_type)
            e_start = cell_idx * vtx_count
            e_end = e_start + vtx_count
            e_type = dataset.element_type

        cell = CellHandle()
        cell.element_type = e_type
        cell.cell_id = wp.int32(id)
        cell.offset_range = wp.vec2i(e_start, e_end)
        return cell

    @staticmethod
    @wp.func
    def get_num_cells(ds: DatasetHandle) -> wp.int32:
        return ds.element_range.y - ds.element_range.x + 1

    @staticmethod
    @wp.func
    def get_num_points(ds: DatasetHandle) -> wp.int32:
        return ds.grid_coords.shape[0]

    @staticmethod
    @wp.func
    def get_point_id_from_idx(ds: DatasetHandle, point_idx: wp.int32) -> wp.int32:
        return point_idx + 1  # SIDS uses 1-based indexing

    @staticmethod
    @wp.func
    def get_point_idx_from_id(ds: DatasetHandle, point_id: wp.int32) -> wp.int32:
        return point_id - 1  # Convert from 1-based to 0-based

    @staticmethod
    @wp.func
    def get_point(ds: DatasetHandle, point_id: wp.int32) -> wp.vec3f:
        return ds.grid_coords[DatasetAPI.get_point_idx_from_id(ds, point_id)]

    @staticmethod
    @wp.func
    def get_cell_link(ds: DatasetHandle, point_id: wp.int32) -> CellLinkHandle:
        """Get the cell link for a given point id.

        Returns a CellLinkHandle that can be queried with CellLinksAPI
        to find which cells use this point.

        Args:
            ds: The dataset to query
            point_id: The point id to get cells for (1-based for SIDS)

        Returns:
            CellLinkHandle containing the point_id
        """
        cell_link = CellLinkHandle()
        cell_link.point_id = point_id
        return cell_link

    @staticmethod
    @wp.func
    def get_field_id_from_idx(dataset: DatasetHandle, local_idx: wp.int32) -> wp.int32:
        return local_idx + 1  # SIDS uses 1-based indexing

    @staticmethod
    @wp.func
    def get_field_idx_from_id(dataset: DatasetHandle, id: wp.int32) -> wp.int32:
        return id - 1  # Convert from 1-based to 0-based

    @staticmethod
    @wp.func
    def evaluate_position(ds: DatasetHandle, position: wp.vec3f, cell: CellHandle) -> InterpolatedCellHandle:
        # Check if cell is valid
        if not CellAPI.is_valid(cell):
            return InterpolatedCellAPI.empty()

        # Dispatch to appropriate shape function based on element type
        if cell.element_type == ET_TETRA_4:
            return _evaluate_tetra_cell(ds, cell, position)
        elif cell.element_type == ET_HEXA_8:
            return (
                _evaluate_voxel_cell(ds, cell, position)
                if ds.hex_is_axis_aligned
                else _evaluate_hexa_cell(ds, cell, position)
            )
        elif cell.element_type == ET_PYRA_5:
            return _evaluate_pyramid_cell(ds, cell, position)
        elif cell.element_type == ET_PENTA_6:
            return _evaluate_penta_cell(ds, cell, position)
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
    def build_cell_locator(data_model, ds: DatasetHandle, device=None):
        """Build a spatial acceleration structure for cell location queries.

        Args:
            data_model: The data model module (should be 'unstructured')
            ds: The dataset
            device: Device to build the locator on

        Returns:
            tuple: (status, locator) - Status code and CellLocator instance
        """
        locator = locators.build_cell_locator(data_model, ds, device)
        if locator is not None:
            ds.cell_bvh_id = locator.get_bvh_id()
            return (True, locator)
        else:
            ds.cell_bvh_id = 0
            return (False, None)

    @staticmethod
    def build_cell_links(data_model, ds: DatasetHandle, device=None):
        """Build the cell links for the dataset.

        Args:
            data_model: The data model module
            ds: The dataset
            device: Device to build the links on

        Returns:
            tuple: (status, links) - Status code and CellLinks instance
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
    """CGNS/SIDS Unstructured Grid data model implementation."""

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
