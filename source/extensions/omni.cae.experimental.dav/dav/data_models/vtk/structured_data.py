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
Shared utilities for VTK structured data models (ImageData, StructuredGrid, RectilinearGrid).
"""

import warp as wp


@wp.struct
class CellLinkHandle:
    """Cell link for a given point in structured grid.

    For structured grids, cell connectivity is implicit based on the grid structure.
    This struct stores the point_id and we compute which cells use this point on-the-fly.
    """

    point_id: wp.int32


@wp.func
def get_hex_vertex_offset(local_idx: wp.int32) -> wp.vec3i:
    """Get the (di, dj, dk) offset for a hexahedron vertex in VTK ordering.

    Args:
        local_idx: Local vertex index (0-7) in VTK hex ordering

    Returns:
        wp.vec3i: The offset from the cell's base point
    """
    offset = wp.vec3i(0, 0, 0)

    if local_idx == 0:  # (0, 0, 0)
        offset = wp.vec3i(0, 0, 0)
    elif local_idx == 1:  # (1, 0, 0)
        offset = wp.vec3i(1, 0, 0)
    elif local_idx == 2:  # (1, 1, 0)
        offset = wp.vec3i(1, 1, 0)
    elif local_idx == 3:  # (0, 1, 0)
        offset = wp.vec3i(0, 1, 0)
    elif local_idx == 4:  # (0, 0, 1)
        offset = wp.vec3i(0, 0, 1)
    elif local_idx == 5:  # (1, 0, 1)
        offset = wp.vec3i(1, 0, 1)
    elif local_idx == 6:  # (1, 1, 1)
        offset = wp.vec3i(1, 1, 1)
    elif local_idx == 7:  # (0, 1, 1)
        offset = wp.vec3i(0, 1, 1)

    return offset


@wp.func
def compute_point_id(point_ijk: wp.vec3i, extent_min: wp.vec3i, extent_dims: wp.vec3i) -> wp.int32:
    """Convert absolute ijk coordinates to point ID (VTK's ComputePointId logic).

    Args:
        point_ijk: Absolute ijk coordinates of the point
        extent_min: Minimum extent of the grid
        extent_dims: Dimensions (number of points) in each direction

    Returns:
        wp.int32: The point ID (0-based)
    """
    relative_ijk = point_ijk - extent_min
    point_id = (relative_ijk.z * extent_dims.y * extent_dims.x) + (relative_ijk.y * extent_dims.x) + relative_ijk.x
    return point_id


@wp.func
def compute_cell_ijk(cell_id: wp.int32, cell_dims: wp.vec3i) -> wp.vec3i:
    """Convert linear cell_id to ijk coordinates (relative to extentMin).

    Args:
        cell_id: The cell ID (0-based)
        cell_dims: Number of cells in each direction

    Returns:
        wp.vec3i: The cell ijk coordinates (relative to extent_min)
    """
    cells_per_layer = cell_dims.x * cell_dims.y
    k = cell_id // cells_per_layer
    remainder = cell_id % cells_per_layer
    j = remainder // cell_dims.x
    i = remainder % cell_dims.x
    return wp.vec3i(i, j, k)


@wp.func
def compute_point_ijk(point_id: wp.int32, point_dims: wp.vec3i) -> wp.vec3i:
    """Convert linear point_id to ijk coordinates (relative to extentMin).

    Args:
        point_id: The point ID (0-based)
        point_dims: Number of points in each direction

    Returns:
        wp.vec3i: The point ijk coordinates (relative to extent_min)
    """
    points_per_layer = point_dims.y * point_dims.x
    k = point_id // points_per_layer
    remainder = point_id % points_per_layer
    j = remainder // point_dims.x
    i = remainder % point_dims.x
    return wp.vec3i(i, j, k)


@wp.func
def get_cell_dims(extent_min: wp.vec3i, extent_max: wp.vec3i) -> wp.vec3i:
    """Get the number of cells in each direction."""
    return extent_max - extent_min


@wp.func
def get_point_dims(extent_min: wp.vec3i, extent_max: wp.vec3i) -> wp.vec3i:
    """Get the number of points in each direction."""
    return extent_max - extent_min + wp.vec3i(1, 1, 1)


@wp.func
def compute_num_cells(extent_min: wp.vec3i, extent_max: wp.vec3i) -> wp.int32:
    """Compute total number of cells from extents.

    Args:
        extent_min: Minimum extent of the grid
        extent_max: Maximum extent of the grid

    Returns:
        wp.int32: Total number of cells
    """
    cell_dims = get_cell_dims(extent_min, extent_max)
    return cell_dims.x * cell_dims.y * cell_dims.z


@wp.func
def compute_point_id_for_cell_vertex(
    cell_ijk: wp.vec3i, local_idx: wp.int32, extent_min: wp.vec3i, extent_dims: wp.vec3i
) -> wp.int32:
    """Get point ID for a cell's vertex given cell ijk and local vertex index.

    Args:
        cell_ijk: Cell ijk coordinates (relative to extent_min)
        local_idx: Local vertex index (0-7)
        extent_min: Minimum extent of the grid
        extent_dims: Dimensions (number of points) in each direction

    Returns:
        wp.int32: The point ID for the vertex
    """
    # Get absolute cell ijk (cell_ijk is relative to extentMin)
    cell_ijk_abs = cell_ijk + extent_min

    # Get vertex offset for this local index (VTK hexahedron ordering)
    vertex_offset = get_hex_vertex_offset(local_idx)

    # Compute absolute point ijk coordinates
    point_ijk = cell_ijk_abs + vertex_offset

    # Convert to linear point ID
    return compute_point_id(point_ijk, extent_min, extent_dims)


# Face API helper functions for structured grids
# All cells in structured grids are hexahedra with 6 quad faces


@wp.func
def get_hex_face_num_points(face_idx: wp.int32) -> wp.int32:
    """Get number of points in a hexahedron face.

    Args:
        face_idx: Local face index (0-5)

    Returns:
        wp.int32: Always 4 for quad faces
    """
    return 4


@wp.func
def get_hex_face_point_local_idx(face_idx: wp.int32, local_idx: wp.int32) -> wp.int32:
    """Get the local point index within a cell for a hexahedron face point.

    Args:
        face_idx: Local face index (0-5)
        local_idx: Local index within the face (0-3)

    Returns:
        wp.int32: The local cell vertex index (0-7)
    """
    # Hexahedron faces (VTK ordering)
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


class CellLinksAPI:
    """Static API for operations on cell links for structured grids.

    For structured grids, cell connectivity is implicit based on grid structure.
    We compute which cells use a given point on-the-fly based on the point's ijk position.

    This class provides the common implementation for both ImageData and StructuredGrid.
    """

    @staticmethod
    @wp.func
    def empty() -> CellLinkHandle:
        """Create an empty cell link.

        Returns:
            CellLinkHandle: An empty cell link with invalid point_id
        """
        cell_link = CellLinkHandle()
        cell_link.point_id = -1
        return cell_link

    @staticmethod
    @wp.func
    def is_valid(cell_link: CellLinkHandle) -> wp.bool:
        """Check if cell link is valid.

        Args:
            cell_link: The cell link to check

        Returns:
            wp.bool: True if valid (point_id >= 0), False otherwise
        """
        return cell_link.point_id >= 0

    @staticmethod
    @wp.func
    def get_point_id(cell_link: CellLinkHandle) -> wp.int32:
        """Get the point ID for a given cell link.

        Args:
            cell_link: The cell link to query

        Returns:
            wp.int32: The point ID
        """
        return cell_link.point_id

    @staticmethod
    @wp.func
    def get_num_cells(cell_link: CellLinkHandle, extent_min: wp.vec3i, extent_max: wp.vec3i) -> wp.int32:
        """Get the number of cells that use this point.

        For structured grids, a point can be used by up to 8 cells (in 3D).
        We compute this implicitly based on the point's position in the grid.

        Args:
            cell_link: The cell link to query
            extent_min: Minimum extent of the grid
            extent_max: Maximum extent of the grid

        Returns:
            wp.int32: Number of cells using this point (0-8)
        """
        if not CellLinksAPI.is_valid(cell_link):
            return 0

        # Get point dimensions and cell dimensions
        point_dims = get_point_dims(extent_min, extent_max)
        cell_dims = get_cell_dims(extent_min, extent_max)

        # Convert point_id to ijk (relative to extent_min)
        point_ijk = compute_point_ijk(cell_link.point_id, point_dims)

        # Count how many cells use this point
        # A point at (i,j,k) can be used by cells at positions (i-1,j-1,k-1) through (i,j,k)
        # We need to check which of these 8 potential cells actually exist
        count = 0
        for di in range(-1, 1):
            for dj in range(-1, 1):
                for dk in range(-1, 1):
                    cell_ijk = point_ijk + wp.vec3i(di, dj, dk)
                    # Check if this cell is valid (within cell bounds)
                    if (
                        cell_ijk.x >= 0
                        and cell_ijk.x < cell_dims.x
                        and cell_ijk.y >= 0
                        and cell_ijk.y < cell_dims.y
                        and cell_ijk.z >= 0
                        and cell_ijk.z < cell_dims.z
                    ):
                        count += 1

        return count

    @staticmethod
    @wp.func
    def get_cell_id(
        cell_link: CellLinkHandle, cell_idx: wp.int32, extent_min: wp.vec3i, extent_max: wp.vec3i
    ) -> wp.int32:
        """Get the cell ID for a given cell index in the cell link.

        Args:
            cell_link: The cell link to query
            cell_idx: Local index within the cells using this point (0-based)
            extent_min: Minimum extent of the grid
            extent_max: Maximum extent of the grid

        Returns:
            wp.int32: The cell ID at the given local index, or -1 if invalid
        """
        if not CellLinksAPI.is_valid(cell_link):
            return -1

        # Get point dimensions and cell dimensions
        point_dims = get_point_dims(extent_min, extent_max)
        cell_dims = get_cell_dims(extent_min, extent_max)

        # Convert point_id to ijk (relative to extent_min)
        point_ijk = compute_point_ijk(cell_link.point_id, point_dims)

        # Enumerate cells that use this point and find the one at cell_idx
        current_idx = 0
        for di in range(-1, 1):
            for dj in range(-1, 1):
                for dk in range(-1, 1):
                    cell_ijk = point_ijk + wp.vec3i(di, dj, dk)
                    # Check if this cell is valid (within cell bounds)
                    if (
                        cell_ijk.x >= 0
                        and cell_ijk.x < cell_dims.x
                        and cell_ijk.y >= 0
                        and cell_ijk.y < cell_dims.y
                        and cell_ijk.z >= 0
                        and cell_ijk.z < cell_dims.z
                    ):
                        if current_idx == cell_idx:
                            # Compute linear cell_id from cell_ijk
                            cell_id = cell_ijk.z * cell_dims.y * cell_dims.x + cell_ijk.y * cell_dims.x + cell_ijk.x
                            return cell_id
                        current_idx += 1

        return -1
