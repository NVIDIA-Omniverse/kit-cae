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
VTK Structured Grid Data Model
================================

This module provides a data model implementation for VTK Structured Grid (vtkStructuredGrid).

Structured grids represent curvilinear grids where points are explicitly stored but
connectivity is implicit based on the grid topology, defined by:
- Points: Explicit 3D coordinates for each grid point (can be warped/non-uniform)
- Extent: Integer range defining the grid dimensions [xmin, xmax, ymin, ymax, zmin, zmax]

All cells are hexahedra with 8 vertices, but unlike image data, points can be positioned
arbitrarily in 3D space (e.g., for cylindrical, spherical, or deformed grids).

Type System
-----------
- **Point IDs**: wp.int32 (0-based, contiguous)
- **Cell IDs**: wp.int32 (0-based, contiguous)
- **Indices**: wp.int32 (same as IDs for structured grids)

Key Features
------------
- Implicit topology: Cell connectivity is computed from grid structure
- Explicit geometry: Point coordinates are stored explicitly
- BVH-based locators: Uses BVH for efficient cell location queries
- Implicit cell links: Point-to-cell adjacency is computed from grid structure
"""

from typing import Any

import warp as wp
from dav import locators
from dav.shape_functions import hexa_8 as hex_shape

from . import structured_data as sd


@wp.struct
class DatasetHandle:
    points: wp.array(dtype=wp.vec3f)
    extent_min: wp.vec3i
    extent_max: wp.vec3i
    cell_bvh_id: wp.uint64


@wp.struct
class CellHandle:
    cell_id: wp.int32
    ijk: wp.vec3i


@wp.struct
class InterpolatedCellHandle:
    cell_id: wp.int32
    inside: bool
    parametric_coords: wp.vec3f
    weights: wp.vec(length=8, dtype=wp.float32)


# Import CellLinkHandle from structured_data module
CellLinkHandle = sd.CellLinkHandle


@wp.func
def _evaluate_hexahedron_cell(ds: DatasetHandle, cell: CellHandle, position: wp.vec3f) -> InterpolatedCellHandle:
    """Evaluate position for hexahedral cell in structured grid.

    Args:
        ds: The dataset
        cell: The cell to evaluate
        position: The position to evaluate in world coordinates

    Returns:
        InterpolatedCellHandle: Interpolated cell information with weights
    """
    i_cell = InterpolatedCellHandle()
    i_cell.cell_id = cell.cell_id

    # Get extent dimensions
    extent_dims = sd.get_point_dims(ds.extent_min, ds.extent_max)

    # Extract 8 points for the hexahedral cell
    p0 = ds.points[sd.compute_point_id_for_cell_vertex(cell.ijk, 0, ds.extent_min, extent_dims)]
    p1 = ds.points[sd.compute_point_id_for_cell_vertex(cell.ijk, 1, ds.extent_min, extent_dims)]
    p2 = ds.points[sd.compute_point_id_for_cell_vertex(cell.ijk, 2, ds.extent_min, extent_dims)]
    p3 = ds.points[sd.compute_point_id_for_cell_vertex(cell.ijk, 3, ds.extent_min, extent_dims)]
    p4 = ds.points[sd.compute_point_id_for_cell_vertex(cell.ijk, 4, ds.extent_min, extent_dims)]
    p5 = ds.points[sd.compute_point_id_for_cell_vertex(cell.ijk, 5, ds.extent_min, extent_dims)]
    p6 = ds.points[sd.compute_point_id_for_cell_vertex(cell.ijk, 6, ds.extent_min, extent_dims)]
    p7 = ds.points[sd.compute_point_id_for_cell_vertex(cell.ijk, 7, ds.extent_min, extent_dims)]

    # Compute parametric coordinates
    i_cell.parametric_coords = hex_shape.compute_parametric_coordinates(p0, p1, p2, p3, p4, p5, p6, p7, position)

    # Check if inside
    i_cell.inside = hex_shape.is_inside(i_cell.parametric_coords)

    # Compute weights
    i_cell.weights = hex_shape.compute_shape_functions(i_cell.parametric_coords)

    return i_cell


class CellAPI:
    """Static API for operations on Structured Grid cells."""

    @staticmethod
    @wp.func
    def is_valid(cell: CellHandle) -> wp.bool:
        """Check if a cell is valid.

        Args:
            cell: The cell to check

        Returns:
            wp.bool: True if the cell has a valid ID (>= 0), False otherwise
        """
        return cell.cell_id >= 0

    @staticmethod
    @wp.func
    def empty() -> CellHandle:
        """Create an empty (invalid) cell.

        Returns:
            CellHandle: An empty cell with ID=-1 and invalid ijk coordinates
        """
        return CellHandle(cell_id=-1, ijk=wp.vec3i(-1, -1, -1))

    @staticmethod
    @wp.func
    def get_cell_id(cell: CellHandle) -> wp.int32:
        """Get the ID of a cell.

        Args:
            cell: The cell to query

        Returns:
            wp.int32: The cell ID (0-based)
        """
        return cell.cell_id

    @staticmethod
    @wp.func
    def get_num_points(cell: CellHandle, ds: DatasetHandle) -> wp.int32:
        """Get the number of points in a cell.

        Args:
            cell: The cell to query
            ds: The dataset containing the cell

        Returns:
            wp.int32: Always 8 for hexahedra
        """
        return 8

    @staticmethod
    @wp.func
    def get_point_id(cell: CellHandle, local_idx: wp.int32, ds: DatasetHandle) -> wp.int32:
        """Get a point ID from a cell's local vertex index.

        Args:
            cell: The cell to query
            local_idx: Local vertex index (0-7)
            ds: The dataset containing the cell

        Returns:
            wp.int32: The global point ID
        """
        # Compute extent dimensions (number of points in each direction)
        extent_dims = sd.get_point_dims(ds.extent_min, ds.extent_max)

        # Get point ID for this cell vertex
        return sd.compute_point_id_for_cell_vertex(cell.ijk, local_idx, ds.extent_min, extent_dims)

    @staticmethod
    @wp.func
    def get_num_faces(cell: CellHandle, ds: DatasetHandle) -> wp.int32:
        """Get the number of faces in a cell.

        Args:
            cell: The cell to query
            ds: The dataset containing the cell

        Returns:
            wp.int32: Always 6 for hexahedra
        """
        return 6

    @staticmethod
    @wp.func
    def get_face_num_points(cell: CellHandle, face_idx: wp.int32, ds: DatasetHandle) -> wp.int32:
        """Get the number of points in a face.

        Args:
            cell: The cell to query
            face_idx: Local face index (0-5)
            ds: The dataset containing the cell

        Returns:
            wp.int32: Always 4 for quadrilateral faces
        """
        return sd.get_hex_face_num_points(face_idx)

    @staticmethod
    @wp.func
    def get_face_point_id(cell: CellHandle, face_idx: wp.int32, local_idx: wp.int32, ds: DatasetHandle) -> wp.int32:
        """Get a point ID from a face.

        Args:
            cell: The cell to query
            face_idx: Local face index (0-5)
            local_idx: Local index within the face (0-3)
            ds: The dataset containing the cell

        Returns:
            wp.int32: The global point ID
        """
        # Get the local point index within the cell
        cell_local_idx = sd.get_hex_face_point_local_idx(face_idx, local_idx)
        # Convert to global point ID
        return CellAPI.get_point_id(cell, cell_local_idx, ds)


class InterpolatedCellAPI:
    """Static API for operations on interpolated cells."""

    @staticmethod
    @wp.func
    def empty() -> InterpolatedCellHandle:
        """Create an empty (invalid) interpolated cell.

        Returns:
            InterpolatedCellHandle: An empty interpolated cell
        """
        i_cell = InterpolatedCellHandle()
        i_cell.cell_id = -1
        i_cell.inside = False
        i_cell.parametric_coords = wp.vec3f(0.0, 0.0, 0.0)
        return i_cell

    @staticmethod
    @wp.func
    def is_valid(i_cell: InterpolatedCellHandle) -> wp.bool:
        """Check if an interpolated cell is valid.

        Args:
            i_cell: The interpolated cell to check

        Returns:
            wp.bool: True if the cell has a valid ID (>= 0), False otherwise
        """
        return i_cell.cell_id >= 0

    @staticmethod
    @wp.func
    def get_cell_id(i_cell: InterpolatedCellHandle) -> wp.int32:
        """Get the cell ID from an interpolated cell.

        Args:
            i_cell: The interpolated cell to query

        Returns:
            wp.int32: The cell ID (0-based)
        """
        return i_cell.cell_id

    @staticmethod
    @wp.func
    def is_inside(i_cell: InterpolatedCellHandle) -> wp.bool:
        """Check if the interpolated position is inside the cell.

        Args:
            i_cell: The interpolated cell to check

        Returns:
            wp.bool: True if the position is inside the cell, False otherwise
        """
        return i_cell.inside

    @staticmethod
    @wp.func
    def get_weight(i_cell: InterpolatedCellHandle, local_idx: wp.int32) -> wp.float32:
        """Get the interpolation weight for a given vertex.

        Args:
            i_cell: The interpolated cell to query
            local_idx: Local vertex index (0-7)

        Returns:
            wp.float32: The interpolation weight at the given vertex
        """
        return i_cell.weights[local_idx]


class CellLinksAPI:
    """Static API for operations on cell links (point-to-cell adjacency)."""

    @staticmethod
    @wp.func
    def empty() -> CellLinkHandle:
        """Create an empty cell link.

        Returns:
            CellLinkHandle: An empty cell link
        """
        return sd.CellLinksAPI.empty()

    @staticmethod
    @wp.func
    def is_valid(cell_link: CellLinkHandle) -> wp.bool:
        """Check if cell link is valid.

        Args:
            cell_link: The cell link to check

        Returns:
            wp.bool: True if the cell link is valid, False otherwise
        """
        return sd.CellLinksAPI.is_valid(cell_link)

    @staticmethod
    @wp.func
    def get_point_id(cell_link: CellLinkHandle) -> wp.int32:
        """Get the point ID for a given cell link.

        Args:
            cell_link: The cell link to query

        Returns:
            wp.int32: The point ID
        """
        return sd.CellLinksAPI.get_point_id(cell_link)

    @staticmethod
    @wp.func
    def get_num_cells(cell_link: CellLinkHandle, ds: DatasetHandle) -> wp.int32:
        """Get the number of cells that use this point.

        Args:
            cell_link: The cell link to query
            ds: The dataset

        Returns:
            wp.int32: The number of cells using this point
        """
        return sd.CellLinksAPI.get_num_cells(cell_link, ds.extent_min, ds.extent_max)

    @staticmethod
    @wp.func
    def get_cell_id(cell_link: CellLinkHandle, cell_idx: wp.int32, ds: DatasetHandle) -> wp.int32:
        """Get the cell ID for a given cell index in the cell link.

        Args:
            cell_link: The cell link to query
            cell_idx: The cell index (0-based)
            ds: The dataset

        Returns:
            wp.int32: The cell ID at the given index
        """
        return sd.CellLinksAPI.get_cell_id(cell_link, cell_idx, ds.extent_min, ds.extent_max)


class DatasetAPI:
    """Static API for dataset operations."""

    @staticmethod
    @wp.func
    def get_cell_id_from_idx(ds: DatasetHandle, cell_idx: wp.int32) -> wp.int32:
        """Get a cell ID from a dataset by local index.

        For structured grid, cell ID == cell index.

        Args:
            ds: The dataset to query
            cell_idx: The local cell index (0-based)

        Returns:
            wp.int32: The cell ID
        """
        return cell_idx

    @staticmethod
    @wp.func
    def get_cell_idx_from_id(ds: DatasetHandle, cell_id: wp.int32) -> wp.int32:
        """Get a cell index from a dataset by cell ID.

        For structured grid, cell ID == cell index.

        Args:
            ds: The dataset to query
            cell_id: The cell ID

        Returns:
            wp.int32: The cell index (0-based)
        """
        return cell_id

    @staticmethod
    @wp.func
    def get_cell(ds: DatasetHandle, cell_id: wp.int32) -> CellHandle:
        """Get a cell from the dataset by ID.

        Args:
            ds: The dataset to query
            cell_id: The cell ID

        Returns:
            CellHandle: The cell at the given ID
        """
        # Calculate cell dimensions
        cell_dims = sd.get_cell_dims(ds.extent_min, ds.extent_max)

        # Convert linear cell_id to ijk (relative to extent_min)
        cell_ijk = sd.compute_cell_ijk(cell_id, cell_dims)

        cell = CellHandle()
        cell.cell_id = cell_id
        cell.ijk = cell_ijk
        return cell

    @staticmethod
    @wp.func
    def get_num_cells(ds: DatasetHandle) -> wp.int32:
        """Get the number of cells in the dataset.

        Args:
            ds: The dataset to query

        Returns:
            wp.int32: The number of cells
        """
        return sd.compute_num_cells(ds.extent_min, ds.extent_max)

    @staticmethod
    @wp.func
    def get_num_points(ds: DatasetHandle) -> wp.int32:
        """Get the number of points in the dataset.

        Args:
            ds: The dataset to query

        Returns:
            wp.int32: The number of points
        """
        point_dims = sd.get_point_dims(ds.extent_min, ds.extent_max)
        return point_dims.x * point_dims.y * point_dims.z

    @staticmethod
    @wp.func
    def get_point_id_from_idx(ds: DatasetHandle, point_idx: wp.int32) -> wp.int32:
        """Get a point ID from a dataset by local index.

        For structured grid, point ID == point index.

        Args:
            ds: The dataset to query
            point_idx: The local point index (0-based)

        Returns:
            wp.int32: The point ID
        """
        return point_idx

    @staticmethod
    @wp.func
    def get_point_idx_from_id(ds: DatasetHandle, point_id: wp.int32) -> wp.int32:
        """Get a point index from a dataset by point ID.

        For structured grid, point ID == point index.

        Args:
            ds: The dataset to query
            point_id: The point ID

        Returns:
            wp.int32: The point index (0-based)
        """
        return point_id

    @staticmethod
    @wp.func
    def get_point(ds: DatasetHandle, point_id: wp.int32) -> wp.vec3f:
        """Get a point from the dataset by ID.

        VTK Structured Grid stores explicit point coordinates.

        Args:
            ds: The dataset to query
            point_id: The point ID

        Returns:
            wp.vec3f: The point coordinates in world space
        """
        return ds.points[point_id]

    @staticmethod
    @wp.func
    def get_cell_link(ds: DatasetHandle, point_id: wp.int32) -> CellLinkHandle:
        """Get the cell link for a given point ID.

        Returns a CellLinkHandle that can be queried with CellLinksAPI
        to find which cells use this point (computed implicitly from grid structure).

        Args:
            ds: The dataset to query
            point_id: The point ID to get cells for (0-based)

        Returns:
            CellLinkHandle: Cell link containing the point_id
        """
        cell_link = CellLinkHandle()
        cell_link.point_id = point_id
        return cell_link

    @staticmethod
    @wp.func
    def evaluate_position(ds: DatasetHandle, position: wp.vec3f, cell: CellHandle) -> InterpolatedCellHandle:
        """Evaluate a position within a cell to get interpolation weights.

        Args:
            ds: The dataset to query
            position: The position to evaluate in world coordinates
            cell: The cell containing the position

        Returns:
            InterpolatedCellHandle: Interpolated cell information with weights
        """
        # Check if cell is valid
        if not CellAPI.is_valid(cell):
            return InterpolatedCellAPI.empty()

        # VTK Structured Grid cells are always hexahedra
        return _evaluate_hexahedron_cell(ds, cell, position)

    @staticmethod
    @wp.func
    def find_cell_containing_point(ds: DatasetHandle, position: wp.vec3f, hint: CellHandle) -> CellHandle:
        """Find the cell containing a given point using BVH.

        Args:
            ds: The dataset to query
            position: The point to locate in world coordinates
            hint: A hint cell to start the search

        Returns:
            CellHandle: The cell containing the point, or empty cell if outside
        """
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
            data_model: The data model module (should be 'structured_grid')
            ds: The dataset
            device: Device to build the locator on

        Returns:
            tuple[bool, Any]: (success, locator) - Success flag and locator instance
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
            tuple[bool, Any]: (status, links) - For VTK Structured Grid, cell links are
                   implicit due to structured grid connectivity, so no explicit
                   data structures are needed
        """
        return (True, None)


# DataModel protocol implementation
class DataModel:
    """VTK Structured Grid data model implementation."""

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
