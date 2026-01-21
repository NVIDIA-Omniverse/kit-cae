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
VTK Image Data Data Model
==========================

This module provides a data model implementation for VTK Image Data (vtkImageData).

Image data represents uniformly spaced rectilinear grids in 3D space, defined by:
- Origin: The position of the first point in world coordinates
- Spacing: Uniform spacing between grid points in x, y, z directions
- Extent: Integer range defining the grid dimensions [xmin, xmax, ymin, ymax, zmin, zmax]

All cells are axis-aligned hexahedra (voxels) with 8 vertices.

Type System
-----------
- **Point IDs**: wp.int32 (0-based, contiguous)
- **Cell IDs**: wp.int32 (0-based, contiguous)
- **Indices**: wp.int32 (same as IDs for image data)

Key Features
------------
- Implicit topology: Cell connectivity is computed on-the-fly from the regular grid structure
- Implicit locators: No BVH needed - cell location is computed directly from position
- Implicit cell links: Point-to-cell adjacency is computed directly from grid structure
- Fast random access to any cell or point
"""

from typing import Any

import warp as wp
from dav.shape_functions import voxel_8

from . import structured_data as sd


@wp.func
def mul(a: wp.vec3f, b: wp.vec3f) -> wp.vec3f:
    return wp.vec3f(a.x * b.x, a.y * b.y, a.z * b.z)


@wp.struct
class DatasetHandle:
    origin: wp.vec3f
    spacing: wp.vec3f
    extent_min: wp.vec3i
    extent_max: wp.vec3i


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
def _compute_parametric_coords(ds: DatasetHandle, position: wp.vec3f, cell_ijk: wp.vec3i) -> wp.vec3f:
    """Compute parametric coordinates (0 to 1) of a position within a cell."""
    # Convert cell ijk (relative) to absolute ijk
    cell_ijk_abs = cell_ijk + ds.extent_min

    # Get the cell's minimum corner point in world coordinates
    cell_min_point = ds.origin + mul(
        wp.vec3f(wp.float32(cell_ijk_abs.x), wp.float32(cell_ijk_abs.y), wp.float32(cell_ijk_abs.z)), ds.spacing
    )

    # Compute parametric coordinates (0 to 1 within the cell)
    return wp.cw_div((position - cell_min_point), ds.spacing)


class CellAPI:
    """Static API for operations on Image Data cells."""

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
        cell = CellHandle()
        cell.cell_id = -1
        cell.ijk = wp.vec3i(-1, -1, -1)
        return cell

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
            wp.int32: Always 8 for voxels (hexahedra)
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

        For image data, cell ID == cell index.

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

        For image data, cell ID == cell index.

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

        For image data, point ID == point index.

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

        For image data, point ID == point index.

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

        Args:
            ds: The dataset to query
            point_id: The point ID

        Returns:
            wp.vec3f: The point coordinates in world space
        """
        # Compute point dimensions (number of points in each direction)
        point_dims = sd.get_point_dims(ds.extent_min, ds.extent_max)

        # Convert point_id to ijk coordinates (relative to extent_min)
        point_ijk_relative = sd.compute_point_ijk(point_id, point_dims)

        # Convert to absolute ijk coordinates
        point_ijk_abs = point_ijk_relative + ds.extent_min

        # Compute physical coordinates: origin + (ijk * spacing)
        point_ijk_float = wp.vec3f(
            wp.float32(point_ijk_abs.x), wp.float32(point_ijk_abs.y), wp.float32(point_ijk_abs.z)
        )
        return ds.origin + mul(point_ijk_float, ds.spacing)

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

        # Compute parametric coordinates
        i_cell = InterpolatedCellHandle()
        i_cell.cell_id = CellAPI.get_cell_id(cell)
        i_cell.parametric_coords = _compute_parametric_coords(ds, position, cell.ijk)

        # Check if the position is inside the cell (parametric coords should be in [0, 1])
        i_cell.inside = (
            i_cell.parametric_coords.x >= 0.0
            and i_cell.parametric_coords.x <= 1.0
            and i_cell.parametric_coords.y >= 0.0
            and i_cell.parametric_coords.y <= 1.0
            and i_cell.parametric_coords.z >= 0.0
            and i_cell.parametric_coords.z <= 1.0
        )

        # Compute trilinear interpolation weights using voxel shape functions
        i_cell.weights = voxel_8.compute_shape_functions(i_cell.parametric_coords)

        return i_cell

    @staticmethod
    @wp.func
    def find_cell_containing_point(ds: DatasetHandle, position: wp.vec3f, hint: CellHandle) -> CellHandle:
        """Find the cell containing a given point.

        Args:
            ds: The dataset to query
            position: The point to locate in world coordinates
            hint: A hint cell to start the search (unused for image data)

        Returns:
            CellHandle: The cell containing the point, or empty cell if outside
        """
        # Convert world position to grid coordinates
        grid_coords = wp.cw_div((position - ds.origin), ds.spacing)

        # Compute cell ijk indices (floor of grid coordinates)
        cell_ijk = wp.vec3i(int(wp.floor(grid_coords.x)), int(wp.floor(grid_coords.y)), int(wp.floor(grid_coords.z)))

        # Get cell dimensions
        cell_dims = sd.get_cell_dims(ds.extent_min, ds.extent_max)

        # Check if cell_ijk is within valid bounds (relative to extent_min)
        cell_ijk_relative = cell_ijk - ds.extent_min

        if (
            cell_ijk_relative.x < 0
            or cell_ijk_relative.x >= cell_dims.x
            or cell_ijk_relative.y < 0
            or cell_ijk_relative.y >= cell_dims.y
            or cell_ijk_relative.z < 0
            or cell_ijk_relative.z >= cell_dims.z
        ):
            # Point is outside the dataset
            return CellAPI.empty()

        # Compute linear cell ID from cell_ijk_relative
        cell_id = (
            (cell_ijk_relative.z * cell_dims.y * cell_dims.x)
            + (cell_ijk_relative.y * cell_dims.x)
            + cell_ijk_relative.x
        )

        # Create and return the cell
        cell = CellHandle()
        cell.cell_id = cell_id
        cell.ijk = cell_ijk_relative
        return cell

    @staticmethod
    def build_cell_locator(data_model, ds: DatasetHandle, device=None) -> tuple[bool, Any]:
        """Build a spatial acceleration structure for cell location queries.

        Args:
            data_model: The data model module (should be 'image_data')
            ds: The dataset
            device: Device to build the locator on

        Returns:
            tuple[bool, Any]: (status, locator) - For VTK Image Data, the cell locator is
                   implicit due to regular grid structure, so no additional
                   data structures are needed
        """
        return (True, None)

    @staticmethod
    def build_cell_links(data_model, ds: DatasetHandle, device=None) -> tuple[bool, Any]:
        """Build the cell links for the dataset.

        Args:
            data_model: The data model module
            ds: The dataset
            device: Device to build the links on

        Returns:
            tuple[bool, Any]: (status, links) - For VTK Image Data, cell links are
                   implicit due to regular grid structure, so no explicit
                   data structures are needed
        """
        return (True, None)


# DataModel protocol implementation
class DataModel:
    """VTK Image Data data model implementation."""

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
