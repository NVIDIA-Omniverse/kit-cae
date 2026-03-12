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

import dav
from dav import locators
from dav.shape_functions import dispatcher as shape_functions_dispatcher

from . import structured_data as sd
from . import vtk_shapes


@wp.struct
class DatasetHandle:
    points: wp.array(dtype=wp.vec3f)
    extent_min: wp.vec3i
    extent_max: wp.vec3i
    cell_bvh_id: wp.uint64


def create_handle(points: wp.array, extent_min: wp.vec3i, extent_max: wp.vec3i) -> DatasetHandle:
    """Create a structured grid dataset handle.

    .. note::
        This function is for advanced use. Most users should use :func:`create_dataset` instead,
        which creates a complete Dataset object ready for use with operators and fields.

    Args:
        points: Array of 3D point coordinates (wp.vec3f). Points must be ordered according to
                the structured grid extent (i varies fastest, then j, then k)
        extent_min: Minimum extent (inclusive) in i, j, k indices
        extent_max: Maximum extent (inclusive) in i, j, k indices

    Returns:
        DatasetHandle: A new structured grid dataset handle

    Raises:
        ValueError: If points array is invalid, dtype is wrong, or extent dimensions don't match point count

    Note:
        The cell_bvh_id will be initialized to 0. Use DatasetAPI.build_cell_locator() to build
        the spatial acceleration structure after creating the dataset.

    Example:
        >>> import warp as wp
        >>> from dav.data_models.vtk.structured_grid import create_handle
        >>> points = wp.array([[0, 0, 0], [1, 0, 0], ...], dtype=wp.vec3f)
        >>> handle = create_handle(
        ...     points=points,
        ...     extent_min=wp.vec3i(0, 0, 0),
        ...     extent_max=wp.vec3i(2, 2, 2)
        ... )
    """
    # Validate points array
    if points is None or points.ndim != 1:
        raise ValueError("points must be a 1D warp array")
    if points.dtype != wp.vec3f:
        raise ValueError(f"points must have dtype wp.vec3f, got {points.dtype}")
    if points.shape[0] == 0:
        raise ValueError("points array cannot be empty")

    # Validate extent_max >= extent_min
    if extent_max.x < extent_min.x or extent_max.y < extent_min.y or extent_max.z < extent_min.z:
        raise ValueError(f"extent_max {extent_max} must be >= extent_min {extent_min} in all dimensions")

    # Validate point count matches extent dimensions
    point_dims = sd.get_point_dims(extent_min, extent_max)
    expected_num_points = point_dims.x * point_dims.y * point_dims.z
    if points.shape[0] != expected_num_points:
        raise ValueError(f"Number of points ({points.shape[0]}) must match extent dimensions ({point_dims.x} × {point_dims.y} × {point_dims.z} = {expected_num_points})")

    handle = DatasetHandle()
    handle.points = points
    handle.extent_min = extent_min
    handle.extent_max = extent_max
    handle.cell_bvh_id = wp.uint64(0)
    return handle


def create_dataset(points: wp.array, extent_min: wp.vec3i, extent_max: wp.vec3i) -> dav.Dataset:
    """Create a structured grid dataset.

    This is the recommended function for creating structured grid datasets. It creates both the
    dataset handle and wraps it in a Dataset object that can be used with operators and fields.

    Args:
        points: Array of 3D point coordinates (wp.vec3f). Points must be ordered according to
                the structured grid extent (i varies fastest, then j, then k)
        extent_min: Minimum extent (inclusive) in i, j, k indices
        extent_max: Maximum extent (inclusive) in i, j, k indices

    Returns:
        Dataset: A new Dataset instance with structured grid data model

    Raises:
        ValueError: If points array is invalid, dtype is wrong, or extent dimensions don't match point count

    Example:
        >>> import warp as wp
        >>> from dav.data_models.vtk.structured_grid import create_dataset
        >>> points = wp.array([[0, 0, 0], [1, 0, 0], ...], dtype=wp.vec3f)
        >>> dataset = create_dataset(
        ...     points=points,
        ...     extent_min=wp.vec3i(0, 0, 0),
        ...     extent_max=wp.vec3i(2, 2, 2)
        ... )
        >>> print(dataset.get_num_cells())
        8
    """
    handle = create_handle(points, extent_min, extent_max)
    return dav.Dataset(DataModel, handle, points.device)


@wp.struct
class CellHandle:
    cell_id: wp.int32
    ijk: wp.vec3i


class CellAPI:
    """Static API for operations on Structured Grid cells."""

    @staticmethod
    @dav.func
    def is_valid(cell: CellHandle) -> wp.bool:
        """Check if a cell is valid.

        Args:
            cell: The cell to check

        Returns:
            wp.bool: True if the cell has a valid ID (>= 0), False otherwise
        """
        return cell.cell_id >= 0

    @staticmethod
    @dav.func
    def empty() -> CellHandle:
        """Create an empty (invalid) cell.

        Returns:
            CellHandle: An empty cell with ID=-1 and invalid ijk coordinates
        """
        return CellHandle(cell_id=-1, ijk=wp.vec3i(-1, -1, -1))

    @staticmethod
    @dav.func
    def get_cell_id(cell: CellHandle) -> wp.int32:
        """Get the ID of a cell.

        Args:
            cell: The cell to query

        Returns:
            wp.int32: The cell ID (0-based)
        """
        return cell.cell_id

    @staticmethod
    @dav.func
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
    @dav.func
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
    @dav.func
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
    @dav.func
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
    @dav.func
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


class CellLinksAPI:
    """Static API for operations on cell links (point-to-cell adjacency)."""

    @staticmethod
    @dav.func
    def is_valid(point_id: wp.int32) -> wp.bool:
        """Check if point ID is valid for cell links.

        Args:
            point_id: The point ID to check

        Returns:
            wp.bool: True if the cell link is valid, False otherwise
        """
        return sd.CellLinksAPI.is_valid(point_id)

    @staticmethod
    @dav.func
    def get_num_cells(point_id: wp.int32, ds: DatasetHandle) -> wp.int32:
        """Get the number of cells that use this point.

        Args:
            point_id: The point ID to query
            ds: The dataset

        Returns:
            wp.int32: The number of cells using this point
        """
        return sd.CellLinksAPI.get_num_cells(point_id, ds.extent_min, ds.extent_max)

    @staticmethod
    @dav.func
    def get_cell_id(point_id: wp.int32, cell_idx: wp.int32, ds: DatasetHandle) -> wp.int32:
        """Get the cell ID for a given cell index in the cell link.

        Args:
            point_id: The point ID to query
            cell_idx: The cell index (0-based)
            ds: The dataset

        Returns:
            wp.int32: The cell ID at the given index
        """
        return sd.CellLinksAPI.get_cell_id(point_id, cell_idx, ds.extent_min, ds.extent_max)


class DatasetAPI:
    """Static API for dataset operations."""

    @staticmethod
    @dav.func
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
    @dav.func
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
    @dav.func
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
    @dav.func
    def get_num_cells(ds: DatasetHandle) -> wp.int32:
        """Get the number of cells in the dataset.

        Args:
            ds: The dataset to query

        Returns:
            wp.int32: The number of cells
        """
        return sd.compute_num_cells(ds.extent_min, ds.extent_max)

    @staticmethod
    @dav.func
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
    @dav.func
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
    @dav.func
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
    @dav.func
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


class DataModelMeta(type):
    def __repr__(cls):
        return "DataModel (VTK Structured Grid)"


class DataModel(metaclass=DataModelMeta):
    """VTK Structured Grid data model implementation."""

    # Handle types
    DatasetHandle = DatasetHandle
    CellHandle = CellHandle
    PointIdHandle = wp.int32
    CellIdHandle = wp.int32

    # API types
    DatasetAPI = DatasetAPI
    CellAPI = CellAPI
    CellLinksAPI = CellLinksAPI


ShapesLibrary = vtk_shapes.get_shapes_library([vtk_shapes.VTKCellTypes.VTK_HEXAHEDRON])
ShapeDispatch = shape_functions_dispatcher.get_shape_dispatcher(DataModel, ShapesLibrary, [vtk_shapes.VTKCellTypes.VTK_HEXAHEDRON])


class CellLocatorAPI:
    """Static API for cell location operations."""

    @staticmethod
    @dav.func
    def evaluate_position(ds: DatasetHandle, position: wp.vec3f, cell: CellHandle) -> wp.vec(length=dav.config.max_points_per_cell, dtype=wp.float32):
        assert CellAPI.is_valid(cell), "Invalid cell handle"
        return ShapeDispatch.get_weights(position, cell, ds, vtk_shapes.VTKCellTypes.VTK_HEXAHEDRON)

    @staticmethod
    @dav.func
    def find_cell_containing_point(ds: DatasetHandle, position: wp.vec3f, hint: CellHandle) -> CellHandle:
        assert ds.cell_bvh_id != 0, "Cell locator not built for dataset"

        if CellAPI.is_valid(hint):
            if ShapeDispatch.is_point_in_cell(position, hint, ds, vtk_shapes.VTKCellTypes.VTK_HEXAHEDRON):
                return hint

        radius = wp.vec3f(1.0e-6, 1.0e-6, 1.0e-6)
        query = wp.bvh_query_aabb(ds.cell_bvh_id, position - radius, position + radius)
        cell_idx = wp.int32(-1)
        while wp.bvh_query_next(query, cell_idx):
            cell_id = DatasetAPI.get_cell_id_from_idx(ds, cell_idx)
            cell = DatasetAPI.get_cell(ds, cell_id)
            assert CellAPI.is_valid(cell), "BVH returned an invalid cell"
            if ShapeDispatch.is_point_in_cell(position, cell, ds, vtk_shapes.VTKCellTypes.VTK_HEXAHEDRON):
                return cell

        return CellAPI.empty()

    @staticmethod
    @dav.func
    def point_in_cell(ds: DatasetHandle, point: wp.vec3f, cell: CellHandle) -> wp.bool:
        assert CellAPI.is_valid(cell), "Invalid cell handle"
        return ShapeDispatch.is_point_in_cell(point, cell, ds, vtk_shapes.VTKCellTypes.VTK_HEXAHEDRON)


DataModel.CellLocatorAPI = CellLocatorAPI


def get_data_model():
    """Factory function to get the data model for VTK Structured Grid."""
    return DataModel
