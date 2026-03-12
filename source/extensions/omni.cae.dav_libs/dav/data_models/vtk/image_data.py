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

import dav
from dav.shape_functions import dispatcher as shape_functions_dispatcher

from . import structured_data as sd
from . import vtk_shapes


@dav.func
def mul(a: wp.vec3f, b: wp.vec3f) -> wp.vec3f:
    return wp.vec3f(a.x * b.x, a.y * b.y, a.z * b.z)


@wp.struct
class DatasetHandle:
    origin: wp.vec3f
    spacing: wp.vec3f
    extent_min: wp.vec3i
    extent_max: wp.vec3i


def create_handle(origin: wp.vec3f, spacing: wp.vec3f, extent_min: wp.vec3i, extent_max: wp.vec3i) -> DatasetHandle:
    """Create an image data dataset handle.

    .. note::
        This function is for advanced use. Most users should use :func:`create_dataset` instead,
        which creates a complete Dataset object ready for use with operators and fields.

    Args:
        origin: The position of the first point in world coordinates
        spacing: Uniform spacing between grid points in x, y, z directions
        extent_min: Minimum extent (inclusive) in i, j, k indices
        extent_max: Maximum extent (inclusive) in i, j, k indices

    Returns:
        DatasetHandle: A new image data dataset handle

    Raises:
        ValueError: If spacing is non-positive or extent_max < extent_min

    Example:
        >>> import warp as wp
        >>> from dav.data_models.vtk.image_data import create_handle
        >>> handle = create_handle(
        ...     origin=wp.vec3f(0.0, 0.0, 0.0),
        ...     spacing=wp.vec3f(1.0, 1.0, 1.0),
        ...     extent_min=wp.vec3i(0, 0, 0),
        ...     extent_max=wp.vec3i(10, 10, 10)
        ... )
    """
    # Validate spacing is positive
    if spacing.x <= 0 or spacing.y <= 0 or spacing.z <= 0:
        raise ValueError(f"Spacing must be positive in all dimensions, got {spacing}")

    # Validate extent_max >= extent_min
    if extent_max.x < extent_min.x or extent_max.y < extent_min.y or extent_max.z < extent_min.z:
        raise ValueError(f"extent_max {extent_max} must be >= extent_min {extent_min} in all dimensions")

    handle = DatasetHandle()
    handle.origin = origin
    handle.spacing = spacing
    handle.extent_min = extent_min
    handle.extent_max = extent_max
    return handle


def create_dataset(origin: wp.vec3f, spacing: wp.vec3f, extent_min: wp.vec3i, extent_max: wp.vec3i, device: str = None) -> dav.Dataset:
    """Create an image data dataset.

    This is the recommended function for creating image data datasets. It creates both the
    dataset handle and wraps it in a Dataset object that can be used with operators and fields.

    Args:
        origin: The position of the first point in world coordinates
        spacing: Uniform spacing between grid points in x, y, z directions
        extent_min: Minimum extent (inclusive) in i, j, k indices
        extent_max: Maximum extent (inclusive) in i, j, k indices
        device: Device to create the dataset on. If None, uses current Warp device.

    Returns:
        dav.Dataset: A new Dataset instance with image data data model

    Raises:
        ValueError: If spacing is non-positive or extent_max < extent_min

    Example:
        >>> import warp as wp
        >>> from dav.data_models.vtk.image_data import create_dataset
        >>> dataset = create_dataset(
        ...     origin=wp.vec3f(0.0, 0.0, 0.0),
        ...     spacing=wp.vec3f(1.0, 1.0, 1.0),
        ...     extent_min=wp.vec3i(0, 0, 0),
        ...     extent_max=wp.vec3i(10, 10, 10)
        ... )
        >>> print(dataset.get_num_cells())
        1000
    """
    handle = create_handle(origin, spacing, extent_min, extent_max)
    return dav.Dataset(DataModel, handle, device)


@wp.struct
class CellHandle:
    cell_id: wp.int32
    ijk: wp.vec3i


class CellAPI:
    """Static API for operations on Image Data cells."""

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
        cell = CellHandle()
        cell.cell_id = -1
        cell.ijk = wp.vec3i(-1, -1, -1)
        return cell

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
            wp.int32: Always 8 for voxels (hexahedra)
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
        return sd.compute_point_id_for_cell_vertex(cell.ijk, local_idx, ds.extent_min, extent_dims, use_voxel_ordering=True)

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

        For image data, cell ID == cell index.

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

        For image data, cell ID == cell index.

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

        For image data, point ID == point index.

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

        For image data, point ID == point index.

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
        point_ijk_float = wp.vec3f(wp.float32(point_ijk_abs.x), wp.float32(point_ijk_abs.y), wp.float32(point_ijk_abs.z))
        return ds.origin + mul(point_ijk_float, ds.spacing)

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


class DataModelMeta(type):
    def __repr__(cls):
        return "DataModel (VTK Image Data)"


class DataModel(metaclass=DataModelMeta):
    """VTK Image Data data model implementation."""

    # Handle types
    DatasetHandle = DatasetHandle
    CellHandle = CellHandle
    PointIdHandle = wp.int32
    CellIdHandle = wp.int32

    # API types
    DatasetAPI = DatasetAPI
    CellAPI = CellAPI
    CellLinksAPI = CellLinksAPI


ShapesLibrary = vtk_shapes.get_shapes_library([vtk_shapes.VTKCellTypes.VTK_VOXEL])
ShapeDispatch = shape_functions_dispatcher.get_shape_dispatcher(DataModel, ShapesLibrary, [vtk_shapes.VTKCellTypes.VTK_VOXEL])


@dav.func
def _image_data_locate_cell(ds: DatasetHandle, position: wp.vec3f) -> CellHandle:
    """Compute the cell containing a given world-space position."""
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
        return CellAPI.empty()

    # Compute linear cell ID from cell_ijk_relative
    cell_id = (cell_ijk_relative.z * cell_dims.y * cell_dims.x) + (cell_ijk_relative.y * cell_dims.x) + cell_ijk_relative.x

    cell = CellHandle()
    cell.cell_id = cell_id
    cell.ijk = cell_ijk_relative
    return cell


class CellLocatorAPI:
    """Static API for cell location operations."""

    @staticmethod
    @dav.func
    def evaluate_position(ds: DatasetHandle, position: wp.vec3f, cell: CellHandle) -> wp.vec(length=dav.config.max_points_per_cell, dtype=wp.float32):
        assert CellAPI.is_valid(cell), "Invalid cell handle"
        return ShapeDispatch.get_weights(position, cell, ds, vtk_shapes.VTKCellTypes.VTK_VOXEL)

    @staticmethod
    @dav.func
    def find_cell_containing_point(ds: DatasetHandle, position: wp.vec3f, hint: CellHandle) -> CellHandle:
        """Find the cell containing a given point.

        Args:
            ds: The dataset to query
            position: The point to locate in world coordinates
            hint: A hint cell to start the search (unused for image data)

        Returns:
            CellHandle: The cell containing the point, or empty cell if outside
        """
        return _image_data_locate_cell(ds, position)

    @staticmethod
    @dav.func
    def point_in_cell(ds: DatasetHandle, point: wp.vec3f, cell: CellHandle) -> wp.bool:
        assert CellAPI.is_valid(cell), "Invalid cell handle"
        found = _image_data_locate_cell(ds, point)
        return CellAPI.is_valid(found) and found.cell_id == cell.cell_id


DataModel.CellLocatorAPI = CellLocatorAPI


def get_data_model():
    """Factory function to get the data model for VTK Image Data."""
    return DataModel
