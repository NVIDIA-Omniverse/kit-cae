# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Gaussian Point Cloud Data Model.

This data model extends the base point cloud model to add Gaussian kernel
interpolation with spatial influence. Points have a radius of influence and
use an exponential decay kernel for smooth interpolation:

    w(d) = exp(-f2 * d²)

where:
    - d is the distance from the point
    - f2 = (sharpness / radius)²

A hash grid is used for efficient spatial queries within the radius.
"""

from typing import Any

import warp as wp

import dav


@wp.struct
class DatasetHandle:
    points: wp.array(dtype=wp.vec3f)
    """Points in the point cloud."""
    radius: wp.float32
    """Radius of influence for each point."""
    f2: wp.float32
    """Gaussian kernel parameter squared: f2 = (sharpness / radius)²."""
    hash_grid_id: wp.uint64
    """ID of the hash grid for spatial queries."""


def create_handle(points: wp.array, radius: float, sharpness: float) -> DatasetHandle:
    """Create a Gaussian point cloud dataset handle.

    .. note::
        This function is for advanced use. Most users should use :func:`create_dataset` instead,
        which creates a complete Dataset object ready for use with operators and fields.

    Args:
        points: Array of 3D point coordinates (wp.vec3f)
        radius: Radius of influence for each point
        sharpness: Sharpness parameter for the Gaussian kernel (controls falloff)

    Returns:
        DatasetHandle: A new Gaussian point cloud dataset handle

    Raises:
        ValueError: If points array is invalid, radius is non-positive, or sharpness is non-positive

    Note:
        The hash_grid_id will be initialized to 0. Use DatasetAPI.build_cell_locator() to build
        the hash grid for spatial queries after creating the dataset.
        Empty point clouds (0 points) are allowed for edge cases.

    Example:
        >>> import warp as wp
        >>> from dav.data_models.custom.gaussian_point_cloud import create_handle
        >>> points = wp.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=wp.vec3f)
        >>> handle = create_handle(points, radius=1.0, sharpness=2.0)
    """
    # Validate points array
    if points is None or points.ndim != 1:
        raise ValueError("points must be a 1D warp array")
    if points.dtype != wp.vec3f:
        raise ValueError(f"points must have dtype wp.vec3f, got {points.dtype}")
    # Note: Allow empty point clouds (0 points) for edge cases

    # Validate radius and sharpness
    if radius <= 0:
        raise ValueError(f"radius must be positive, got {radius}")
    if sharpness <= 0:
        raise ValueError(f"sharpness must be positive, got {sharpness}")

    handle = DatasetHandle()
    handle.points = points
    handle.radius = wp.float32(radius)
    handle.f2 = wp.float32((sharpness / radius) ** 2)
    handle.hash_grid_id = wp.uint64(0)
    return handle


def create_dataset(points: wp.array, radius: float, sharpness: float) -> dav.Dataset:
    """Create a Gaussian point cloud dataset.

    This is the recommended function for creating Gaussian point cloud datasets. It creates both the
    dataset handle and wraps it in a Dataset object that can be used with operators and fields.

    Args:
        points: Array of 3D point coordinates (wp.vec3f)
        radius: Radius of influence for each point
        sharpness: Sharpness parameter for the Gaussian kernel (controls falloff)

    Returns:
        dav.Dataset: A new Dataset instance with Gaussian point cloud data model

    Raises:
        ValueError: If points array is invalid, radius is non-positive, or sharpness is non-positive

    Note:
        Empty point clouds (0 points) are allowed for edge cases.

    Example:
        >>> import warp as wp
        >>> from dav.data_models.custom.gaussian_point_cloud import create_dataset
        >>> points = wp.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=wp.vec3f)
        >>> dataset = create_dataset(points, radius=1.0, sharpness=2.0)
        >>> print(dataset.get_num_points())
        3
    """
    handle = create_handle(points, radius, sharpness)
    return dav.Dataset(dav.DataModel, handle, points.device)


CellHandle = wp.bool
PointIdHandle = wp.int32
CellIdHandle = wp.int32


class CellAPI:
    @staticmethod
    @dav.func
    def empty() -> CellHandle:
        """Create an empty cell."""
        return False

    @staticmethod
    @dav.func
    def is_valid(cell: CellHandle) -> wp.bool:
        """Check if a cell is valid."""
        return cell

    @staticmethod
    @dav.func
    def get_cell_id(cell: CellHandle) -> wp.int32:
        return -1

    @staticmethod
    @dav.func
    def get_num_points(cell: CellHandle, ds: DatasetHandle) -> wp.int32:
        return 0

    @staticmethod
    @dav.func
    def get_point_id(cell: CellHandle, local_idx: wp.int32, ds: DatasetHandle) -> PointIdHandle:
        return -1

    @staticmethod
    @dav.func
    def get_num_faces(cell: CellHandle, ds: DatasetHandle) -> wp.int32:
        return 0

    @staticmethod
    @dav.func
    def get_face_num_points(cell: CellHandle, face_idx: wp.int32, ds: DatasetHandle) -> wp.int32:
        return 0

    @staticmethod
    @dav.func
    def get_face_point_id(cell: CellHandle, face_idx: wp.int32, local_idx: wp.int32, ds: DatasetHandle) -> PointIdHandle:
        return -1


class DatasetAPI:
    """Dataset API for Gaussian point clouds with spatial influence."""

    @staticmethod
    @dav.func
    def get_num_cells(ds: DatasetHandle) -> wp.int32:
        """Get the number of cells in a dataset."""
        return 0

    @staticmethod
    @dav.func
    def get_num_points(ds: DatasetHandle) -> wp.int32:
        """Get the number of points in a dataset."""
        return ds.points.shape[0]

    @staticmethod
    @dav.func
    def get_point_id_from_idx(ds: DatasetHandle, local_idx: wp.int32) -> wp.int32:
        """Get the point id from a dataset by local index."""
        return local_idx

    @staticmethod
    @dav.func
    def get_point_idx_from_id(ds: DatasetHandle, id: wp.int32) -> wp.int32:
        """Get the point index from a dataset by point id."""
        return id

    @staticmethod
    @dav.func
    def get_cell_id_from_idx(ds: DatasetHandle, local_idx: wp.int32) -> wp.int32:
        """Get the cell id from a dataset by local index."""
        return -1

    @staticmethod
    @dav.func
    def get_cell_idx_from_id(ds: DatasetHandle, id: wp.int32) -> wp.int32:
        """Get the cell index from a dataset by cell id."""
        return -1

    @staticmethod
    @dav.func
    def get_cell(ds: DatasetHandle, id: wp.int32) -> wp.bool:
        """Get the cell from a dataset by cell id."""
        return False

    @staticmethod
    @dav.func
    def get_point(ds: DatasetHandle, id: wp.int32) -> wp.vec3f:
        """Get the point from a dataset by point id."""
        return ds.points[id]

    @staticmethod
    def build_cell_locator(data_model, ds: DatasetHandle, device=None) -> tuple[bool, Any]:
        """Build the cell locator for a dataset."""
        return (True, None)

    @staticmethod
    def build_cell_links(data_model, ds: DatasetHandle, device=None) -> tuple[bool, Any]:
        """Build the cell links for a dataset."""
        return (True, None)

    @staticmethod
    def create_interpolated_field_api(field_model: dav.FieldModel) -> type[dav.InterpolatedFieldAPI]:
        """Create an interpolated field API for a field model."""

        mul_f32 = dav.utils.get_mul_by_f32_function(type(field_model.FieldAPI.zero()))

        class Meta(type):
            def __repr__(cls):
                return f"GaussianPointCloudInterpolatedFieldAPI[{field_model}]"

        # Standalone helper defined before GaussianPointCloudInterpolatedFieldAPI so that
        # get() can call it without referencing the class mid-class-body (empty closure cell).
        @dav.func
        def _get_point_value(ds: DatasetHandle, field: field_model.FieldHandle, position: wp.vec3f):
            """Get the Gaussian-weighted field value at a position."""
            value = field_model.FieldAPI.zero()
            pt_idx = wp.int32(0)
            weight_sum = wp.float32(0.0)
            query = wp.hash_grid_query(ds.hash_grid_id, position, ds.radius)
            while wp.hash_grid_query_next(query, pt_idx):
                pt_id = DatasetAPI.get_point_id_from_idx(ds, pt_idx)
                point_coords = DatasetAPI.get_point(ds, pt_id)
                dist = wp.length(point_coords - position)
                if dist <= ds.radius:
                    d2 = dist * dist
                    w = wp.exp(-1.0 * ds.f2 * d2)
                    pt_value = field_model.FieldAPI.get(field, pt_idx)
                    value += mul_f32(pt_value, w)
                    weight_sum += w
            if weight_sum > wp.float32(0.0):
                reciprocal = wp.float32(1.0) / weight_sum
                return mul_f32(value, reciprocal)
            else:
                return value

        class GaussianPointCloudInterpolatedFieldAPI(metaclass=Meta):
            @staticmethod
            @dav.func
            def get(ds: DatasetHandle, field: field_model.FieldHandle, cell: wp.bool, position: wp.vec3f):
                """Get the value of a field at a position."""
                assert CellAPI.is_valid(cell), "Cell is invalid"
                assert field_model.FieldAPI.get_association(field) == wp.static(dav.AssociationType.VERTEX.value), "Only vertex-associated fields are supported for interpolation"
                return _get_point_value(ds, field, position)

        return GaussianPointCloudInterpolatedFieldAPI


class CellLocatorAPI:
    @staticmethod
    @dav.func
    def find_cell_containing_point(ds: DatasetHandle, position: wp.vec3f, hint: wp.bool) -> wp.bool:
        """Find the cell containing a point in a dataset.

        For Gaussian point clouds, returns True if at least one point of influence
        exists for the position (i.e., within radius), otherwise returns False.
        """
        if ds.hash_grid_id == 0:
            wp.printf("ERROR: Hash grid not built for dataset\n")
            return False

        pt_idx = wp.int32(0)
        query = wp.hash_grid_query(ds.hash_grid_id, position, ds.radius)
        while wp.hash_grid_query_next(query, pt_idx):
            pt_id = DatasetAPI.get_point_id_from_idx(ds, pt_idx)
            point_coords = DatasetAPI.get_point(ds, pt_id)
            dist = wp.length(point_coords - position)
            if dist <= ds.radius:
                return True
        return False

    @staticmethod
    @dav.func
    def point_in_cell(ds: DatasetHandle, point: wp.vec3f, cell: wp.bool) -> wp.bool:
        return False


class DataModelMeta(type):
    def __repr__(cls):
        return "DataModel (Gaussian Point Cloud)"


class DataModel(metaclass=DataModelMeta):
    """Gaussian Point Cloud data model implementation.

    Extends the base point cloud model to add Gaussian kernel interpolation
    with configurable radius and sharpness.
    """

    # Handle types
    DatasetHandle = DatasetHandle
    CellHandle = CellHandle
    PointIdHandle = PointIdHandle
    CellIdHandle = CellIdHandle

    # API types
    DatasetAPI = DatasetAPI
    CellAPI = CellAPI
    CellLocatorAPI = CellLocatorAPI


def get_data_model():
    """Get the Gaussian Point Cloud data model."""
    return DataModel
