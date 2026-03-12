# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Point Cloud Data Model.

This is a base data model for point clouds with zero radius.
Points are discrete samples in 3D space without spatial influence.
Field values are only defined exactly at point locations.

This model can be extended for point clouds with spatial influence
(e.g., Gaussian, SPH kernels) that provide interpolation between points.
"""

from typing import Any

import warp as wp

import dav


@wp.struct
class DatasetHandle:
    points: wp.array(dtype=wp.vec3f)
    """Points in the point cloud."""


def create_handle(points: wp.array) -> DatasetHandle:
    """Create a point cloud dataset handle.

    .. note::
        This function is for advanced use. Most users should use :func:`create_dataset` instead,
        which creates a complete Dataset object ready for use with operators and fields.

    Args:
        points: Array of 3D point coordinates (wp.vec3f)

    Returns:
        DatasetHandle: A new point cloud dataset handle

    Raises:
        ValueError: If points array is invalid or has wrong dtype

    Note:
        Empty point clouds (0 points) are allowed for edge cases.

    Example:
        >>> import warp as wp
        >>> from dav.data_models.custom.point_cloud import create_handle
        >>> points = wp.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=wp.vec3f)
        >>> handle = create_handle(points)
    """
    # Validate points array
    if points is None or points.ndim != 1:
        raise ValueError("points must be a 1D warp array")
    if points.dtype != wp.vec3f:
        raise ValueError(f"points must have dtype wp.vec3f, got {points.dtype}")
    # Note: Allow empty point clouds (0 points) for edge cases

    handle = DatasetHandle()
    handle.points = points
    return handle


def create_dataset(points: wp.array) -> dav.Dataset:
    """Create a point cloud dataset.

    This is the recommended function for creating point cloud datasets. It creates both the
    dataset handle and wraps it in a Dataset object that can be used with operators and fields.

    Args:
        points: Array of 3D point coordinates (wp.vec3f)

    Returns:
        dav.Dataset: A new Dataset instance with point cloud data model

    Raises:
        ValueError: If points array is invalid or has wrong dtype

    Note:
        Empty point clouds (0 points) are allowed for edge cases.

    Example:
        >>> import warp as wp
        >>> from dav.data_models.custom.point_cloud import create_dataset
        >>> points = wp.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=wp.vec3f)
        >>> dataset = create_dataset(points)
        >>> print(dataset.get_num_points())
        3
    """
    handle = create_handle(points)
    return dav.Dataset(DataModel, handle, points.device)


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
        return False

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
    def get_point(ds: DatasetHandle, id: wp.int32) -> wp.vec3f:
        """Get the point from a dataset by point id."""
        return ds.points[id]

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
    def build_cell_locator(data_model, ds: DatasetHandle, device=None) -> tuple[bool, Any]:
        """Build the cell locator for a dataset.

        For point clouds with zero radius, no locator is needed.
        """
        return (True, None)

    @staticmethod
    def build_cell_links(data_model, ds: DatasetHandle, device=None) -> tuple[bool, Any]:
        """Build the cell links for a dataset.

        For point clouds with zero radius, no cell links are needed.
        """
        return (True, None)

    @staticmethod
    def create_interpolated_field_api(field_model: dav.FieldModel) -> type[dav.InterpolatedFieldAPI]:
        """Create an interpolated field API for a field model.

        For point clouds with zero radius, interpolation is not supported.
        Field values can only be queried exactly at point locations.
        """

        class PointCloudInterpolatedFieldAPI:
            @staticmethod
            @dav.func
            def get(ds: DatasetHandle, field: field_model.FieldHandle, cell: wp.bool, position: wp.vec3f):
                """Get the value of a field at a position.

                For zero-radius point clouds, this returns zero since there's
                no valid cell and no interpolation is possible.
                """
                if not CellAPI.is_valid(cell):
                    # No valid cell - cannot interpolate
                    return field_model.FieldAPI.zero()
                else:
                    # Should never reach here for zero-radius point clouds
                    wp.printf("WARNING: Point cloud with zero radius does not support interpolation\n")
                    return field_model.FieldAPI.zero()

        return PointCloudInterpolatedFieldAPI


class CellLocatorAPI:
    def evaluate_position(dataset: DatasetHandle, position: wp.vec3f, cell: CellHandle) -> wp.vec(length=dav.config.max_points_per_cell, dtype=wp.float32):
        return wp.vec(length=dav.config.max_points_per_cell, dtype=wp.float32)

    @staticmethod
    @dav.func
    def find_cell_containing_point(ds: DatasetHandle, position: wp.vec3f, hint: wp.bool) -> wp.bool:
        """Find the cell containing a point in a dataset.

        For a point cloud with zero radius, there are no cells,
        so this always returns False (invalid cell).
        """
        return False

    @staticmethod
    @dav.func
    def point_in_cell(ds: DatasetHandle, point: wp.vec3f, cell: wp.bool) -> wp.bool:
        return False


class DataModelMeta(type):
    def __repr__(cls):
        return "DataModel (Point Cloud)"


class DataModel(metaclass=DataModelMeta):
    """Point Cloud data model implementation (zero radius)."""

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
    """Get the Point Cloud data model."""
    return DataModel
