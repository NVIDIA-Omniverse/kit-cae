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
Curves Data Model (UsdGeomBasisCurves-based)
============================================

This is a partial implementation for now since we only really need
it to output dataset from operators like streamlines. We can expand it in the future as needed.

"""

import warp as wp

import dav


@wp.struct
class DatasetHandle:
    points: wp.array(dtype=wp.vec3f)
    curve_vertex_counts: wp.array(dtype=wp.int32)


@wp.struct
class CellHandle:
    to_do: wp.int32  # Curves don't have cell handles, but we need this to satisfy the protocol


class DatasetAPI:
    """
    Parital implementation of the DatasetAPI protocol.
    """

    @staticmethod
    @dav.func
    def get_num_cells(handle: DatasetHandle) -> int:
        """Get number of curves (cells) in the dataset."""
        return len(handle.curve_vertex_counts)

    @staticmethod
    @dav.func
    def get_num_points(handle: DatasetHandle) -> int:
        """Get total number of points across all curves."""
        return len(handle.points)

    @staticmethod
    @dav.func
    def get_point_id_from_idx(ds: DatasetHandle, point_idx: wp.int32) -> wp.int32:
        """Get point ID from point index (same for surface meshes)."""
        return point_idx

    @staticmethod
    @dav.func
    def get_point_idx_from_id(ds: DatasetHandle, point_id: wp.int32) -> wp.int32:
        """Get point index from point ID (same for surface meshes)."""
        return point_id

    @staticmethod
    @dav.func
    def get_point(ds: DatasetHandle, point_id: wp.int32) -> wp.vec3f:
        """Get a vertex position by point ID."""
        return ds.points[point_id]


class DataModelMeta(type):
    def __repr__(cls):
        return "DataModel (Curves)"


class DataModel(metaclass=DataModelMeta):
    DatasetHandle = DatasetHandle
    PointIdHandle = wp.int32
    CellIdHandle = wp.int32
    CellHandle = CellHandle

    DatasetAPI = DatasetAPI


def create_dataset(points: wp.array(dtype=wp.vec3f), curve_vertex_counts: wp.array(dtype=wp.int32)) -> DatasetHandle:
    """
    Create a Curves dataset.

    Args:
        points: Array of curve points (shape [num_points])
        curve_vertex_counts: Array of vertex counts for each curve (shape [num_curves])

    Returns:
        DatasetHandle containing the curves data
    """
    if points.ndim != 1 or points.dtype != wp.vec3f:
        raise ValueError("Points array must be 1D with dtype wp.vec3f")
    if curve_vertex_counts.ndim != 1 or curve_vertex_counts.dtype != wp.int32:
        raise ValueError("Curve vertex counts array must be 1D with dtype wp.int32")
    if points.device != curve_vertex_counts.device:
        raise ValueError("Points and curve vertex counts must be on the same device")

    handle = DatasetHandle()
    handle.points = points
    handle.curve_vertex_counts = curve_vertex_counts

    return dav.Dataset(DataModel, handle, device=points.device)


def get_data_model():
    """Get the Curves data model."""
    return DataModel
