# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Gaussian Point Cloud Data Model."""

from typing import Any

import warp as wp
from dav.fields.typing import AssociationType, FieldModel, InterpolatedFieldAPI


@wp.struct
class GaussianPointCloudDatasetHandle:
    points: wp.array(dtype=wp.vec3f)
    """Points in the point cloud."""
    radius: wp.float32
    """Radius of the points."""
    f2: wp.float32
    """Gaussian kernel parameter squared."""
    hash_grid_id: wp.uint64
    """ID of the hash grid for the point cloud."""


class CellAPI:
    @staticmethod
    @wp.func
    def empty() -> wp.bool:
        """Create an empty cell."""
        return False

    @staticmethod
    @wp.func
    def is_valid(cell: wp.bool) -> wp.bool:
        """Check if a cell is valid."""
        return cell


class DatasetAPI:
    @staticmethod
    @wp.func
    def get_num_cells(ds: GaussianPointCloudDatasetHandle) -> wp.int32:
        """Get the number of cells in a dataset."""
        return 0

    @staticmethod
    @wp.func
    def get_num_points(ds: GaussianPointCloudDatasetHandle) -> wp.int32:
        """Get the number of points in a dataset."""
        return ds.points.shape[0]

    @staticmethod
    @wp.func
    def get_point_id_from_idx(ds: GaussianPointCloudDatasetHandle, local_idx: wp.int32) -> wp.int32:
        """Get the point id from a dataset by local index."""
        return local_idx

    @staticmethod
    @wp.func
    def get_point_idx_from_id(ds: GaussianPointCloudDatasetHandle, id: wp.int32) -> wp.int32:
        """Get the point index from a dataset by point id."""
        return id

    @staticmethod
    @wp.func
    def get_point(ds: GaussianPointCloudDatasetHandle, id: wp.int32) -> wp.vec3f:
        """Get the point from a dataset by point id."""
        return ds.points[id]

    @staticmethod
    @wp.func
    def find_cell_containing_point(ds: GaussianPointCloudDatasetHandle, position: wp.vec3f, hint: wp.bool) -> wp.bool:
        """Find the cell containing a point in a dataset. If atleast point point of influence exists
        for the position, then return True, otherwise return False."""
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
    def build_cell_locator(data_model, ds: GaussianPointCloudDatasetHandle, device=None) -> tuple[bool, Any]:
        """Build the cell locator for a dataset."""
        return (True, None)

    @staticmethod
    def build_cell_links(data_model, ds: GaussianPointCloudDatasetHandle, device=None) -> tuple[bool, Any]:
        """Build the cell links for a dataset."""
        return (True, None)

    @staticmethod
    def create_interpolated_field_api(field_model: FieldModel) -> type[InterpolatedFieldAPI]:
        """Create an interpolated field API for a field model."""

        class PointCloudInterpolatedFieldAPI:
            @staticmethod
            @wp.func
            def _get_point_value(
                ds: GaussianPointCloudDatasetHandle, field: field_model.FieldHandle, position: wp.vec3f
            ):
                """Get the value of a field at a position."""

                value = field_model.FieldAPI.zero()
                pt_idx = wp.int32(0)
                weight_sum = field_model.FieldAPI.zero_s()
                query = wp.hash_grid_query(ds.hash_grid_id, position, ds.radius)
                while wp.hash_grid_query_next(query, pt_idx):
                    pt_id = DatasetAPI.get_point_id_from_idx(ds, pt_idx)
                    point_coords = DatasetAPI.get_point(ds, pt_id)
                    dist = wp.length(point_coords - position)
                    if dist <= ds.radius:
                        d2 = dist * dist
                        w = type(field_model.FieldAPI.zero_s())(wp.exp(-1.0 * ds.f2 * d2))
                        pt_value = field_model.FieldAPI.get(field, pt_idx)
                        value += w * pt_value
                        weight_sum += w
                return value / weight_sum if weight_sum > 0 else value

            @staticmethod
            @wp.func
            def get(
                ds: GaussianPointCloudDatasetHandle, field: field_model.FieldHandle, cell: wp.bool, position: wp.vec3f
            ):
                """Get the value of a field at a position."""
                if not CellAPI.is_valid(cell):
                    wp.printf("Cell is not valid\n")
                    return field_model.FieldAPI.zero()
                elif field_model.FieldAPI.get_association(field) == wp.static(AssociationType.VERTEX.value):
                    return PointCloudInterpolatedFieldAPI._get_point_value(ds, field, position)
                else:
                    wp.printf("Unsupported association type: %d\n", field_model.FieldAPI.get_association(field))
                    return field_model.FieldAPI.zero()

        return PointCloudInterpolatedFieldAPI


class DataModel:
    """Gaussian Point Cloud data model implementation."""

    # Handle types
    DatasetHandle = GaussianPointCloudDatasetHandle
    CellHandle = wp.bool
    InterpolatedCellHandle = wp.bool
    CellLinkHandle = wp.bool
    PointIdHandle = wp.int32
    CellIdHandle = wp.int32

    # API types
    DatasetAPI = DatasetAPI
    CellAPI = CellAPI
