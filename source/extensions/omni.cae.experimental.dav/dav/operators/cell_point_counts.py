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
This module provides functions to compute the number of points per cell in a dataset.
This is useful for understanding the topology and connectivity of cells.
"""

import dav
import warp as wp


@dav.cached
def get_kernel(data_model: dav.DataModel):
    @wp.kernel(enable_backward=False)
    def cell_point_counts(ds: data_model.DatasetHandle, counts: wp.array(dtype=wp.int32)):
        cell_idx = wp.tid()
        cell_id = data_model.DatasetAPI.get_cell_id_from_idx(ds, cell_idx)
        cell = data_model.DatasetAPI.get_cell(ds, cell_id)

        if data_model.CellAPI.is_valid(cell):
            num_points = data_model.CellAPI.get_num_points(cell, ds)
            counts[cell_idx] = num_points
        else:
            counts[cell_idx] = 0

    return cell_point_counts


def compute(dataset: dav.DatasetLike, field_name: str = "points_per_cell") -> dav.DatasetLike:
    """
    Compute the number of points per cell in the dataset.

    This operator counts how many vertices/points each cell contains.
    For example, a tetrahedral cell would return 4, a hexahedral cell
    would return 8, etc.

    Args:
        dataset (dav.DatasetLike): The dataset containing cells and points.
        field_name (str): Name for the computed field (default: "points_per_cell").

    Returns:
        dav.DatasetLike: A new dataset (shallow copy) containing the computed
                 points per cell field as an integer cell-associated field.
    """
    device = dataset.device
    nb_cells = dataset.get_num_cells()

    with dav.scoped_timer("cell_point_counts.allocate_results"):
        counts = wp.zeros((nb_cells,), dtype=wp.int32, device=device)
    with dav.scoped_timer("cell_point_counts.get_kernel"):
        kernel = get_kernel(dataset.data_model)
    with dav.scoped_timer("cell_point_counts.launch", cuda_filter=wp.TIMING_ALL):
        wp.launch(kernel, dim=nb_cells, inputs=[dataset.handle, counts], device=device)

    field = dav.Field.from_array(counts, dav.AssociationType.CELL)
    result = dataset.shallow_copy()
    result.add_field(field_name, field)
    return result
