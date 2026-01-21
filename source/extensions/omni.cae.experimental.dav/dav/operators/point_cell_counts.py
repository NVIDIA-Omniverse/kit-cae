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
This module provides functions to compute the number of cells using each point in a dataset.
This is useful for understanding mesh connectivity and identifying shared vertices.
"""

import dav
import warp as wp


@dav.cached
def get_kernel(data_model: dav.DataModel):
    @wp.kernel(enable_backward=False)
    def point_cell_counts(ds: data_model.DatasetHandle, counts: wp.array(dtype=wp.int32)):
        cell_idx = wp.tid()
        cell_id = data_model.DatasetAPI.get_cell_id_from_idx(ds, cell_idx)
        cell = data_model.DatasetAPI.get_cell(ds, cell_id)

        if data_model.CellAPI.is_valid(cell):
            num_points = data_model.CellAPI.get_num_points(cell, ds)
            for i in range(num_points):
                pt_id = data_model.CellAPI.get_point_id(cell, i, ds)
                pt_idx = data_model.DatasetAPI.get_point_idx_from_id(ds, pt_id)
                # Use atomic add to avoid race conditions when multiple cells share points
                wp.atomic_add(counts, pt_idx, 1)

    return point_cell_counts


def compute(dataset: dav.DatasetLike, field_name: str = "cells_per_point") -> dav.DatasetLike:
    """
    Compute the number of cells using each point in the dataset.

    This operator counts how many cells reference each vertex/point.
    For example, in a 2D quad mesh, interior points are typically shared
    by 4 cells, while boundary points are shared by fewer cells.

    Args:
        dataset (dav.DatasetLike): The dataset containing cells and points.
        field_name (str): Name for the computed field (default: "cells_per_point").

    Returns:
        dav.DatasetLike: A new dataset (shallow copy) containing the computed
                 cells per point field as an integer point-associated field.
    """
    device = dataset.device
    nb_points = dataset.get_num_points()
    nb_cells = dataset.get_num_cells()

    with dav.scoped_timer("point_cell_counts.allocate_results"):
        counts = wp.zeros((nb_points,), dtype=wp.int32, device=device)
    with dav.scoped_timer("point_cell_counts.get_kernel"):
        kernel = get_kernel(dataset.data_model)
    with dav.scoped_timer("point_cell_counts.launch", cuda_filter=wp.TIMING_ALL):
        wp.launch(kernel, dim=nb_cells, inputs=[dataset.handle, counts], device=device)

    field = dav.Field.from_array(counts, dav.AssociationType.VERTEX)
    result = dataset.shallow_copy()
    result.add_field(field_name, field)

    return result
