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
This module provides functions to compute the sizes of cells in a dataset.
Cell size is defined as the length of the diagonal of the cell's bounding box.
"""

import dav
import warp as wp


@dav.cached
def get_kernel(data_model: dav.DataModel):
    @wp.kernel(enable_backward=False)
    def cell_sizes(ds: data_model.DatasetHandle, sizes: wp.array(dtype=wp.float32)):
        cell_idx = wp.tid()
        cell_id = data_model.DatasetAPI.get_cell_id_from_idx(ds, cell_idx)
        cell = data_model.DatasetAPI.get_cell(ds, cell_id)

        if data_model.CellAPI.is_valid(cell):
            num_points = data_model.CellAPI.get_num_points(cell, ds)
            min_bound = wp.vec3f(1e10, 1e10, 1e10)
            max_bound = wp.vec3f(-1e10, -1e10, -1e10)
            for i in range(num_points):
                pt_id = data_model.CellAPI.get_point_id(cell, i, ds)
                pt = data_model.DatasetAPI.get_point(ds, pt_id)
                min_bound = wp.min(min_bound, pt)
                max_bound = wp.max(max_bound, pt)
            diag = max_bound - min_bound
            sizes[cell_idx] = wp.length(diag)
        else:
            sizes[cell_idx] = 0.0

    return cell_sizes


def compute(dataset: dav.DatasetLike, field_name: str = "cell_sizes") -> dav.DatasetLike:
    """
    Compute the sizes of all cells in the dataset.

    Args:
        dataset (dav.DatasetLike): The dataset containing cells and points.
        field_name (str): Name for the computed field (default: "cell_sizes").

    Returns:
        dav.DatasetLike: A new dataset (shallow copy) containing the computed cell sizes field.
    """
    device = dataset.device
    nb_cells = dataset.get_num_cells()

    with dav.scoped_timer("cell_sizes.allocate_results"):
        sizes = wp.zeros((nb_cells,), dtype=wp.float32, device=device)
    with dav.scoped_timer("cell_sizes.get_kernel"):
        kernel = get_kernel(dataset.data_model)
    with dav.scoped_timer("cell_sizes.launch", cuda_filter=wp.TIMING_ALL):
        wp.launch(kernel, dim=nb_cells, inputs=[dataset.handle, sizes], device=device)

    field = dav.Field.from_array(sizes, dav.AssociationType.CELL)
    result = dataset.shallow_copy()
    result.add_field(field_name, field)
    return result
