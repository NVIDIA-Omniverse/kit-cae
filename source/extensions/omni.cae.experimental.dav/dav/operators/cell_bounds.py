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
This module provides functions to compute the bounds of cells in a dataset.
"""

import dav
import warp as wp


@dav.cached
def get_kernel(data_model: dav.DataModel):
    @wp.kernel(enable_backward=False)
    def nb_cells_kernel(ds: data_model.DatasetHandle, nb_cells: wp.array(dtype=wp.int32)):
        nb_cells[0] = data_model.DatasetAPI.get_num_cells(ds)

    @wp.kernel(enable_backward=False)
    def cell_bounds_kernel(
        ds: data_model.DatasetHandle, mins: wp.array(dtype=wp.vec3f), maxs: wp.array(dtype=wp.vec3f)
    ):
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
            mins[cell_idx] = min_bound
            maxs[cell_idx] = max_bound
        else:
            mins[cell_idx] = wp.vec3f(0.0, 0.0, 0.0)
            maxs[cell_idx] = wp.vec3f(0.0, 0.0, 0.0)

    return cell_bounds_kernel, nb_cells_kernel


def compute_from_data_model(
    data_model: dav.DataModel, dataset_handle: dav.DatasetHandle, device
) -> tuple[wp.array, wp.array]:
    """
    Compute the bounds of all cells in the dataset (low-level function).

    Args:
        data_model: The data model defining dataset operations
        dataset_handle: The dataset handle
        device: The device to run the computation on

    Returns:
        Tuple[wp.array, wp.array]: Two arrays containing the min and max bounds for each cell.
    """
    kernel, nb_cells_kernel = get_kernel(data_model)
    nb_cells = wp.zeros(1, dtype=wp.int32, device=device)
    wp.launch(nb_cells_kernel, dim=1, inputs=[dataset_handle, nb_cells], device=device)
    nb_cells = int(nb_cells.numpy()[0])
    mins = wp.empty((nb_cells,), dtype=wp.vec3f, device=device)
    maxs = wp.empty((nb_cells,), dtype=wp.vec3f, device=device)

    with dav.scoped_timer("cell_bounds.launch", cuda_filter=wp.TIMING_ALL):
        wp.launch(kernel, dim=nb_cells, inputs=[dataset_handle, mins, maxs], device=device)
    return mins, maxs


def compute(
    dataset: dav.DatasetLike, min_field_name: str = "cell_bounds_min", max_field_name: str = "cell_bounds_max"
) -> dav.DatasetLike:
    """
    Compute the bounds of all cells in the dataset.

    Args:
        dataset (dav.DatasetLike): The dataset containing cells and points.
        min_field_name (str): Name for the minimum bounds field (default: "cell_bounds_min").
        max_field_name (str): Name for the maximum bounds field (default: "cell_bounds_max").

    Returns:
        dav.DatasetLike: A new dataset (shallow copy) containing the computed cell bounds fields.
    """
    # Compute bounds using low-level function
    mins, maxs = compute_from_data_model(dataset.data_model, dataset.handle, dataset.device)

    # Create fields from arrays
    min_field = dav.Field.from_array(mins, dav.AssociationType.CELL)
    max_field = dav.Field.from_array(maxs, dav.AssociationType.CELL)

    # Create shallow copy and add fields
    result = dataset.shallow_copy()
    result.add_field(min_field_name, min_field)
    result.add_field(max_field_name, max_field)

    return result
