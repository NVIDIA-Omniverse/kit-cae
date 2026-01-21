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
This module computes the centroids of cells in a dataset.
"""

from logging import getLogger

import dav
import warp as wp

logger = getLogger(__name__)


@dav.cached
def get_kernel(data_model: dav.DataModel):
    @wp.kernel(enable_backward=False)
    def centroid(ds: data_model.DatasetHandle, geo_center: wp.array(dtype=wp.vec3f)):
        cell_idx = wp.tid()
        cell_id = data_model.DatasetAPI.get_cell_id_from_idx(ds, cell_idx)
        # wp.printf("Computing centroid for cell index: %d, cell id: %d\n", cell_idx, cell_id)
        cell = data_model.DatasetAPI.get_cell(ds, cell_id)

        # iterate over cell's points to compute centroid
        if data_model.CellAPI.is_valid(cell):
            num_points = data_model.CellAPI.get_num_points(cell, ds)
            # wp.printf("Cell index: %d has %d points\n", cell_idx, num_points)
            acc = wp.vec3f(0.0, 0.0, 0.0)
            for i in range(num_points):
                pt_id = data_model.CellAPI.get_point_id(cell, i, ds)
                pt = data_model.DatasetAPI.get_point(ds, pt_id)
                acc += pt
            geo_center[cell_idx] = acc / wp.float32(num_points)
        else:
            geo_center[cell_idx] = wp.vec3f(0.0, 0.0, 0.0)

    return centroid


def compute(dataset: dav.DatasetLike, field_name: str = "cell_centers") -> dav.DatasetLike:
    """
    Compute the centroids of all cells in the dataset.

    Args:
        dataset (dav.DatasetLike): The dataset containing cells and points.
        field_name (str): Name for the computed field (default: "cell_centers").

    Returns:
        dav.DatasetLike: A new dataset (shallow copy) containing the computed cell centroids field.
    """
    device = dataset.device
    nb_cells = dataset.get_num_cells()

    with dav.scoped_timer("centroid.allocate_results"):
        geo_center = wp.zeros(nb_cells, dtype=wp.vec3f, device=device)
    with dav.scoped_timer("centroid.get_kernel"):
        kernel = get_kernel(dataset.data_model)
    with dav.scoped_timer("centroid.launch", cuda_filter=wp.TIMING_ALL):
        wp.launch(kernel, dim=nb_cells, inputs=[dataset.handle, geo_center], device=device)

    field = dav.Field.from_array(geo_center, dav.AssociationType.CELL)
    result = dataset.shallow_copy()
    result.add_field(field_name, field)
    return result
