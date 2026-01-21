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
This module provides the point splats operator for splatting field values at points.
"""

from logging import getLogger

import dav
import warp as wp
from dav.data_models.custom import gaussian_point_cloud

logger = getLogger(__name__)


@dav.cached
def get_kernel(data_model: dav.DataModel):
    @wp.kernel(enable_backward=False)
    def point_splats(ds: data_model.DatasetHandle, positions: wp.array(dtype=wp.vec3f)):
        pt_idx = wp.tid()
        pt_id = data_model.DatasetAPI.get_point_id_from_idx(ds, pt_idx)
        positions[pt_idx] = data_model.DatasetAPI.get_point(ds, pt_id)

    return point_splats


def compute(dataset: dav.DatasetLike, radius: wp.float32, sharpness: wp.float32) -> dav.DatasetLike:
    """
    Splat field values at points in the dataset.

    Args:
        dataset: The dataset to splat.
        radius: The radius of the points.
        sharpness: The sharpness of the points.

    Returns:
        The dataset with the splatted points.
    """
    device = dataset.device
    nb_points = dataset.get_num_points()

    if radius <= 0.0:
        raise ValueError("Radius must be positive")
    if sharpness <= 0.0:
        raise ValueError("Sharpness must be positive")

    with dav.scoped_timer("point_splats.allocate_results"):
        positions = wp.empty((nb_points,), dtype=wp.vec3f, device=device)
    with dav.scoped_timer("point_splats.get_kernel"):
        kernel = get_kernel(dataset.data_model)
    with dav.scoped_timer("point_splats.launch", cuda_filter=wp.TIMING_ALL):
        wp.launch(kernel, dim=nb_points, inputs=[dataset.handle], outputs=[positions], device=device)

    # Now create output dataset.
    data_model = gaussian_point_cloud.DataModel
    ds_handle = data_model.DatasetHandle()
    ds_handle.points = positions
    ds_handle.radius = radius
    ds_handle.f2 = (sharpness / radius) ** 2

    min_bounds, max_bounds = dataset.get_bounds()
    resolution = (max_bounds - min_bounds) / radius
    hash_grid = wp.HashGrid(dim_x=int(resolution[0]), dim_y=int(resolution[1]), dim_z=int(resolution[2]), device=device)
    hash_grid.build(positions, radius)

    ds_handle.hash_grid_id = hash_grid.id
    result_dataset = dav.Dataset(data_model=data_model, handle=ds_handle, device=device, hash_grid=hash_grid)
    for field_name, field in dataset.fields.items():
        if field.association == dav.AssociationType.CELL:
            logger.warning(f"Skipping cell-centered field: {field_name}")
            continue
        result_dataset.add_field(field_name, field)
    return result_dataset
