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

The operator takes any dataset and creates a point cloud representation where each
point in the original dataset becomes a splat. When radius > 0, it creates a Gaussian
point cloud with the specified radius and sharpness for smooth spatial influence.
When radius = 0, it creates a simple point cloud with zero radius (no spatial influence).
This is useful for visualizing and interpolating point-based data.
"""

from logging import getLogger

import warp as wp

import dav
from dav.data_models.custom import gaussian_point_cloud, point_cloud

logger = getLogger(__name__)


@dav.cached(aot="operators.point_splats")
def get_kernel(data_model: dav.DataModel):
    @dav.kernel(module="unique")
    @dav.utils.set_qualname("dav_point_splats_kernel")
    def point_splats(ds: data_model.DatasetHandle, positions: wp.array(dtype=wp.vec3f)):
        pt_idx = wp.tid()
        pt_id = data_model.DatasetAPI.get_point_id_from_idx(ds, pt_idx)
        positions[pt_idx] = data_model.DatasetAPI.get_point(ds, pt_id)

    return point_splats


def compute(dataset: dav.DatasetLike, radius: wp.float32, sharpness: wp.float32) -> dav.DatasetLike:
    """
    Splat field values at points in the dataset using a Gaussian kernel.

    This operator converts any dataset into a point cloud. When radius > 0, it
    creates a Gaussian point cloud where each point has a radius of influence
    and uses an exponential decay kernel for smooth interpolation: w(d) = exp(-f2 * d²)
    where f2 = (sharpness/radius)². When radius = 0, it creates a regular point cloud
    with zero radius (no spatial influence).

    Args:
        dataset: The dataset to splat. Points are extracted from this dataset.
        radius: The radius of influence for each point (must be non-negative).
                When radius = 0, creates a zero-radius point cloud.
                When radius > 0, creates a Gaussian point cloud.
        sharpness: The sharpness parameter controlling kernel decay (must be positive
                   when radius > 0, ignored when radius = 0).
                   Higher values result in faster decay and sharper transitions.

    Returns:
        A point cloud dataset with the splatted points and fields.
        When radius = 0: PointCloudDataModel
        When radius > 0: GaussianPointCloudDataModel

    Raises:
        ValueError: If radius is negative, or if sharpness is not positive when radius > 0.
    """
    device = dataset.device
    nb_points = dataset.get_num_points()

    if radius < 0.0:
        raise ValueError("Radius must be non-negative")
    if radius > 0.0 and sharpness <= 0.0:
        raise ValueError("Sharpness must be positive when radius > 0")

    with dav.scoped_timer("point_splats.allocate_results"):
        positions = wp.zeros((nb_points,), dtype=wp.vec3f, device=device)
    with dav.scoped_timer("point_splats.get_kernel"):
        kernel = get_kernel(dataset.data_model)
    with dav.scoped_timer("point_splats.launch", cuda_filter=wp.TIMING_ALL):
        wp.launch(kernel, dim=nb_points, inputs=[dataset.handle], outputs=[positions], device=device)

    # Create output dataset based on radius
    if radius == 0.0:
        # Create a zero-radius point cloud
        result_dataset = point_cloud.create_dataset(points=positions)
    else:
        # Create a Gaussian point cloud
        ds_handle = gaussian_point_cloud.create_handle(points=positions, radius=radius, sharpness=sharpness)

        min_bounds, max_bounds = dataset.get_bounds()
        resolution = (max_bounds - min_bounds) / radius
        hash_grid = wp.HashGrid(dim_x=int(resolution[0]), dim_y=int(resolution[1]), dim_z=int(resolution[2]), device=device)
        hash_grid.build(positions, radius)

        ds_handle.hash_grid_id = hash_grid.id
        result_dataset = dav.Dataset(data_model=gaussian_point_cloud.DataModel, handle=ds_handle, device=device, hash_grid=hash_grid)
    return result_dataset


if dav.config.compile_kernels_aot:
    from dav.core import aot

    for data_model in aot.get_data_models(specialization="operators.point_splats.dataset"):
        logger.info(f"Compiling kernels for data model: {data_model}")
        kernel = get_kernel(data_model)
        wp.compile_aot_module(kernel.module, device=aot.get_devices())
