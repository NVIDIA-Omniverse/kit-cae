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

This module computes the spatial bounds of a dataset.

"""

from logging import getLogger

import dav
import warp as wp

logger = getLogger(__name__)


@dav.cached
def get_kernel(data_model: dav.DataModel):
    @wp.kernel(enable_backward=False)
    def bounds(
        ds: data_model.DatasetHandle, min_bounds: wp.array(dtype=wp.vec3f), max_bounds: wp.array(dtype=wp.vec3f)
    ):
        point_idx = wp.tid()
        point_id = data_model.DatasetAPI.get_point_id_from_idx(ds, point_idx)
        pt = data_model.DatasetAPI.get_point(ds, point_id)

        # Use atomic min/max operations with vector values
        wp.atomic_min(min_bounds, 0, pt)
        wp.atomic_max(max_bounds, 0, pt)

    return bounds


def compute(
    dataset: dav.DatasetLike, min_field_name: str = "bounds_min", max_field_name: str = "bounds_max"
) -> dav.DatasetLike:
    """
    Compute the spatial bounds of the dataset.

    Args:
        dataset (dav.DatasetLike): The dataset containing points.
        min_field_name (str): Name for the minimum bounds field (default: "bounds_min").
        max_field_name (str): Name for the maximum bounds field (default: "bounds_max").

    Returns:
        dav.DatasetLike: A new dataset (shallow copy) containing the computed bounds fields.
                 Each field is a single-element vec3f array with NOT_SPECIFIED association.
    """
    device = dataset.device
    num_points = dataset.get_num_points()

    # Initialize bounds arrays with extreme values using warp constants
    min_bounds = wp.array(
        [wp.float32(3.4028235e38), wp.float32(3.4028235e38), wp.float32(3.4028235e38)], dtype=wp.vec3f, device=device
    )
    max_bounds = wp.array(
        [wp.float32(-3.4028235e38), wp.float32(-3.4028235e38), wp.float32(-3.4028235e38)], dtype=wp.vec3f, device=device
    )

    # Launch kernel to compute bounds using atomic operations
    with dav.scoped_timer("bounds.get_kernel"):
        kernel = get_kernel(dataset.data_model)
    with dav.scoped_timer("bounds.launch", cuda_filter=wp.TIMING_ALL):
        wp.launch(kernel, dim=num_points, inputs=[dataset.handle, min_bounds, max_bounds], device=device)

    # Create fields from arrays (single-element arrays for dataset-level data)
    min_field = dav.Field.from_array(min_bounds, dav.AssociationType.NOT_SPECIFIED)
    max_field = dav.Field.from_array(max_bounds, dav.AssociationType.NOT_SPECIFIED)

    # Create shallow copy and add fields
    result = dataset.shallow_copy()
    result.add_field(min_field_name, min_field)
    result.add_field(max_field_name, max_field)

    return result
