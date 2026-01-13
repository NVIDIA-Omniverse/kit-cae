# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import math
from logging import getLogger

import numpy as np
import warp as wp

from . import array_utils
from .typing import FieldArrayLike

logger = getLogger(__name__)


@wp.func
def get_linear_index(ijk: wp.vec3i, dims: wp.vec3i):
    return (ijk[2] * dims[1] + ijk[1]) * dims[0] + ijk[0]


@wp.kernel
def copy_dense_volume_fortran_to_nano_vdb_f(
    volume: wp.uint64,
    values: wp.array(dtype=wp.float32),
    volume_ijk_min: wp.vec3i,
    values_ijk_min: wp.vec3i,
    values_dims: wp.vec3i,
):
    i, j, k = wp.tid()
    ijk = wp.vec3i(i, j, k) + volume_ijk_min
    in_ijk = ijk - values_ijk_min
    in_idx = get_linear_index(in_ijk, values_dims)
    wp.volume_store_f(volume, ijk[0], ijk[1], ijk[2], values[in_idx])
    # if i == 0 and j == 0 and k == 0:
    #     wp.printf("tid=(%d, %d, %d)\n", i, j, k)
    #     wp.printf("ijk=(%d, %d, %d)\n", ijk[0], ijk[1], ijk[2])


def voxelize(
    data: FieldArrayLike,
    in_extents: tuple[tuple[int, int, int], tuple[int, int, int]],
    out_extents: tuple[tuple[int, int, int], tuple[int, int, int]],
) -> wp.Volume:

    in_extents = np.array(in_extents, dtype=np.int32)
    out_extents = np.array(out_extents, dtype=np.int32)
    logger.info("in_extents: %s", in_extents)
    logger.info("out_extents: %s", out_extents)
    logger.info("data.shape: %s", data.shape)

    min_extents = np.maximum(in_extents[0], out_extents[0])
    max_extents = np.minimum(in_extents[1], out_extents[1])
    logger.info("target_out_extents: %s %s", min_extents, max_extents)

    # validate out_extent; must be subextent.
    if np.any(max_extents < min_extents):
        raise ValueError("out_extents '{out_extents}' must overlap '{in_extents}'")

    in_dims = in_extents[1] - in_extents[0] + 1
    if np.prod(in_dims) != data.shape:
        raise ValueError("input data size mismatch!")

    dims = max_extents - min_extents + 1
    assert np.prod(dims) > 0

    # we use this custom logic instead of `wp.Volume.load_from_numpy` because
    # this handles cases where the volume has min_ext != 0.
    volume = wp.Volume.allocate(
        min=tuple(min_extents), max=tuple(max_extents), voxel_size=1.0, bg_value=0.0, points_in_world_space=False
    )
    wp.launch(
        copy_dense_volume_fortran_to_nano_vdb_f,
        dim=dims.tolist(),
        inputs=[
            volume.id,
            wp.array(data, dtype=wp.float32),
            wp.vec3i(*min_extents),
            wp.vec3i(*in_extents[0]),
            wp.vec3i(*in_dims),
        ],
    )
    return volume
