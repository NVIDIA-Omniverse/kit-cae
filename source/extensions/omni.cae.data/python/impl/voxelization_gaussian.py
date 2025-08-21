# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
#  its affiliates is strictly prohibited.

from logging import getLogger

import numpy as np
import warp as wp
from carb.settings import get_settings
from numpy.typing import ArrayLike
from omni.cae.data import Range3i

logger = getLogger(__name__)


def get_batch_size() -> int:
    settings = get_settings()
    key = "/persistent/exts/omni.cae.data/warpVoxelizationBatchSize"
    value = settings.get_as_int(key)
    logger.info("Using batch size (specified using setting '%s') of %d", key, value)
    return value


@wp.struct
class GaussianKernel:
    radius: float
    f2: float

    @staticmethod
    def create(radius: float, sharpness: float):
        k = GaussianKernel()
        k.radius = radius
        k.f2 = (sharpness / radius) ** 2
        return k


@wp.kernel
def gaussian_f(
    tiles: wp.array2d(dtype=wp.int32),
    hgrid: wp.uint64,
    coords: wp.array(dtype=wp.vec3f),
    field: wp.array(dtype=wp.float32),
    kernel: GaussianKernel,
    spacing: wp.float32,
    ovol: wp.uint64,
    osumvol: wp.uint64,
    iter: wp.int32,
):
    ti, tj, tk, t = wp.tid()
    i = ti + tiles[t][0]
    j = tj + tiles[t][1]
    k = tk + tiles[t][2]

    f_ijk = wp.vec3f(float(i), float(j), float(k))

    # get world-coordinate for the current volume pos
    # p = wp.volume_index_to_world(ovol, f_ijk)
    p = f_ijk * spacing

    index = int(0)

    if iter == 0:
        sum = wp.float32(0)
        value = wp.float32(0)
    else:
        sum = wp.volume_lookup_f(osumvol, i, j, k)
        value = wp.volume_lookup_f(ovol, i, j, k)

    query = wp.hash_grid_query(hgrid, p, kernel.radius)
    while wp.hash_grid_query_next(query, index):
        neighbor = coords[index]
        d = wp.length(p - neighbor)
        if d <= kernel.radius:
            d2 = d * d
            w = wp.exp(-1.0 * kernel.f2 * d2)
            value += w * field[index]
            sum = sum + w
    wp.volume_store_f(ovol, i, j, k, value)
    wp.volume_store_f(osumvol, i, j, k, sum)


@wp.kernel
def normalize_f(tiles: wp.array2d(dtype=wp.int32), ovol: wp.uint64, osumvol: wp.uint64):
    ti, tj, tk, t = wp.tid()
    i = ti + tiles[t][0]
    j = tj + tiles[t][1]
    k = tk + tiles[t][2]

    s = wp.volume_lookup_f(osumvol, i, j, k)
    if s != 0.0:
        v = wp.volume_lookup_f(ovol, i, j, k)
        v = v / s
        wp.volume_store_f(ovol, i, j, k, v)


@wp.kernel
def gaussian_vec3f(
    tiles: wp.array2d(dtype=wp.int32),
    hgrid: wp.uint64,
    coords: wp.array(dtype=wp.vec3f),
    field: wp.array(dtype=wp.vec3f),
    kernel: GaussianKernel,
    spacing: wp.float32,
    ovol: wp.uint64,
    osumvol: wp.uint64,
    iter: wp.int32,
):
    ti, tj, tk, t = wp.tid()
    i = ti + tiles[t][0]
    j = tj + tiles[t][1]
    k = tk + tiles[t][2]

    f_ijk = wp.vec3f(float(i), float(j), float(k))

    # get world-coordinate for the current volume pos
    # p = wp.volume_index_to_world(ovol, f_ijk)
    p = f_ijk * spacing

    index = int(0)

    if iter == 0:
        sum = wp.float32(0)
        value = wp.vec3f(0.0, 0.0, 0.0)
    else:
        sum = wp.volume_lookup_f(osumvol, i, j, k)
        value = wp.volume_lookup_v(ovol, i, j, k)

    query = wp.hash_grid_query(hgrid, p, kernel.radius)
    while wp.hash_grid_query_next(query, index):
        neighbor = coords[index]
        d = wp.length(p - neighbor)
        if d <= kernel.radius:
            d2 = d * d
            w = wp.exp(-1.0 * kernel.f2 * d2)
            f = field[index]
            value += wp.vec3f(w * f.x, w * f.y, w * f.z)
            sum = sum + w
    wp.volume_store_v(ovol, i, j, k, value)
    wp.volume_store_f(osumvol, i, j, k, sum)


@wp.kernel
def normalize_vec3f(tiles: wp.array2d(dtype=wp.int32), ovol: wp.uint64, osumvol: wp.uint64):
    ti, tj, tk, t = wp.tid()
    i = ti + tiles[t][0]
    j = tj + tiles[t][1]
    k = tk + tiles[t][2]

    s = wp.volume_lookup_f(osumvol, i, j, k)
    if s != 0.0:
        v = wp.volume_lookup_v(ovol, i, j, k)
        v = wp.vec3f(v.x / s, v.y / s, v.z / s)
        wp.volume_store_v(ovol, i, j, k, v)


def voxelize(
    coords: ArrayLike,
    field: ArrayLike,
    extents: Range3i,
    voxel_size: float,
    radius_factor: float,
    sharpness: float = 2.0,
    batch_size: int = 1000000,
):

    extents = extents.numpy()
    resolution = (extents[1] - extents[0]) + 1

    if np.any(resolution <= 5):
        raise RuntimeError(f"Too small resolution ({resolution})")

    radius = radius_factor * voxel_size
    kernel = GaussianKernel.create(radius, sharpness)

    if field.ndim == 1 or field.ndim == 2 and field.shape[1] == 1:
        wp_field = wp.array(field, dtype=wp.float32)
        bg_value = 0.0
        wp_kernel = gaussian_f
        wp_normalization_kernel = normalize_f
    elif field.ndim == 2 and field.shape[1] == 3:
        wp_field = wp.array(field, dtype=wp.vec3f)
        bg_value = [0.0, 0.0, 0.0]
        wp_kernel = gaussian_vec3f
        wp_normalization_kernel = normalize_vec3f
    else:
        logger.error("Unsupported array with shape %s", str(field.shape))
        return None

    wp_coords = wp.array(coords, wp.vec3f, copy=False)

    ovol = wp.Volume.allocate(
        min=extents[0].tolist(),
        max=extents[1].tolist(),
        voxel_size=voxel_size,
        bg_value=bg_value,
        points_in_world_space=False,
    )
    sumvol = wp.Volume.allocate(
        min=extents[0].tolist(),
        max=extents[1].tolist(),
        voxel_size=voxel_size,
        bg_value=0.0,
        points_in_world_space=False,
    )

    tiles = ovol.get_tiles()

    max_nb_points = batch_size
    for cc, i in enumerate(range(0, wp_coords.shape[0], max_nb_points)):
        s = i
        e = min(i + max_nb_points, wp_coords.shape[0])
        logger.info(f"Voxelizing batch: {cc} of {wp_coords.shape[0] // max_nb_points + 1}, point range: {s:,} : {e:,}")
        igrid = wp.HashGrid(dim_x=resolution[0], dim_y=resolution[1], dim_z=resolution[2])
        igrid.build(points=wp_coords[s:e], radius=kernel.radius)
        wp.launch(
            wp_kernel,
            (8, 8, 8, len(tiles)),
            inputs=[tiles, igrid.id, wp_coords[s:e], wp_field[s:e], kernel, voxel_size, ovol.id, sumvol.id, cc],
        )
        del igrid
        wp.synchronize_device()

    wp.launch(wp_normalization_kernel, dim=(8, 8, 8, len(tiles)), inputs=[tiles, ovol.id, sumvol.id])
    logger.info("voxelization complete")
    return ovol
