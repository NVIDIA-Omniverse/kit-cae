# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
#  its affiliates is strictly prohibited.

__all__ = ["voxelize"]

from logging import getLogger

import numpy as np
import warp as wp
from numpy.typing import ArrayLike
from omni.cae.data import IFieldArray

logger = getLogger(__name__)

SPH_CUBIC = 101
SPH_QUARTIC = 102
SPH_QUINTIC = 103
WENDLAND_QUINTIC = 104


@wp.struct
class SPHKernel:
    kind: wp.int32
    spatial_step: wp.float32  # also know as smoothing length
    dimension: wp.int32  # spatial dimension of the kernel
    cutoff_factor: wp.float32  # varies across each kernel, e.g. cubic=2, quartic=2.5, quintic=3
    cutoff: wp.float32  # the spatial step * cutoff factor
    sigma: wp.float32  # normalization constant
    dist_norm: wp.float32  # distance normalization factor 1/(spatial step)
    norm_factor: wp.float32  # dimensional normalization factor sigma/(spatial step)^Dimension
    default_volume: wp.float32  # if mass and density arrays not specified, use this

    @staticmethod
    def create(radius: float, kind: int):
        k = SPHKernel()
        k.kind = kind
        k.spatial_step = wp.float32(radius)
        k.dimension = wp.int32(3)
        if kind == SPH_CUBIC:
            k.cutoff_factor = wp.float32(2.0)
            k.sigma = wp.float32(1.0 / np.pi)
        elif kind == SPH_QUARTIC:
            k.cutoff_factor = wp.float32(2.5)
            k.sigma = wp.float32(1.0 / (20.0 * np.pi))
        elif kind == SPH_QUINTIC:
            k.cutoff_factor = wp.float32(3)
            k.sigma = wp.float32(1.0 / (120 * np.pi))
        elif kind == WENDLAND_QUINTIC:
            k.cutoff_factor = wp.float32(2.0)
            k.sigma = wp.float32(21.0 / (16.0 * np.pi))
        else:
            raise RuntimeError(f"Unknown kind {kind}")
        k.cutoff = k.cutoff_factor * k.spatial_step
        k.dist_norm = wp.float32(1.0 / k.spatial_step)
        k.norm_factor = wp.float32(k.sigma * np.power(k.dist_norm, k.dimension))
        k.default_volume = wp.float32(np.power(k.spatial_step, k.dimension))
        logger.info(
            f"""SPH Kernel:
    kind = {k.kind}
    spatial_step = {k.spatial_step}
    dimension = {k.dimension}
    cutoff_factor = {k.cutoff_factor}
    cutoff = {k.cutoff}
    sigma = {k.sigma}
    dist_norm = {k.dist_norm}
    norm_factor = {k.norm_factor}
    default_volume = {k.default_volume}
"""
        )
        return k


@wp.func
def compute_sph_function_weight(t: wp.int32, d: wp.float32) -> wp.float32:
    """
    Compute weighting factor given a normalized distance from a sample point.
    """
    if t == SPH_CUBIC:
        tmp1 = wp.float32(2.0 - wp.min(d, 2.0))
        tmp2 = wp.float32(1.0 - wp.min(d, 1.0))
        return 0.25 * tmp1 * tmp1 * tmp1 - tmp2 * tmp2 * tmp2
    elif t == SPH_QUARTIC:
        tmp1 = wp.float32(2.5 - wp.min(d, 2.5))
        tmp2 = wp.float32(1.5 - wp.min(d, 1.5))
        tmp3 = wp.float32(0.5 - wp.min(d, 0.5))
        return wp.pow(tmp1, 4.0) - 5.0 * wp.pow(tmp2, 4.0) + 10.0 * wp.pow(tmp3, 4.0)
    elif t == SPH_QUINTIC:
        tmp1 = wp.float32(3.0 - wp.min(d, 3.0))
        tmp2 = wp.float32(2.0 - wp.min(d, 2.0))
        tmp3 = wp.float32(1.0 - wp.min(d, 1.0))
        return wp.pow(tmp1, 5.0) - 6.0 * wp.pow(tmp2, 5.0) + 15.0 * wp.pow(tmp3, 5.0)
    elif t == WENDLAND_QUINTIC:
        if d >= 2.0:
            return wp.float32(0.0)
        else:
            tmp = wp.float32(1.0 - 0.5 * d)
            return wp.pow(tmp, 4.0) * (1.0 + 2.0 * d)
    return wp.float32(0.0)


@wp.kernel
def voxelize_sph_f(
    tiles: wp.array2d(dtype=wp.int32),
    hgrid: wp.uint64,
    coords: wp.array(dtype=wp.vec3f),
    field: wp.array(dtype=wp.float32),
    sph_kernel: SPHKernel,
    spacing: wp.float32,
    ovol: wp.uint64,
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
    value = wp.float32(0)
    query = wp.hash_grid_query(hgrid, p, sph_kernel.cutoff)
    while wp.hash_grid_query_next(query, index):
        neighbor = coords[index]
        d = wp.length(p - neighbor)
        if d <= sph_kernel.cutoff:
            KW = compute_sph_function_weight(sph_kernel.kind, d * sph_kernel.dist_norm)
            weight = sph_kernel.norm_factor * KW * sph_kernel.default_volume
            value += weight * field[index]
    wp.volume_store_f(ovol, i, j, k, value)


@wp.kernel
def voxelize_sph_vec3f(
    tiles: wp.array2d(dtype=wp.int32),
    hgrid: wp.uint64,
    coords: wp.array(dtype=wp.vec3f),
    field: wp.array(dtype=wp.vec3f),
    sph_kernel: SPHKernel,
    spacing: wp.float32,
    ovol: wp.uint64,
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
    value = wp.vec3f(0.0, 0.0, 0.0)
    query = wp.hash_grid_query(hgrid, p, sph_kernel.cutoff)
    while wp.hash_grid_query_next(query, index):
        neighbor = coords[index]
        d = wp.length(p - neighbor)
        if d <= sph_kernel.cutoff:
            KW = compute_sph_function_weight(sph_kernel.kind, d * sph_kernel.dist_norm)
            weight = sph_kernel.norm_factor * KW * sph_kernel.default_volume
            f = field[index]
            value += wp.vec3f(weight * f.x, weight * f.y, weight * f.z)
    wp.volume_store_v(ovol, i, j, k, value)


def voxelize_sph(
    coords: ArrayLike,
    field: ArrayLike,
    extents: tuple[tuple[int, int, int], tuple[int, int, int]],
    voxel_size: float,
    radius_factor: float,
) -> wp.Volume:
    """
    Initially, we're accepting numpy arrays. Eventually, this will work with
    numpy.ndarray or cupy.ndarray.
    """

    # if coords.ndim != 2 or coords.shape[1] != 3:
    #     raise RuntimeError(f"Unsupported coords array dim ({coords.ndim}) or shape ({coords.shape})")

    extents = np.array(extents)
    resolution = (extents[1] - extents[0]) + 1

    if np.any(resolution <= 5):
        raise RuntimeError(f"Too small resolution ({resolution})")

    radius = radius_factor * voxel_size
    kernel = SPHKernel.create(radius, SPH_QUINTIC)

    wp_coords = wp.array(coords, wp.vec3f, copy=False)
    igrid = wp.HashGrid(dim_x=resolution[0], dim_y=resolution[1], dim_z=resolution[2])
    igrid.build(points=wp_coords, radius=kernel.cutoff)

    if field.ndim == 1 or field.ndim == 2 and field.shape[1] == 1:
        wp_field = wp.array(field, dtype=wp.float32)
        # acc_kernel = accumulate_f
        vox_kernel = voxelize_sph_f
        bg_value = 0.0
    elif field.ndim == 2 and field.shape[1] == 3:
        wp_field = wp.array(field, dtype=wp.vec3f)
        # acc_kernel = accumulate_vec3f
        vox_kernel = voxelize_sph_vec3f
        bg_value = [0.0, 0.0, 0.0]
    else:
        logger.error("Unsupported array with shape %s", str(field.shape))
        return None

    ovol = wp.Volume.allocate(
        min=extents[0].tolist(),
        max=extents[1].tolist(),
        voxel_size=voxel_size,
        bg_value=bg_value,
        points_in_world_space=False,
    )
    tiles = ovol.get_tiles()
    wp.launch(
        vox_kernel, (8, 8, 8, len(tiles)), inputs=[tiles, igrid.id, wp_coords, wp_field, kernel, voxel_size, ovol.id]
    )
    return ovol


def voxelize(
    coords: list[IFieldArray],
    field: IFieldArray,
    extents: tuple[tuple[int, int, int], tuple[int, int, int]],
    voxel_size: float,
    radius_factor: float,
    deviceId: int,
) -> IFieldArray:
    with wp.ScopedDevice(device=wp.get_cuda_device(deviceId)):
        if len(coords) == 3:
            coords = np.vstack(coords).T
        else:
            coords = coords[0]
        result = voxelize_sph(coords, field, extents, voxel_size, radius_factor)
    if not result:
        return None
    return IFieldArray.from_numpy(result.array().numpy().view(np.uint64))
