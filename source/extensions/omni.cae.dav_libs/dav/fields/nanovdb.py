# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""NanoVDB-based field implementation for volume data.

This module provides field models for NanoVDB volume data using wp.Volume.
It supports float32 and vec3f data types with both 'ij' (Fortran) and 'xy' (Cartesian) indexing.
"""

__all__ = ["get_field_model", "allocate_nanovdb_volume"]


import warp as wp

import dav


def allocate_nanovdb_volume(min_ijk: wp.vec3i, max_ijk: wp.vec3i, voxel_size: wp.vec3f, bg_value, device, translation: wp.vec3f = wp.vec3f(0.0)) -> wp.Volume:
    """Allocate a NanoVDB volume covering the given ijk extent.

    Uses tile-based allocation (faster than ``wp.Volume.allocate`` for large volumes).

    Args:
        min_ijk: Minimum ijk coordinates (inclusive).
        max_ijk: Maximum ijk coordinates (inclusive).
        voxel_size: Voxel size in world space.
        bg_value: Background fill value (must match the volume dtype).
        device: Warp device to allocate on.
        translation: Index-to-world translation for the NanoVDB transform.

    Returns:
        wp.Volume: Allocated NanoVDB volume.
    """
    import numpy as np

    tile_min = min_ijk / 8  # integer division via wp.vec3i
    tile_max = max_ijk / 8
    tiles_shape = tile_max - tile_min + wp.vec3i(1, 1, 1)
    tiles = wp.array((np.indices((tiles_shape[0], tiles_shape[1], tiles_shape[2])).reshape(3, -1).T + [tile_min[0], tile_min[1], tile_min[2]]) * 8, dtype=wp.vec3i, device=device)
    return wp.Volume.allocate_by_tiles(tiles, voxel_size=voxel_size, bg_value=bg_value, translation=translation, device=device)


@wp.struct
class FieldHandleNanoVDB:
    """Field containing NanoVDB volume data.

    This handle type is shared across all NanoVDB field types since
    volume_id is storage-type agnostic.
    """

    association: wp.int32
    volume_id: wp.uint64
    dims: wp.vec3i  # Volume dimensions for index to ijk conversion
    origin: wp.vec3i  # Volume origin (Ni, Nj, Nk)


@dav.func
def _volume_lookup_f(volume_id: wp.uint64, i: wp.int32, j: wp.int32, k: wp.int32) -> wp.float32:
    return wp.volume_lookup_f(volume_id, i, j, k)


@dav.func
def _volume_store_f(volume_id: wp.uint64, i: wp.int32, j: wp.int32, k: wp.int32, value: wp.float32):
    wp.volume_store_f(volume_id, i, j, k, value)


@dav.func
def _volume_lookup_v(volume_id: wp.uint64, i: wp.int32, j: wp.int32, k: wp.int32) -> wp.vec3f:
    return wp.volume_lookup_v(volume_id, i, j, k)


@dav.func
def _volume_store_v(volume_id: wp.uint64, i: wp.int32, j: wp.int32, k: wp.int32, value: wp.vec3f):
    wp.volume_store_v(volume_id, i, j, k, value)


# Standalone helper defined before FieldAPINanoVDB so that get/set can call it
# without referencing FieldAPINanoVDB mid-class-body (empty closure cell).
@dav.func
def _idx_to_ijk_ij(idx: wp.int32, dims: wp.vec3i, origin: wp.vec3i) -> wp.vec3i:
    """Convert linear index to (i, j, k) coordinates for 'ij' (Fortran/column-major) indexing.

    In Fortran order, the FIRST dimension varies fastest:
    idx = i + j * Ni + k * Ni * Nj
    """
    k = idx // (dims[0] * dims[1])
    remainder = idx % (dims[0] * dims[1])
    j = remainder // dims[0]
    i = remainder % dims[0]
    return wp.vec3i(i, j, k) + origin


@dav.cached
def get_field_model(dtype) -> dav.FieldModel:
    """Get a FieldModel for NanoVDB volumes.

    Args:
        dtype: Data type (wp.float32 or wp.vec3f)

    Returns:
        dav.FieldModel with FieldHandle and FieldAPI types

    Raises:
        ValueError: If dtype is not supported

    Example:
        >>> model = get_field_model(wp.float32)
        >>> # Use model.FieldHandle and model.FieldAPI in kernels
    """
    if dtype not in [wp.float32, wp.vec3f]:
        raise ValueError(f"Unsupported dtype: {dtype}. NanoVDB only supports wp.float32 and wp.vec3f")

    # this can be updated once we require Warp 1.10 or later
    if dtype == wp.float32:
        volume_lookup = _volume_lookup_f
        volume_store = _volume_store_f

    else:  # dtype == wp.vec3f
        assert dtype == wp.vec3f, "Logic error: expected dtype to be wp.vec3f if not wp.float32"
        volume_lookup = _volume_lookup_v
        volume_store = _volume_store_v

    class FieldAPINanoVDB:
        @staticmethod
        @dav.func
        def get_association(field: FieldHandleNanoVDB) -> wp.int32:
            """Get the association type of the field."""
            return field.association

        @staticmethod
        @dav.func
        def get_count(field: FieldHandleNanoVDB) -> wp.int32:
            """Get the number of elements in the field."""
            return field.dims[0] * field.dims[1] * field.dims[2]

        @staticmethod
        @dav.func
        def get(field: FieldHandleNanoVDB, idx: wp.int32) -> dtype:
            """Get value at linear index."""
            ijk = _idx_to_ijk_ij(idx, field.dims, field.origin)

            # this can be updated once we require Warp 1.10 or later
            return volume_lookup(field.volume_id, ijk[0], ijk[1], ijk[2])

        @staticmethod
        @dav.func
        def set(field: FieldHandleNanoVDB, idx: wp.int32, value: dtype):
            """Set value at linear index."""
            ijk = _idx_to_ijk_ij(idx, field.dims, field.origin)
            volume_store(field.volume_id, ijk[0], ijk[1], ijk[2], value)

        @staticmethod
        @dav.func
        def zero() -> dtype:
            """Get a zero value of the appropriate type."""
            if wp.static(dtype == wp.float32):
                return wp.float32(0.0)
            elif wp.static(dtype == wp.vec3f):
                return wp.vec3f(0.0, 0.0, 0.0)

        @staticmethod
        @dav.func
        def zero_s() -> wp.float32:
            """Get a zero scalar value of the appropriate type."""
            return wp.float32(0.0)

        @staticmethod
        @dav.func
        def zero_vec3() -> wp.vec3f:
            """Get a zero vec3 value of the appropriate type."""
            return wp.vec3f(0.0, 0.0, 0.0)

    class FieldModelMeta(type):
        def __repr__(cls):
            return f"FieldModelNanoVDB(dtype={dtype.__name__})"

    class FieldModel(metaclass=FieldModelMeta):
        FieldHandle = FieldHandleNanoVDB
        FieldAPI = FieldAPINanoVDB

    return FieldModel
