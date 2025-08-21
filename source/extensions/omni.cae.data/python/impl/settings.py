# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
#  its affiliates is strictly prohibited.

__all__ = [
    "get_enable_cache",
    "get_enable_intermediate_cache",
    "get_voxelization_impl",
    "get_warp_voxelization_batch_size",
    "get_flow_voxelization_max_blocks",
    "get_default_max_voxel_grid_resolution",
    "get_warp_voxelization_radius_factor",
    "get_streamline_impl",
]


from carb.settings import get_settings


class SettingsKeys:
    ENABLE_CACHE = "/persistent/exts/omni.cae.data/enableCache"
    ENABLE_INTERMEDIATE_CACHE = "/persistent/exts/omni.cae.data/enableIntermediateCache"
    VOXELIZATION_IMPL = "/persistent/exts/omni.cae.data/voxelizationImpl"
    WARP_VOXELIZATION_BATCH_SIZE = "/persistent/exts/omni.cae.data/warpVoxelizationBatchSize"
    WARP_VOXELIZATION_RADIUS_FACTOR = "/persistent/exts/omni.cae.data/warpVoxelizationRadiusFactor"
    FLOW_VOXELIZATION_MAX_BLOCKS = "/persistent/exts/omni.cae.data/flowVoxelizationMaxBlocks"
    DEFAULT_MAX_VOXEL_GRID_RESOLUTION = "/persistent/exts/omni.cae.data/defaultMaxVoxelGridResolution"
    STREAMLINE_IMPL = "/persistent/exts/omni.cae.data/streamlinesImpl"


def get_enable_cache() -> bool:
    return get_settings().get_as_bool(SettingsKeys.ENABLE_CACHE)


def get_enable_intermediate_cache() -> bool:
    return get_settings().get_as_bool(SettingsKeys.ENABLE_INTERMEDIATE_CACHE)


def get_voxelization_impl() -> str:
    return get_settings().get_as_string(SettingsKeys.VOXELIZATION_IMPL)


def get_warp_voxelization_batch_size() -> int:
    return get_settings().get_as_int(SettingsKeys.WARP_VOXELIZATION_BATCH_SIZE)


def get_flow_voxelization_max_blocks() -> int:
    return get_settings().get_as_int(SettingsKeys.FLOW_VOXELIZATION_MAX_BLOCKS)


def get_default_max_voxel_grid_resolution() -> int:
    return max(1, get_settings().get_as_int(SettingsKeys.DEFAULT_MAX_VOXEL_GRID_RESOLUTION))


def get_warp_voxelization_radius_factor() -> float:
    return get_settings().get_as_float(SettingsKeys.WARP_VOXELIZATION_RADIUS_FACTOR)


def get_streamline_impl() -> str:
    return get_settings().get_as_string(SettingsKeys.STREAMLINE_IMPL)
