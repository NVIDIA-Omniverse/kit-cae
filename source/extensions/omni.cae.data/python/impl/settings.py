# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

__all__ = [
    "get_cache_mode",
    "get_default_max_voxel_grid_resolution",
    "get_downconvert_64bit",
    "get_flow_voxelization_max_blocks",
    "get_streamline_impl",
    "get_voxelization_impl",
    "get_warp_voxelization_batch_size",
    "get_warp_voxelization_radius_factor",
    "is_legacy_stages_enabled",
    "is_legacy_ui_enabled",
    "override_setting",
    "SettingsKeys",
]


from carb.settings import get_settings


class SettingsKeys:
    ENABLE_LEGACY_UI = "/persistent/exts/omni.cae.data/enableLegacyUI"
    ENABLE_LEGACY_STAGES = "/persistent/exts/omni.cae.data/enableLegacyStages"
    CACHE_MODE = "/persistent/exts/omni.cae.data/cacheMode"
    DOWN_CONVERT_64BIT = "/persistent/exts/omni.cae.data/downConvert64Bit"
    VOXELIZATION_IMPL = "/persistent/exts/omni.cae.data/voxelizationImpl"
    WARP_VOXELIZATION_BATCH_SIZE = "/persistent/exts/omni.cae.data/warpVoxelizationBatchSize"
    WARP_VOXELIZATION_RADIUS_FACTOR = "/persistent/exts/omni.cae.data/warpVoxelizationRadiusFactor"
    FLOW_VOXELIZATION_MAX_BLOCKS = "/persistent/exts/omni.cae.data/flowVoxelizationMaxBlocks"
    DEFAULT_MAX_VOXEL_GRID_RESOLUTION = "/persistent/exts/omni.cae.data/defaultMaxVoxelGridResolution"
    STREAMLINE_IMPL = "/persistent/exts/omni.cae.data/streamlinesImpl"

    # Non-persistent warp config overrides (set in kit .toml / launch args, not persisted)
    WARP_SKIP_BLACKWELL_PTX_OVERRIDE = "/exts/omni.cae.data/warp/skipBlackwellPtxOverride"
    WARP_MODE = "/exts/omni.cae.data/warp/mode"
    WARP_VERIFY_FP = "/exts/omni.cae.data/warp/verifyFp"
    WARP_VERIFY_CUDA = "/exts/omni.cae.data/warp/verifyCuda"
    WARP_VERBOSE = "/exts/omni.cae.data/warp/verbose"
    WARP_VERBOSE_WARNINGS = "/exts/omni.cae.data/warp/verboseWarnings"
    WARP_PTX_TARGET_ARCH = "/exts/omni.cae.data/warp/ptxTargetArch"
    WARP_MAX_UNROLL = "/exts/omni.cae.data/warp/maxUnroll"
    WARP_CUDA_OUTPUT = "/exts/omni.cae.data/warp/cudaOutput"


def get_cache_mode() -> str:
    return get_settings().get_as_string(SettingsKeys.CACHE_MODE)


def get_downconvert_64bit() -> bool:
    return get_settings().get_as_bool(SettingsKeys.DOWN_CONVERT_64BIT)


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


def is_legacy_ui_enabled() -> bool:
    """Check if legacy UI elements should be shown (requires restart to take effect)."""
    return get_settings().get_as_bool(SettingsKeys.ENABLE_LEGACY_UI)


def is_legacy_stages_enabled() -> bool:
    """Check if legacy stages support is enabled (requires restart to take effect)."""
    return get_settings().get_as_bool(SettingsKeys.ENABLE_LEGACY_STAGES)


class override_setting:
    """
    Context manager to temporarily override a setting and restore it on exit.

    Parameters
    ----------
    setting_key : str
        The setting key to override (can use SettingsKeys constants)
    value : any
        The value to set

    Usage
    -----
    with override_setting(SettingsKeys.DOWN_CONVERT_64BIT, False):
        # Setting is False here
        assert not get_downconvert_64bit()
    # Setting is restored to original value here

    # Can also be used with string keys
    with override_setting("/persistent/exts/omni.cae.data/downConvert64Bit", False):
        # work with the overridden setting
        pass
    """

    def __init__(self, setting_key: str, value):
        self.setting_key = setting_key
        self.new_value = value
        self.old_value = None
        self.settings = get_settings()

    def __enter__(self):
        # Save the current value
        self.old_value = self.settings.get(self.setting_key)
        # Set the new value
        self.settings.set(self.setting_key, self.new_value)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore the old value
        if self.old_value is not None:
            self.settings.set(self.setting_key, self.old_value)
        else:
            # If there was no previous value, destroy the setting
            self.settings.destroy_item(self.setting_key)
        return False
