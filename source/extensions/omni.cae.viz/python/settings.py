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
    "get_default_operator_enabled",
    "get_default_device_for_auto",
    "get_default_bounding_box_device",
    "get_default_bounding_box_use_point_bounds",
]


from carb.settings import get_settings


class SettingsKeys:
    DEFAULT_OPERATOR_ENABLED = "/persistent/exts/omni.cae.viz/defaultOperatorEnabled"
    DEFAULT_DEVICE_FOR_AUTO = "/persistent/exts/omni.cae.viz/defaultDeviceForAuto"
    DEFAULT_BOUNDING_BOX_DEVICE = "/persistent/exts/omni.cae.viz/defaultBoundingBoxDevice"
    DEFAULT_BOUNDING_BOX_USE_POINT_BOUNDS = "/persistent/exts/omni.cae.viz/defaultBoundingBoxUsePointBounds"


def get_default_operator_enabled() -> bool:
    """Get whether operators should be created enabled by default."""
    return get_settings().get_as_bool(SettingsKeys.DEFAULT_OPERATOR_ENABLED)


def get_default_device_for_auto() -> str:
    """Get the default device to use when device is set to 'auto'."""
    return get_settings().get_as_string(SettingsKeys.DEFAULT_DEVICE_FOR_AUTO)


def get_default_bounding_box_device() -> str:
    """Get the default device to use when computing bounding boxes."""
    return get_settings().get_as_string(SettingsKeys.DEFAULT_BOUNDING_BOX_DEVICE)


def get_default_bounding_box_use_point_bounds() -> bool:
    """Get whether to always use point bounds for bounding box computation."""
    return get_settings().get_as_bool(SettingsKeys.DEFAULT_BOUNDING_BOX_USE_POINT_BOUNDS)
