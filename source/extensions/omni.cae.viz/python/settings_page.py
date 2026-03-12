# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import omni.ui as ui
import warp as wp
from omni.kit.window.preferences import PreferenceBuilder, SettingType

from .settings import SettingsKeys


def _get_available_device_strings() -> list[str]:
    """Get list of available devices as strings (e.g., 'cpu', 'cuda:0', 'cuda:1')."""
    try:
        devices = wp.get_devices()
        device_strings = []
        for device in devices:
            device_strings.append(str(device))
        return device_strings if device_strings else ["cpu"]
    except Exception:
        # Fallback to cpu if warp is not initialized or fails
        return ["cpu"]


class SettingsPage(PreferenceBuilder):
    def __init__(self):
        super().__init__("CAE")

    def build(self):
        available_devices = _get_available_device_strings()

        with ui.VStack(height=0):
            with self.add_frame("CAE Visualization (omni.cae.viz)"):
                with ui.VStack():
                    self.create_setting_widget(
                        "Default Operator Enabled",
                        SettingsKeys.DEFAULT_OPERATOR_ENABLED,
                        SettingType.BOOL,
                        tooltip="When enabled, operators are created in the enabled state by default. When disabled, operators must be manually enabled after creation.",
                    )
                    self.spacer()
                    self.create_setting_widget_combo(
                        "Default Device for 'auto'",
                        SettingsKeys.DEFAULT_DEVICE_FOR_AUTO,
                        available_devices,
                        setting_is_index=False,
                        tooltip="The device to use when CaeVizOperatorAPI.device is set to 'auto'. Typically 'cuda:0' for GPU or 'cpu' for CPU.",
                    )
            self.spacer()
            with self.add_frame("Bounding Box Computation (omni.cae.viz)"):
                with ui.VStack():
                    self.label("Settings for bounding box computation when creating visualization primitives.")
                    self.spacer()
                    self.create_setting_widget_combo(
                        "Bounding Box Device",
                        SettingsKeys.DEFAULT_BOUNDING_BOX_DEVICE,
                        available_devices,
                        setting_is_index=False,
                        tooltip="Device to use when computing bounding boxes for datasets. 'cpu' is recommended for most cases.",
                    )
                    self.create_setting_widget(
                        "Always Use Point Bounds",
                        SettingsKeys.DEFAULT_BOUNDING_BOX_USE_POINT_BOUNDS,
                        SettingType.BOOL,
                        tooltip="When enabled, always use point bounds for bounding box computation. When disabled, cell bounds will be used if cells are present in the dataset.",
                    )
