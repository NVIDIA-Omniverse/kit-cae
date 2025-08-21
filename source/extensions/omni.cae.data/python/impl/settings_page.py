# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
#  its affiliates is strictly prohibited.

import omni.ui as ui
from omni.kit.window.preferences import PreferenceBuilder, SettingType

from .settings import SettingsKeys


class SettingsPage(PreferenceBuilder):
    def __init__(self):
        super().__init__("CAE")

        # def on_change(item, event_type):
        #     if event_type == carb.settings.ChangeEventType.CHANGED:
        #         omni.kit.window.preferences.rebuild_pages()

        # self._subscriptions = []
        # self._subscriptions.append(omni.kit.app.SettingChangeSubscription(
        #     PERSISTENT_SETTINGS_PREFIX + "/exts/omni.cae.data/enableCache", on_change))

    def build(self):
        with ui.VStack(height=0):
            with self.add_frame("CAE Data Delegate (omni.cae.data)"):
                with ui.VStack():
                    self.create_setting_widget(
                        "Enable Cache",
                        SettingsKeys.ENABLE_CACHE,
                        SettingType.BOOL,
                        tooltip="Enable caching of input data.",
                    )
                    self.create_setting_widget(
                        "Enable Intermediate Results Cache",
                        SettingsKeys.ENABLE_INTERMEDIATE_CACHE,
                        SettingType.BOOL,
                        tooltip="Enable caching of intermediate results.",
                    )
            self.spacer()
            with self.add_frame("Voxelization (omni.cae.data)"):
                with ui.VStack():
                    self.label("Changes to any of these options will only impact the next voxelization request.")
                    self.spacer()
                    self.create_setting_widget_combo(
                        "Voxelization Type",
                        SettingsKeys.VOXELIZATION_IMPL,
                        ["GaussianWarp", "Flow"],
                        setting_is_index=False,
                        tooltip="Voxelization implementation to use.",
                    )
                    self.create_setting_widget(
                        "Default max voxel grid resolution",
                        SettingsKeys.DEFAULT_MAX_VOXEL_GRID_RESOLUTION,
                        SettingType.INT,
                        tooltip="Default voxel grid resolution to use when creating new prims that will use voxelization.",
                    )
                    self.spacer()
                    self.create_setting_widget(
                        "Flow Voxelization Max Blocks", SettingsKeys.FLOW_VOXELIZATION_MAX_BLOCKS, SettingType.INT
                    )
                    self.spacer()
                    self.create_setting_widget(
                        "Warp Voxelization Batch Size",
                        SettingsKeys.WARP_VOXELIZATION_BATCH_SIZE,
                        SettingType.INT,
                        tooltip="Batch size for voxelization when using Warp implementations.",
                    )
                    self.create_setting_widget(
                        "Warp Voxelization Radius Factor",
                        SettingsKeys.WARP_VOXELIZATION_RADIUS_FACTOR,
                        SettingType.FLOAT,
                        tooltip="Radius factor for voxelization when using Warp implementations.",
                    )
            self.spacer()
            with self.add_frame("Advection (omni.cae.data)"):
                with ui.VStack():
                    self.label("Changes to any of these options will only impact the next streamline request.")
                    self.spacer()
                    self.create_setting_widget_combo(
                        "Streamline Type",
                        SettingsKeys.STREAMLINE_IMPL,
                        ["VTK", "Warp"],
                        setting_is_index=False,
                        tooltip="Streamline implementation to use.",
                    )
