# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import carb
import omni.ui as ui
from omni.kit.window.preferences import PreferenceBuilder


class SettingsPage(PreferenceBuilder):
    def __init__(self):
        super().__init__("CAE")
        self._save_path_model = ui.SimpleStringModel("aot_config.json")

    def build(self):
        with ui.VStack(height=0):
            with self.add_frame("AOT Configuration Recorder (omni.cae.dav)"):
                with ui.VStack(spacing=4):
                    self.label(
                        "Export the current recorder state (all compiled kernel specializations) to a JSON file."
                    )
                    self.spacer()
                    with ui.HStack(height=0, spacing=4):
                        ui.Label("Output Path", width=ui.Pixel(90), height=ui.Pixel(22))
                        ui.StringField(model=self._save_path_model, height=ui.Pixel(22))
                    self.spacer()
                    ui.Button(
                        "Save AOT Config",
                        height=ui.Pixel(28),
                        clicked_fn=self._on_save_clicked,
                        tooltip="Save the current recorder state to the specified JSON file.",
                    )

    def _on_save_clicked(self):
        path = self._save_path_model.as_string.strip()
        if not path:
            carb.log_warn("[omni.cae.dav] AOT config save path is empty — please enter a file path.")
            return
        try:
            from dav.core.recorder import save_config_from_cache

            save_config_from_cache(path)
            carb.log_info(f"[omni.cae.dav] AOT config saved to: {path}")
        except Exception as e:
            carb.log_error(f"[omni.cae.dav] Failed to save AOT config to '{path}': {e}")
