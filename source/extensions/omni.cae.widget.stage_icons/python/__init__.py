# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

__all__ = ["Extension"]

from pathlib import Path

from carb.settings import get_settings
from omni.ext import IExt


class Extension(IExt):

    def on_startup(self, ext_id):
        from omni.kit.widget.stage import StageIcons

        self._prim_types = []
        stage_icons = StageIcons()

        current_path = Path(__file__).parent
        icon_path = current_path.parent.parent.parent.parent.joinpath("icons")

        style = get_settings().get_as_string("/persistent/app/window/uiStyle") or "NvidiaDark"

        # Read all the svg files in the directory
        icons = {icon.stem: str(icon) for icon in icon_path.joinpath(style).glob("*.svg")}
        for prim_type, filename in icons.items():
            stage_icons.set(prim_type, filename)
            self._prim_types.append(prim_type)

    def on_shutdown(self):
        from omni.kit.widget.stage import StageIcons

        stage_icons = StageIcons()
        for prim_type in self._prim_types:
            stage_icons.set(prim_type, None)
