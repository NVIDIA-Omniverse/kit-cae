# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
#  its affiliates is strictly prohibited.

from omni.ext import IExt
from omni.kit.app import get_app

from . import cache


class Extension(IExt):
    def on_startup(self, extId):
        self._extId = extId
        cache._initialize()

        from omni.kit.commands import register_all_commands_in_module

        from . import commands

        register_all_commands_in_module(commands)

        # initialize Warp
        import warp as wp

        wp.init()

        self._preference_page = None
        self._hooks = []
        manager = get_app().get_extension_manager()
        self._hooks.append(
            manager.subscribe_to_extension_enable(
                on_enable_fn=lambda _: self._register_page(),
                on_disable_fn=lambda _: self._unregister_page(),
                ext_name="omni.kit.window.preferences",
                hook_name="omni.cae.data omni.kit.window.preferences listener",
            )
        )

    def on_shutdown(self):
        from omni.kit.commands import unregister_module_commands

        from . import commands

        cache._finalize()

        unregister_module_commands(commands)
        self._unregister_page()

    def _register_page(self):
        from omni.kit.window.preferences import register_page

        from .settings_page import SettingsPage

        self._preference_page = register_page(SettingsPage())

    def _unregister_page(self):
        if self._preference_page is not None:
            from omni.kit.window.preferences import unregister_page

            unregister_page(self._preference_page)
            self._preference_page = None
