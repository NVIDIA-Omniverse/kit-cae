# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import logging

import warp as wp
from omni.ext import IExt
from omni.kit.app import get_app

logger = logging.getLogger(__name__)


class Extension(IExt):
    def on_startup(self, extId):
        from omni.kit.commands import register_all_commands_in_module

        from . import cache, commands

        self._extId = extId
        cache._initialize()

        register_all_commands_in_module(commands)

        # initialize Warp
        self.initialize_warp()

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

        from . import cache, commands

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

    def initialize_warp(self):
        from carb.settings import get_settings

        from .settings import SettingsKeys

        # Required for DAV.
        wp.config.enable_vector_component_overwrites = True
        wp.init()

        settings = get_settings()

        cuda_device = wp.get_cuda_device()
        cuda_arch = cuda_device.arch
        if cuda_arch >= 100:
            # CUDA 12.* currently used by Warp-Kit is too slow compiling kernels.
            # Can be suppressed via /exts/omni.cae.data/warp/skipBlackwellPtxOverride
            if not settings.get_as_bool(SettingsKeys.WARP_SKIP_BLACKWELL_PTX_OVERRIDE):
                logger.info(
                    "Blackwell GPU detected (arch %d): forcing wp.config.cuda_output='ptx', ptx_target_arch=90",
                    cuda_arch,
                )
                wp.config.cuda_output = "ptx"
                wp.config.ptx_target_arch = 90
            else:
                logger.info(
                    "Blackwell GPU detected (arch %d): skipping PTX override (skipBlackwellPtxOverride is set)",
                    cuda_arch,
                )

        # Apply optional non-persistent warp config overrides from settings.
        # Set e.g. /exts/omni.cae.data/warp/verbose = true in a .kit file to
        # pass that value through to wp.config.verbose (and so on for each key).
        _WARP_CONFIG_SETTINGS = (
            (SettingsKeys.WARP_MODE, "mode"),
            (SettingsKeys.WARP_VERIFY_FP, "verify_fp"),
            (SettingsKeys.WARP_VERIFY_CUDA, "verify_cuda"),
            (SettingsKeys.WARP_VERBOSE, "verbose"),
            (SettingsKeys.WARP_VERBOSE_WARNINGS, "verbose_warnings"),
            (SettingsKeys.WARP_PTX_TARGET_ARCH, "ptx_target_arch"),
            (SettingsKeys.WARP_MAX_UNROLL, "max_unroll"),
            (SettingsKeys.WARP_CUDA_OUTPUT, "cuda_output"),
        )
        for setting_key, config_attr in _WARP_CONFIG_SETTINGS:
            value = settings.get(setting_key)
            if value is not None:
                logger.info(
                    "Applying warp config override: wp.config.%s = %r (from %s)", config_attr, value, setting_key
                )
                setattr(wp.config, config_attr, value)
