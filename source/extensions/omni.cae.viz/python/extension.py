# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
Extension that provides CAE visualization algorithms and utilities.
"""

import os
from logging import getLogger

import omni.ext
from omni.client import add_default_search_path, remove_default_search_path
from omni.client.utils import make_file_url_if_possible
from omni.kit.app import get_app
from omni.kit.commands import register_all_commands_in_module, unregister_module_commands

from . import create_commands, faces, flow_emitters, index_volume, points, streamlines
from .listener import Listener
from .operator import register_module_operators, unregister_module_operators

logger = getLogger(__name__)


class Extension(omni.ext.IExt):
    """Extension class for omni.cae.viz"""

    @staticmethod
    def get_materials_path(extId: str) -> str:
        """
        Return the path to the CAE materials file.

        Returns
        -------
        str
            The path to the CAE materials file.
        """
        # Get the absolute path of the materials directory.
        ext_path = get_app().get_extension_manager().get_extension_path(extId)
        materials_path = os.path.join(ext_path, "material_library")
        if not os.path.exists(materials_path):
            raise RuntimeError(f'Materials path "{materials_path}" not found for extension {extId}')
        return make_file_url_if_possible(materials_path)

    def on_startup(self, extId):
        """Called when the extension is started."""
        logger.info(f"Starting extension {extId}")
        self._extId = extId

        # Create listener for stage events
        self._listener = Listener()
        logger.info("Listener created and subscribed to stage events")

        # Import and register all operator modules
        operator_count = register_module_operators(streamlines)
        operator_count += register_module_operators(index_volume)
        operator_count += register_module_operators(points)
        operator_count += register_module_operators(faces)
        operator_count += register_module_operators(flow_emitters)
        logger.info(f"Registered {operator_count} operators")

        register_all_commands_in_module(create_commands)

        # Add the materials path to the omni.client search path.
        if materials_path := self.get_materials_path(extId):
            add_default_search_path(materials_path)
            self._materials_path = materials_path
        else:
            raise RuntimeError(f"Materials path not found for extension {extId}")

        # Register settings page
        self._preference_page = None
        self._hooks = []
        manager = get_app().get_extension_manager()
        self._hooks.append(
            manager.subscribe_to_extension_enable(
                on_enable_fn=lambda _: self._register_page(),
                on_disable_fn=lambda _: self._unregister_page(),
                ext_name="omni.kit.window.preferences",
                hook_name="omni.cae.viz omni.kit.window.preferences listener",
            )
        )

    def on_shutdown(self):
        """Called when the extension is shutdown."""
        logger.info(f"Shutting down extension {self._extId}")

        if self._materials_path:
            remove_default_search_path(self._materials_path)
            self._materials_path = None

        unregister_module_commands(create_commands)

        # Unregister all operators
        operator_count = unregister_module_operators(streamlines)
        operator_count += unregister_module_operators(index_volume)
        operator_count += unregister_module_operators(points)
        operator_count += unregister_module_operators(faces)
        operator_count += unregister_module_operators(flow_emitters)
        logger.info(f"Unregistered {operator_count} operators")

        # Unregister settings page
        self._unregister_page()

        # Cleanup listener
        if self._listener:
            self._listener.finalize()
            self._listener = None
            logger.info("Listener cleaned up")

    def _register_page(self):
        from omni.kit.window.preferences import register_page

        from .settings_page import SettingsPage

        self._preference_page = register_page(SettingsPage())

    def _unregister_page(self):
        if self._preference_page is not None:
            from omni.kit.window.preferences import unregister_page

            unregister_page(self._preference_page)
            self._preference_page = None
