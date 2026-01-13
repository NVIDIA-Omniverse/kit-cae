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


import os.path
from logging import getLogger

import omni.client
from omni.cae.algorithms.core import Factory, get_factory
from omni.client.utils import make_file_url_if_possible
from omni.ext import IExt
from omni.kit.app import get_app

from .algorithms import NanoVdbSlice, NanoVdbVolume, Slice, Volume, VolumeSlice

logger = getLogger(__name__)


class Extension(IExt):
    def on_startup(self, extId):
        self._extId = extId
        factory: Factory = get_factory()

        # tell the factory all algorithms we can handle.
        factory.register_create_callback("Scope", ["CaeIndeXSliceAPI"], Slice, self._extId)
        factory.register_create_callback("Scope", ["CaeIndeXNanoVdbSliceAPI"], NanoVdbSlice, self._extId)
        factory.register_create_callback("Scope", ["CaeIndeXVolumeSliceAPI"], VolumeSlice, self._extId)
        factory.register_create_callback("Volume", ["CaeIndeXVolumeAPI"], Volume, self._extId)
        factory.register_create_callback("Volume", ["CaeIndeXNanoVdbVolumeAPI"], NanoVdbVolume, self._extId)

        # register commands
        self.register_commands()

        # add materials path to the omni.client search path
        # so stage can find the XAC materials included in this extension.
        ext_path = get_app().get_extension_manager().get_extension_path(extId)
        materials_path = make_file_url_if_possible(os.path.join(ext_path, "data"))
        omni.client.add_default_search_path(materials_path)
        self._path = materials_path

    def on_shutdown(self):
        self.unregister_commands()

        # cleanup factory
        factory: Factory = get_factory()
        factory.unregister_all(self._extId)

        # Remove the materials path from the omni.client search path.
        omni.client.remove_default_search_path(self._path)

    def get_commands(self):
        try:
            from omni.kit import commands

            return commands
        except ImportError:
            return None

    def register_commands(self):
        if kit_commands := self.get_commands():
            from . import commands

            kit_commands.register_all_commands_in_module(commands)

    def unregister_commands(self):
        if kit_commands := self.get_commands():
            from . import commands

            kit_commands.unregister_module_commands(commands)
