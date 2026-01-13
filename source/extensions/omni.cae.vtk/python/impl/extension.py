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

from omni.ext import IExt


class Extension(IExt):
    def on_startup(self, extId):
        from omni.cae.data import get_data_delegate_registry

        from .delegate import VTKDataDelegate

        self._registry = get_data_delegate_registry()
        self._delegate = VTKDataDelegate(extId)
        self._registry.register_data_delegate(self._delegate)

        # register commands
        if kit_commands := self.get_commands():
            from . import commands, index_commands

            kit_commands.register_all_commands_in_module(commands)
            kit_commands.register_all_commands_in_module(index_commands)

    def on_shutdown(self):
        self._registry.deregister_data_delegate(self._delegate)
        del self._delegate

        if kit_commands := self.get_commands():
            from . import commands, index_commands

            kit_commands.unregister_module_commands(commands)
            kit_commands.unregister_module_commands(index_commands)

    def get_commands(self):
        try:
            from omni.kit import commands

            return commands
        except ImportError:
            return None
