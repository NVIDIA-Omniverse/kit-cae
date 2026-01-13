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


from logging import getLogger

from omni.cae.algorithms.core import Factory, get_factory
from omni.ext import IExt

from .algorithms import DataSetEmitter

logger = getLogger(__name__)


class Extension(IExt):
    def on_startup(self, extId):
        self._extId = extId
        factory: Factory = get_factory()

        # tell the factory all algorithms we can handle.
        factory.register_create_callback(
            "FlowEmitterNanoVdb", ["CaeFlowDataSetEmitterAPI"], DataSetEmitter, self._extId
        )

        # register commands
        self.register_commands()

    def on_shutdown(self):
        self.unregister_commands()

        # cleanup factory
        factory: Factory = get_factory()
        factory.unregister_all(self._extId)

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
