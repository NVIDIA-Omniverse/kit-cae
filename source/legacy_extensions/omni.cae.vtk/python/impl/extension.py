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

from omni.ext import IExt

logger = getLogger(__name__)


class Extension(IExt):
    def on_startup(self, extId):
        try:
            import vtkmodules  # noqa: F401 — ensure VTK is available
        except ImportError:
            logger.error(
                "VTK packages not available. VTK operator commands will not function.\n"
                "To install optional dependencies and relaunch:\n"
                "  Linux:   ./repo.sh pip_download\n"
                "  Windows: repo.bat pip_download\n"
                "See docs/Build.md for details."
            )
            self._vtk_available = False
            return

        self._vtk_available = True
        if kit_commands := self.get_commands():
            from . import commands, index_commands

            kit_commands.register_all_commands_in_module(commands)
            kit_commands.register_all_commands_in_module(index_commands)

    def on_shutdown(self):
        if not getattr(self, "_vtk_available", False):
            return

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
