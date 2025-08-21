# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
#  its affiliates is strictly prohibited.

from logging import getLogger

from omni.ext import IExt

logger = getLogger(__name__)


class Extension(IExt):

    def on_startup(self, extId):
        from omni.kit.commands import register_all_commands_in_module

        from . import commands

        self._modules = [commands]

        try:
            from . import index_commands

            self._modules.append(index_commands)
        except ImportError:
            logger.warning("omni.cae.index is not enabled. Hence skipping enabling SIDS / IndeX support")

        try:
            from . import vtk_commands

            self._modules.append(vtk_commands)
        except ImportError:
            logger.warning("omni.cae.vtk is not enabled. Hence skipping enabling SIDS / VTK support")

        for m in self._modules:
            register_all_commands_in_module(m)

    def on_shutdown(self):
        from omni.kit.commands import unregister_module_commands

        for m in self._modules:
            unregister_module_commands(m)
