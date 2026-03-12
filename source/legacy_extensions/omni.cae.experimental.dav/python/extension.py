# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from logging import getLogger

import omni.ext
from omni.kit.commands import register_all_commands_in_module, unregister_module_commands

from . import commands

logger = getLogger(__name__)


class Extension(omni.ext.IExt):
    def on_startup(self, ext_id):
        try:
            import dav
        except ImportError:
            logger.error(
                "[omni.cae.experimental.dav] Failed to import dav module. " "Make sure the dav package is installed."
            )
            return

        register_all_commands_in_module(commands)

    def on_shutdown(self):
        unregister_module_commands(commands)
