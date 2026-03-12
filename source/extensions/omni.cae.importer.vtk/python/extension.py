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

import omni.kit.tool.asset_importer as ai
from omni.ext import IExt

logger = getLogger(__name__)


class Extension(IExt):
    def on_startup(self, ext_id):
        try:
            from .importers import VTKImporter
        except ImportError:
            logger.error(
                "VTK packages not available. VTK file importer will not function.\n"
                "To install optional dependencies and relaunch:\n"
                "  Linux:   ./repo.sh pip_download\n"
                "  Windows: repo.bat pip_download\n"
                "See docs/Build.md for details."
            )
            self._importers = []
            return

        self._importers = [VTKImporter()]
        for importer in self._importers:
            ai.register_importer(importer)

    def on_shutdown(self):
        for importer in self._importers:
            ai.remove_importer(importer)
