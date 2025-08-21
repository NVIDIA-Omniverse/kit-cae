# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
#  its affiliates is strictly prohibited.

import omni.kit.tool.asset_importer as ai
from omni.ext import IExt

from .importers import VTKImporter


class Extension(IExt):
    def on_startup(self, ext_id):
        self._importers = []
        self._importers.append(VTKImporter())

        for importer in self._importers:
            ai.register_importer(importer)

    def on_shutdown(self):
        for importer in self._importers:
            ai.remove_importer(importer)
