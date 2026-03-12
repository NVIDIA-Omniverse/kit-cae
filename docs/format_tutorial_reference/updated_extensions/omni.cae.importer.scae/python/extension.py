# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import omni.ext
import omni.kit.tool.asset_importer as ai

from .importer import ScaeAssetImporter


class Extension(omni.ext.IExt):
    def on_startup(self, ext_id):
        self._importer = ScaeAssetImporter()
        ai.register_importer(self._importer)

    def on_shutdown(self):
        ai.remove_importer(self._importer)
        del self._importer
