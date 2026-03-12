# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import omni.ext


class Extension(omni.ext.IExt):
    """Extension class for the Ensight asset importer."""

    def on_startup(self, _ext_id):
        import omni.kit.tool.asset_importer as ai

        from .importer import EnSightGoldImporter

        self._ensight_importer = EnSightGoldImporter()
        ai.register_importer(self._ensight_importer)

    def on_shutdown(self):
        import omni.kit.tool.asset_importer as ai

        ai.remove_importer(self._ensight_importer)
        self._ensight_importer = None
