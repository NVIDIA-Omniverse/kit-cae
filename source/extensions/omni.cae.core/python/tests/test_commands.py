# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

import omni.kit.test
from omni.cae.core import commands
from pxr import Usd


class TestCommands(omni.kit.test.AsyncTestCase):
    def test_type_names_exclude_tf_root_but_preserve_schema_ancestors(self):
        stage = Usd.Stage.CreateInMemory()
        prim = stage.DefinePrim("/LegacyDataset", "CaeDataSet")

        type_names = commands._get_type_names(prim)

        self.assertIn("OmniCaeDataSet", type_names)
        self.assertIn("UsdTyped", type_names)
        self.assertNotIn("TfType::_Root", type_names)

    async def test_missing_legacy_dataset_command_reports_candidates_and_bundle_hint(self):
        stage = Usd.Stage.CreateInMemory()
        prim = stage.DefinePrim("/LegacyDataset", "CaeDataSet")

        with self.assertRaises(NotImplementedError) as context:
            await commands.execute("MissingCommand", prim)

        diagnostic = str(context.exception)
        self.assertRegex(diagnostic, r"CaeDataSetMissingCommand.*omni\.cae\.legacy\.bundle")
        self.assertNotIn("TfType::_Root", diagnostic)
