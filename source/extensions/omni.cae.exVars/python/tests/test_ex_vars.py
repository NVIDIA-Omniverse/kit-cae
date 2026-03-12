# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import carb.settings
import omni.cae.exVars
import omni.kit.test
from omni.cae.testing import get_test_stage_path, new_stage

_SETTINGS_PATH = "/exts/omni.cae.exVars/variables"
_DATA_ROOT_VALUE = "/test/root"


class TestExVars(omni.kit.test.AsyncTestCase):

    async def setUp(self):
        self._settings = carb.settings.get_settings()
        self._settings.set_string(f"{_SETTINGS_PATH}/DATA_ROOT", _DATA_ROOT_VALUE)

    async def tearDown(self):
        self._settings.destroy_item(_SETTINGS_PATH)

    async def test_expression_vars_applied_to_session_layer(self):
        """Expression variables from settings are written to the session layer on stage attach."""
        async with new_stage(get_test_stage_path("ex_vars_test.usda")) as stage:
            session_layer = stage.GetSessionLayer()
            ex_vars = session_layer.expressionVariables

            self.assertIsNotNone(ex_vars)
            self.assertIn("DATA_ROOT", ex_vars)
            self.assertEqual(ex_vars["DATA_ROOT"], _DATA_ROOT_VALUE)

            # Confirm the stage prim's asset attribute uses the expression variable
            prim = stage.GetPrimAtPath("/World/TestPrim")
            self.assertTrue(prim.IsValid())
            attr = prim.GetAttribute("dataFile")
            self.assertTrue(attr.IsValid())
            asset_path = attr.Get()
            self.assertIsNotNone(asset_path)
            self.assertIn("DATA_ROOT", asset_path.path)
