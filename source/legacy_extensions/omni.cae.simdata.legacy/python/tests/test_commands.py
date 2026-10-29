# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

import omni.kit.commands
import omni.kit.test


class TestLegacyCommandRegistration(omni.kit.test.AsyncTestCase):
    def test_mesh_conversion_uses_active_command_without_ambiguity(self):
        command_class = omni.kit.commands.get_command_class("UsdGeomMeshConvertToSimDataSet")

        self.assertIsNotNone(command_class)
        self.assertEqual(command_class.__module__, "omni.cae.simdata.mesh_commands")

    def test_sids_conversion_remains_registered(self):
        command_class = omni.kit.commands.get_command_class("CaeSidsUnstructuredConvertToSimDataSet")

        self.assertIsNotNone(command_class)
        self.assertEqual(command_class.__module__, "omni.cae.simdata_legacy.commands")
