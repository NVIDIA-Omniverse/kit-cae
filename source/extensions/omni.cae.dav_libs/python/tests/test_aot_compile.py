# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

from unittest.mock import patch

import dav
import omni.kit.test
from dav import aot_compile
from dav.core import aot


class TestAotCompile(omni.kit.test.AsyncTestCase):
    async def test_masked_operator_modules(self):
        """Aliases select owning modules without rewriting recorder specialization keys."""
        configuration = {
            "devices": ["cpu"],
            "operators": {
                "probe": {},
                "probe_masked": {},
                "voxelization": {},
                "voxelization_masked": {},
            },
        }
        with (
            patch.object(dav.config, "compile_kernels_aot", True),
            patch.object(aot, "configuration", configuration),
            patch.object(aot_compile.importlib, "import_module") as importer,
        ):
            aot_compile.compile()
        self.assertEqual(
            [call.args[0] for call in importer.call_args_list],
            [
                "dav.operators.probe",
                "dav.operators.probe",
                "dav.operators.voxelization",
                "dav.operators.voxelization",
            ],
        )
        self.assertEqual(
            list(configuration["operators"]),
            [
                "probe",
                "probe_masked",
                "voxelization",
                "voxelization_masked",
            ],
        )

    async def test_unknown_operator_still_raises(self):
        configuration = {"devices": ["cpu"], "operators": {"missing_operator": {}}}
        with (
            patch.object(dav.config, "compile_kernels_aot", True),
            patch.object(aot, "configuration", configuration),
        ):
            with self.assertRaises(ModuleNotFoundError):
                aot_compile.compile()
