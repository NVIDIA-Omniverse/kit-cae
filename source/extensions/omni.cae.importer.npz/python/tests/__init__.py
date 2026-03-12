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

import omni.kit.app
import omni.kit.test
from omni.cae.data.commands import ComputeBounds
from omni.cae.importer.npz import import_to_stage
from omni.cae.schema import cae
from omni.cae.testing import get_test_data_path
from omni.usd import get_context
from pxr import Gf, Usd, UsdUtils

logger = getLogger(__name__)


class TestNPZImporter(omni.kit.test.AsyncTestCase):

    async def test_npz_importer(self):
        usd_context = get_context()
        await usd_context.new_stage_async()
        stage = usd_context.get_stage()
        await import_to_stage(get_test_data_path("disk_out_ref.npz"), "/World/disk_out_ref_npz")

        self.assertIsNotNone(stage.GetPrimAtPath("/World/disk_out_ref_npz/NumPyDataSet"))
        self.assertIsNotNone(stage.GetPrimAtPath("/World/disk_out_ref_npz/NumPyArrays"))
        self.assertIsNotNone(stage.GetPrimAtPath("/World/disk_out_ref_npz/NumPyArrays/AsH3"))

        # Fix field associations
        array_base_path = "/World/disk_out_ref_npz/NumPyArrays"
        array_paths = [f"{array_base_path}/{base}" for base in ["AsH3", "CH4", "GaMe3", "H2", "Pres", "Temp", "V"]]
        for array_path in array_paths:
            array_prim = stage.GetPrimAtPath(array_path)
            print(array_prim, array_path)
            self.assertIsNotNone(array_prim)
            self.assertTrue(array_prim.IsA(cae.FieldArray))
            array = cae.FieldArray(array_prim)
            array.GetFieldAssociationAttr().Set(cae.Tokens.vertex)

        # compute bounds
        bds = await ComputeBounds.invoke(
            stage.GetPrimAtPath("/World/disk_out_ref_npz/NumPyDataSet"), Usd.TimeCode.EarliestTime()
        )
        self.assertEqual(bds.GetMin(), Gf.Vec3d(-5.75, -5.75, -10.0))
        self.assertEqual(bds.GetMax(), Gf.Vec3d(5.75, 5.75, 10.15999984741211))

        usd_context.close_stage()
        del stage
