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

import omni.kit.test
from omni.cae.data import usd_utils
from omni.cae.schema import cae, sids
from omni.cae.testing import get_test_data_path
from omni.usd import get_context
from pxr import Usd

logger = getLogger(__name__)


class TestUsdUtils(omni.kit.test.AsyncTestCase):

    async def test_with_hex_polyhedra(self):
        usd_context = get_context()
        await usd_context.open_stage_async(get_test_data_path("hex_polyhedra.cgns"))
        stage: Usd.Stage = usd_context.get_stage()

        gcPrim = stage.GetPrimAtPath("/World/hex_polyhedra_cgns/Base/Zone/GridCoordinates")
        self.assertTrue(gcPrim.IsValid())
        coords = usd_utils.get_target_paths(gcPrim, cae.Tokens.caePointCloudCoordinates)
        self.assertEqual(len(coords), 3)

        with self.assertRaises(usd_utils.QuietableException):
            _ = usd_utils.get_target_path(gcPrim, sids.Tokens.caeSidsGridCoordinates)

        none_field = usd_utils.get_target_path(gcPrim, sids.Tokens.caeSidsGridCoordinates, quiet=True)
        self.assertIsNone(none_field, "None should be returned on quiet=True")

        prim = stage.GetPrimAtPath("/World/hex_polyhedra_cgns/Base/Zone/ElementsNfaces")
        self.assertTrue(prim.IsValid())

        array = await usd_utils.get_vecN_from_relationship(
            prim, sids.Tokens.caeSidsGridCoordinates, 3, Usd.TimeCode.EarliestTime()
        )
        self.assertEqual(array.shape[1], 3)
        self.assertEqual(array.shape[0], 35937)

        field = await usd_utils.get_array_from_relationship(
            prim, "field:PointDistanceToCenter", Usd.TimeCode.EarliestTime()
        )
        self.assertEqual(field.shape[0], 35937)

        fields = await usd_utils.get_arrays_from_relationship(
            prim, "field:CellDistanceToCenter", Usd.TimeCode.EarliestTime()
        )
        self.assertEqual(len(fields), 1)
        self.assertEqual(fields[0].shape[0], 32768)

        cellfieldPath = "/World/hex_polyhedra_cgns/Base/Zone/SolutionCellCenter/CellDistanceToCenter"
        fname = usd_utils.get_field_name(prim, stage.GetPrimAtPath(cellfieldPath))
        self.assertEqual(fname, "CellDistanceToCenter")

        with self.assertRaises(usd_utils.QuietableException):
            _ = usd_utils.get_field_name(prim, gcPrim)
