# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import pathlib

import dav
import omni.cae.dav as cae_dav
import omni.kit.test
import omni.usd
import warp as wp
from omni.cae.schema import sids
from omni.cae.testing import get_test_data_path
from pxr import Usd


class Test(omni.kit.test.AsyncTestCase):
    async def setUp(self):
        self._test_data = str(pathlib.Path(__file__).parent.joinpath("data"))

    async def tearDown(self) -> None:
        pass

    def get_local_test_scene_path(self, relative_path: str) -> str:
        "compute the absolute path of the test data"
        return self._test_data + "/" + relative_path

    async def test_dav_import(self):
        """Test that DAV can be imported successfully"""
        try:
            import dav

            self.assertTrue(True, "DAV imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import DAV: {e}")

    async def test_sids_ngon_cell_field_subset(self):
        """Cell-centered CGNS fields on NGON_n sections are remapped through NFACE_n."""
        usd_context = omni.usd.get_context()
        await usd_context.open_stage_async(get_test_data_path("hex_polyhedra.cgns"))
        stage = usd_context.get_stage()

        zone_path = "/World/hex_polyhedra_cgns/Base/Zone"
        ngon = stage.GetPrimAtPath(f"{zone_path}/ElementsNgons")
        nface = stage.GetPrimAtPath(f"{zone_path}/ElementsNfaces")
        self.assertTrue(ngon.IsValid())
        self.assertTrue(nface.IsValid())
        self.assertTrue(ngon.HasRelationship("field:CellDistanceToCenter"))

        ngon_field = await cae_dav.GetField.invoke(
            ngon, "CellDistanceToCenter", device="cpu", timeCode=Usd.TimeCode.EarliestTime()
        )
        ngon_field_cached = await cae_dav.GetField.invoke(
            ngon, "CellDistanceToCenter", device="cpu", timeCode=Usd.TimeCode.EarliestTime()
        )
        ngon_field_device_cached = await cae_dav.GetField.invoke(
            ngon, "CellDistanceToCenter", device=wp.get_device("cpu"), timeCode=Usd.TimeCode.EarliestTime()
        )
        nface_field = await cae_dav.GetField.invoke(
            nface, "CellDistanceToCenter", device="cpu", timeCode=Usd.TimeCode.EarliestTime()
        )

        ngon_count = (
            ngon.GetAttribute(sids.Tokens.caeSidsElementRangeEnd).Get()
            - ngon.GetAttribute(sids.Tokens.caeSidsElementRangeStart).Get()
            + 1
        )
        nface_count = (
            nface.GetAttribute(sids.Tokens.caeSidsElementRangeEnd).Get()
            - nface.GetAttribute(sids.Tokens.caeSidsElementRangeStart).Get()
            + 1
        )

        self.assertEqual(ngon_field.association, dav.AssociationType.CELL)
        self.assertEqual(ngon_field.size, ngon_count)
        self.assertEqual(ngon_field_cached.association, dav.AssociationType.CELL)
        self.assertEqual(ngon_field_cached.size, ngon_count)
        self.assertEqual(ngon_field_device_cached.association, dav.AssociationType.CELL)
        self.assertEqual(ngon_field_device_cached.size, ngon_count)
        self.assertEqual(nface_field.association, dav.AssociationType.CELL)
        self.assertEqual(nface_field.size, nface_count)

        usd_context.close_stage()
