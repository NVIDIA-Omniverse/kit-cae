# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
#  its affiliates is strictly prohibited.

from logging import getLogger

import omni.kit.app
import omni.kit.test
from omni.cae.asset_importer.cgns import CGNSAssetImporter
from omni.cae.data.commands import ComputeBounds
from omni.cae.schema import cae
from omni.cae.testing import get_test_data_path
from omni.usd import get_context
from pxr import Gf, Usd, UsdUtils

logger = getLogger(__name__)


class TestCGNSImporter(omni.kit.test.AsyncTestCase):
    async def importCGNS(self, cgns_path):
        importer = CGNSAssetImporter()
        result = await importer.convert_assets([cgns_path], import_as_reference=False)
        stage_id = next(iter(result.values()))
        cache: Usd.StageCache = UsdUtils.StageCache.Get()
        cached_stage = cache.Find(Usd.StageCache.Id.FromString(stage_id))
        # cached_stage.Export("/tmp/stage.usda")
        return cached_stage

    async def test_cgns_importer(self):
        usd_context = get_context()
        stage = await self.importCGNS(get_test_data_path("StaticMixer.cgns"))
        await usd_context.attach_stage_async(stage)
        dataset_path = "/World/StaticMixer_cgns/Base/StaticMixer/B1_P3"
        self.assert_(stage.GetPrimAtPath(dataset_path))

        # compute bounds
        bds = await ComputeBounds.invoke(stage.GetPrimAtPath(dataset_path), Usd.TimeCode.EarliestTime())
        self.assertEqual(bds.GetMin(), Gf.Vec3d(-2.0, -3.0, -2.0))
        self.assertEqual(bds.GetMax(), Gf.Vec3d(2, 3, 2))

        usd_context.close_stage()
        del stage
