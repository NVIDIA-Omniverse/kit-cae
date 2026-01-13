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
from omni.cae.schema import cae
from omni.cae.testing import get_test_data_path
from omni.usd import get_context
from pxr import Gf, Usd, UsdUtils

from ..importers import VTKImporter

logger = getLogger(__name__)


class TestVTKImporter(omni.kit.test.AsyncTestCase):
    async def importVTI(self, path):
        importer = VTKImporter()
        result = await importer.convert_assets([path], import_as_reference=False)
        stage_id = next(iter(result.values()))
        cache: Usd.StageCache = UsdUtils.StageCache.Get()
        cached_stage = cache.Find(Usd.StageCache.Id.FromString(stage_id))
        # cached_stage.Export("/tmp/stage.usda")
        return cached_stage

    async def test_vti(self):
        usd_context = get_context()
        stage = await self.importVTI(get_test_data_path("headsq.vti"))
        await usd_context.attach_stage_async(stage)
        dataset_path = "/World/headsq_vti/VTKImageData"
        dataset_prim = stage.GetPrimAtPath(dataset_path)
        self.assertTrue(dataset_prim)

        self.assertTrue(dataset_prim.IsA(cae.DataSet))
        self.assertTrue(dataset_prim.HasAPI(cae.DenseVolumeAPI))

        denseVolumeAPI = cae.DenseVolumeAPI(dataset_prim)
        self.assertEqual(denseVolumeAPI.GetMinExtentAttr().Get(), Gf.Vec3i(0, 0, 0))
        self.assertEqual(denseVolumeAPI.GetMaxExtentAttr().Get(), Gf.Vec3i(255, 255, 93))
        self.assertEqual(denseVolumeAPI.GetSpacingAttr().Get(), Gf.Vec3f(1, 1, 2))

        # compute bounds
        bds = await ComputeBounds.invoke(stage.GetPrimAtPath(dataset_path), Usd.TimeCode.EarliestTime())
        self.assertEqual(bds.GetMin(), Gf.Vec3d(0, 0, 0))
        self.assertEqual(bds.GetMax(), Gf.Vec3d(255, 255, 93 * 2))

        usd_context.close_stage()
        del stage
