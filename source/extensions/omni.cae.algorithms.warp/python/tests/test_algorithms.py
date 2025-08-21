# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
#  its affiliates is strictly prohibited.

import asyncio
from logging import getLogger

import numpy as np
import omni.kit.commands
from omni.cae.testing import get_test_data_path
from omni.kit.test import AsyncTestCase
from omni.usd import get_context
from pxr import Usd, UsdGeom
from usdrt import Usd as UsdRt
from usdrt import UsdGeom as UsdGeomRt

logger = getLogger(__name__)


class TestNanoVDBStreamline(AsyncTestCase):

    async def test_static_mixer(self):
        usd_context = get_context()
        await usd_context.open_stage_async(get_test_data_path("StaticMixer.cgns"))
        stage: Usd.Stage = usd_context.get_stage()

        zone_path = "/World/StaticMixer_cgns/Base/StaticMixer"
        dataset_path = f"{zone_path}/B1_P3"
        viz_path = "/World/CAE/Streamlines"
        sphere_path: str = "/World/CAE/Sphere"
        sphere_scale = 0.01

        omni.kit.commands.execute("CreateCaeNanoVdbStreamlines", dataset_path=dataset_path, prim_path=viz_path)
        omni.kit.commands.execute("CreateMeshPrim", prim_type="Sphere", prim_path=sphere_path)
        omni.kit.commands.execute(
            "TransformPrimSRT", path=sphere_path, new_scale=[sphere_scale, sphere_scale, sphere_scale]
        )

        viz_prim: Usd.Prim = stage.GetPrimAtPath(viz_path)
        self.assert_(viz_prim.IsValid())
        viz_prim.GetRelationship("omni:cae:warp:streamlines:seeds").SetTargets([sphere_path])
        flow_solution_prim_path = "/World/StaticMixer_cgns/Base/StaticMixer/Flow_Solution"
        velocity_targets = [
            f"{flow_solution_prim_path}/VelocityX",
            f"{flow_solution_prim_path}/VelocityY",
            f"{flow_solution_prim_path}/VelocityZ",
        ]
        viz_prim.GetRelationship("omni:cae:warp:streamlines:velocity").SetTargets(velocity_targets)

        # let the stage update
        for i in range(10):
            await asyncio.sleep(0.1)
            await omni.kit.app.get_app().next_update_async()

        rt_stage = UsdRt.Stage.Attach(get_context().get_stage_id())
        curves = UsdGeomRt.BasisCurves(rt_stage.GetPrimAtPath(viz_path))
        pts = np.asarray(curves.GetPointsAttr().Get())
        self.assertIsNotNone(pts)
        self.assertEqual(pts.shape, (24100, 3))
