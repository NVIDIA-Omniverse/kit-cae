# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import omni.kit.app
import omni.kit.commands
import omni.kit.test
import omni.usd

# from omni.kit.viewport.utility import get_active_viewport
from omni.cae.testing import get_test_data_path
from pxr import Sdf, Usd


class TestIndexCommands(omni.kit.test.AsyncTestCase):

    def get_data_path(self, relative_path: str) -> str:
        return get_test_data_path(relative_path)

    async def load(self, dataset: str):
        source_layer: Sdf.Layer = Sdf.Layer.FindOrOpen(self.get_data_path(dataset), {})
        stage = Usd.Stage.Open(source_layer)
        usd_context = omni.usd.get_context()
        await usd_context.attach_stage_async(stage)
        return stage

    async def test_uniform(self):
        stage = await self.load("hex_uniform.cgns")

        dataset_path = "/World/hex_uniform_cgns/Base/Zone/ElementsUniform"
        omni.kit.commands.execute("CreateCaeIndeXVolume", dataset_path=dataset_path)

        viz_prim = stage.GetPrimAtPath(dataset_path + "/IndeXVolume")
        self.assertIsNotNone(viz_prim)

        viz_prim.GetRelationship("omni:cae:index:volume:field").SetTargets(
            ["/World/hex_uniform_cgns/Base/Zone/SolutionVertex/PointDistanceToCenter"]
        )

        colormap_prim = stage.GetPrimAtPath(dataset_path + "/IndeXVolume/Material/Colormap")
        self.assertIsNotNone(colormap_prim)
        colormap_prim.GetAttribute("domain").Set((0.0, 3.0))

        for _ in range(30000):
            await omni.kit.app.get_app().next_update_async()

        # # https://docs.omniverse.nvidia.com/kit/docs/omni.usd/1.11.2+106/omni.usd.commands/omni.usd.commands.FramePrimsCommand.html
        # viewport = get_active_viewport()
        # camera_path = viewport.camera_path
        # omni.kit.commands.execute(
        #     "FramePrimsCommand", prim_to_move=camera_path, prims_to_frame=[str(viz_prim.GetPath())], zoom=0.05
        # )
