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
import warp as wp
from omni.cae.data import Range3i
from omni.cae.data.impl.command_types import Voxelize
from omni.cae.testing import get_test_data_path, get_test_stage_path
from omni.usd import get_context
from pxr import Usd

logger = getLogger(__name__)


class TestOmniCaeVoxelization(omni.kit.test.AsyncTestCase):

    async def test_point_cloud_voxelization(self):
        usd_context = get_context()
        await usd_context.open_stage_async(get_test_stage_path("disk_out_ref_point_cloud.usda"))
        stage: Usd.Stage = usd_context.get_stage()

        volume: wp.Volume = await Voxelize.invoke(
            stage.GetPrimAtPath("/Root/disk_out_ref_npz/NumPyDataSet"),
            ["Temp"],
            bbox=Range3i((-10, -10, -10), (10, 10, 10)),
            voxel_size=0.1,
            device_ordinal=0,
            timeCode=Usd.TimeCode.EarliestTime(),
        )
        self.assertIsNotNone(volume)
        self.assertGreater(volume.get_voxel_count(), 0)
        omni.usd.get_context().close_stage()

    async def test_polyhedra_point_data_voxelization(self):
        usd_context = get_context()
        await usd_context.open_stage_async(get_test_data_path("wedge_polyhedra.cgns"))
        stage: Usd.Stage = usd_context.get_stage()

        volume: wp.Volume = await Voxelize.invoke(
            stage.GetPrimAtPath("/World/wedge_polyhedra_cgns/Base/Zone/ElementsNfaces"),
            ["PointDistanceToCenter"],
            bbox=None,
            voxel_size=1.0,
            device_ordinal=0,
            timeCode=Usd.TimeCode.EarliestTime(),
        )
        self.assertIsNotNone(volume)
        self.assertGreater(volume.get_voxel_count(), 0)
        omni.usd.get_context().close_stage()

    async def test_polyhedra_cell_data_voxelization(self):
        usd_context = get_context()
        await usd_context.open_stage_async(get_test_data_path("wedge_polyhedra.cgns"))
        stage: Usd.Stage = usd_context.get_stage()

        volume: wp.Volume = await Voxelize.invoke(
            stage.GetPrimAtPath("/World/wedge_polyhedra_cgns/Base/Zone/ElementsNfaces"),
            ["CellDistanceToCenter"],
            bbox=None,
            voxel_size=1.0,
            device_ordinal=0,
            timeCode=Usd.TimeCode.EarliestTime(),
        )
        self.assertIsNotNone(volume)
        self.assertGreater(volume.get_voxel_count(), 0)
        omni.usd.get_context().close_stage()

    async def test_dense_grid_voxelization(self):
        usd_context = get_context()
        await usd_context.open_stage_async(get_test_stage_path("headsq.usda"))
        stage: Usd.Stage = usd_context.get_stage()

        volume: wp.Volume = await Voxelize.invoke(
            stage.GetPrimAtPath("/World/headsq/headsq_vti/VTIDataSet"),
            ["Scalars_"],
            bbox=Range3i((0, 0, 0), (100, 100, 100)),
            voxel_size=0.001,
            device_ordinal=0,
            timeCode=Usd.TimeCode.EarliestTime(),
        )
        self.assertIsNotNone(volume)
        self.assertGreater(volume.get_voxel_count(), 0)
        omni.usd.get_context().close_stage()
