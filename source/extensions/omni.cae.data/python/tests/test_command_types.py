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
import omni.kit.test
import warp as wp
from omni.cae.data import IJKExtents
from omni.cae.data.commands import (
    ComputeBounds,
    ComputeIJKExtents,
    ConvertToMesh,
    ConvertToPointCloud,
    GenerateStreamlines,
    Mesh,
    Streamlines,
)
from omni.cae.testing import get_test_data_path, get_test_stage_path
from omni.usd import get_context
from pxr import Gf, Usd

logger = getLogger(__name__)


class TestConvertToPointCloud(omni.kit.test.AsyncTestCase):

    async def test_CaePointCloudAPI(self):
        usd_context = get_context()
        await usd_context.open_stage_async(get_test_stage_path("disk_out_ref_point_cloud.usda"))
        stage: Usd.Stage = usd_context.get_stage()
        dataset: Usd.Prim = stage.GetPrimAtPath("/Root/disk_out_ref_npz/NumPyDataSet")
        result = await ConvertToPointCloud.invoke(dataset, ["Pres", "Temp", "V"], Usd.TimeCode.EarliestTime())
        self.assertIsNotNone(result)
        self.assertEqual(len(result.fields), 3)
        self.assertIsNotNone(result.points)
        self.assertEqual(result.points.shape, [8499, 3])
        self.assertEqual(result.fields["Pres"].shape[0], 8499)
        self.assertEqual(result.fields["Temp"].shape[0], 8499)
        self.assertEqual(result.fields["V"].shape, [8499, 3])

        result = await ConvertToPointCloud.invoke(dataset, [], Usd.TimeCode.EarliestTime())
        self.assertIsNotNone(result)
        self.assertEqual(len(result.fields), 0)
        self.assertIsNotNone(result.points)

        with self.assertRaises(ValueError):
            _ = await ConvertToPointCloud.invoke(dataset, ["missing"], Usd.TimeCode.EarliestTime())
        usd_context.close_stage()

    async def test_sids_polyhedra(self):

        for dname in ["hex_polyhedra", "hex_polyhedra_small", "wedge_polyhedra"]:
            usd_context = get_context()
            await usd_context.open_stage_async(get_test_data_path(f"{dname}.cgns"))
            stage: Usd.Stage = usd_context.get_stage()
            prim_path = f"/World/{dname}_cgns/Base/Zone/ElementsNfaces"
            dataset: Usd.Prim = stage.GetPrimAtPath(prim_path)
            self.assertTrue(dataset.IsValid())

            result = await ConvertToPointCloud.invoke(dataset, [], Usd.TimeCode.EarliestTime())
            self.assertIsNotNone(result)
            self.assertIsNotNone(result.points)
            self.assertEqual(len(result.fields), 0)

            result = await ConvertToPointCloud.invoke(dataset, ["PointDistanceToCenter"], Usd.TimeCode.EarliestTime())
            self.assertIsNotNone(result)
            self.assertIsNotNone(result.points)
            self.assertEqual(len(result.fields), 1)
            self.assertEqual(result.fields["PointDistanceToCenter"].shape[0], result.points.shape[0])

            result = await ConvertToPointCloud.invoke(dataset, ["CellDistanceToCenter"], Usd.TimeCode.EarliestTime())
            self.assertIsNotNone(result)
            self.assertIsNotNone(result.points)
            self.assertEqual(len(result.fields), 1)
            self.assertEqual(result.fields["CellDistanceToCenter"].shape[0], result.points.shape[0])

            result = await ConvertToPointCloud.invoke(
                dataset, ["PointDistanceToCenter", "CellDistanceToCenter"], Usd.TimeCode.EarliestTime()
            )
            self.assertIsNotNone(result)
            self.assertIsNotNone(result.points)
            self.assertEqual(len(result.fields), 2)
            self.assertEqual(result.fields["PointDistanceToCenter"].shape[0], result.points.shape[0])
            self.assertEqual(result.fields["CellDistanceToCenter"].shape[0], result.points.shape[0])

            usd_context.close_stage()
            await asyncio.sleep(0.1)

    async def test_sids_uniform(self):
        usd_context = get_context()
        await usd_context.open_stage_async(get_test_data_path("hex_uniform.cgns"))
        stage: Usd.Stage = usd_context.get_stage()
        prim_path = "/World/hex_uniform_cgns/Base/Zone/ElementsUniform"
        dataset: Usd.Prim = stage.GetPrimAtPath(prim_path)
        self.assertTrue(dataset.IsValid())

        result = await ConvertToPointCloud.invoke(dataset, [], Usd.TimeCode.EarliestTime())
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.points)
        self.assertEqual(len(result.fields), 0)

        result = await ConvertToPointCloud.invoke(dataset, ["PointDistanceToCenter"], Usd.TimeCode.EarliestTime())
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.points)
        self.assertEqual(len(result.fields), 1)
        self.assertEqual(result.fields["PointDistanceToCenter"].shape[0], result.points.shape[0])
        self.assertAlmostEqual(np.amin(result.fields["PointDistanceToCenter"]), 0.8660254, places=5)
        self.assertAlmostEqual(np.amax(result.fields["PointDistanceToCenter"]), 2.598076, places=5)

        result = await ConvertToPointCloud.invoke(dataset, ["CellDistanceToCenter"], Usd.TimeCode.EarliestTime())
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.points)
        self.assertEqual(len(result.fields), 1)
        self.assertEqual(result.fields["CellDistanceToCenter"].shape[0], result.points.shape[0])
        self.assertAlmostEqual(np.amin(result.fields["CellDistanceToCenter"]), 1.0640972, places=5)
        self.assertAlmostEqual(np.amax(result.fields["CellDistanceToCenter"]), 2.1281943, places=5)

        result = await ConvertToPointCloud.invoke(
            dataset, ["PointDistanceToCenter", "CellDistanceToCenter"], Usd.TimeCode.EarliestTime()
        )
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.points)
        self.assertEqual(len(result.fields), 2)
        self.assertEqual(result.fields["PointDistanceToCenter"].shape[0], result.points.shape[0])
        self.assertEqual(result.fields["CellDistanceToCenter"].shape[0], result.points.shape[0])

        usd_context.close_stage()
        await asyncio.sleep(0.1)

    async def test_sids_mixed(self):
        usd_context = get_context()
        await usd_context.open_stage_async(get_test_data_path("hex_mixed.cgns"))
        stage: Usd.Stage = usd_context.get_stage()
        prim_path = "/World/hex_mixed_cgns/Base/Zone/ElementsMixed"
        dataset: Usd.Prim = stage.GetPrimAtPath(prim_path)
        self.assertTrue(dataset.IsValid())

        result = await ConvertToPointCloud.invoke(dataset, [], Usd.TimeCode.EarliestTime())
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.points)
        self.assertEqual(len(result.fields), 0)

        result = await ConvertToPointCloud.invoke(dataset, ["PointDistanceToCenter"], Usd.TimeCode.EarliestTime())
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.points)
        self.assertEqual(len(result.fields), 1)
        self.assertEqual(result.fields["PointDistanceToCenter"].shape[0], result.points.shape[0])

        result = await ConvertToPointCloud.invoke(dataset, ["CellDistanceToCenter"], Usd.TimeCode.EarliestTime())
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.points)
        self.assertEqual(len(result.fields), 1)
        self.assertEqual(result.fields["CellDistanceToCenter"].shape[0], result.points.shape[0])

        result = await ConvertToPointCloud.invoke(
            dataset, ["PointDistanceToCenter", "CellDistanceToCenter"], Usd.TimeCode.EarliestTime()
        )
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.points)
        self.assertEqual(len(result.fields), 2)
        self.assertEqual(result.fields["PointDistanceToCenter"].shape[0], result.points.shape[0])
        self.assertEqual(result.fields["CellDistanceToCenter"].shape[0], result.points.shape[0])

        usd_context.close_stage()
        await asyncio.sleep(0.1)


class TestComputeBounds(omni.kit.test.AsyncTestCase):

    async def test_point_cloud(self):
        usd_context = get_context()
        await usd_context.open_stage_async(get_test_stage_path("disk_out_ref_point_cloud.usda"))
        stage: Usd.Stage = usd_context.get_stage()
        dataset: Usd.Prim = stage.GetPrimAtPath("/Root/disk_out_ref_npz/NumPyDataSet")
        bds: Gf.Range3d = await ComputeBounds.invoke(dataset, Usd.TimeCode.EarliestTime())
        self.assertIsNotNone(bds)
        self.assertIsInstance(bds, Gf.Range3d)
        self.assertEqual(bds, Gf.Range3d((-5.75, -5.75, -10.0), (5.75, 5.75, 10.15999984741211)))

    async def test_polyhedra(self):
        usd_context = get_context()
        await usd_context.open_stage_async(get_test_data_path("hex_polyhedra.cgns"))
        stage: Usd.Stage = usd_context.get_stage()

        zone_path = "/World/hex_polyhedra_cgns/Base/Zone"

        dataset: Usd.Prim = stage.GetPrimAtPath(f"{zone_path}/GridCoordinates")
        self.assertTrue(dataset.IsValid())

        bds: Gf.Range3d = await ComputeBounds.invoke(dataset, Usd.TimeCode.EarliestTime())
        self.assertIsNotNone(bds)
        self.assertIsInstance(bds, Gf.Range3d)
        self.assertEqual(bds, Gf.Range3d((0, 0, 0), (32, 32, 32)))

        dataset: Usd.Prim = stage.GetPrimAtPath(f"{zone_path}/ElementsNgons")
        self.assertTrue(dataset.IsValid())

        bds: Gf.Range3d = await ComputeBounds.invoke(dataset, Usd.TimeCode.EarliestTime())
        self.assertIsNotNone(bds)
        self.assertIsInstance(bds, Gf.Range3d)
        self.assertEqual(bds, Gf.Range3d((0, 0, 0), (32, 32, 32)))

        dataset: Usd.Prim = stage.GetPrimAtPath(f"{zone_path}/ElementsNfaces")
        self.assertTrue(dataset.IsValid())

        bds: Gf.Range3d = await ComputeBounds.invoke(dataset, Usd.TimeCode.EarliestTime())
        self.assertIsNotNone(bds)
        self.assertIsInstance(bds, Gf.Range3d)
        self.assertEqual(bds, Gf.Range3d((0, 0, 0), (32, 32, 32)))

        usd_context.close_stage()
        await asyncio.sleep(0.1)

    async def test_static_mixer(self):
        usd_context = get_context()
        await usd_context.open_stage_async(get_test_data_path("StaticMixer.cgns"))
        stage: Usd.Stage = usd_context.get_stage()

        zone_path = "/World/StaticMixer_cgns/Base/StaticMixer"

        dataset: Usd.Prim = stage.GetPrimAtPath(f"{zone_path}/GridCoordinates")
        self.assertTrue(dataset.IsValid())

        bds: Gf.Range3d = await ComputeBounds.invoke(dataset, Usd.TimeCode.EarliestTime())
        self.assertEqual(bds, Gf.Range3d((-2, -3, -2), (2, 3, 2)))

        dataset: Usd.Prim = stage.GetPrimAtPath(f"{zone_path}/B1_P3")
        self.assertTrue(dataset.IsValid())

        bds: Gf.Range3d = await ComputeBounds.invoke(dataset, Usd.TimeCode.EarliestTime())
        self.assertEqual(bds, Gf.Range3d((-2, -3, -2), (2, 3, 2)))

        dataset: Usd.Prim = stage.GetPrimAtPath(f"{zone_path}/StaticMixer_Default")
        self.assertTrue(dataset.IsValid())

        bds: Gf.Range3d = await ComputeBounds.invoke(dataset, Usd.TimeCode.EarliestTime())
        self.assertEqual(bds, Gf.Range3d((-2, -3, -2), (2, 3, 2)))

        dataset: Usd.Prim = stage.GetPrimAtPath(f"{zone_path}/in1")
        self.assertTrue(dataset.IsValid())

        bds: Gf.Range3d = await ComputeBounds.invoke(dataset, Usd.TimeCode.EarliestTime())
        self.assertEqual(bds, Gf.Range3d((-1.5, -3, 0.5), (-0.5, -3.0, 1.5)))

        dataset: Usd.Prim = stage.GetPrimAtPath(f"{zone_path}/in2")
        self.assertTrue(dataset.IsValid())

        bds: Gf.Range3d = await ComputeBounds.invoke(dataset, Usd.TimeCode.EarliestTime())
        self.assertEqual(bds, Gf.Range3d(Gf.Vec3d(0.5, 3.0, 0.5), Gf.Vec3d(1.5, 3.0, 1.5)))

        dataset: Usd.Prim = stage.GetPrimAtPath(f"{zone_path}/out")
        self.assertTrue(dataset.IsValid())

        bds: Gf.Range3d = await ComputeBounds.invoke(dataset, Usd.TimeCode.EarliestTime())
        self.assertEqual(bds, Gf.Range3d(Gf.Vec3d(-0.5, -0.5, -2.0), Gf.Vec3d(0.5, 0.5, -2.0)))

        usd_context.close_stage()
        await asyncio.sleep(0.1)

    async def test_dense_volume(self):
        usd_context = get_context()
        await usd_context.open_stage_async(get_test_stage_path("headsq.usda"))
        stage: Usd.Stage = usd_context.get_stage()

        dataset: Usd.Prim = stage.GetPrimAtPath("/World/headsq/headsq_vti/VTIDataSet")
        self.assertTrue(dataset.IsValid())

        bds: Gf.Range3d = await ComputeBounds.invoke(dataset, Usd.TimeCode.EarliestTime())
        self.assertEqual(bds, Gf.Range3d((0, 0, 0), (255, 255, 93 * 2)))

        usd_context.close_stage()
        await asyncio.sleep(0.1)


class TestComputeIJKExtents(omni.kit.test.AsyncTestCase):

    async def test_dense_volume(self):
        usd_context = get_context()
        await usd_context.open_stage_async(get_test_stage_path("headsq.usda"))
        stage: Usd.Stage = usd_context.get_stage()

        dataset: Usd.Prim = stage.GetPrimAtPath("/World/headsq/headsq_vti/VTIDataSet")
        self.assertTrue(dataset.IsValid())

        extents: IJKExtents = await ComputeIJKExtents.invoke(
            dataset, max_dims=(100, 100, 100), timeCode=Usd.TimeCode.EarliestTime()
        )
        # note: ComputeIJKExtents intentionally ignores max_dims for DenseVolumes
        self.assertEqual(extents, IJKExtents((0, 0, 0), (255, 255, 93), (1, 1, 2)))

        sub_extents: IJKExtents = await ComputeIJKExtents.invoke(
            dataset,
            max_dims=(100, 100, 100),
            roi=Gf.Range3d((0, 0, 0), (50, 50, 50)),
            timeCode=Usd.TimeCode.EarliestTime(),
        )
        self.assertEqual(sub_extents, IJKExtents((0, 0, 0), (50, 50, 25), (1, 1, 2)))

        usd_context.close_stage()
        await asyncio.sleep(0.1)

    async def test_point_cloud(self):
        usd_context = get_context()
        await usd_context.open_stage_async(get_test_stage_path("disk_out_ref_point_cloud.usda"))
        stage: Usd.Stage = usd_context.get_stage()

        dataset: Usd.Prim = stage.GetPrimAtPath("/Root/disk_out_ref_npz/NumPyDataSet")
        self.assertTrue(dataset.IsValid())

        extents: IJKExtents = await ComputeIJKExtents.invoke(
            dataset, max_dims=(100, 100, 100), timeCode=Usd.TimeCode.EarliestTime()
        )
        self.assertEqual(
            extents,
            IJKExtents(
                min=(-28, -28, -49),
                max=(28, 28, 50),
                spacing=(0.2036363620950718, 0.2036363620950718, 0.2036363620950718),
            ),
        )

        sub_extents: IJKExtents = await ComputeIJKExtents.invoke(
            dataset,
            max_dims=(100, 100, 100),
            roi=Gf.Range3d((0, 0, 0), (50, 50, 50)),
            timeCode=Usd.TimeCode.EarliestTime(),
        )
        self.assertEqual(
            sub_extents,
            IJKExtents(
                min=(0, 0, 0), max=(56, 56, 99), spacing=(0.1026262610849708, 0.1026262610849708, 0.1026262610849708)
            ),
        )

        # clean up
        usd_context.close_stage()
        await asyncio.sleep(0.1)

    async def test_sids(self):
        usd_context = get_context()
        await usd_context.open_stage_async(get_test_data_path("hex_mixed.cgns"))
        stage: Usd.Stage = usd_context.get_stage()
        prim_path = "/World/hex_mixed_cgns/Base/Zone/ElementsMixed"

        dataset: Usd.Prim = stage.GetPrimAtPath(prim_path)
        self.assertTrue(dataset.IsValid())
        extents: IJKExtents = await ComputeIJKExtents.invoke(
            dataset, max_dims=(100, 100, 100), timeCode=Usd.TimeCode.EarliestTime()
        )
        self.assertEqual(
            extents,
            IJKExtents(
                min=(0, 0, 0),
                max=(20, 20, 99),
                spacing=(0.050505050505050504, 0.050505050505050504, 0.050505050505050504),
            ),
        )

        sub_extents: IJKExtents = await ComputeIJKExtents.invoke(
            dataset,
            max_dims=(100, 100, 100),
            roi=Gf.Range3d((0, 0, 0), (5, 500, 2)),
            timeCode=Usd.TimeCode.EarliestTime(),
        )
        self.assertEqual(
            sub_extents,
            IJKExtents(
                min=(0, 0, 0),
                max=(49, 49, 99),
                spacing=(0.020202020202020204, 0.020202020202020204, 0.020202020202020204),
            ),
        )

        # clean up
        usd_context.close_stage()
        await asyncio.sleep(0.1)


class TestConvertToMesh(omni.kit.test.AsyncTestCase):

    async def test_static_mixer(self):
        """
        Test that we can convert a static mixer into a mesh using VTK's external faces algorithm.
        """
        usd_context = get_context()
        await usd_context.open_stage_async(get_test_data_path("StaticMixer.cgns"))
        stage: Usd.Stage = usd_context.get_stage()

        zone_path = "/World/StaticMixer_cgns/Base/StaticMixer"

        dataset: Usd.Prim = stage.GetPrimAtPath(f"{zone_path}/B1_P3")
        self.assertTrue(dataset.IsValid())

        mesh = await ConvertToMesh.invoke(dataset, [], Usd.TimeCode.EarliestTime())
        self.assertIsNotNone(mesh)
        self.assertIsInstance(mesh, Mesh)
        self.assertEqual(mesh.points.shape, (2922, 3))
        self.assertEqual(mesh.normals.shape, (2922, 3))
        self.assertEqual(mesh.faceVertexCounts.shape, (1630, 1))
        self.assertEqual(mesh.faceVertexIndices.shape, (4890, 1))

    async def test_polyhedra(self):
        """
        Test that we can convert a polyhedra into a mesh using VTK's external faces algorithm.
        While this is not currently supported, we should raise appropriate exceptions to confirm.
        """
        usd_context = get_context()
        await usd_context.open_stage_async(get_test_data_path("hex_polyhedra.cgns"))
        stage: Usd.Stage = usd_context.get_stage()

        zone_path = "/World/hex_polyhedra_cgns/Base/Zone"

        dataset: Usd.Prim = stage.GetPrimAtPath(f"{zone_path}/ElementsNgons")
        self.assertTrue(dataset.IsValid())

        with self.assertRaises(NotImplementedError):
            _ = await ConvertToMesh.invoke(dataset, [], Usd.TimeCode.EarliestTime())

        dataset: Usd.Prim = stage.GetPrimAtPath(f"{zone_path}/ElementsNfaces")
        self.assertTrue(dataset.IsValid())

        with self.assertRaises(NotImplementedError):
            _ = await ConvertToMesh.invoke(dataset, [], Usd.TimeCode.EarliestTime())


class TestGenerateStreamlines(omni.kit.test.AsyncTestCase):

    def assertListAlmostEquals(self, list1, list2, places=7):
        self.assertEqual(len(list1), len(list2))
        for a, b in zip(list1, list2):
            self.assertAlmostEqual(a, b, places=places)

    async def test_static_mixer_vtk(self):
        usd_context = get_context()
        await usd_context.open_stage_async(get_test_data_path("StaticMixer.cgns"))
        stage: Usd.Stage = usd_context.get_stage()

        zone_path = "/World/StaticMixer_cgns/Base/StaticMixer"

        dataset: Usd.Prim = stage.GetPrimAtPath(f"{zone_path}/B1_P3")
        self.assertTrue(dataset.IsValid())

        in1: Usd.Prim = stage.GetPrimAtPath(f"{zone_path}/in1")
        self.assertTrue(in1.IsValid())

        seeds = await ConvertToPointCloud.invoke(in1, [], Usd.TimeCode.EarliestTime())

        GenerateStreamlines.override_impl("VTK")
        with self.assertRaises(ValueError):
            streamlines = await GenerateStreamlines.invoke(
                dataset,
                seeds.points,
                ["Velocity_X", "Velocity_Y", "Velocity_Z"],
                "Temperature",
                0.1,
                100,
                Usd.TimeCode.EarliestTime(),
            )
        streamlines = await GenerateStreamlines.invoke(
            dataset,
            seeds.points,
            ["VelocityX", "VelocityY", "VelocityZ"],
            "Temperature",
            0.1,
            100,
            Usd.TimeCode.EarliestTime(),
        )

        self.assertIsNotNone(streamlines)
        self.assertIsInstance(streamlines, Streamlines)
        self.assertEquals(streamlines.points.shape, (408, 3))
        self.assertEquals(
            np.amin(streamlines.points, axis=0).tolist(), [-1.5977600812911987, -3.0, -1.6546809673309326]
        )
        self.assertEquals(
            np.amax(streamlines.points, axis=0).tolist(), [1.2705073356628418, 1.662825345993042, 1.1933543682098389]
        )

        self.assertEqual(streamlines.curveVertexCounts.shape, (4, 1))
        self.assertEqual(np.amin(streamlines.curveVertexCounts).tolist(), 102)
        self.assertEqual(np.amax(streamlines.curveVertexCounts).tolist(), 102)

        self.assertEqual(streamlines.fields["scalar"].shape, (408,))
        self.assertAlmostEqual(np.amin(streamlines.fields["scalar"]), 296.36197, 3)
        self.assertEqual(np.amax(streamlines.fields["scalar"]), 315.0)

        self.assertEqual(streamlines.fields["time"].shape, (408,))
        self.assertEqual(np.amin(streamlines.fields["time"]), 0.0)
        self.assertAlmostEqual(np.amax(streamlines.fields["time"]), 7.37515, 2)

    async def test_yf17_warp(self):
        usd_context = get_context()
        await usd_context.open_stage_async(get_test_data_path("yf17_hdf5.cgns"))
        stage: Usd.Stage = usd_context.get_stage()
        zone_path = "/World/yf17_hdf5_cgns/Base/Zone1"

        dataset: Usd.Prim = stage.GetPrimAtPath(f"{zone_path}/GridElements")
        self.assertTrue(dataset.IsValid(), f"'{zone_path}/GridElements' is not valid Prim in stage.")

        intake: Usd.Prim = stage.GetPrimAtPath(f"{zone_path}/intake")
        self.assertTrue(intake.IsValid())
        seeds = await ConvertToPointCloud.invoke(intake, [], Usd.TimeCode.EarliestTime())

        GenerateStreamlines.override_impl("Warp")
        streamlines = await GenerateStreamlines.invoke(
            dataset,
            wp.array(seeds.points, copy=False, device="cuda", dtype=wp.vec3f),
            ["VelocityX", "VelocityY", "VelocityZ"],
            "Density",
            0.5,
            10,
            Usd.TimeCode.EarliestTime(),
        )

        self.assertIsNotNone(streamlines)
        streamlines.points = streamlines.points.numpy()

        self.assertEquals(streamlines.points.shape, (418, 3))
        self.assertListAlmostEquals(
            np.amin(streamlines.points, axis=0).tolist(),
            [-54.117576599121094, 0.33262574672698975, -0.8075870871543884],
            places=1,
        )
        self.assertListAlmostEquals(
            np.amax(streamlines.points, axis=0).tolist(),
            [14.900866508483887, 1.2461183071136475, 0.33379295468330383],
            places=1,
        )

        self.assertEqual(streamlines.curveVertexCounts.shape, (22,))
        self.assertEqual(np.amin(streamlines.curveVertexCounts).tolist(), 19)
        self.assertEqual(np.amax(streamlines.curveVertexCounts).tolist(), 19)

        self.assertEqual(streamlines.fields["scalar"].shape, (418,))
        self.assertAlmostEqual(np.amin(streamlines.fields["scalar"].numpy()), 0.0, 1)
        self.assertAlmostEqual(np.amax(streamlines.fields["scalar"].numpy()), 299.9464, 2)

        self.assertEqual(streamlines.fields["time"].shape, (418,))
        self.assertEqual(np.amin(streamlines.fields["time"].numpy()), -8.5)
        self.assertAlmostEqual(np.amax(streamlines.fields["time"].numpy()), 8.5, 1)
