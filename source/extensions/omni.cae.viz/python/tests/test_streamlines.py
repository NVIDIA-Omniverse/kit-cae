# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import asyncio
import json
from logging import getLogger

import numpy as np
import omni.kit.test
import warp as wp
from omni.cae.data import usd_utils
from omni.cae.data.commands import execute_command
from omni.cae.importer.cgns import import_to_stage
from omni.cae.schema import viz as cae_viz
from omni.cae.testing import get_test_data_path, get_vtrt_array_as_numpy, new_stage, wait_for_update
from omni.kit.app import get_app
from pxr import Usd
from usdrt import UsdGeom as UsdGeomRT

logger = getLogger(__name__)


class TestStreamlines(omni.kit.test.AsyncTestCase):
    tolerance = 1e-5

    @staticmethod
    def skip_if_kit_108():
        """Skip test for Kit 108 where some behaviors differ."""
        app_version = get_app().get_kit_version()
        # Parse version string like "108.0.0+feature.221586.5941509b"
        major_version = int(app_version.split(".")[0])
        if major_version < 109:
            return True
        return False

    async def _streamlines_static_mixer(self, streamlines_type: str, use_colors: bool = False):
        async with new_stage() as stage:
            await import_to_stage(get_test_data_path("StaticMixer.cgns"), "/World/StaticMixer")
            base_path: str = "/World/StaticMixer/Base/StaticMixer"

            dataset_path: str = f"{base_path}/B1_P3"
            viz_path: str = f"/World/CAE/Streamlines_B1_P3"
            sphere_path: str = f"/World/CAE/Sphere"
            await execute_command(
                "CreateCaeVizStreamlines", dataset_path=dataset_path, prim_path=viz_path, type=streamlines_type
            )
            await execute_command("CreateCaeVizMeshPrim", prim_type="UnitSphere", prim_path=sphere_path)
            await execute_command("TransformPrimSRT", path=sphere_path, new_scale=[0.2, 0.2, 0.2])
            await wait_for_update()

            viz_prim: Usd.Prim = stage.GetPrimAtPath(viz_path)
            sphere_prim: Usd.Prim = stage.GetPrimAtPath(sphere_path)
            self.assertTrue(viz_prim.IsValid())
            self.assertTrue(sphere_prim.IsValid())

            streamlines_api: cae_viz.StreamlinesAPI = cae_viz.StreamlinesAPI(viz_prim)
            streamlines_api.GetDirectionAttr().Set(cae_viz.Tokens.forward)

            ds_api: cae_viz.DatasetSelectionAPI = cae_viz.DatasetSelectionAPI(viz_prim, "seeds")
            ds_api.GetTargetRel().SetTargets({sphere_prim.GetPath()})

            vs_api = cae_viz.FieldSelectionAPI(viz_prim, "velocities")
            vs_api.GetTargetRel().SetTargets(
                [
                    f"{base_path}/Flow_Solution/VelocityX",
                    f"{base_path}/Flow_Solution/VelocityY",
                    f"{base_path}/Flow_Solution/VelocityZ",
                ]
            )

            if use_colors:
                vs_api = cae_viz.FieldSelectionAPI(viz_prim, "colors")
                vs_api.GetTargetRel().SetTargets([f"{base_path}/Flow_Solution/Temperature"])

            await wait_for_update()

            # Get forward direction points
            usdrt_curves = UsdGeomRT.BasisCurves(usd_utils.get_prim_rt(viz_prim))
            points_forward = get_vtrt_array_as_numpy(usdrt_curves.GetPointsAttr())

            # Verify times and rnd primvars are present
            times_attr = usdrt_curves.GetPrim().GetAttribute("primvars:times")
            self.assertIsNotNone(times_attr, "times primvar should be present")
            times_forward = get_vtrt_array_as_numpy(times_attr)
            self.assertGreater(len(times_forward), 0, "times primvar should have values")

            rnd_attr = usdrt_curves.GetPrim().GetAttribute("primvars:rnd")
            self.assertIsNotNone(rnd_attr, "rnd primvar should be present")
            rnd_forward = get_vtrt_array_as_numpy(rnd_attr)
            self.assertGreater(len(rnd_forward), 0, "rnd primvar should have values")

            colors_forward = None
            if use_colors:
                colors_attr = usdrt_curves.GetPrim().GetAttribute("primvars:colors")
                self.assertIsNotNone(colors_attr, "colors primvar should be present")
                colors_forward = get_vtrt_array_as_numpy(colors_attr)

            # Switch to backward direction
            streamlines_api.GetDirectionAttr().Set(cae_viz.Tokens.backward)
            await wait_for_update()
            points_backward = get_vtrt_array_as_numpy(usdrt_curves.GetPointsAttr())
            colors_backward = None
            if use_colors:
                colors_attr = usdrt_curves.GetPrim().GetAttribute("primvars:colors")
                self.assertIsNotNone(colors_attr, "colors primvar should be present")
                colors_backward = get_vtrt_array_as_numpy(colors_attr)

            # Move the sphere to the right
            await execute_command("TransformPrimSRT", path=sphere_path, new_translation=[0.1, 0, 0])
            await wait_for_update()
            points_moved = get_vtrt_array_as_numpy(usdrt_curves.GetPointsAttr())
            colors_moved = None
            if use_colors:
                colors_attr = usdrt_curves.GetPrim().GetAttribute("primvars:colors")
                self.assertIsNotNone(colors_attr, "colors primvar should be present")
                colors_moved = get_vtrt_array_as_numpy(colors_attr)

            # Return summary values for assertions in the test methods
            result = {
                "forward": {
                    "min": points_forward.min(axis=0).tolist(),
                    "max": points_forward.max(axis=0).tolist(),
                    "shape": points_forward.shape,
                },
                "backward": {
                    "min": points_backward.min(axis=0).tolist(),
                    "max": points_backward.max(axis=0).tolist(),
                    "shape": points_backward.shape,
                },
                "moved": {
                    "min": points_moved.min(axis=0).tolist(),
                    "max": points_moved.max(axis=0).tolist(),
                    "shape": points_moved.shape,
                },
            }

            # Add color information if colors were used
            if use_colors:
                if colors_forward is not None:
                    result["forward"]["colors_min"] = colors_forward.min(axis=0).tolist()
                    result["forward"]["colors_max"] = colors_forward.max(axis=0).tolist()
                    result["forward"]["colors_shape"] = colors_forward.shape
                if colors_backward is not None:
                    result["backward"]["colors_min"] = colors_backward.min(axis=0).tolist()
                    result["backward"]["colors_max"] = colors_backward.max(axis=0).tolist()
                    result["backward"]["colors_shape"] = colors_backward.shape
                if colors_moved is not None:
                    result["moved"]["colors_min"] = colors_moved.min(axis=0).tolist()
                    result["moved"]["colors_max"] = colors_moved.max(axis=0).tolist()
                    result["moved"]["colors_shape"] = colors_moved.shape

            logger.info("Streamlines result: %s", json.dumps(result, indent=4))
            return result

    async def test_streamlines_static_mixer_standard(self):
        result = await self._streamlines_static_mixer("standard")

        # Forward direction assertions
        np.testing.assert_allclose(
            result["forward"]["min"], [-0.19455785, -0.19781476, -1.9918408], atol=self.tolerance
        )
        np.testing.assert_allclose(result["forward"]["max"], [0.19890438, 0.19781476, 0.2], atol=self.tolerance)
        self.assertEqual(result["forward"]["shape"], (2992, 3))

        # Backward direction assertions
        np.testing.assert_allclose(result["backward"]["min"], [-1.9416908, -1.9246704, -0.2], atol=self.tolerance)
        np.testing.assert_allclose(result["backward"]["max"], [1.9306283, 1.8990393, 1.9998803], atol=self.tolerance)
        self.assertEqual(result["backward"]["shape"], (34724, 3))

        if not self.skip_if_kit_108():
            # Moved sphere assertions
            np.testing.assert_allclose(result["moved"]["min"], [-1.983881, -2.990228, -0.2], atol=self.tolerance)
            np.testing.assert_allclose(result["moved"]["max"], [1.9660718, 2.9283297, 1.9999539], atol=self.tolerance)
            self.assertEqual(result["moved"]["shape"], (31858, 3))

    async def test_streamlines_static_mixer_nanovdb(self):
        result = await self._streamlines_static_mixer("nanovdb")

        # Forward direction assertions
        np.testing.assert_allclose(
            result["forward"]["min"], [-0.19460653, -0.19781476, -1.9648020], atol=self.tolerance
        )
        np.testing.assert_allclose(result["forward"]["max"], [0.19890438, 0.19781476, 0.2], atol=self.tolerance)
        self.assertEqual(result["forward"]["shape"], (38400, 3))

        # Backward direction assertions
        np.testing.assert_allclose(result["backward"]["min"], [-0.50252163, -0.50247568, -0.2], atol=self.tolerance)
        np.testing.assert_allclose(result["backward"]["max"], [0.57305855, 0.59693217, 1.9457186], atol=self.tolerance)
        self.assertEqual(result["backward"]["shape"], (38400, 3))

        if not self.skip_if_kit_108():
            # Moved sphere assertions
            np.testing.assert_allclose(result["moved"]["min"], [-0.69733274, -0.8208583, -0.2], atol=self.tolerance)
            np.testing.assert_allclose(result["moved"]["max"], [0.71503311, 0.68206424, 1.9481354], atol=self.tolerance)
            self.assertEqual(result["moved"]["shape"], (38400, 3))

    async def test_streamlines_static_mixer_standard_with_colors(self):
        result = await self._streamlines_static_mixer("standard", use_colors=True)

        # Points assertions (same as without colors)
        np.testing.assert_allclose(
            result["forward"]["min"], [-0.19455785, -0.19781476, -1.9918408], atol=self.tolerance
        )
        np.testing.assert_allclose(result["forward"]["max"], [0.19890438, 0.19781476, 0.2], atol=self.tolerance)
        self.assertEqual(result["forward"]["shape"], (2992, 3))

        # Color assertions for forward direction
        self.assertIsNotNone(result["forward"].get("colors_min"))
        np.testing.assert_allclose(result["forward"]["colors_min"], [299.43332], atol=self.tolerance)
        np.testing.assert_allclose(result["forward"]["colors_max"], [300.4997], atol=self.tolerance)
        self.assertEqual(result["forward"]["colors_shape"], (2992, 1))

        # Backward direction assertions
        np.testing.assert_allclose(result["backward"]["min"], [-1.9416908, -1.9246704, -0.2], atol=self.tolerance)
        np.testing.assert_allclose(result["backward"]["max"], [1.9306283, 1.8990393, 1.9998803], atol=self.tolerance)
        self.assertEqual(result["backward"]["shape"], (34724, 3))

        # Color assertions for backward direction
        np.testing.assert_allclose(result["backward"]["colors_min"], 289.355, atol=1e-3)
        np.testing.assert_allclose(result["backward"]["colors_max"], 307.960, atol=1e-3)
        self.assertEqual(result["backward"]["colors_shape"], (34724, 1))

        if not self.skip_if_kit_108():
            # Skip moved sphere assertions for Kit 108 since TramsformPrimSRT doesn't seem t have same effect.
            # Moved sphere assertions
            np.testing.assert_allclose(result["moved"]["min"], [-1.983881, -2.990228, -0.2], atol=self.tolerance)
            np.testing.assert_allclose(result["moved"]["max"], [1.9660718, 2.9283297, 1.9999539], atol=self.tolerance)
            self.assertEqual(result["moved"]["shape"], (31858, 3))

            # Color assertions for moved sphere
            np.testing.assert_allclose(result["moved"]["colors_min"], [284.99997], atol=self.tolerance)
            np.testing.assert_allclose(result["moved"]["colors_max"], [315.00027], atol=self.tolerance)
            self.assertEqual(result["moved"]["colors_shape"], (31858, 1))

    async def test_streamlines_static_mixer_nanovdb_with_colors(self):
        result = await self._streamlines_static_mixer("nanovdb", use_colors=True)

        # Points assertions (same as without colors)
        np.testing.assert_allclose(
            result["forward"]["min"], [-0.19460653, -0.19781476, -1.9648020], atol=self.tolerance
        )
        np.testing.assert_allclose(result["forward"]["max"], [0.19890438, 0.19781476, 0.2], atol=self.tolerance)
        self.assertEqual(result["forward"]["shape"], (38400, 3))

        # Color assertions for forward direction
        self.assertIsNotNone(result["forward"].get("colors_min"))
        self.assertAlmostEqual(result["forward"]["colors_min"][0], 285.0, places=3)
        self.assertAlmostEqual(result["forward"]["colors_max"][0], 300.540, places=3)
        self.assertEqual(result["forward"]["colors_shape"], (38400, 1))

    async def test_streamlines_static_mixer_standard_with_widths(self):
        async with new_stage() as stage:
            await import_to_stage(get_test_data_path("StaticMixer.cgns"), "/World/StaticMixer")
            base_path: str = "/World/StaticMixer/Base/StaticMixer"

            dataset_path: str = f"{base_path}/B1_P3"
            viz_path: str = f"/World/CAE/Streamlines_B1_P3"
            sphere_path: str = f"/World/CAE/Sphere"
            await execute_command(
                "CreateCaeVizStreamlines", dataset_path=dataset_path, prim_path=viz_path, type="standard"
            )
            await execute_command("CreateCaeVizMeshPrim", prim_type="UnitSphere", prim_path=sphere_path)
            await execute_command("TransformPrimSRT", path=sphere_path, new_scale=[0.2, 0.2, 0.2])
            await wait_for_update()

            viz_prim: Usd.Prim = stage.GetPrimAtPath(viz_path)
            sphere_prim: Usd.Prim = stage.GetPrimAtPath(sphere_path)
            self.assertTrue(viz_prim.IsValid())
            self.assertTrue(sphere_prim.IsValid())

            streamlines_api: cae_viz.StreamlinesAPI = cae_viz.StreamlinesAPI(viz_prim)
            streamlines_api.GetDirectionAttr().Set(cae_viz.Tokens.forward)

            ds_api: cae_viz.DatasetSelectionAPI = cae_viz.DatasetSelectionAPI(viz_prim, "seeds")
            ds_api.GetTargetRel().SetTargets({sphere_prim.GetPath()})

            vs_api = cae_viz.FieldSelectionAPI(viz_prim, "velocities")
            vs_api.GetTargetRel().SetTargets(
                [
                    f"{base_path}/Flow_Solution/VelocityX",
                    f"{base_path}/Flow_Solution/VelocityY",
                    f"{base_path}/Flow_Solution/VelocityZ",
                ]
            )

            # first; pass constant width and confirm that's what we get.
            streamlines_api.GetWidthAttr().Set(0.05)
            await wait_for_update()

            # Get forward direction points
            usdrt_curves = UsdGeomRT.BasisCurves(usd_utils.get_prim_rt(viz_prim))
            widths = get_vtrt_array_as_numpy(usdrt_curves.GetPrim().GetAttribute("primvars:widths"))
            np.testing.assert_allclose(widths, 0.05, atol=self.tolerance)

            # now; pass field-specific width and confirm that's what we get.
            vs_api = cae_viz.FieldSelectionAPI(viz_prim, "widths")
            vs_api.GetTargetRel().SetTargets([f"{base_path}/Flow_Solution/Temperature"])

            mapping_api = cae_viz.FieldMappingAPI(viz_prim, "widths")
            mapping_api.GetRangeAttr().Set((0.045, 0.1))

            await wait_for_update()

            widths = get_vtrt_array_as_numpy(usdrt_curves.GetPrim().GetAttribute("primvars:widths"))
            self.assertTrue(widths.min() >= 0.045)
            self.assertTrue(widths.max() <= 0.1)
            self.assertAlmostEqual(widths.mean(), 0.072453074, places=3)

            domain = mapping_api.GetDomainAttr().Get()
            self.assertAlmostEqual(domain[0], 285.0, places=3)
            self.assertAlmostEqual(domain[1], 315.0, places=3)

            # change range and confirm that's what we get.
            mapping_api.GetRangeAttr().Set((0.01, 0.05))
            await wait_for_update()
            widths = get_vtrt_array_as_numpy(usdrt_curves.GetPrim().GetAttribute("primvars:widths"))
            self.assertTrue(widths.min() >= 0.01)
            self.assertTrue(widths.max() <= 0.05)
            self.assertAlmostEqual(widths.mean(), 0.02996587, places=3)

            # change array to Pres; drop rescale range and confirm that range remains unchanged.
            rescale_range_api = cae_viz.RescaleRangeAPI(viz_prim, "widths")
            rescale_range_api.GetIncludesRel().SetTargets([])

            vs_api.GetTargetRel().SetTargets([f"{base_path}/Flow_Solution/Pressure"])
            await wait_for_update()

            widths = get_vtrt_array_as_numpy(usdrt_curves.GetPrim().GetAttribute("primvars:widths"))
            self.assertTrue(widths.min() >= (0.01 - self.tolerance))
            self.assertTrue(widths.max() <= (0.05 + self.tolerance))
            self.assertAlmostEqual(widths.mean(), 0.042299, places=3)

            domain = mapping_api.GetDomainAttr().Get()
            self.assertAlmostEqual(domain[0], 285.0, places=3)
            self.assertAlmostEqual(domain[1], 315.0, places=3)

    async def test_streamlines_seeds_outside_bounds(self):
        """Test that streamlines with seeds outside dataset bounds doesn't raise errors."""
        async with new_stage() as stage:
            await import_to_stage(get_test_data_path("StaticMixer.cgns"), "/World/StaticMixer")
            base_path: str = "/World/StaticMixer/Base/StaticMixer"

            dataset_path: str = f"{base_path}/B1_P3"
            viz_path: str = f"/World/CAE/Streamlines_Outside"
            sphere_path: str = f"/World/CAE/SphereOutside"

            # Create streamlines and sphere
            await execute_command(
                "CreateCaeVizStreamlines", dataset_path=dataset_path, prim_path=viz_path, type="standard"
            )
            await execute_command("CreateCaeVizMeshPrim", prim_type="UnitSphere", prim_path=sphere_path)

            # Move sphere far outside the dataset bounds
            await execute_command(
                "TransformPrimSRT", path=sphere_path, new_translation=[100, 100, 100], new_scale=[0.2, 0.2, 0.2]
            )
            await wait_for_update()

            viz_prim: Usd.Prim = stage.GetPrimAtPath(viz_path)
            sphere_prim: Usd.Prim = stage.GetPrimAtPath(sphere_path)
            self.assertTrue(viz_prim.IsValid())
            self.assertTrue(sphere_prim.IsValid())

            streamlines_api: cae_viz.StreamlinesAPI = cae_viz.StreamlinesAPI(viz_prim)
            streamlines_api.GetDirectionAttr().Set(cae_viz.Tokens.forward)

            ds_api: cae_viz.DatasetSelectionAPI = cae_viz.DatasetSelectionAPI(viz_prim, "seeds")
            ds_api.GetTargetRel().SetTargets({sphere_prim.GetPath()})

            vs_api = cae_viz.FieldSelectionAPI(viz_prim, "velocities")
            vs_api.GetTargetRel().SetTargets(
                [
                    f"{base_path}/Flow_Solution/VelocityX",
                    f"{base_path}/Flow_Solution/VelocityY",
                    f"{base_path}/Flow_Solution/VelocityZ",
                ]
            )

            # This should complete without raising errors (though streamlines will be empty/invisible)
            await wait_for_update()

            # Verify the prim is still valid and invisible (since no streamlines were generated)
            usdrt_curves = UsdGeomRT.BasisCurves(usd_utils.get_prim_rt(viz_prim))
            attr = usdrt_curves.GetVisibilityAttr()
            visibility = usdrt_curves.GetVisibilityAttr().Get()
            # When seeds are outside bounds, prim should be invisible due to QuietableException
            self.assertEqual(
                visibility,
                UsdGeomRT.Tokens.invisible,
                "Streamlines prim should be invisible when seeds are outside dataset bounds",
            )

    async def test_streamlines_nanovdb_point_cloud_applies_splatting(self):
        """Test that NanoVDB streamlines apply DatasetGaussianSplattingAPI for point cloud data.

        Point clouds have no cells, so the voxelization kernel cannot sample them directly
        (CellLocatorAPI.find_cell_containing_point always returns False). The streamlines
        command must detect this (nb_cells <= 0) and apply DatasetGaussianSplattingAPI
        to splat point values onto a grid before voxelization.
        """
        import os
        import tempfile

        async with new_stage() as stage:
            # Create a minimal point cloud NPZ
            n_points = 100
            rng = np.random.default_rng(42)
            coords = rng.uniform(-1.0, 1.0, (n_points, 3)).astype(np.float32)
            velocity = np.tile(np.array([1.0, 0.0, 0.0], dtype=np.float32), (n_points, 1))

            with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
                npz_path = f.name
                np.savez(f, coordinates=coords, velocity=velocity)

            try:
                from omni.cae.importer.npz import import_to_stage as import_npz

                await import_npz(npz_path, "/World/PointCloud", schema_type="Point Cloud")
                await wait_for_update()

                dataset_path = "/World/PointCloud/NumPyDataSet"
                viz_path = "/World/CAE/Streamlines_PointCloud"

                await execute_command(
                    "CreateCaeVizStreamlines", dataset_path=dataset_path, prim_path=viz_path, type="nanovdb"
                )
                await wait_for_update()

                viz_prim = stage.GetPrimAtPath(viz_path)
                self.assertTrue(viz_prim.IsValid())

                # The fix: for point cloud sources (nb_cells=0), the command should apply
                # DatasetGaussianSplattingAPI alongside DatasetVoxelizationAPI.
                self.assertTrue(
                    viz_prim.HasAPI(cae_viz.DatasetVoxelizationAPI, "source"),
                    "NanoVDB streamlines should have DatasetVoxelizationAPI applied",
                )
                self.assertTrue(
                    viz_prim.HasAPI(cae_viz.DatasetGaussianSplattingAPI, "source"),
                    "NanoVDB streamlines on point cloud should have DatasetGaussianSplattingAPI applied",
                )
            finally:
                os.unlink(npz_path)

    async def test_streamlines_nanovdb_mesh_no_splatting(self):
        """Test that NanoVDB streamlines do NOT apply DatasetGaussianSplattingAPI for mesh data.

        Mesh data has cells, so voxelization can sample directly — no splatting needed.
        """
        async with new_stage() as stage:
            await import_to_stage(get_test_data_path("StaticMixer.cgns"), "/World/StaticMixer")
            base_path = "/World/StaticMixer/Base/StaticMixer"
            dataset_path = f"{base_path}/B1_P3"
            viz_path = "/World/CAE/Streamlines_Mesh"

            await execute_command(
                "CreateCaeVizStreamlines", dataset_path=dataset_path, prim_path=viz_path, type="nanovdb"
            )
            await wait_for_update()

            viz_prim = stage.GetPrimAtPath(viz_path)
            self.assertTrue(viz_prim.IsValid())

            self.assertTrue(
                viz_prim.HasAPI(cae_viz.DatasetVoxelizationAPI, "source"),
                "NanoVDB streamlines should have DatasetVoxelizationAPI applied",
            )
            self.assertFalse(
                viz_prim.HasAPI(cae_viz.DatasetGaussianSplattingAPI, "source"),
                "NanoVDB streamlines on mesh should NOT have DatasetGaussianSplattingAPI applied",
            )
