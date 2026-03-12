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

import numpy as np
import omni.kit.test
import warp as wp
from omni.cae.data import usd_utils
from omni.cae.data.commands import execute_command
from omni.cae.importer.cgns import import_to_stage
from omni.cae.schema import viz as cae_viz
from omni.cae.testing import get_test_data_path, get_vtrt_array_as_numpy, new_stage, wait_for_update
from omni.kit.app import get_app
from omni.timeline import get_timeline_interface
from omni.usd import get_stage_next_free_path
from pxr import Usd, UsdShade
from usdrt import UsdGeom as UsdGeomRT


class TestPoints(omni.kit.test.AsyncTestCase):
    tolerance = 1e-5

    async def forward_frames(self, timeline, frames: int):
        for i in range(frames):
            timeline.forward_one_frame()
        await wait_for_update()

    async def create_points(self, stage: Usd.Stage, dataset_path: str) -> Usd.Prim:
        ds_prim = stage.GetPrimAtPath(dataset_path)
        self.assertIsNotNone(ds_prim, "Dataset prim should be valid")

        viz_path = get_stage_next_free_path(stage, f"/World/CAE/Points_{ds_prim.GetName()}", False)

        await execute_command("CreateCaeVizPoints", dataset_path=dataset_path, prim_path=viz_path)
        await wait_for_update()
        prim = stage.GetPrimAtPath(viz_path)
        self.assertIsNotNone(prim, "Points prim should be valid")
        return prim

    async def test_points_static_mixer(self):
        async with new_stage() as stage:
            await import_to_stage(get_test_data_path("StaticMixer.cgns"), "/World/StaticMixer")
            b1_p3_path = "/World/StaticMixer/Base/StaticMixer/B1_P3"
            prim_b1_p3 = await self.create_points(stage, b1_p3_path)
            self.assertTrue(prim_b1_p3.HasAPI(cae_viz.PointsAPI))
            points_api = cae_viz.PointsAPI(prim_b1_p3)

            # let's validate attributes on the prim.
            prim_rt = usd_utils.get_prim_rt(prim_b1_p3)
            points_rt = UsdGeomRT.Points(prim_rt)

            np_points = get_vtrt_array_as_numpy(points_rt.GetPointsAttr())
            self.assertIsNotNone(np_points)
            self.assertEqual(np_points.shape[0], 2786)
            self.assertEqual(np_points.shape[1], 3)
            np.testing.assert_allclose(np_points.min(axis=0).tolist(), [-2.0, -3.0, -2.0], atol=self.tolerance)
            np.testing.assert_allclose(np_points.max(axis=0).tolist(), [2.0, 3.0, 2.0], atol=self.tolerance)

            # toggle use_cell_points and confirm that the points are same (for B1_P3, it should be true)
            points_api.GetUseCellPointsAttr().Set(True)
            await wait_for_update()
            np_points = get_vtrt_array_as_numpy(points_rt.GetPointsAttr())
            self.assertIsNotNone(np_points)
            self.assertEqual(np_points.shape[0], 2786)
            self.assertEqual(np_points.shape[1], 3)
            np.testing.assert_allclose(np_points.min(axis=0).tolist(), [-2.0, -3.0, -2.0], atol=self.tolerance)
            np.testing.assert_allclose(np_points.max(axis=0).tolist(), [2.0, 3.0, 2.0], atol=self.tolerance)

            # set max_count to 100 and confirm that the points are 100
            points_api.GetMaxCountAttr().Set(100)
            await wait_for_update()
            np_points = get_vtrt_array_as_numpy(points_rt.GetPointsAttr())
            self.assertIsNotNone(np_points)
            self.assertEqual(np_points.shape[0], 100)
            self.assertEqual(np_points.shape[1], 3)

            # set to 0 and confirm that the points are all passed
            points_api.GetMaxCountAttr().Set(0)
            await wait_for_update()
            np_points = get_vtrt_array_as_numpy(points_rt.GetPointsAttr())
            self.assertIsNotNone(np_points)
            self.assertEqual(np_points.shape[0], 2786)
            self.assertEqual(np_points.shape[1], 3)

            # Now do the same for `in1` and it should change based on whether cell points are used or not.
            in1_path = "/World/StaticMixer/Base/StaticMixer/in1"
            prim_in1 = await self.create_points(stage, in1_path)
            self.assertTrue(prim_in1.HasAPI(cae_viz.PointsAPI))
            in1_points_api = cae_viz.PointsAPI(prim_in1)
            await wait_for_update()
            in1_points_rt = UsdGeomRT.Points(usd_utils.get_prim_rt(prim_in1))
            np_points = get_vtrt_array_as_numpy(in1_points_rt.GetPointsAttr())
            self.assertIsNotNone(np_points)
            self.assertEqual(np_points.shape[0], 12)
            self.assertEqual(np_points.shape[1], 3)
            np.testing.assert_allclose(np_points.min(axis=0).tolist(), [-1.5, -3, 0.5], atol=self.tolerance)
            np.testing.assert_allclose(np_points.max(axis=0).tolist(), [-0.5, -3, 1.5], atol=self.tolerance)

            # toggle use_cell_points and confirm that the points are different
            in1_points_api.GetUseCellPointsAttr().Set(False)
            await wait_for_update()
            np_points = get_vtrt_array_as_numpy(in1_points_rt.GetPointsAttr())
            self.assertIsNotNone(np_points)
            self.assertEqual(np_points.shape[0], 2786)
            self.assertEqual(np_points.shape[1], 3)
            np.testing.assert_allclose(np_points.min(axis=0).tolist(), [-2.0, -3.0, -2.0], atol=self.tolerance)
            np.testing.assert_allclose(np_points.max(axis=0).tolist(), [2.0, 3.0, 2.0], atol=self.tolerance)

    async def test_points_static_mixer_with_colors(self):
        async with new_stage() as stage:
            await import_to_stage(get_test_data_path("StaticMixer.cgns"), "/World/StaticMixer")
            b1_p3_path = "/World/StaticMixer/Base/StaticMixer/B1_P3"
            temp_path = "/World/StaticMixer/Base/StaticMixer/Flow_Solution/Temperature"
            prim_b1_p3 = await self.create_points(stage, b1_p3_path)
            self.assertTrue(prim_b1_p3.HasAPI(cae_viz.PointsAPI))
            points_api = cae_viz.PointsAPI(prim_b1_p3)

            # confirm no colors are present
            prim_rt = usd_utils.get_prim_rt(prim_b1_p3)
            points_rt = UsdGeomRT.Points(prim_rt)
            self.assertFalse(points_rt.GetPrim().GetAttribute("primvars:colors").IsValid())

            # confirm shader's enable_coloring is false
            shader = UsdShade.Shader(prim_b1_p3.GetPrimAtPath("Materials/ScalarColor/Shader"))
            self.assertTrue(shader.GetInput("enable_coloring").Get() == False)

            # add colors
            cae_viz.FieldSelectionAPI.Apply(prim_b1_p3, "colors")
            fs_api = cae_viz.FieldSelectionAPI(prim_b1_p3, "colors")
            fs_api.CreateTargetRel().SetTargets([temp_path])
            await wait_for_update()
            self.assertTrue(prim_rt.GetAttribute("primvars:colors").IsValid())

            np_colors = get_vtrt_array_as_numpy(prim_rt.GetAttribute("primvars:colors"))
            self.assertIsNotNone(np_colors)
            self.assertEqual(np_colors.shape[0], 2786)
            self.assertEqual(np_colors.shape[1], 1)
            np.testing.assert_allclose(np_colors.min(axis=0).tolist(), 285.0, atol=self.tolerance)
            np.testing.assert_allclose(np_colors.max(axis=0).tolist(), 315.000458, atol=self.tolerance)

            # confirm shader's enable_coloring is true
            shader = UsdShade.Shader(prim_b1_p3.GetPrimAtPath("Materials/ScalarColor/Shader"))
            self.assertTrue(shader.GetInput("enable_coloring").Get() == True)

            # Now remove colors, and confirm that shader's enable_coloring is false
            fs_api.GetTargetRel().ClearTargets(False)
            await wait_for_update()

            # this will not drop the primvar, but the shader's enable_coloring will be false
            self.assertTrue(prim_rt.GetAttribute("primvars:colors").IsValid())
            self.assertTrue(shader.GetInput("enable_coloring").Get() == False)

    async def test_points_static_mixer_with_widths(self):
        async with new_stage() as stage:
            await import_to_stage(get_test_data_path("StaticMixer.cgns"), "/World/StaticMixer")
            b1_p3_path = "/World/StaticMixer/Base/StaticMixer/B1_P3"
            temp_path = "/World/StaticMixer/Base/StaticMixer/Flow_Solution/Temperature"
            pres_path = "/World/StaticMixer/Base/StaticMixer/Flow_Solution/Pressure"
            prim_b1_p3 = await self.create_points(stage, b1_p3_path)
            self.assertTrue(prim_b1_p3.HasAPI(cae_viz.PointsAPI))
            points_api = cae_viz.PointsAPI(prim_b1_p3)

            prim_rt = usd_utils.get_prim_rt(prim_b1_p3)
            points_rt = UsdGeomRT.Points(prim_rt)

            # Test 1: Set constant width and verify it's applied uniformly
            points_api.GetWidthAttr().Set(0.05)
            await wait_for_update()

            widths = get_vtrt_array_as_numpy(prim_rt.GetAttribute("primvars:widths"))
            self.assertIsNotNone(widths)
            np.testing.assert_allclose(widths, 0.05, atol=self.tolerance)

            # Test 2: Add field-based widths using Temperature field
            cae_viz.FieldSelectionAPI.Apply(prim_b1_p3, "widths")
            fs_api = cae_viz.FieldSelectionAPI(prim_b1_p3, "widths")
            fs_api.CreateTargetRel().SetTargets([temp_path])

            # Add field mapping with custom range
            cae_viz.FieldMappingAPI.Apply(prim_b1_p3, "widths")
            mapping_api = cae_viz.FieldMappingAPI(prim_b1_p3, "widths")
            mapping_api.GetRangeAttr().Set((0.045, 0.1))

            await wait_for_update()

            widths = get_vtrt_array_as_numpy(prim_rt.GetAttribute("primvars:widths"))
            self.assertIsNotNone(widths)
            self.assertEqual(widths.shape[0], 2786)
            self.assertGreaterEqual(widths.min(), 0.045 - self.tolerance)
            self.assertLessEqual(widths.max(), 0.1 + self.tolerance)
            self.assertAlmostEqual(widths.mean(), 0.07245971, places=3)

            # Verify domain was computed from the Temperature field
            domain = mapping_api.GetDomainAttr().Get()
            self.assertIsNotNone(domain)
            self.assertAlmostEqual(domain[0], 285.0, places=3)
            self.assertAlmostEqual(domain[1], 315.0, places=3)

            # Test 3: Change range and verify widths are remapped
            mapping_api.GetRangeAttr().Set((0.01, 0.05))
            await wait_for_update()

            widths = get_vtrt_array_as_numpy(prim_rt.GetAttribute("primvars:widths"))
            self.assertIsNotNone(widths)
            self.assertGreaterEqual(widths.min(), 0.01 - self.tolerance)
            self.assertLessEqual(widths.max(), 0.05 + self.tolerance)
            self.assertAlmostEqual(widths.mean(), 0.029970698, places=3)

            # Test 4: Change to Pressure field, drop rescale range, verify domain remains unchanged
            cae_viz.RescaleRangeAPI.Apply(prim_b1_p3, "widths")
            rescale_range_api = cae_viz.RescaleRangeAPI(prim_b1_p3, "widths")
            rescale_range_api.GetIncludesRel().SetTargets([])

            fs_api.GetTargetRel().SetTargets([pres_path])
            await wait_for_update()

            widths = get_vtrt_array_as_numpy(prim_rt.GetAttribute("primvars:widths"))
            self.assertIsNotNone(widths)
            self.assertGreaterEqual(widths.min(), 0.01 - self.tolerance)
            self.assertLessEqual(widths.max(), 0.05 + self.tolerance)
            self.assertAlmostEqual(widths.mean(), 0.04946224, places=3)

            # Domain should remain unchanged from Temperature
            domain = mapping_api.GetDomainAttr().Get()
            self.assertIsNotNone(domain)
            self.assertAlmostEqual(domain[0], 285.0, places=3)
            self.assertAlmostEqual(domain[1], 315.0, places=3)

            # Test 5: Remove field-based widths and verify constant width is used again
            fs_api.GetTargetRel().ClearTargets(False)
            await wait_for_update()

            widths = get_vtrt_array_as_numpy(prim_rt.GetAttribute("primvars:widths"))
            self.assertIsNotNone(widths)
            # Should fall back to the constant width of 0.05 that was set earlier
            np.testing.assert_allclose(widths, 0.05, atol=self.tolerance)

    async def test_points_animated_beam(self):
        async with new_stage(path=get_test_data_path("animated_beam/animated_beam.usda")) as stage:
            # set up timeline; we use time_codes_per_second of 1.0 for convenience
            timeline = get_timeline_interface()
            timeline.set_time_codes_per_second(1.0)
            timeline.set_start_time(0.0)
            timeline.set_end_time(190.0)
            timeline.set_current_time(0.0)
            await wait_for_update()

            ds_path = "/World/animated_beam_vtu/VTKUnstructuredGrid"
            prim_ds = await self.create_points(stage, ds_path)
            self.assertTrue(prim_ds.HasAPI(cae_viz.PointsAPI))
            points_api = cae_viz.PointsAPI(prim_ds)

            # let's validate attributes on the prim.
            prim_rt = usd_utils.get_prim_rt(prim_ds)
            points_rt = UsdGeomRT.Points(prim_rt)

            np_points = get_vtrt_array_as_numpy(points_rt.GetPointsAttr()).copy()
            self.assertIsNotNone(np_points)
            self.assertEqual(np_points.shape[0], 12221)
            self.assertEqual(np_points.shape[1], 3)
            np.testing.assert_allclose(np_points.min(axis=0).tolist(), [-5, 0, -5], atol=self.tolerance)
            np.testing.assert_allclose(np_points.max(axis=0).tolist(), [5, 100, 5], atol=self.tolerance)

            # forward frame by 5 and confirm nothing changed
            await self.forward_frames(timeline, 5)
            np_points_new = get_vtrt_array_as_numpy(points_rt.GetPointsAttr())
            self.assertIsNotNone(np_points_new)
            np.testing.assert_allclose(np_points_new, np_points, atol=self.tolerance)

            # forward frame by 5 more and confirm points changed
            await self.forward_frames(timeline, 5)
            np_points_new = get_vtrt_array_as_numpy(points_rt.GetPointsAttr())
            self.assertIsNotNone(np_points_new)
            self.assertFalse(np.allclose(np_points_new, np_points, atol=self.tolerance))
            np.testing.assert_allclose(np_points_new.min(axis=0).tolist(), [-4.47368, 0, -5], atol=self.tolerance)
            np.testing.assert_allclose(np_points_new.max(axis=0).tolist(), [15.5526, 100, 5], atol=self.tolerance)

            # forward frame by 10 more and confirm points changed
            await self.forward_frames(timeline, 10)
            np_points_new = get_vtrt_array_as_numpy(points_rt.GetPointsAttr())
            self.assertIsNotNone(np_points_new)
            self.assertFalse(np.allclose(np_points_new, np_points, atol=self.tolerance))

            timeline.set_current_time(0.0)
            await wait_for_update()

            # now add temporal and confirm points are interpolated
            cae_viz.OperatorTemporalAPI.Apply(prim_ds)
            temporal_api = cae_viz.OperatorTemporalAPI(prim_ds)
            temporal_api.CreateEnableFieldInterpolationAttr().Set(True)
            await wait_for_update()

            np_points_new = get_vtrt_array_as_numpy(points_rt.GetPointsAttr())
            self.assertIsNotNone(np_points_new)
            np.testing.assert_allclose(np_points_new, np_points, atol=self.tolerance)
            np.testing.assert_allclose(np_points.min(axis=0).tolist(), [-5, 0, -5], atol=self.tolerance)
            np.testing.assert_allclose(np_points.max(axis=0).tolist(), [5, 100, 5], atol=self.tolerance)

            # forward frame by 5 and confirm points are interpolated
            await self.forward_frames(timeline, 5)
            np_points_new = get_vtrt_array_as_numpy(points_rt.GetPointsAttr())
            self.assertIsNotNone(np_points_new)
            self.assertFalse(np.allclose(np_points_new, np_points, atol=self.tolerance))
            np.testing.assert_allclose(np_points_new.min(axis=0).tolist(), [-4.736842, 0, -5], atol=self.tolerance)
            np.testing.assert_allclose(np_points_new.max(axis=0).tolist(), [10.276299, 100, 5], atol=self.tolerance)

    async def test_points_animated_beam_with_colors_and_widths(self):
        """Test time-varying colors (RTData) and static widths (displ) with temporal interpolation."""
        async with new_stage(path=get_test_data_path("animated_beam/animated_beam.usda")) as stage:
            # set up timeline; we use time_codes_per_second of 1.0 for convenience
            timeline = get_timeline_interface()
            timeline.set_time_codes_per_second(1.0)
            timeline.set_start_time(0.0)
            timeline.set_end_time(190.0)
            timeline.set_current_time(0.0)
            await wait_for_update()

            ds_path = "/World/animated_beam_vtu/VTKUnstructuredGrid"
            rtdata_path = "/World/animated_beam_vtu/PointData/RTData"
            displ_path = "/World/animated_beam_vtu/PointData/displ"

            prim_ds = await self.create_points(stage, ds_path)
            self.assertTrue(prim_ds.HasAPI(cae_viz.PointsAPI))
            points_api = cae_viz.PointsAPI(prim_ds)

            prim_rt = usd_utils.get_prim_rt(prim_ds)
            points_rt = UsdGeomRT.Points(prim_rt)

            # Add colors mapped to RTData (time-varying)
            colors_fs_api = cae_viz.FieldSelectionAPI(prim_ds, "colors")
            colors_fs_api.CreateTargetRel().SetTargets([rtdata_path])
            await wait_for_update()

            # Add widths mapped to displ (static field)
            widths_fs_api = cae_viz.FieldSelectionAPI(prim_ds, "widths")
            widths_fs_api.CreateTargetRel().SetTargets([displ_path])
            await wait_for_update()

            # Get initial colors and widths at time 0
            np_colors_t0 = get_vtrt_array_as_numpy(prim_rt.GetAttribute("primvars:colors")).copy()
            np_widths_t0 = get_vtrt_array_as_numpy(prim_rt.GetAttribute("primvars:widths")).copy()
            self.assertIsNotNone(np_colors_t0)
            self.assertIsNotNone(np_widths_t0)
            self.assertEqual(np_colors_t0.shape[0], 12221)
            self.assertEqual(np_widths_t0.shape[0], 12221)

            # Forward frame by 5 and confirm colors changed but widths remain same (without interpolation)
            await self.forward_frames(timeline, 5)
            np_colors_t5 = get_vtrt_array_as_numpy(prim_rt.GetAttribute("primvars:colors"))
            np_widths_t5 = get_vtrt_array_as_numpy(prim_rt.GetAttribute("primvars:widths"))
            np.testing.assert_allclose(np_colors_t5, np_colors_t0, atol=self.tolerance)
            np.testing.assert_allclose(np_widths_t5, np_widths_t0, atol=self.tolerance)

            # Forward frame by 5 more (to time 10) and confirm colors changed
            await self.forward_frames(timeline, 5)
            np_colors_t10 = get_vtrt_array_as_numpy(prim_rt.GetAttribute("primvars:colors"))
            np_widths_t10 = get_vtrt_array_as_numpy(prim_rt.GetAttribute("primvars:widths"))
            self.assertFalse(np.allclose(np_colors_t10, np_colors_t0, atol=self.tolerance))
            np.testing.assert_allclose(np_widths_t10, np_widths_t0, atol=self.tolerance)

            # Reset timeline and enable temporal interpolation
            timeline.set_current_time(0.0)
            await wait_for_update()

            cae_viz.OperatorTemporalAPI.Apply(prim_ds)
            temporal_api = cae_viz.OperatorTemporalAPI(prim_ds)
            temporal_api.CreateEnableFieldInterpolationAttr().Set(True)
            await wait_for_update()

            # Verify we're back at time 0
            np_colors_t0_interp = get_vtrt_array_as_numpy(prim_rt.GetAttribute("primvars:colors"))
            np_widths_t0_interp = get_vtrt_array_as_numpy(prim_rt.GetAttribute("primvars:widths"))
            np.testing.assert_allclose(np_colors_t0_interp, np_colors_t0, atol=self.tolerance)
            np.testing.assert_allclose(np_widths_t0_interp, np_widths_t0, atol=self.tolerance)

            # Forward frame by 5 (halfway between time 0 and time 10)
            # With interpolation, colors should be interpolated between t0 and t10
            await self.forward_frames(timeline, 5)
            np_colors_t5_interp = get_vtrt_array_as_numpy(prim_rt.GetAttribute("primvars:colors"))
            np_widths_t5_interp = get_vtrt_array_as_numpy(prim_rt.GetAttribute("primvars:widths"))
            self.assertFalse(np.allclose(np_colors_t5_interp, np_colors_t0, atol=self.tolerance))
            self.assertFalse(np.allclose(np_colors_t5_interp, np_colors_t10, atol=self.tolerance))

            # the "lerp" still adds a small error for widths; we should allow for that for now.
            # ultimately, we want to skip lerp if the field is static.
            np.testing.assert_allclose(np_widths_t5_interp, np_widths_t0, atol=1e-2)

    async def create_glyphs(self, stage: Usd.Stage, dataset_path: str) -> Usd.Prim:
        ds_prim = stage.GetPrimAtPath(dataset_path)
        self.assertIsNotNone(ds_prim, "Dataset prim should be valid")

        viz_path = get_stage_next_free_path(stage, f"/World/CAE/Glyphs_{ds_prim.GetName()}", False)

        await execute_command("CreateCaeVizGlyphs", dataset_path=dataset_path, prim_path=viz_path)
        await wait_for_update()
        prim = stage.GetPrimAtPath(viz_path)
        self.assertIsNotNone(prim, "Glyphs prim should be valid")
        return prim

    async def test_glyphs_static_mixer(self):
        async with new_stage() as stage:
            await import_to_stage(get_test_data_path("StaticMixer.cgns"), "/World/StaticMixer")
            b1_p3_path = "/World/StaticMixer/Base/StaticMixer/B1_P3"
            prim_b1_p3 = await self.create_glyphs(stage, b1_p3_path)
            self.assertTrue(prim_b1_p3.HasAPI(cae_viz.GlyphsAPI))
            glyphs_api = cae_viz.GlyphsAPI(prim_b1_p3)

            # let's validate attributes on the prim.
            prim_rt = usd_utils.get_prim_rt(prim_b1_p3)
            glyphs_rt = UsdGeomRT.PointInstancer(prim_rt)

            np_positions = get_vtrt_array_as_numpy(glyphs_rt.GetPositionsAttr())
            self.assertIsNotNone(np_positions)
            self.assertEqual(np_positions.shape[0], 2786)
            self.assertEqual(np_positions.shape[1], 3)
            np.testing.assert_allclose(np_positions.min(axis=0).tolist(), [-2.0, -3.0, -2.0], atol=self.tolerance)
            np.testing.assert_allclose(np_positions.max(axis=0).tolist(), [2.0, 3.0, 2.0], atol=self.tolerance)

            # verify proto indices
            np_proto_indices = get_vtrt_array_as_numpy(glyphs_rt.GetProtoIndicesAttr())
            self.assertIsNotNone(np_proto_indices)
            self.assertEqual(np_proto_indices.shape[0], 2786)
            np.testing.assert_array_equal(np_proto_indices, 0)

            # toggle use_cell_points and confirm that the positions are same (for B1_P3, it should be true)
            glyphs_api.GetUseCellPointsAttr().Set(True)
            await wait_for_update()
            np_positions = get_vtrt_array_as_numpy(glyphs_rt.GetPositionsAttr())
            self.assertIsNotNone(np_positions)
            self.assertEqual(np_positions.shape[0], 2786)
            self.assertEqual(np_positions.shape[1], 3)
            np.testing.assert_allclose(np_positions.min(axis=0).tolist(), [-2.0, -3.0, -2.0], atol=self.tolerance)
            np.testing.assert_allclose(np_positions.max(axis=0).tolist(), [2.0, 3.0, 2.0], atol=self.tolerance)

            # set max_count to 100 and confirm that the positions are 100
            glyphs_api.GetMaxCountAttr().Set(100)
            await wait_for_update()
            np_positions = get_vtrt_array_as_numpy(glyphs_rt.GetPositionsAttr())
            self.assertIsNotNone(np_positions)
            self.assertEqual(np_positions.shape[0], 100)
            self.assertEqual(np_positions.shape[1], 3)

            # set to 0 and confirm that the positions are all passed
            glyphs_api.GetMaxCountAttr().Set(0)
            await wait_for_update()
            np_positions = get_vtrt_array_as_numpy(glyphs_rt.GetPositionsAttr())
            self.assertIsNotNone(np_positions)
            self.assertEqual(np_positions.shape[0], 2786)
            self.assertEqual(np_positions.shape[1], 3)

            # Now do the same for `in1` and it should change based on whether cell points are used or not.
            in1_path = "/World/StaticMixer/Base/StaticMixer/in1"
            prim_in1 = await self.create_glyphs(stage, in1_path)
            self.assertTrue(prim_in1.HasAPI(cae_viz.GlyphsAPI))
            in1_glyphs_api = cae_viz.GlyphsAPI(prim_in1)
            await wait_for_update()
            in1_glyphs_rt = UsdGeomRT.PointInstancer(usd_utils.get_prim_rt(prim_in1))
            np_positions = get_vtrt_array_as_numpy(in1_glyphs_rt.GetPositionsAttr())
            self.assertIsNotNone(np_positions)
            self.assertEqual(np_positions.shape[0], 12)
            self.assertEqual(np_positions.shape[1], 3)
            np.testing.assert_allclose(np_positions.min(axis=0).tolist(), [-1.5, -3, 0.5], atol=self.tolerance)
            np.testing.assert_allclose(np_positions.max(axis=0).tolist(), [-0.5, -3, 1.5], atol=self.tolerance)

            # toggle use_cell_points and confirm that the positions are different
            in1_glyphs_api.GetUseCellPointsAttr().Set(False)
            await wait_for_update()
            np_positions = get_vtrt_array_as_numpy(in1_glyphs_rt.GetPositionsAttr())
            self.assertIsNotNone(np_positions)
            self.assertEqual(np_positions.shape[0], 2786)
            self.assertEqual(np_positions.shape[1], 3)
            np.testing.assert_allclose(np_positions.min(axis=0).tolist(), [-2.0, -3.0, -2.0], atol=self.tolerance)
            np.testing.assert_allclose(np_positions.max(axis=0).tolist(), [2.0, 3.0, 2.0], atol=self.tolerance)

    async def test_glyphs_static_mixer_with_colors(self):
        async with new_stage() as stage:
            await import_to_stage(get_test_data_path("StaticMixer.cgns"), "/World/StaticMixer")
            b1_p3_path = "/World/StaticMixer/Base/StaticMixer/B1_P3"
            temp_path = "/World/StaticMixer/Base/StaticMixer/Flow_Solution/Temperature"
            prim_b1_p3 = await self.create_glyphs(stage, b1_p3_path)
            self.assertTrue(prim_b1_p3.HasAPI(cae_viz.GlyphsAPI))
            glyphs_api = cae_viz.GlyphsAPI(prim_b1_p3)

            # confirm no colors are present
            prim_rt = usd_utils.get_prim_rt(prim_b1_p3)
            glyphs_rt = UsdGeomRT.PointInstancer(prim_rt)
            self.assertFalse(glyphs_rt.GetPrim().GetAttribute("primvars:colors").IsValid())

            # confirm shader's enable_coloring is false
            shader = UsdShade.Shader(prim_b1_p3.GetPrimAtPath("Materials/ScalarColor/Shader"))
            self.assertTrue(shader.GetInput("enable_coloring").Get() == False)

            # add colors
            colors_fs_api = cae_viz.FieldSelectionAPI(prim_b1_p3, "colors")
            colors_fs_api.CreateTargetRel().SetTargets([temp_path])
            await wait_for_update()
            self.assertTrue(prim_rt.GetAttribute("primvars:colors").IsValid())

            np_colors = get_vtrt_array_as_numpy(prim_rt.GetAttribute("primvars:colors"))
            self.assertIsNotNone(np_colors)
            self.assertEqual(np_colors.shape[0], 2786)
            self.assertEqual(np_colors.shape[1], 1)
            np.testing.assert_allclose(np_colors.min(axis=0).tolist(), 285.0, atol=self.tolerance)
            np.testing.assert_allclose(np_colors.max(axis=0).tolist(), 315.000458, atol=self.tolerance)

            # confirm shader's enable_coloring is true
            shader = UsdShade.Shader(prim_b1_p3.GetPrimAtPath("Materials/ScalarColor/Shader"))
            self.assertTrue(shader.GetInput("enable_coloring").Get() == True)

            # Now remove colors, and confirm that shader's enable_coloring is false
            colors_fs_api.GetTargetRel().ClearTargets(False)
            await wait_for_update()

            # this will not drop the primvar, but the shader's enable_coloring will be false
            self.assertTrue(prim_rt.GetAttribute("primvars:colors").IsValid())
            self.assertTrue(shader.GetInput("enable_coloring").Get() == False)

    async def test_glyphs_static_mixer_with_scales(self):
        async with new_stage() as stage:
            await import_to_stage(get_test_data_path("StaticMixer.cgns"), "/World/StaticMixer")
            b1_p3_path = "/World/StaticMixer/Base/StaticMixer/B1_P3"
            temp_path = "/World/StaticMixer/Base/StaticMixer/Flow_Solution/Temperature"
            pres_path = "/World/StaticMixer/Base/StaticMixer/Flow_Solution/Pressure"
            prim_b1_p3 = await self.create_glyphs(stage, b1_p3_path)
            self.assertTrue(prim_b1_p3.HasAPI(cae_viz.GlyphsAPI))

            prim_rt = usd_utils.get_prim_rt(prim_b1_p3)
            glyphs_rt = UsdGeomRT.PointInstancer(prim_rt)

            # Initially (no scales field), scales should be filled with the default scale (1.0)
            scales = get_vtrt_array_as_numpy(glyphs_rt.GetScalesAttr())
            self.assertIsNotNone(scales)
            self.assertEqual(scales.shape[0], 2786)
            self.assertEqual(scales.shape[1], 3)
            np.testing.assert_allclose(scales, 1.0, atol=self.tolerance)

            # Add field-based scales using Temperature field
            scales_fs_api = cae_viz.FieldSelectionAPI(prim_b1_p3, "scales")
            scales_fs_api.CreateTargetRel().SetTargets([temp_path])

            # Add field mapping with custom range
            cae_viz.FieldMappingAPI.Apply(prim_b1_p3, "scales")
            mapping_api = cae_viz.FieldMappingAPI(prim_b1_p3, "scales")
            mapping_api.GetRangeAttr().Set((0.5, 2.0))
            mapping_api.GetDomainAttr().Set((285, 315.000458))

            await wait_for_update()

            scales = get_vtrt_array_as_numpy(glyphs_rt.GetScalesAttr())
            self.assertIsNotNone(scales)
            self.assertEqual(scales.shape[0], 2786)
            self.assertEqual(scales.shape[1], 3)
            # Since Temperature is a scalar, it gets converted to vec3f with same value for all components
            np.testing.assert_allclose(scales.min(axis=0), 0.5, atol=self.tolerance)
            np.testing.assert_allclose(scales.max(axis=0), 2.0, atol=self.tolerance)

            # Verify domain was computed from the Temperature field
            domain = mapping_api.GetDomainAttr().Get()
            self.assertIsNotNone(domain)
            self.assertAlmostEqual(domain[0], 285.0, places=3)
            self.assertAlmostEqual(domain[1], 315.0, places=3)

            # Change range and verify scales are remapped
            mapping_api.GetRangeAttr().Set((0.1, 1.0))
            await wait_for_update()

            scales = get_vtrt_array_as_numpy(glyphs_rt.GetScalesAttr())
            self.assertIsNotNone(scales)
            self.assertTrue(np.all(scales.min(axis=0) >= 0.1))
            self.assertTrue(np.all(scales.max(axis=0) <= 1.0))

            # Change to Pressure field, drop rescale range, verify domain remains unchanged
            cae_viz.RescaleRangeAPI.Apply(prim_b1_p3, "scales")
            rescale_range_api = cae_viz.RescaleRangeAPI(prim_b1_p3, "scales")
            rescale_range_api.GetIncludesRel().SetTargets([])

            scales_fs_api.GetTargetRel().SetTargets([pres_path])
            await wait_for_update()

            scales = get_vtrt_array_as_numpy(glyphs_rt.GetScalesAttr())
            self.assertIsNotNone(scales)
            self.assertTrue(np.all(scales.min(axis=0) >= (0.1 - self.tolerance)))
            self.assertTrue(np.all(scales.max(axis=0) <= (1.0 + self.tolerance)))

            # Domain should remain unchanged from Temperature
            domain = mapping_api.GetDomainAttr().Get()
            self.assertIsNotNone(domain)
            self.assertAlmostEqual(domain[0], 285.0, places=3)
            self.assertAlmostEqual(domain[1], 315.0, places=3)

            # Remove field-based scales and verify scales fall back to the default scale (1.0)
            scales_fs_api.GetTargetRel().ClearTargets(False)
            await wait_for_update()

            scales = get_vtrt_array_as_numpy(glyphs_rt.GetScalesAttr())
            self.assertIsNotNone(scales)
            self.assertEqual(scales.shape[0], 2786)
            self.assertEqual(scales.shape[1], 3)
            np.testing.assert_allclose(scales, 1.0, atol=self.tolerance)

    async def test_glyphs_default_scale(self):
        async with new_stage() as stage:
            await import_to_stage(get_test_data_path("StaticMixer.cgns"), "/World/StaticMixer")
            b1_p3_path = "/World/StaticMixer/Base/StaticMixer/B1_P3"
            temp_path = "/World/StaticMixer/Base/StaticMixer/Flow_Solution/Temperature"
            prim_b1_p3 = await self.create_glyphs(stage, b1_p3_path)
            self.assertTrue(prim_b1_p3.HasAPI(cae_viz.GlyphsAPI))
            glyphs_api = cae_viz.GlyphsAPI(prim_b1_p3)

            prim_rt = usd_utils.get_prim_rt(prim_b1_p3)
            glyphs_rt = UsdGeomRT.PointInstancer(prim_rt)

            # Default scale is 1.0; all glyph instances should be uniformly scaled to 1.0
            scales = get_vtrt_array_as_numpy(glyphs_rt.GetScalesAttr())
            self.assertIsNotNone(scales)
            self.assertEqual(scales.shape[0], 2786)
            self.assertEqual(scales.shape[1], 3)
            np.testing.assert_allclose(scales, 1.0, atol=self.tolerance)

            # Change default scale to 0.5 and verify all instances are updated
            glyphs_api.GetScaleAttr().Set(0.5)
            await wait_for_update()

            scales = get_vtrt_array_as_numpy(glyphs_rt.GetScalesAttr())
            self.assertIsNotNone(scales)
            self.assertEqual(scales.shape[0], 2786)
            np.testing.assert_allclose(scales, 0.5, atol=self.tolerance)

            # Change default scale to 2.0 and verify all instances are updated
            glyphs_api.GetScaleAttr().Set(2.0)
            await wait_for_update()

            scales = get_vtrt_array_as_numpy(glyphs_rt.GetScalesAttr())
            self.assertIsNotNone(scales)
            np.testing.assert_allclose(scales, 2.0, atol=self.tolerance)

            # Add a field-based scales selection; field values should override the default scale
            cae_viz.FieldSelectionAPI.Apply(prim_b1_p3, "scales")
            scales_fs_api = cae_viz.FieldSelectionAPI(prim_b1_p3, "scales")
            scales_fs_api.CreateTargetRel().SetTargets([temp_path])
            cae_viz.FieldMappingAPI.Apply(prim_b1_p3, "scales")
            mapping_api = cae_viz.FieldMappingAPI(prim_b1_p3, "scales")
            mapping_api.GetRangeAttr().Set((0.1, 1.0))
            mapping_api.GetDomainAttr().Set((285, 315.000458))
            await wait_for_update()

            scales = get_vtrt_array_as_numpy(glyphs_rt.GetScalesAttr())
            self.assertIsNotNone(scales)
            self.assertEqual(scales.shape[0], 2786)
            # Field-based scales should be in [0.1, 1.0], not the default 2.0
            self.assertTrue(np.all(scales <= 1.0 + self.tolerance))
            self.assertTrue(np.all(scales >= 0.1 - self.tolerance))

            # Remove the field; default scale (2.0) should be restored
            scales_fs_api.GetTargetRel().ClearTargets(False)
            await wait_for_update()

            scales = get_vtrt_array_as_numpy(glyphs_rt.GetScalesAttr())
            self.assertIsNotNone(scales)
            self.assertEqual(scales.shape[0], 2786)
            np.testing.assert_allclose(scales, 2.0, atol=self.tolerance)

    async def test_glyphs_static_mixer_with_orientations(self):
        async with new_stage() as stage:
            await import_to_stage(get_test_data_path("StaticMixer.cgns"), "/World/StaticMixer")
            b1_p3_path = "/World/StaticMixer/Base/StaticMixer/B1_P3"
            velocity_x_path = "/World/StaticMixer/Base/StaticMixer/Flow_Solution/VelocityX"
            velocity_y_path = "/World/StaticMixer/Base/StaticMixer/Flow_Solution/VelocityY"
            velocity_z_path = "/World/StaticMixer/Base/StaticMixer/Flow_Solution/VelocityZ"
            prim_b1_p3 = await self.create_glyphs(stage, b1_p3_path)
            self.assertTrue(prim_b1_p3.HasAPI(cae_viz.GlyphsAPI))
            glyphs_api = cae_viz.GlyphsAPI(prim_b1_p3)

            prim_rt = usd_utils.get_prim_rt(prim_b1_p3)
            glyphs_rt = UsdGeomRT.PointInstancer(prim_rt)

            # Initially, orientations should be empty
            orientations_attr = glyphs_rt.GetOrientationsAttr()
            self.assertTrue(orientations_attr.IsValid())
            orientations_value = orientations_attr.Get()
            self.assertEqual(len(orientations_value), 0)

            # Add field-based orientations using Velocity field (as eulerAngles)
            orientations_fs_api = cae_viz.FieldSelectionAPI(prim_b1_p3, "orientations")
            orientations_fs_api.CreateTargetRel().SetTargets([velocity_x_path, velocity_y_path, velocity_z_path])

            # Set orientations mode to eulerAngles
            glyphs_api.GetOrientationsModeAttr().Set("eulerAngles")
            await wait_for_update()

            orientations = get_vtrt_array_as_numpy(glyphs_rt.GetOrientationsAttr())
            self.assertIsNotNone(orientations)
            self.assertEqual(orientations.shape[0], 2786)
            self.assertEqual(orientations.shape[1], 4)  # quaternions are vec4

            # Verify quaternions are normalized (w^2 + x^2 + y^2 + z^2 = 1)
            norms = np.linalg.norm(orientations, axis=1)
            np.testing.assert_allclose(norms, 1.0, atol=1e-3)

            # Store eulerAngles-based orientations
            euler_orientations = orientations.copy()

            # Remove orientations field and verify orientations are empty
            orientations_fs_api.GetTargetRel().ClearTargets(False)
            await wait_for_update()

            orientations_value = glyphs_rt.GetOrientationsAttr().Get()
            self.assertEqual(len(orientations_value), 0)

    async def test_glyphs_animated_beam(self):
        async with new_stage(path=get_test_data_path("animated_beam/animated_beam.usda")) as stage:
            # set up timeline; we use time_codes_per_second of 1.0 for convenience
            timeline = get_timeline_interface()
            timeline.set_time_codes_per_second(1.0)
            timeline.set_start_time(0.0)
            timeline.set_end_time(190.0)
            timeline.set_current_time(0.0)
            await wait_for_update()

            ds_path = "/World/animated_beam_vtu/VTKUnstructuredGrid"
            prim_ds = await self.create_glyphs(stage, ds_path)
            self.assertTrue(prim_ds.HasAPI(cae_viz.GlyphsAPI))
            glyphs_api = cae_viz.GlyphsAPI(prim_ds)

            # let's validate attributes on the prim.
            prim_rt = usd_utils.get_prim_rt(prim_ds)
            glyphs_rt = UsdGeomRT.PointInstancer(prim_rt)

            np_positions = get_vtrt_array_as_numpy(glyphs_rt.GetPositionsAttr()).copy()
            self.assertIsNotNone(np_positions)
            self.assertEqual(np_positions.shape[0], 12221)
            self.assertEqual(np_positions.shape[1], 3)
            np.testing.assert_allclose(np_positions.min(axis=0).tolist(), [-5, 0, -5], atol=self.tolerance)
            np.testing.assert_allclose(np_positions.max(axis=0).tolist(), [5, 100, 5], atol=self.tolerance)

            # forward frame by 5 and confirm nothing changed
            await self.forward_frames(timeline, 5)
            np_positions_new = get_vtrt_array_as_numpy(glyphs_rt.GetPositionsAttr())
            self.assertIsNotNone(np_positions_new)
            np.testing.assert_allclose(np_positions_new, np_positions, atol=self.tolerance)

            # forward frame by 5 more and confirm positions changed
            await self.forward_frames(timeline, 5)
            np_positions_new = get_vtrt_array_as_numpy(glyphs_rt.GetPositionsAttr())
            self.assertIsNotNone(np_positions_new)
            self.assertFalse(np.allclose(np_positions_new, np_positions, atol=self.tolerance))
            np.testing.assert_allclose(np_positions_new.min(axis=0).tolist(), [-4.47368, 0, -5], atol=self.tolerance)
            np.testing.assert_allclose(np_positions_new.max(axis=0).tolist(), [15.5526, 100, 5], atol=self.tolerance)

            # Reset and enable temporal interpolation
            timeline.set_current_time(0.0)
            await wait_for_update()

            cae_viz.OperatorTemporalAPI.Apply(prim_ds)
            temporal_api = cae_viz.OperatorTemporalAPI(prim_ds)
            temporal_api.CreateEnableFieldInterpolationAttr().Set(True)
            await wait_for_update()

            np_positions_new = get_vtrt_array_as_numpy(glyphs_rt.GetPositionsAttr())
            self.assertIsNotNone(np_positions_new)
            np.testing.assert_allclose(np_positions_new, np_positions, atol=self.tolerance)

            # forward frame by 5 and confirm positions are interpolated
            await self.forward_frames(timeline, 5)
            np_positions_new = get_vtrt_array_as_numpy(glyphs_rt.GetPositionsAttr())
            self.assertIsNotNone(np_positions_new)
            self.assertFalse(np.allclose(np_positions_new, np_positions, atol=self.tolerance))
            np.testing.assert_allclose(np_positions_new.min(axis=0).tolist(), [-4.736842, 0, -5], atol=self.tolerance)
            np.testing.assert_allclose(np_positions_new.max(axis=0).tolist(), [10.276299, 100, 5], atol=self.tolerance)

    async def test_glyphs_animated_beam_with_colors_and_scales(self):
        """Test time-varying colors (RTData) and static scales (displ) with temporal interpolation."""
        async with new_stage(path=get_test_data_path("animated_beam/animated_beam.usda")) as stage:
            # set up timeline
            timeline = get_timeline_interface()
            timeline.set_time_codes_per_second(1.0)
            timeline.set_start_time(0.0)
            timeline.set_end_time(190.0)
            timeline.set_current_time(0.0)
            await wait_for_update()

            ds_path = "/World/animated_beam_vtu/VTKUnstructuredGrid"
            rtdata_path = "/World/animated_beam_vtu/PointData/RTData"
            displ_path = "/World/animated_beam_vtu/PointData/displ"

            prim_ds = await self.create_glyphs(stage, ds_path)
            self.assertTrue(prim_ds.HasAPI(cae_viz.GlyphsAPI))

            prim_rt = usd_utils.get_prim_rt(prim_ds)
            glyphs_rt = UsdGeomRT.PointInstancer(prim_rt)

            # Add colors mapped to RTData (time-varying)
            colors_fs_api = cae_viz.FieldSelectionAPI(prim_ds, "colors")
            colors_fs_api.CreateTargetRel().SetTargets([rtdata_path])
            await wait_for_update()

            # Add scales mapped to displ (static field)
            scales_fs_api = cae_viz.FieldSelectionAPI(prim_ds, "scales")
            scales_fs_api.CreateTargetRel().SetTargets([displ_path])
            await wait_for_update()

            # Get initial colors and scales at time 0
            np_colors_t0 = get_vtrt_array_as_numpy(prim_rt.GetAttribute("primvars:colors")).copy()
            np_scales_t0 = get_vtrt_array_as_numpy(glyphs_rt.GetScalesAttr()).copy()
            self.assertIsNotNone(np_colors_t0)
            self.assertIsNotNone(np_scales_t0)
            self.assertEqual(np_colors_t0.shape[0], 12221)
            self.assertEqual(np_scales_t0.shape[0], 12221)
            self.assertEqual(np_scales_t0.shape[1], 3)  # vec3f

            # Forward frame by 5
            await self.forward_frames(timeline, 5)
            np_colors_t5 = get_vtrt_array_as_numpy(prim_rt.GetAttribute("primvars:colors"))
            np_scales_t5 = get_vtrt_array_as_numpy(glyphs_rt.GetScalesAttr())
            np.testing.assert_allclose(np_colors_t5, np_colors_t0, atol=self.tolerance)
            np.testing.assert_allclose(np_scales_t5, np_scales_t0, atol=self.tolerance)

            # Forward frame by 5 more (to time 10) and confirm colors changed
            await self.forward_frames(timeline, 5)
            np_colors_t10 = get_vtrt_array_as_numpy(prim_rt.GetAttribute("primvars:colors"))
            np_scales_t10 = get_vtrt_array_as_numpy(glyphs_rt.GetScalesAttr())
            self.assertFalse(np.allclose(np_colors_t10, np_colors_t0, atol=self.tolerance))
            np.testing.assert_allclose(np_scales_t10, np_scales_t0, atol=self.tolerance)

            # Reset timeline and enable temporal interpolation
            timeline.set_current_time(0.0)
            await wait_for_update()

            cae_viz.OperatorTemporalAPI.Apply(prim_ds)
            temporal_api = cae_viz.OperatorTemporalAPI(prim_ds)
            temporal_api.CreateEnableFieldInterpolationAttr().Set(True)
            await wait_for_update()

            # Verify we're back at time 0
            np_colors_t0_interp = get_vtrt_array_as_numpy(prim_rt.GetAttribute("primvars:colors"))
            np_scales_t0_interp = get_vtrt_array_as_numpy(glyphs_rt.GetScalesAttr())
            np.testing.assert_allclose(np_colors_t0_interp, np_colors_t0, atol=self.tolerance)
            np.testing.assert_allclose(np_scales_t0_interp, np_scales_t0, atol=self.tolerance)

            # Forward frame by 5 (halfway between time 0 and time 10)
            # With interpolation, colors should be interpolated
            await self.forward_frames(timeline, 5)
            np_colors_t5_interp = get_vtrt_array_as_numpy(prim_rt.GetAttribute("primvars:colors"))
            np_scales_t5_interp = get_vtrt_array_as_numpy(glyphs_rt.GetScalesAttr())
            self.assertFalse(np.allclose(np_colors_t5_interp, np_colors_t0, atol=self.tolerance))
            self.assertFalse(np.allclose(np_colors_t5_interp, np_colors_t10, atol=self.tolerance))

            # scales may have small interpolation errors; we should allow for that for now
            np.testing.assert_allclose(np_scales_t5_interp, np_scales_t0, atol=1e-2)
