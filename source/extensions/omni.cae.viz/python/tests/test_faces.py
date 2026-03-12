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
from omni.usd import get_stage_next_free_path
from pxr import Usd, UsdShade
from usdrt import UsdGeom as UsdGeomRT


class TestFaces(omni.kit.test.AsyncTestCase):
    tolerance = 1e-5

    async def create_faces(self, stage: Usd.Stage, dataset_path: str, external_only: bool = True) -> Usd.Prim:
        ds_prim = stage.GetPrimAtPath(dataset_path)
        self.assertIsNotNone(ds_prim, "Dataset prim should be valid")

        viz_path = get_stage_next_free_path(stage, f"/World/CAE/Faces_{ds_prim.GetName()}", False)
        await execute_command("CreateCaeVizFaces", dataset_path=dataset_path, prim_path=viz_path)

        prim = stage.GetPrimAtPath(viz_path)
        self.assertIsNotNone(prim, "Faces prim should be valid")
        faces_api = cae_viz.FacesAPI(prim)
        faces_api.CreateExternalOnlyAttr().Set(external_only)

        await wait_for_update()
        return prim

    async def test_faces_static_mixer(self):
        async with new_stage() as stage:
            await import_to_stage(get_test_data_path("StaticMixer.cgns"), "/World/StaticMixer")
            dataset_path = "/World/StaticMixer/Base/StaticMixer/StaticMixer_Default"
            prim = await self.create_faces(stage, dataset_path)
            self.assertTrue(prim.HasAPI(cae_viz.FacesAPI))

            # Get the UsdGeomMesh
            prim_rt = usd_utils.get_prim_rt(prim)
            mesh_rt = UsdGeomRT.Mesh(prim_rt)

            # Verify points are populated
            np_points = get_vtrt_array_as_numpy(mesh_rt.GetPointsAttr())
            self.assertIsNotNone(np_points)
            self.assertEqual(np_points.shape[1], 3, "Points should have 3 components (x, y, z)")
            self.assertEqual(np_points.shape[0], 802, "Should have 802 points")

            # Verify point ranges
            np.testing.assert_allclose(np_points.min(axis=0).tolist(), [-2.0, -3.0, -2.0], atol=self.tolerance)
            np.testing.assert_allclose(np_points.max(axis=0).tolist(), [2.0, 3.0, 2.0], atol=self.tolerance)

            # Verify face vertex counts
            np_face_vertex_counts = get_vtrt_array_as_numpy(mesh_rt.GetFaceVertexCountsAttr())
            self.assertIsNotNone(np_face_vertex_counts)
            self.assertEqual(np_face_vertex_counts.shape[0], 1582, "Should have 1582 faces")
            # Each face should have exactly 3 vertices
            self.assertTrue(np.allclose(np_face_vertex_counts, 3), "All faces should have exactly 3 vertices")

            # Verify face vertex indices
            np_face_vertex_indices = get_vtrt_array_as_numpy(mesh_rt.GetFaceVertexIndicesAttr())
            self.assertIsNotNone(np_face_vertex_indices)
            self.assertGreater(np_face_vertex_indices.shape[0], 0, "Should have face vertex indices")
            # Sum of face vertex counts should equal the number of face vertex indices
            self.assertEqual(
                np_face_vertex_counts.sum(),
                np_face_vertex_indices.shape[0],
                "Sum of face vertex counts should match number of face vertex indices",
            )
            # All indices should be valid (within range of point count)
            self.assertTrue(np.all(np_face_vertex_indices >= 0), "All face vertex indices should be non-negative")
            self.assertTrue(
                np.all(np_face_vertex_indices < np_points.shape[0]),
                f"All face vertex indices should be less than point count ({np_points.shape[0]})",
            )

    async def test_faces_static_mixer_with_colors(self):
        async with new_stage() as stage:
            await import_to_stage(get_test_data_path("StaticMixer.cgns"), "/World/StaticMixer")
            dataset_path = "/World/StaticMixer/Base/StaticMixer/StaticMixer_Default"
            temp_path = "/World/StaticMixer/Base/StaticMixer/Flow_Solution/Temperature"
            prim = await self.create_faces(stage, dataset_path)
            self.assertTrue(prim.HasAPI(cae_viz.FacesAPI))

            # confirm no colors are present initially
            prim_rt = usd_utils.get_prim_rt(prim)
            mesh_rt = UsdGeomRT.Mesh(prim_rt)
            self.assertFalse(mesh_rt.GetPrim().GetAttribute("primvars:colors").IsValid())

            # confirm shader's enable_coloring is false
            shader = UsdShade.Shader(prim.GetPrimAtPath("Materials/ScalarColor/Shader"))
            self.assertTrue(shader.GetInput("enable_coloring").Get() == False)

            # add colors
            colors_fs_api = cae_viz.FieldSelectionAPI(prim, "colors")
            colors_fs_api.CreateTargetRel().SetTargets([temp_path])
            await wait_for_update()
            self.assertTrue(prim_rt.GetAttribute("primvars:colors").IsValid())

            np_colors = get_vtrt_array_as_numpy(prim_rt.GetAttribute("primvars:colors"))
            self.assertIsNotNone(np_colors)
            self.assertGreater(np_colors.shape[0], 0, "Should have color values")
            self.assertEqual(np_colors.shape[1], 1, "Colors should be scalar (1 component)")

            # Verify color range is within Temperature field range
            np.testing.assert_allclose(np_colors.min(axis=0).tolist(), 285.0, atol=self.tolerance)
            np.testing.assert_allclose(np_colors.max(axis=0).tolist(), 315.000458, atol=self.tolerance)

            # confirm shader's enable_coloring is true
            shader = UsdShade.Shader(prim.GetPrimAtPath("Materials/ScalarColor/Shader"))
            self.assertTrue(shader.GetInput("enable_coloring").Get() == True)

            # Now remove colors, and confirm that shader's enable_coloring is false
            colors_fs_api.GetTargetRel().ClearTargets(False)
            await wait_for_update()

            # this will not drop the primvar, but the shader's enable_coloring will be false
            self.assertTrue(prim_rt.GetAttribute("primvars:colors").IsValid())
            self.assertTrue(shader.GetInput("enable_coloring").Get() == False)

    async def test_faces_animated_beam(self):
        """Test Faces operator with animated beam dataset to verify temporal updates and interpolation."""
        async with new_stage(path=get_test_data_path("animated_beam/animated_beam.usda")) as stage:
            from omni.timeline import get_timeline_interface

            # set up timeline; we use time_codes_per_second of 1.0 for convenience
            timeline = get_timeline_interface()
            timeline.set_time_codes_per_second(1.0)
            timeline.set_start_time(0.0)
            timeline.set_end_time(190.0)
            timeline.set_current_time(0.0)
            await wait_for_update()

            ds_path = "/World/animated_beam_vtu/VTKUnstructuredGrid"
            rtdata_path = "/World/animated_beam_vtu/PointData/RTData"
            prim = await self.create_faces(stage, ds_path, external_only=False)
            self.assertTrue(prim.HasAPI(cae_viz.FacesAPI))

            prim_rt = usd_utils.get_prim_rt(prim)
            mesh_rt = UsdGeomRT.Mesh(prim_rt)

            # Add colors mapped to RTData (time-varying)
            colors_fs_api = cae_viz.FieldSelectionAPI(prim, "colors")
            colors_fs_api.CreateTargetRel().SetTargets([rtdata_path])
            await wait_for_update()

            # Get initial points and colors at time 0
            np_points_t0 = get_vtrt_array_as_numpy(mesh_rt.GetPointsAttr()).copy()
            np_colors_t0 = get_vtrt_array_as_numpy(prim_rt.GetAttribute("primvars:colors")).copy()
            self.assertIsNotNone(np_points_t0)
            self.assertIsNotNone(np_colors_t0)
            self.assertGreater(np_points_t0.shape[0], 0, "Should have points")
            self.assertEqual(np_points_t0.shape[1], 3, "Points should have 3 components")
            self.assertEqual(np_colors_t0.shape[0], np_points_t0.shape[0], "Colors should match point count")

            # Verify initial bounds
            np.testing.assert_allclose(np_points_t0.min(axis=0).tolist(), [-5, 0, -5], atol=self.tolerance)
            np.testing.assert_allclose(np_points_t0.max(axis=0).tolist(), [5, 100, 5], atol=self.tolerance)

            # Forward frame by 5 and confirm nothing changed (no keyframe yet)
            for i in range(5):
                timeline.forward_one_frame()
            await wait_for_update()

            np_points_t5 = get_vtrt_array_as_numpy(mesh_rt.GetPointsAttr())
            np_colors_t5 = get_vtrt_array_as_numpy(prim_rt.GetAttribute("primvars:colors"))
            np.testing.assert_allclose(np_points_t5, np_points_t0, atol=self.tolerance)
            np.testing.assert_allclose(np_colors_t5, np_colors_t0, atol=self.tolerance)

            # Forward frame by 5 more (to time 10) and confirm both points and colors changed
            for i in range(5):
                timeline.forward_one_frame()
            await wait_for_update()

            np_points_t10 = get_vtrt_array_as_numpy(mesh_rt.GetPointsAttr()).copy()
            np_colors_t10 = get_vtrt_array_as_numpy(prim_rt.GetAttribute("primvars:colors")).copy()
            self.assertIsNotNone(np_points_t10)
            self.assertIsNotNone(np_colors_t10)

            # Points should have changed (geometry is time-varying)
            self.assertFalse(np.allclose(np_points_t10, np_points_t0, atol=self.tolerance))
            np.testing.assert_allclose(np_points_t10.min(axis=0).tolist(), [-4.47368, 0, -5], atol=self.tolerance)
            np.testing.assert_allclose(np_points_t10.max(axis=0).tolist(), [15.5526, 100, 5], atol=self.tolerance)

            # Colors should have changed (RTData is time-varying)
            self.assertFalse(np.allclose(np_colors_t10, np_colors_t0, atol=self.tolerance))

            # Reset timeline and enable temporal interpolation
            timeline.set_current_time(0.0)
            await wait_for_update()

            cae_viz.OperatorTemporalAPI.Apply(prim)
            temporal_api = cae_viz.OperatorTemporalAPI(prim)
            temporal_api.CreateEnableFieldInterpolationAttr().Set(True)
            await wait_for_update()

            # Verify we're back at time 0
            np_points_t0_interp = get_vtrt_array_as_numpy(mesh_rt.GetPointsAttr())
            np_colors_t0_interp = get_vtrt_array_as_numpy(prim_rt.GetAttribute("primvars:colors"))
            np.testing.assert_allclose(np_points_t0_interp, np_points_t0, atol=self.tolerance)
            np.testing.assert_allclose(np_colors_t0_interp, np_colors_t0, atol=self.tolerance)

            # Forward frame by 5 (halfway between time 0 and time 10)
            # With interpolation, both points and colors should be interpolated
            for i in range(5):
                timeline.forward_one_frame()
            await wait_for_update()

            np_points_t5_interp = get_vtrt_array_as_numpy(mesh_rt.GetPointsAttr())
            np_colors_t5_interp = get_vtrt_array_as_numpy(prim_rt.GetAttribute("primvars:colors"))
            self.assertIsNotNone(np_points_t5_interp)
            self.assertIsNotNone(np_colors_t5_interp)

            # Points should be different from both t0 and t10 (interpolated)
            self.assertFalse(np.allclose(np_points_t5_interp, np_points_t0, atol=self.tolerance))
            self.assertFalse(np.allclose(np_points_t5_interp, np_points_t10, atol=self.tolerance))
            np.testing.assert_allclose(
                np_points_t5_interp.min(axis=0).tolist(), [-4.736842, 0, -5], atol=self.tolerance
            )
            np.testing.assert_allclose(
                np_points_t5_interp.max(axis=0).tolist(), [10.276299, 100, 5], atol=self.tolerance
            )

            # Colors should also be interpolated (different from both t0 and t10)
            self.assertFalse(np.allclose(np_colors_t5_interp, np_colors_t0, atol=self.tolerance))
            self.assertFalse(np.allclose(np_colors_t5_interp, np_colors_t10, atol=self.tolerance))

            # Forward to time 10 and verify points and colors match the keyframe
            for i in range(5):
                timeline.forward_one_frame()
            await wait_for_update()

            np_points_t10_interp = get_vtrt_array_as_numpy(mesh_rt.GetPointsAttr())
            np_colors_t10_interp = get_vtrt_array_as_numpy(prim_rt.GetAttribute("primvars:colors"))

            # Points and colors at time 10 should match the keyframe values
            np.testing.assert_allclose(np_points_t10_interp, np_points_t10, atol=self.tolerance)
            np.testing.assert_allclose(np_colors_t10_interp, np_colors_t10, atol=self.tolerance)

    async def test_faces_cell_data_colors(self):
        """Test Faces operator with both point and cell data coloring."""
        async with new_stage() as stage:
            await import_to_stage(get_test_data_path("hex_timesteps.cgns"), "/World/hex_timesteps")
            dataset_path = "/World/hex_timesteps/Base/Zone/ElementsUniform"
            point_field_path = "/World/hex_timesteps/Base/Zone/SolutionVertex0001/PointSinusoid"
            cell_field_path = "/World/hex_timesteps/Base/Zone/SolutionCellCenter0001/CellSinusoid"

            prim = await self.create_faces(stage, dataset_path, external_only=False)
            self.assertTrue(prim.HasAPI(cae_viz.FacesAPI))

            prim_rt = usd_utils.get_prim_rt(prim)
            mesh_rt = UsdGeomRT.Mesh(prim_rt)

            # Get the mesh topology info
            np_points = get_vtrt_array_as_numpy(mesh_rt.GetPointsAttr())
            np_face_vertex_counts = get_vtrt_array_as_numpy(mesh_rt.GetFaceVertexCountsAttr())
            self.assertIsNotNone(np_points)
            self.assertIsNotNone(np_face_vertex_counts)
            num_points = np_points.shape[0]
            num_faces = np_face_vertex_counts.shape[0]

            # Initially, no colors should be present
            self.assertFalse(prim_rt.GetAttribute("primvars:colors").IsValid())

            # Test 1: Color by point data
            colors_fs_api = cae_viz.FieldSelectionAPI(prim, "colors")
            colors_fs_api.CreateTargetRel().SetTargets([point_field_path])
            await wait_for_update()

            # Verify colors primvar exists and has point data characteristics
            self.assertTrue(prim_rt.GetAttribute("primvars:colors").IsValid())
            np_colors = get_vtrt_array_as_numpy(prim_rt.GetAttribute("primvars:colors"))
            self.assertIsNotNone(np_colors)

            # Colors should match number of points
            self.assertEqual(np_colors.shape[0], num_points, "Colors should match number of points for point data")
            self.assertEqual(np_colors.shape[1], 1, "Colors should be scalar (1 component)")

            # Check interpolation attribute - should be "vertex" for point data
            colors_interp = prim_rt.GetAttribute("primvars:colors:interpolation").Get()
            self.assertEqual(colors_interp, "vertex", "Interpolation should be 'vertex' for point data")

            # Test 2: Switch to cell data
            colors_fs_api.GetTargetRel().SetTargets([cell_field_path])
            await wait_for_update()

            # Verify colors primvar still exists but now has cell data characteristics
            self.assertTrue(prim_rt.GetAttribute("primvars:colors").IsValid())
            np_colors_cell = get_vtrt_array_as_numpy(prim_rt.GetAttribute("primvars:colors"))
            self.assertIsNotNone(np_colors_cell)

            # Colors should match number of faces (cells)
            self.assertEqual(np_colors_cell.shape[0], num_faces, "Colors should match number of faces for cell data")
            self.assertEqual(np_colors_cell.shape[1], 1, "Colors should be scalar (1 component)")

            # Check interpolation attribute - should be "uniform" for cell data
            colors_interp_cell = prim_rt.GetAttribute("primvars:colors:interpolation").Get()
            self.assertEqual(colors_interp_cell, "uniform", "Interpolation should be 'uniform' for cell data")

            # Verify the colors are different between point and cell data
            # (they should be, since they're different fields with different values)
            self.assertFalse(
                np.allclose(
                    np_colors[: min(len(np_colors), len(np_colors_cell))],
                    np_colors_cell[: min(len(np_colors), len(np_colors_cell))],
                    atol=self.tolerance,
                ),
                "Point and cell colors should be different",
            )
