# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import hashlib

import numpy as np
import omni.kit.test
from omni.cae.data import usd_utils
from omni.cae.data.commands import execute_command
from omni.cae.importer.cgns import import_to_stage
from omni.cae.schema import viz as cae_viz
from omni.cae.testing import get_test_data_path, get_vtrt_array_as_numpy, new_stage, wait_for_update
from omni.cae.viz import slice as _slice_mod
from pxr import Gf, Usd
from usdrt import Sdf as SdfRt
from usdrt import UsdGeom as UsdGeomRt


class TestSlicePureFunctions(omni.kit.test.AsyncTestCase):
    """Unit tests for the pure mathematical helpers in the slice module."""

    def test_compute_plane_identity(self):
        """Identity transform → origin position, +Y normal."""
        pos, normal = _slice_mod._compute_plane(np.eye(4))
        np.testing.assert_allclose(pos, [0, 0, 0], atol=1e-7)
        np.testing.assert_allclose(normal, [0, 1, 0], atol=1e-7)

    def test_compute_plane_translation(self):
        """Translation in the matrix's row 3 is returned as position."""
        xform = np.eye(4)
        xform[3, :3] = [3.0, 4.0, 5.0]
        pos, normal = _slice_mod._compute_plane(xform)
        np.testing.assert_allclose(pos, [3, 4, 5], atol=1e-7)
        np.testing.assert_allclose(normal, [0, 1, 0], atol=1e-7)

    def test_compute_plane_normal_is_unit_length(self):
        """A non-unit row-1 vector is normalised before being returned."""
        xform = np.eye(4)
        xform[1, :3] = [0, 3, 0]  # scaled +Y
        _, normal = _slice_mod._compute_plane(xform)
        np.testing.assert_allclose(np.linalg.norm(normal), 1.0, atol=1e-7)
        np.testing.assert_allclose(normal, [0, 1, 0], atol=1e-7)

    def test_compute_plane_arbitrary_normal(self):
        """A rotated row-1 gives the corresponding normalised normal."""
        xform = np.eye(4)
        xform[1, :3] = [1.0, 0.0, 0.0]  # +Y axis now points along +X
        _, normal = _slice_mod._compute_plane(xform)
        np.testing.assert_allclose(normal, [1, 0, 0], atol=1e-7)

    # ------------------------------------------------------------------
    # _compute_tight_fit_quad
    # ------------------------------------------------------------------

    def test_tight_fit_quad_x_slice(self):
        """X-slice through a unit cube: all corners at x=0, y/z span ±0.5."""
        bounds = Gf.Range3d(Gf.Vec3d(-0.5, -0.5, -0.5), Gf.Vec3d(0.5, 0.5, 0.5))
        corners = _slice_mod._compute_tight_fit_quad(
            center=np.array([0.0, 0.0, 0.0]),
            normal=np.array([1.0, 0.0, 0.0]),
            bounds=bounds,
        )
        self.assertEqual(corners.shape, (4, 3))
        np.testing.assert_allclose(corners[:, 0], 0.0, atol=1e-6, err_msg="All corners should lie on x=0")
        np.testing.assert_allclose(sorted(corners[:, 1]), [-0.5, -0.5, 0.5, 0.5], atol=1e-6)
        np.testing.assert_allclose(sorted(corners[:, 2]), [-0.5, -0.5, 0.5, 0.5], atol=1e-6)

    def test_tight_fit_quad_y_slice(self):
        """Y-slice through a unit cube: all corners at y=0, x/z span ±0.5."""
        bounds = Gf.Range3d(Gf.Vec3d(-0.5, -0.5, -0.5), Gf.Vec3d(0.5, 0.5, 0.5))
        corners = _slice_mod._compute_tight_fit_quad(
            center=np.array([0.0, 0.0, 0.0]),
            normal=np.array([0.0, 1.0, 0.0]),
            bounds=bounds,
        )
        self.assertEqual(corners.shape, (4, 3))
        np.testing.assert_allclose(corners[:, 1], 0.0, atol=1e-6, err_msg="All corners should lie on y=0")
        self.assertAlmostEqual(corners[:, 0].min(), -0.5, places=5)
        self.assertAlmostEqual(corners[:, 0].max(), 0.5, places=5)

    def test_tight_fit_quad_no_intersection(self):
        """Plane outside the AABB returns a degenerate quad at center."""
        bounds = Gf.Range3d(Gf.Vec3d(-0.5, -0.5, -0.5), Gf.Vec3d(0.5, 0.5, 0.5))
        center = np.array([10.0, 0.0, 0.0])  # far outside
        corners = _slice_mod._compute_tight_fit_quad(
            center=center,
            normal=np.array([1.0, 0.0, 0.0]),
            bounds=bounds,
        )
        self.assertEqual(corners.shape, (4, 3))
        np.testing.assert_allclose(corners, np.tile(center, (4, 1)), atol=1e-6)

    def test_tight_fit_quad_oblique_stays_inside_aabb(self):
        """An oblique 45° slice stays within the AABB."""
        bounds = Gf.Range3d(Gf.Vec3d(-1.0, -1.0, -1.0), Gf.Vec3d(1.0, 1.0, 1.0))
        normal = np.array([1.0, 1.0, 0.0]) / np.sqrt(2)
        corners = _slice_mod._compute_tight_fit_quad(
            center=np.array([0.0, 0.0, 0.0]),
            normal=normal,
            bounds=bounds,
        )
        self.assertEqual(corners.shape, (4, 3))
        self.assertTrue(np.all(corners >= -1.01), "Corners below AABB lower bound")
        self.assertTrue(np.all(corners <= 1.01), "Corners above AABB upper bound")

    # ------------------------------------------------------------------
    # _create_probe_grid
    # ------------------------------------------------------------------

    def test_probe_grid_shape(self):
        """Probe grid has shape (height * width, 3)."""
        corners = np.array(
            [[0.0, -0.5, -0.5], [0.0, 0.5, -0.5], [0.0, 0.5, 0.5], [0.0, -0.5, 0.5]],
            dtype=np.float32,
        )
        grid = _slice_mod._create_probe_grid(corners, width=4, height=8)
        self.assertEqual(grid.shape, (32, 3))

    def test_probe_grid_pixel_centres(self):
        """For a unit square quad, probes land at pixel centres (0.25, 0.75)."""
        # Identity quad in XY plane: P00=(0,0,0), P10=(1,0,0), P11=(1,1,0), P01=(0,1,0)
        corners = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        )
        grid = _slice_mod._create_probe_grid(corners, width=2, height=2)
        # Row-major: v varies slowest, so order is (u=0.25,v=0.25), (u=0.75,v=0.25), ...
        expected = np.array(
            [[0.25, 0.25, 0.0], [0.75, 0.25, 0.0], [0.25, 0.75, 0.0], [0.75, 0.75, 0.0]],
            dtype=np.float32,
        )
        np.testing.assert_allclose(grid, expected, atol=1e-6)

    def test_probe_grid_all_points_on_plane(self):
        """All probe positions for a planar quad lie on the same plane (z=0 here)."""
        corners = np.array(
            [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]],
            dtype=np.float32,
        )
        grid = _slice_mod._create_probe_grid(corners, width=10, height=10)
        np.testing.assert_allclose(grid[:, 2], 0.0, atol=1e-6, err_msg="All probe z-coords should be 0")

    def test_probe_grid_covers_interior(self):
        """Probe positions cover the interior of the quad, not the edges."""
        corners = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        )
        grid = _slice_mod._create_probe_grid(corners, width=4, height=4)
        # No probe should touch the boundary (all coords strictly inside (0,1))
        self.assertTrue(np.all(grid[:, :2] > 0.0), "Probes should be strictly inside lower bound")
        self.assertTrue(np.all(grid[:, :2] < 1.0), "Probes should be strictly inside upper bound")


class TestPlanarSlice(omni.kit.test.AsyncTestCase):
    """Integration tests for the PlanarSlice operator end-to-end."""

    async def test_planar_slice_creates_rt_quad_prim(self):
        """The operator creates an RT sub-prim with 4 corner points within the dataset bounds."""
        async with new_stage() as stage:
            await import_to_stage(get_test_data_path("StaticMixer.cgns"), "/World/StaticMixer")
            base_path = "/World/StaticMixer/Base/StaticMixer"
            dataset_path = f"{base_path}/B1_P3"
            slice_path = "/World/CAE/PlanarSlice_Test"

            await execute_command("CreateCaeVizPlanarSlice", dataset_path=dataset_path, prim_path=slice_path)

            slice_prim: Usd.Prim = stage.GetPrimAtPath(slice_path)
            self.assertTrue(slice_prim.IsValid())
            self.assertTrue(slice_prim.HasAPI(cae_viz.PlanarSliceAPI))
            self.assertTrue(slice_prim.HasAPI(cae_viz.OperatorAPI))

            # Point the colors field at a scalar field so the operator can run
            cae_viz.FieldSelectionAPI(slice_prim, "colors").GetTargetRel().SetTargets(
                [f"{base_path}/Flow_Solution/Temperature"]
            )
            await wait_for_update()

            # The operator writes its output to RT sub-prims at /CaePlanarSlice/slice_{h}_{slot}
            h = hashlib.sha512(slice_path.encode()).hexdigest()[:8]
            rt_stage = usd_utils.get_prim_rt(slice_prim).GetStage()
            quad_prim = rt_stage.GetPrimAtPath(SdfRt.Path(f"/CaePlanarSlice/slice_{h}_0"))

            self.assertTrue(quad_prim.IsValid(), "RT quad sub-prim should exist for slot 0 (free mode)")

            points = get_vtrt_array_as_numpy(UsdGeomRt.Mesh(quad_prim).GetPointsAttr())
            self.assertIsNotNone(points, "Quad points attribute should be readable")
            self.assertEqual(points.shape, (4, 3), "Quad should have exactly 4 corners in 3D")

            # Corners must be in the vicinity of the dataset (StaticMixer spans roughly ±2 units)
            self.assertTrue(np.all(np.abs(points) < 5.0), "All corners should be near the dataset bounds")

    def _set_mode(self, slice_prim: Usd.Prim, mode: str):
        """Set the PlanarSliceAPI mode, authoring the attribute if needed."""
        api = cae_viz.PlanarSliceAPI(slice_prim)
        attr = api.GetModeAttr()
        if not attr.IsAuthored():
            attr = api.CreateModeAttr()
        attr.Set(mode)

    async def test_planar_slice_mode_xyz_creates_three_prims(self):
        """In 'xyz' mode the operator creates RT quad prims for all three axis-aligned planes."""
        async with new_stage() as stage:
            await import_to_stage(get_test_data_path("StaticMixer.cgns"), "/World/StaticMixer")
            base_path = "/World/StaticMixer/Base/StaticMixer"
            dataset_path = f"{base_path}/B1_P3"
            slice_path = "/World/CAE/PlanarSlice_XYZ"

            await execute_command("CreateCaeVizPlanarSlice", dataset_path=dataset_path, prim_path=slice_path)

            slice_prim: Usd.Prim = stage.GetPrimAtPath(slice_path)
            cae_viz.FieldSelectionAPI(slice_prim, "colors").GetTargetRel().SetTargets(
                [f"{base_path}/Flow_Solution/Temperature"]
            )
            self._set_mode(slice_prim, "xyz")
            await wait_for_update()

            h = hashlib.sha512(slice_path.encode()).hexdigest()[:8]
            rt_stage = usd_utils.get_prim_rt(slice_prim).GetStage()

            for slot in range(3):
                quad_prim = rt_stage.GetPrimAtPath(SdfRt.Path(f"/CaePlanarSlice/slice_{h}_{slot}"))
                self.assertTrue(quad_prim.IsValid(), f"RT quad prim for slot {slot} should exist in xyz mode")
                points = get_vtrt_array_as_numpy(UsdGeomRt.Mesh(quad_prim).GetPointsAttr())
                self.assertIsNotNone(points, f"Points for slot {slot} should be readable")
                self.assertEqual(points.shape, (4, 3), f"Slot {slot} quad should have 4 corners")

    async def test_planar_slice_axis_planes_are_orthogonal(self):
        """In 'xyz' mode the three planes are mutually orthogonal (axis-aligned normals)."""
        async with new_stage() as stage:
            await import_to_stage(get_test_data_path("StaticMixer.cgns"), "/World/StaticMixer")
            base_path = "/World/StaticMixer/Base/StaticMixer"
            dataset_path = f"{base_path}/B1_P3"
            slice_path = "/World/CAE/PlanarSlice_Ortho"

            await execute_command("CreateCaeVizPlanarSlice", dataset_path=dataset_path, prim_path=slice_path)

            slice_prim: Usd.Prim = stage.GetPrimAtPath(slice_path)
            cae_viz.FieldSelectionAPI(slice_prim, "colors").GetTargetRel().SetTargets(
                [f"{base_path}/Flow_Solution/Temperature"]
            )
            self._set_mode(slice_prim, "xyz")
            await wait_for_update()

            h = hashlib.sha512(slice_path.encode()).hexdigest()[:8]
            rt_stage = usd_utils.get_prim_rt(slice_prim).GetStage()

            normals = []
            for slot in range(3):
                quad_prim = rt_stage.GetPrimAtPath(SdfRt.Path(f"/CaePlanarSlice/slice_{h}_{slot}"))
                self.assertTrue(quad_prim.IsValid())
                pts = get_vtrt_array_as_numpy(UsdGeomRt.Mesh(quad_prim).GetPointsAttr())
                self.assertIsNotNone(pts)
                # Compute quad normal from two edge vectors
                e1 = pts[1] - pts[0]
                e2 = pts[3] - pts[0]
                n = np.cross(e1.astype(np.float64), e2.astype(np.float64))
                norm = np.linalg.norm(n)
                if norm > 1e-9:
                    normals.append(n / norm)

            self.assertEqual(len(normals), 3, "Should have 3 valid normals")

            # Each pair of normals should be (nearly) orthogonal
            for i in range(3):
                for j in range(i + 1, 3):
                    dot = abs(np.dot(normals[i], normals[j]))
                    self.assertLess(dot, 0.1, f"Normals {i} and {j} should be orthogonal (dot={dot:.4f})")
