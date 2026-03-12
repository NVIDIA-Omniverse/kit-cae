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

import numpy as np
import omni.kit.test
from omni.cae.data import usd_utils
from omni.cae.data.commands import execute_command
from omni.cae.importer.vtk import import_to_stage
from omni.cae.schema import cae
from omni.cae.schema import viz as cae_viz
from omni.cae.testing import get_test_data_path, get_vtrt_array_as_numpy, new_stage, wait_for_update
from omni.kit.app import get_app
from pxr import Gf
from usdrt import UsdGeom as UsdGeomRT

logger = getLogger(__name__)


class TestVTKImporter(omni.kit.test.AsyncTestCase):

    async def test_vti(self):
        async with new_stage() as stage:
            await import_to_stage(get_test_data_path("headsq.vti"), "/World/headsq_vti")
            dataset_path = "/World/headsq_vti/VTKImageData"
            dataset_prim = stage.GetPrimAtPath(dataset_path)
            self.assertTrue(dataset_prim)

            self.assertTrue(dataset_prim.IsA(cae.DataSet))
            self.assertTrue(dataset_prim.HasAPI(cae.DenseVolumeAPI))

            denseVolumeAPI = cae.DenseVolumeAPI(dataset_prim)
            self.assertEqual(denseVolumeAPI.GetMinExtentAttr().Get(), Gf.Vec3i(0, 0, 0))
            self.assertEqual(denseVolumeAPI.GetMaxExtentAttr().Get(), Gf.Vec3i(255, 255, 93))
            self.assertEqual(denseVolumeAPI.GetSpacingAttr().Get(), Gf.Vec3f(1, 1, 2))

    async def test_polyhedra(self):
        """
        Test importing a VTK Unstructured Grid with polyhedron cells.
        """
        async with new_stage() as stage:
            await import_to_stage(get_test_data_path("multicomb_0_polyhedra.vtu"), "/World/multicomb_0_polyhedra")
            base_path = "/World/multicomb_0_polyhedra"
            dataset_path = f"{base_path}/VTKUnstructuredGrid"
            dataset_prim = stage.GetPrimAtPath(dataset_path)
            self.assertTrue(dataset_prim)
            self.assertTrue(dataset_prim.IsA(cae.DataSet))

            # create bbox
            bbox_path = "/World/CAE/BoundingBox_Multicomb"
            await execute_command("CreateCaeVizBoundingBox", dataset_paths=[dataset_path], prim_path=bbox_path)
            await wait_for_update()

            bbox = usd_utils.get_bounds(stage.GetPrimAtPath(bbox_path))
            self.assertIsNotNone(bbox, "Bounding box is None")
            self.assertIsInstance(bbox, Gf.Range3d, "Bounding box is not a Gf.Range3d")

            print("Bounding box min:", bbox.GetMin(), "max:", bbox.GetMax())

            np.testing.assert_array_almost_equal(
                np.array(bbox.GetMin()),
                np.array([0.0, -5.1778, 23.3311]),
                decimal=3,
            )
            np.testing.assert_array_almost_equal(
                np.array(bbox.GetMax()),
                np.array([6.9158, 5.1778, 33.11483]),
                decimal=3,
            )

            # show faces and then color with "Temp_n"
            faces_path = "/World/CAE/Faces"
            await execute_command("CreateCaeVizFaces", dataset_path=dataset_path, prim_path=faces_path)

            prim = stage.GetPrimAtPath(faces_path)
            self.assertIsNotNone(prim, "Faces prim should be valid")

            # confirm no colors are present initially
            prim_rt = usd_utils.get_prim_rt(prim)
            mesh_rt = UsdGeomRT.Mesh(prim_rt)
            self.assertFalse(mesh_rt.GetPrim().GetAttribute("primvars:colors").IsValid())

            faces_api = cae_viz.FacesAPI(prim)
            faces_api.CreateExternalOnlyAttr().Set(False)

            colors_api = cae_viz.FieldSelectionAPI(prim, "colors")
            colors_api.CreateTargetRel().SetTargets([f"{base_path}/PointData/Density"])
            await wait_for_update()

            self.assertTrue(prim_rt.GetAttribute("primvars:colors").IsValid())
            np_colors = get_vtrt_array_as_numpy(prim_rt.GetAttribute("primvars:colors"))
            self.assertIsNotNone(np_colors)
            self.assertGreater(np_colors.shape[0], 0, "Should have color values")
            self.assertEqual(np_colors.shape[1], 1, "Colors should be scalar (1 component)")
            np.testing.assert_almost_equal(
                np_colors.min(), 0.2009, decimal=2, err_msg="Min color value should be close to 0.2009"
            )
            np.testing.assert_almost_equal(
                np_colors.max(), 0.7104, decimal=2, err_msg="Max color value should be close to 0.7104"
            )
