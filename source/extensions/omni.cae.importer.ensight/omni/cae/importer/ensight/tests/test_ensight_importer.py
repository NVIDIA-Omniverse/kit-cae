# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import numpy as np
import omni.kit.test
from omni.cae.data import usd_utils
from omni.cae.data.commands import execute_command
from omni.cae.importer.ensight import import_to_stage
from omni.cae.schema import viz as cae_viz
from omni.cae.testing import get_test_data_path, get_vtrt_array_as_numpy, new_stage, wait_for_update
from omni.kit.app import get_app
from pxr import Gf
from usdrt import UsdGeom as UsdGeomRT


class TestEnSightImporter(omni.kit.test.AsyncTestCase):
    tolerance = 1e-3

    async def test_disk_out_ref_hex(self):
        async with new_stage() as stage:
            await import_to_stage(get_test_data_path("EnSight/disk_out_ref.0.case"), "/World/disk_out_ref")
            await wait_for_update()

            dataset_path = "/World/disk_out_ref/VTK_Part"
            viz_path = "/World/CAE/BoundingBox_VTK_Part"

            # let's show bounding box first.
            await execute_command("CreateCaeVizBoundingBox", dataset_paths=[dataset_path], prim_path=viz_path)
            await wait_for_update()

            bbox = usd_utils.get_bounds(stage.GetPrimAtPath(viz_path))
            self.assertIsNotNone(bbox, "Bounding box is None")
            self.assertIsInstance(bbox, Gf.Range3d, "Bounding box is not a Gf.Range3d")

            np.testing.assert_array_almost_equal(
                np.array(bbox.GetMin()),
                np.array([-5.75, -5.75, -10.0]),
                decimal=3,
            )
            np.testing.assert_array_almost_equal(
                np.array(bbox.GetMax()),
                np.array([5.75, 5.75, 10.15999]),
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
            colors_api.CreateTargetRel().SetTargets([f"{dataset_path}/Variables/Temp_n"])
            await wait_for_update()

            self.assertTrue(prim_rt.GetAttribute("primvars:colors").IsValid())
            np_colors = get_vtrt_array_as_numpy(prim_rt.GetAttribute("primvars:colors"))
            self.assertIsNotNone(np_colors)
            self.assertGreater(np_colors.shape[0], 0, "Should have color values")
            self.assertEqual(np_colors.shape[1], 1, "Colors should be scalar (1 component)")
            np.testing.assert_almost_equal(
                np_colors.min(), 293.15, decimal=2, err_msg="Min color value should be close to 293.15"
            )
            np.testing.assert_almost_equal(
                np_colors.max(), 913.15, decimal=2, err_msg="Max color value should be close to 913.15"
            )
