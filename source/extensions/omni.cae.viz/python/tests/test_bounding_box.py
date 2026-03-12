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

import carb.settings
import numpy as np
import omni.kit.test
from omni.cae.data import usd_utils
from omni.cae.data.commands import execute_command
from omni.cae.importer.cgns import import_to_stage
from omni.cae.testing import get_test_data_path, new_stage, wait_for_update
from omni.kit.app import get_app
from pxr import Gf, Usd


class TestBoundingBox(omni.kit.test.AsyncTestCase):
    tolerance = 1e-5

    async def test_bounding_box_sids(self):
        async with new_stage() as stage:
            await import_to_stage(get_test_data_path("StaticMixer.cgns"), "/World/StaticMixer")
            base_path: str = "/World/StaticMixer/Base/StaticMixer"

            dataset_names: list[str] = ["in1", "in2", "out", "B1_P3", "StaticMixer_Default", "GridCoordinates"]
            expected_bds = {
                "in1": Gf.Range3d((-1.5, -3, 0.5), (-0.5, -3.0, 1.5)),
                "in2": Gf.Range3d(Gf.Vec3d(0.5, 3.0, 0.5), Gf.Vec3d(1.5, 3.0, 1.5)),
                "out": Gf.Range3d(Gf.Vec3d(-0.5, -0.5, -2.0), Gf.Vec3d(0.5, 0.5, -2.0)),
                "B1_P3": Gf.Range3d((-2, -3, -2), (2, 3, 2)),
                "StaticMixer_Default": Gf.Range3d((-2, -3, -2), (2, 3, 2)),
                "GridCoordinates": Gf.Range3d((-2, -3, -2), (2, 3, 2)),
            }

            for ds_name in dataset_names:
                dataset_path = f"{base_path}/{ds_name}"
                viz_path = f"/World/CAE/BoundingBox_{ds_name}"
                assert stage.GetPrimAtPath(dataset_path).IsValid(), f"Dataset {dataset_path} is invalid"
                await execute_command("CreateCaeVizBoundingBox", dataset_paths=[dataset_path], prim_path=viz_path)
                await wait_for_update()

                bbox = usd_utils.get_bounds(stage.GetPrimAtPath(viz_path))
                self.assertIsNotNone(bbox, f"Bounding box for {ds_name} is None")
                self.assertIsInstance(bbox, Gf.Range3d, f"Bounding box for {ds_name} is not a Gf.Range3d")
                self.assertEqual(bbox, expected_bds[ds_name], f"Bounding box for {ds_name} is incorrect")

    async def test_bounding_box_sids_multiple(self):
        async with new_stage() as stage:
            await import_to_stage(get_test_data_path("StaticMixer.cgns"), "/World/StaticMixer")
            base_path: str = "/World/StaticMixer/Base/StaticMixer"

            dataset_names: list[str] = ["in1", "in2"]
            dataset_paths: list[str] = [f"{base_path}/{ds_name}" for ds_name in dataset_names]
            viz_path = f"/World/CAE/BoundingBox_in1_in2"
            await execute_command("CreateCaeVizBoundingBox", dataset_paths=dataset_paths, prim_path=viz_path)
            await wait_for_update()

            expected_bds = Gf.Range3d((-1.5, -3, 0.5), (1.5, 3.0, 1.5))  # union of the two bounding boxes

            bbox = usd_utils.get_bounds(stage.GetPrimAtPath(viz_path))
            self.assertIsNotNone(bbox, "Bounding box is None")
            self.assertIsInstance(bbox, Gf.Range3d, "Bounding box is not a Gf.Range3d")
            self.assertEqual(bbox, expected_bds, "Bounding box is incorrect")

    async def test_bounding_box_use_point_bounds(self):
        """Test that when use_point_bounds setting is enabled, all datasets use point bounds for bounding box.
        For StaticMixer, all bounding boxes should match GridCoordinates bounds since all datasets share the same points.
        """
        settings = carb.settings.get_settings()
        setting_path = "/persistent/exts/omni.cae.viz/defaultBoundingBoxUsePointBounds"

        # Save original setting value
        original_value = settings.get(setting_path)

        try:
            # Enable use_point_bounds setting
            settings.set(setting_path, True)

            async with new_stage() as stage:
                await import_to_stage(get_test_data_path("StaticMixer.cgns"), "/World/StaticMixer")
                base_path: str = "/World/StaticMixer/Base/StaticMixer"

                # Expected bounds: all datasets should have same bounds as GridCoordinates when using point bounds
                expected_bounds = Gf.Range3d((-2, -3, -2), (2, 3, 2))

                dataset_names: list[str] = ["in1", "in2", "out", "B1_P3", "StaticMixer_Default", "GridCoordinates"]

                for ds_name in dataset_names:
                    dataset_path = f"{base_path}/{ds_name}"
                    viz_path = f"/World/CAE/BoundingBox_PointBounds_{ds_name}"
                    assert stage.GetPrimAtPath(dataset_path).IsValid(), f"Dataset {dataset_path} is invalid"
                    await execute_command("CreateCaeVizBoundingBox", dataset_paths=[dataset_path], prim_path=viz_path)
                    await wait_for_update()

                    bbox = usd_utils.get_bounds(stage.GetPrimAtPath(viz_path))
                    self.assertIsNotNone(bbox, f"Bounding box for {ds_name} is None")
                    self.assertIsInstance(bbox, Gf.Range3d, f"Bounding box for {ds_name} is not a Gf.Range3d")

                    # All datasets should have the same bounds as GridCoordinates when using point bounds
                    np.testing.assert_allclose(
                        [bbox.min[0], bbox.min[1], bbox.min[2]],
                        list(expected_bounds.min),
                        atol=self.tolerance,
                        err_msg=f"Bounding box min for {ds_name} is incorrect",
                    )
                    np.testing.assert_allclose(
                        [bbox.max[0], bbox.max[1], bbox.max[2]],
                        list(expected_bounds.max),
                        atol=self.tolerance,
                        err_msg=f"Bounding box max for {ds_name} is incorrect",
                    )

        finally:
            # Restore original setting value
            if original_value is not None:
                settings.set(setting_path, original_value)
            else:
                settings.set(setting_path, False)
