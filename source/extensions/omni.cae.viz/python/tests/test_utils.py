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
import omni.cae.dav as cae_dav
import omni.kit.test
import omni.usd
import warp as wp
from omni.cae.data import settings, usd_utils
from omni.cae.importer.npz import import_to_stage
from omni.cae.schema import cae
from omni.cae.schema import viz as cae_viz
from omni.cae.testing import get_test_data_path, new_stage
from omni.cae.viz import utils
from omni.cae.viz.utils import RtSubPrimGuard
from omni.kit.app import get_app
from omni.usd import get_context
from pxr import Gf, Usd, UsdGeom
from usdrt import Sdf as SdfRT
from usdrt import Usd as UsdRT


class TestUtils(omni.kit.test.AsyncTestCase):

    async def setUp(self):
        self.datasets = {}
        usd_context = get_context()
        await usd_context.new_stage_async()
        self.stage = usd_context.get_stage()

        # load the test data
        await import_to_stage(get_test_data_path("disk_out_ref.npz"), "/World/disk_out_ref_npz")
        await import_to_stage(get_test_data_path("disk_out_ref_tet.npz"), "/World/disk_out_ref_tet_npz")

        self.datasets["disk_out_ref_npz"] = {
            "dataset": "/World/disk_out_ref_npz/NumPyDataSet",
            "nb_cells": 7_472,
            "nb_points": 8_499,
            "bds": (wp.vec3f(-5.75, -5.75, -10.0), wp.vec3f(5.75, 5.75, 10.16)),
            "fields": {
                "Temp": "/World/disk_out_ref_npz/NumPyArrays/Temp",
                "V": "/World/disk_out_ref_npz/NumPyArrays/V",
                "AsH3": "/World/disk_out_ref_npz/NumPyArrays/AsH3",
                "CH4": "/World/disk_out_ref_npz/NumPyArrays/CH4",
                "GaMe3": "/World/disk_out_ref_npz/NumPyArrays/GaMe3",
                "H2": "/World/disk_out_ref_npz/NumPyArrays/H2",
                "Pres": "/World/disk_out_ref_npz/NumPyArrays/Pres",
            },
            "field_ranges": {
                "Temp": (293.15, 913.15),
                "V_2": (-21.1259765625, 6.7119011878967285),
                "V_Mag": [0, 22.418820167105352],
            },
        }

        self.datasets["disk_out_ref_tet_npz"] = {
            "dataset": "/World/disk_out_ref_tet_npz/NumPyDataSet",
            "nb_cells": 44_832,
            "nb_points": 8_499,
            "bds": (wp.vec3f(-5.75, -5.75, -10.0), wp.vec3f(5.75, 5.75, 10.16)),
            "fields": {
                "Temp": "/World/disk_out_ref_tet_npz/NumPyArrays/Temp",
                "V": "/World/disk_out_ref_tet_npz/NumPyArrays/V",
                "AsH3": "/World/disk_out_ref_tet_npz/NumPyArrays/AsH3",
                "CH4": "/World/disk_out_ref_tet_npz/NumPyArrays/CH4",
                "GaMe3": "/World/disk_out_ref_tet_npz/NumPyArrays/GaMe3",
                "H2": "/World/disk_out_ref_tet_npz/NumPyArrays/H2",
                "Pres": "/World/disk_out_ref_tet_npz/NumPyArrays/Pres",
            },
            "field_ranges": {
                "Temp": (293.15, 913.15),
                "V_2": (-21.1259765625, 6.7119011878967285),
                "V_Mag": [0, 22.418820167105352],
            },
        }

        # fix associations for fields.
        for dataset in self.datasets.values():
            for field_name, field_path in dataset["fields"].items():
                field_prim = self.stage.GetPrimAtPath(field_path)
                field_array = cae.FieldArray(field_prim)
                field_array.GetFieldAssociationAttr().Set(cae.Tokens.vertex)

    async def tearDown(self):
        del self.stage
        usd_context = get_context()
        await usd_context.close_stage_async()

    async def test_utils_get_input_dataset(self):
        operator_prim = self.stage.DefinePrim("/World/OperatorTestInputDataset")
        # cae_viz.OperatorAPI.Apply(operator_prim)
        cae_viz.DatasetSelectionAPI.Apply(operator_prim, "source")
        dataset_selection_api_source = cae_viz.DatasetSelectionAPI(operator_prim, "source")
        dataset_selection_api_source.CreateTargetRel().SetTargets({self.datasets["disk_out_ref_npz"]["dataset"]})

        cae_viz.DatasetSelectionAPI.Apply(operator_prim, "other_dataset")
        dataset_selection_api_other_dataset = cae_viz.DatasetSelectionAPI(operator_prim, "other_dataset")
        dataset_selection_api_other_dataset.CreateTargetRel().SetTargets(
            {self.datasets["disk_out_ref_tet_npz"]["dataset"]}
        )

        cae_viz.DatasetSelectionAPI.Apply(operator_prim, "empty_dataset")

        dataset = await utils.get_input_dataset(
            operator_prim, "source", timeCode=Usd.TimeCode.EarliestTime(), device="cpu"
        )
        self.assertIsNotNone(dataset, "Dataset should not be None")
        self.assertEqual(
            dataset.get_num_points(), self.datasets["disk_out_ref_npz"]["nb_points"], "Mismatch in number of points"
        )
        self.assertEqual(
            dataset.get_num_cells(), self.datasets["disk_out_ref_npz"]["nb_cells"], "Mismatch in number of cells"
        )

        # we don't expect any fields to be present
        self.assertEqual(
            len(dataset.get_field_names()), 0, "Expected 0 fields in the dataset, got {len(dataset.get_field_names())}"
        )

        # we expect the bounds to be correct
        bds = dataset.get_bounds()
        self.assertEqual(bds[0], self.datasets["disk_out_ref_npz"]["bds"][0], "Mismatch in bounds min")
        self.assertEqual(bds[1], self.datasets["disk_out_ref_npz"]["bds"][1], "Mismatch in bounds max")

        other_dataset = await utils.get_input_dataset(
            operator_prim, "other_dataset", timeCode=Usd.TimeCode.EarliestTime(), device="cpu"
        )
        self.assertIsNotNone(other_dataset, "Other dataset should not be None")
        self.assertEqual(
            other_dataset.get_num_points(),
            self.datasets["disk_out_ref_tet_npz"]["nb_points"],
            "Mismatch in number of points",
        )
        self.assertEqual(
            other_dataset.get_num_cells(),
            self.datasets["disk_out_ref_tet_npz"]["nb_cells"],
            "Mismatch in number of cells",
        )

        # we don't expect any fields to be present
        self.assertEqual(
            len(other_dataset.get_field_names()),
            0,
            "Expected 0 fields in the dataset, got {len(other_dataset.get_field_names())}",
        )

        # we expect the bounds to be correct
        bds = other_dataset.get_bounds()
        self.assertEqual(bds[0], self.datasets["disk_out_ref_tet_npz"]["bds"][0], "Mismatch in bounds min")
        self.assertEqual(bds[1], self.datasets["disk_out_ref_tet_npz"]["bds"][1], "Mismatch in bounds max")

        with self.assertRaises(
            ValueError, msg="Expected ValueError when getting input dataset for non-existent instance"
        ):
            await utils.get_input_dataset(
                operator_prim, "bogus_dataset", timeCode=Usd.TimeCode.EarliestTime(), device="cpu"
            )

        with self.assertRaises(
            usd_utils.QuietableException, msg="Expected QuietableException when getting input dataset for empty dataset"
        ):
            await utils.get_input_dataset(
                operator_prim, "empty_dataset", timeCode=Usd.TimeCode.EarliestTime(), device="cpu"
            )

    # disabled till we start support suc collections.
    # async def test_utils_get_input_collection_dataset(self):

    #     operator_prim = self.stage.DefinePrim("/World/OperatorTestCollectionDataset")
    #     # cae_viz.OperatorAPI.Apply(operator_prim)
    #     cae_viz.DatasetSelectionAPI.Apply(operator_prim, "collection_dataset")
    #     dataset_selection_api_collection_dataset = cae_viz.DatasetSelectionAPI(operator_prim, "collection_dataset")
    #     dataset_selection_api_collection_dataset.CreateTargetRel().SetTargets(
    #         {self.datasets["disk_out_ref_npz"]["dataset"], self.datasets["disk_out_ref_tet_npz"]["dataset"]}
    #     )

    #     collection_dataset = await utils.get_input_dataset(
    #         operator_prim, "collection_dataset", timeCode=Usd.TimeCode.EarliestTime(), device="cpu"
    #     )
    #     self.assertIsNotNone(collection_dataset, "Collection dataset should not be None")
    #     self.assertEqual(
    #         collection_dataset.get_num_points(),
    #         self.datasets["disk_out_ref_npz"]["nb_points"] + self.datasets["disk_out_ref_tet_npz"]["nb_points"],
    #         "Mismatch in number of points",
    #     )
    #     self.assertEqual(
    #         collection_dataset.get_num_cells(),
    #         self.datasets["disk_out_ref_npz"]["nb_cells"] + self.datasets["disk_out_ref_tet_npz"]["nb_cells"],
    #         "Mismatch in number of cells",
    #     )

    #     # we don't expect any fields to be present
    #     self.assertEqual(
    #         len(collection_dataset.get_field_names()),
    #         0,
    #         "Expected 0 fields in the dataset, got {len(collection_dataset.get_field_names())}",
    #     )

    #     # we expect the bounds to be correct
    #     bds = collection_dataset.get_bounds()
    #     self.assertEqual(bds[0], self.datasets["disk_out_ref_npz"]["bds"][0], "Mismatch in bounds min")
    #     self.assertEqual(bds[1], self.datasets["disk_out_ref_npz"]["bds"][1], "Mismatch in bounds max")

    async def test_utils_get_input_dataset_with_fields(self):
        with settings.override_setting(settings.SettingsKeys.DOWN_CONVERT_64BIT, False):
            operator_prim = self.stage.DefinePrim("/World/OperatorTestInputDatasetWithFields")
            # cae_viz.OperatorAPI.Apply(operator_prim)
            cae_viz.DatasetSelectionAPI.Apply(operator_prim, "source")
            dataset_selection_api_source = cae_viz.DatasetSelectionAPI(operator_prim, "source")
            dataset_selection_api_source.CreateTargetRel().SetTargets({self.datasets["disk_out_ref_npz"]["dataset"]})

            cae_viz.FieldSelectionAPI.Apply(operator_prim, "Temp")
            field_selection_api_temp = cae_viz.FieldSelectionAPI(operator_prim, "Temp")
            field_selection_api_temp.CreateTargetRel().SetTargets({self.datasets["disk_out_ref_npz"]["fields"]["Temp"]})

            cae_viz.FieldSelectionAPI.Apply(operator_prim, "V")
            field_selection_api_v = cae_viz.FieldSelectionAPI(operator_prim, "V")
            field_selection_api_v.CreateTargetRel().SetTargets({self.datasets["disk_out_ref_npz"]["fields"]["V"]})

            # V_2 is the second component of the V field
            cae_viz.FieldSelectionAPI.Apply(operator_prim, "V_2")
            field_selection_api_v_2 = cae_viz.FieldSelectionAPI(operator_prim, "V_2")
            field_selection_api_v_2.CreateTargetRel().SetTargets({self.datasets["disk_out_ref_npz"]["fields"]["V"]})
            field_selection_api_v_2.CreateModeAttr().Set(cae_viz.Tokens.selected_component)
            field_selection_api_v_2.CreateComponentIndexAttr().Set(2)

            # V_Mag is the vector magnitude of the V field
            cae_viz.FieldSelectionAPI.Apply(operator_prim, "V_Mag")
            field_selection_api_v_mag = cae_viz.FieldSelectionAPI(operator_prim, "V_Mag")
            field_selection_api_v_mag.CreateTargetRel().SetTargets({self.datasets["disk_out_ref_npz"]["fields"]["V"]})
            field_selection_api_v_mag.CreateModeAttr().Set(cae_viz.Tokens.vector_magnitude)

            dataset_prim = self.stage.GetPrimAtPath(self.datasets["disk_out_ref_npz"]["dataset"])
            temp_direct = await cae_dav.GetField.invoke(
                dataset_prim, "Temp", timeCode=Usd.TimeCode.EarliestTime(), device="cpu"
            )
            self.assertEqual(temp_direct.dtype, wp.float64, "Mismatch in direct scalar field dtype")
            self.assertAlmostEqual(
                temp_direct.get_range()[0],
                self.datasets["disk_out_ref_npz"]["field_ranges"]["Temp"][0],
                3,
                "Mismatch in direct scalar field range min",
            )
            self.assertAlmostEqual(
                temp_direct.get_range()[1],
                self.datasets["disk_out_ref_npz"]["field_ranges"]["Temp"][1],
                3,
                "Mismatch in direct scalar field range max",
            )

            v_direct = await cae_dav.GetField.invoke(
                dataset_prim, "V", timeCode=Usd.TimeCode.EarliestTime(), device="cpu"
            )
            self.assertEqual(v_direct.dtype, wp.vec3d, "Mismatch in direct vector field dtype")

            with self.assertRaises(
                usd_utils.QuietableException,
                msg="Expected QuietableException when getting input dataset for non-existent field",
            ):
                await utils.get_input_dataset(
                    operator_prim,
                    "source",
                    timeCode=Usd.TimeCode.EarliestTime(),
                    device="cpu",
                    required_fields={"Bogus"},
                )

            dataset = await utils.get_input_dataset(
                operator_prim,
                "source",
                timeCode=Usd.TimeCode.EarliestTime(),
                device="cpu",
                required_fields={"Temp", "V", "V_2", "V_Mag"},
            )
            self.assertIsNotNone(dataset, "Dataset should not be None")
            self.assertEqual(
                len(dataset.get_field_names()),
                4,
                "Expected 4 fields in the dataset, got {len(dataset.get_field_names())}",
            )

            self.assertEqual(dataset.get_field_names()[0], "Temp", "Mismatch in field name")
            self.assertEqual(dataset.get_field_names()[1], "V", "Mismatch in field name")
            self.assertEqual(dataset.get_field_names()[2], "V_2", "Mismatch in field name")
            self.assertEqual(dataset.get_field_names()[3], "V_Mag", "Mismatch in field name")

            temp_field = dataset.get_field("Temp")
            self.assertAlmostEqual(
                temp_field.get_range()[0],
                self.datasets["disk_out_ref_npz"]["field_ranges"]["Temp"][0],
                3,
                "Mismatch in field range min",
            )
            self.assertAlmostEqual(
                temp_field.get_range()[1],
                self.datasets["disk_out_ref_npz"]["field_ranges"]["Temp"][1],
                3,
                "Mismatch in field range max",
            )
            self.assertEqual(temp_field.dtype, wp.float64, "Mismatch in field dtype")

            v_field = dataset.get_field("V")
            self.assertEqual(v_field.dtype, wp.vec3d, "Mismatch in field dtype")

            v_2_field = dataset.get_field("V_2")
            self.assertEqual(v_2_field.dtype, wp.float64, "Mismatch in field dtype")
            self.assertAlmostEqual(
                v_2_field.get_range()[0],
                self.datasets["disk_out_ref_npz"]["field_ranges"]["V_2"][0],
                3,
                "Mismatch in field range min",
            )
            self.assertAlmostEqual(
                v_2_field.get_range()[1],
                self.datasets["disk_out_ref_npz"]["field_ranges"]["V_2"][1],
                3,
                "Mismatch in field range max",
            )

            v_mag_field = dataset.get_field("V_Mag")
            self.assertEqual(v_mag_field.dtype, wp.float64, "Mismatch in field dtype")
            self.assertAlmostEqual(
                v_mag_field.get_range()[0],
                self.datasets["disk_out_ref_npz"]["field_ranges"]["V_Mag"][0],
                3,
                "Mismatch in field range min",
            )
            self.assertAlmostEqual(
                v_mag_field.get_range()[1],
                self.datasets["disk_out_ref_npz"]["field_ranges"]["V_Mag"][1],
                3,
                "Mismatch in field range max",
            )

    async def test_utils_get_input_dataset_with_voxelization(self):
        operator_prim = self.stage.DefinePrim("/World/OperatorTestInputDatasetWithVoxelization")
        # cae_viz.OperatorAPI.Apply(operator_prim)
        cae_viz.DatasetSelectionAPI.Apply(operator_prim, "source")
        dataset_selection_api_source = cae_viz.DatasetSelectionAPI(operator_prim, "source")
        dataset_selection_api_source.CreateTargetRel().SetTargets({self.datasets["disk_out_ref_npz"]["dataset"]})

        cae_viz.DatasetVoxelizationAPI.Apply(operator_prim, "source")

        cae_viz.FieldSelectionAPI.Apply(operator_prim, "Temp")
        field_selection_api_temp = cae_viz.FieldSelectionAPI(operator_prim, "Temp")
        field_selection_api_temp.CreateTargetRel().SetTargets({self.datasets["disk_out_ref_npz"]["fields"]["Temp"]})

        cae_viz.FieldSelectionAPI.Apply(operator_prim, "V")
        field_selection_api_v = cae_viz.FieldSelectionAPI(operator_prim, "V")
        field_selection_api_v.CreateTargetRel().SetTargets({self.datasets["disk_out_ref_npz"]["fields"]["V"]})

        with self.assertRaises(RuntimeError, msg="Expected RuntimeError when getting input dataset on cpu"):
            await utils.get_input_dataset(
                operator_prim,
                "source",
                timeCode=Usd.TimeCode.EarliestTime(),
                device="cpu",
                required_fields={"Temp", "V"},
            )
        dataset = await utils.get_input_dataset(
            operator_prim,
            "source",
            timeCode=Usd.TimeCode.EarliestTime(),
            device="cuda:0",
            required_fields={"Temp", "V"},
        )
        self.assertIsNotNone(dataset, "Dataset should not be None")
        self.assertEqual(
            len(dataset.get_field_names()), 3, "Expected 3 fields in the dataset, got {len(dataset.get_field_names())}"
        )
        self.assertEqual(dataset.get_field_names()[0], "Temp", "Mismatch in field name")
        self.assertEqual(dataset.get_field_names()[1], "cae_mask", "Mismatch in field name")
        self.assertEqual(dataset.get_field_names()[2], "V", "Mismatch in field name")

        temp_field = dataset.get_field("Temp")
        self.assertEqual(temp_field.dtype, wp.float32, "Mismatch in field dtype")
        self.assertAlmostEqual(
            temp_field.get_range()[0],
            self.datasets["disk_out_ref_npz"]["field_ranges"]["Temp"][0],
            3,
            msg="Mismatch in field range min",
        )
        self.assertAlmostEqual(
            temp_field.get_range()[1],
            self.datasets["disk_out_ref_npz"]["field_ranges"]["Temp"][1],
            3,
            msg="Mismatch in field range max",
        )
        self.assertTrue(isinstance(temp_field.get_data(), wp.Volume), "Mismatch in field data type")

        v_field = dataset.get_field("V")
        self.assertEqual(v_field.dtype, wp.vec3f, "Mismatch in field dtype")
        self.assertTrue(isinstance(v_field.get_data(), wp.Volume), "Mismatch in field data type")

    async def test_utils_get_input_dataset_with_vox_voxel_size_mode(self):
        operator_prim = self.stage.DefinePrim("/World/OperatorTestInputDatasetWithVoxelization")
        # cae_viz.OperatorAPI.Apply(operator_prim)
        cae_viz.DatasetSelectionAPI.Apply(operator_prim, "source")
        dataset_selection_api_source = cae_viz.DatasetSelectionAPI(operator_prim, "source")
        dataset_selection_api_source.CreateTargetRel().SetTargets({self.datasets["disk_out_ref_npz"]["dataset"]})

        cae_viz.DatasetVoxelizationAPI.Apply(operator_prim, "source")
        dataset_voxelization_api_source = cae_viz.DatasetVoxelizationAPI(operator_prim, "source")
        dataset_voxelization_api_source.CreateVoxelSizeModeAttr().Set(cae_viz.Tokens.voxelSize)
        dataset_voxelization_api_source.CreateVoxelSizeAttr().Set((0.1, 0.1, 0.1))

        cae_viz.FieldSelectionAPI.Apply(operator_prim, "Temp")
        field_selection_api_temp = cae_viz.FieldSelectionAPI(operator_prim, "Temp")
        field_selection_api_temp.CreateTargetRel().SetTargets({self.datasets["disk_out_ref_npz"]["fields"]["Temp"]})

        dataset = await utils.get_input_dataset(
            operator_prim, "source", timeCode=Usd.TimeCode.EarliestTime(), device="cuda:0", required_fields={"Temp"}
        )
        self.assertIsNotNone(dataset, "Dataset should not be None")
        self.assertEqual(
            len(dataset.get_field_names()), 2, "Expected 2 fields in the dataset, got {len(dataset.get_field_names())}"
        )
        self.assertEqual(dataset.get_field_names()[0], "Temp", "Mismatch in field name")
        self.assertEqual(dataset.get_field_names()[1], "cae_mask", "Mismatch in field name")

        temp_field = dataset.get_field("Temp")
        self.assertEqual(temp_field.dtype, wp.float32, "Mismatch in field dtype")
        self.assertTrue(isinstance(temp_field.get_data(), wp.Volume), "Mismatch in field data type")
        self.assertEqual(temp_field.get_data().get_voxel_size(), wp.vec3f(0.1), "Mismatch in voxel size")
        self.assertAlmostEqual(
            temp_field.get_range()[0],
            self.datasets["disk_out_ref_npz"]["field_ranges"]["Temp"][0],
            3,
            msg="Mismatch in field range min",
        )
        self.assertAlmostEqual(
            temp_field.get_range()[1],
            self.datasets["disk_out_ref_npz"]["field_ranges"]["Temp"][1],
            3,
            msg="Mismatch in field range max",
        )

    async def test_utils_voxelization_parameters_field_centering(self):
        data_bounds = Gf.Range3d(Gf.Vec3d(-1.0, -1.0, -1.0), Gf.Vec3d(1.0, 1.0, 1.0))

        default_prim = self.stage.DefinePrim("/World/OperatorVoxelizationDefaultCentering")
        cae_viz.DatasetVoxelizationAPI.Apply(default_prim, "source")
        default_api = cae_viz.DatasetVoxelizationAPI(default_prim, "source")

        default_params = utils._get_voxelization_parameters(
            default_prim, "source", data_bounds, Usd.TimeCode.EarliestTime()
        )
        self.assertEqual(default_api.GetFieldCenteringAttr().Get(), cae_viz.Tokens.point)
        self.assertEqual(default_params.field_centering, "point")

        cell_prim = self.stage.DefinePrim("/World/OperatorVoxelizationCellCentering")
        cae_viz.DatasetVoxelizationAPI.Apply(cell_prim, "source")
        cell_api = cae_viz.DatasetVoxelizationAPI(cell_prim, "source")
        cell_api.CreateFieldCenteringAttr().Set(cae_viz.Tokens.cell)

        cell_params = utils._get_voxelization_parameters(cell_prim, "source", data_bounds, Usd.TimeCode.EarliestTime())
        self.assertEqual(cell_params.field_centering, "cell")

        invalid_prim = self.stage.DefinePrim("/World/OperatorVoxelizationInvalidCentering")
        cae_viz.DatasetVoxelizationAPI.Apply(invalid_prim, "source")
        invalid_api = cae_viz.DatasetVoxelizationAPI(invalid_prim, "source")
        invalid_api.CreateFieldCenteringAttr().Set("node")

        with self.assertRaisesRegex(ValueError, "Unsupported voxelization field centering"):
            utils._get_voxelization_parameters(invalid_prim, "source", data_bounds, Usd.TimeCode.EarliestTime())

    async def test_utils_get_input_dataset_with_subset(self):
        """Verify CaeVizDatasetSubsetAPI restricts the input dataset to cells inside the ROI bounds."""
        dataset_info = self.datasets["disk_out_ref_npz"]
        total_cells = dataset_info["nb_cells"]

        # Axis-aligned ROI cube centered at origin with extents (-2, 2) on each axis —
        # a strict subset of the dataset bounds (~ (-5.75, -5.75, -10) to (5.75, 5.75, 10.16)).
        roi_path = "/World/SubsetROI"
        roi_cube = UsdGeom.Cube.Define(self.stage, roi_path)
        roi_cube.CreateSizeAttr(4.0)

        counts = {}
        for mode in ("all", "any", "centroid"):
            operator_prim = self.stage.DefinePrim(f"/World/OperatorSubset_{mode}")
            cae_viz.DatasetSelectionAPI.Apply(operator_prim, "source")
            cae_viz.DatasetSelectionAPI(operator_prim, "source").CreateTargetRel().SetTargets({dataset_info["dataset"]})

            cae_viz.DatasetSubsetAPI.Apply(operator_prim, "source")
            subset_api = cae_viz.DatasetSubsetAPI(operator_prim, "source")
            subset_api.CreateRoiRel().SetTargets({roi_path})
            subset_api.CreateModeAttr().Set(mode)

            cae_viz.FieldSelectionAPI.Apply(operator_prim, "Temp")
            cae_viz.FieldSelectionAPI(operator_prim, "Temp").CreateTargetRel().SetTargets(
                {dataset_info["fields"]["Temp"]}
            )

            dataset = await utils.get_input_dataset(
                operator_prim,
                "source",
                timeCode=Usd.TimeCode.EarliestTime(),
                device="cuda:0",
                required_fields={"Temp"},
            )
            self.assertIsNotNone(dataset, f"mode={mode}: dataset should not be None")
            count = dataset.get_num_cells()
            counts[mode] = count

            self.assertGreater(count, 0, f"mode={mode}: subset should contain at least one cell")
            self.assertLess(count, total_cells, f"mode={mode}: subset should exclude at least one cell")
            self.assertTrue(dataset.has_field("Temp"), f"mode={mode}: Temp should pass through")
            # cell_idx is an internal indirection used by pass_fields and must not leak to callers.
            self.assertFalse(dataset.has_field("cell_idx"), f"mode={mode}: cell_idx should not be exposed")

        # For convex cells inside a convex (AABB) ROI, every cell selected by "all" has its
        # centroid inside the box (centroid is a convex combination of vertices) and at least
        # one vertex inside (trivially). So "all" is a subset of both "centroid" and "any".
        self.assertLessEqual(counts["all"], counts["centroid"], '"all" should be a subset of "centroid"')
        self.assertLessEqual(counts["all"], counts["any"], '"all" should be a subset of "any"')

    async def test_utils_get_input_dataset_with_subset_inflate_bounds(self):
        """Verify inflateBounds expands the effective ROI before cell selection."""
        dataset_info = self.datasets["disk_out_ref_npz"]

        # Tight ROI — extents (-0.5, 0.5) on each axis.
        roi_path = "/World/SubsetROITight"
        roi_cube = UsdGeom.Cube.Define(self.stage, roi_path)
        roi_cube.CreateSizeAttr(1.0)

        def build_operator(name: str, inflate: int) -> Usd.Prim:
            operator_prim = self.stage.DefinePrim(f"/World/{name}")
            cae_viz.DatasetSelectionAPI.Apply(operator_prim, "source")
            cae_viz.DatasetSelectionAPI(operator_prim, "source").CreateTargetRel().SetTargets({dataset_info["dataset"]})
            cae_viz.DatasetSubsetAPI.Apply(operator_prim, "source")
            subset_api = cae_viz.DatasetSubsetAPI(operator_prim, "source")
            subset_api.CreateRoiRel().SetTargets({roi_path})
            subset_api.CreateModeAttr().Set("any")
            subset_api.CreateInflateBoundsAttr().Set(inflate)
            return operator_prim

        tight_prim = build_operator("OperatorSubsetTight", 0)
        inflated_prim = build_operator("OperatorSubsetInflated", 400)

        tight_ds = await utils.get_input_dataset(
            tight_prim, "source", timeCode=Usd.TimeCode.EarliestTime(), device="cuda:0"
        )
        inflated_ds = await utils.get_input_dataset(
            inflated_prim, "source", timeCode=Usd.TimeCode.EarliestTime(), device="cuda:0"
        )

        self.assertGreater(
            inflated_ds.get_num_cells(),
            tight_ds.get_num_cells(),
            "Inflating ROI bounds should select more cells",
        )
        self.assertLess(
            inflated_ds.get_num_cells(),
            dataset_info["nb_cells"],
            "Inflated ROI should still exclude some cells from the full dataset",
        )

    async def test_utils_get_input_dataset_caching(self):
        """Test that get_input_dataset properly caches and invalidates cache."""
        operator_prim = self.stage.DefinePrim("/World/OperatorTestCaching")

        # Setup dataset selection
        cae_viz.DatasetSelectionAPI.Apply(operator_prim, "source")
        dataset_selection_api = cae_viz.DatasetSelectionAPI(operator_prim, "source")
        dataset_selection_api.CreateTargetRel().SetTargets({self.datasets["disk_out_ref_npz"]["dataset"]})

        # Setup field selection
        cae_viz.FieldSelectionAPI.Apply(operator_prim, "Temp")
        field_selection_api = cae_viz.FieldSelectionAPI(operator_prim, "Temp")
        field_selection_api.CreateTargetRel().SetTargets({self.datasets["disk_out_ref_npz"]["fields"]["Temp"]})

        # First call - should NOT be cached
        dataset1 = await utils.get_input_dataset(
            operator_prim,
            "source",
            timeCode=Usd.TimeCode.EarliestTime(),
            device="cpu",
            required_fields={"Temp"},
        )
        self.assertIsNotNone(dataset1, "First dataset should not be None")
        self.assertEqual(len(dataset1.get_field_names()), 1, "Should have 1 field")

        # Second call with same parameters - should be cached
        dataset2 = await utils.get_input_dataset(
            operator_prim,
            "source",
            timeCode=Usd.TimeCode.EarliestTime(),
            device="cpu",
            required_fields={"Temp"},
        )
        self.assertIsNotNone(dataset2, "Second dataset should not be None")
        # Verify it's the same cached object
        self.assertIs(dataset1, dataset2, "Second call should return cached dataset")

        # Modify a property on the prim - this should invalidate the cache
        field_selection_api.GetModeAttr().Set(cae_viz.Tokens.unchanged)
        await get_app().next_update_async()

        # Third call - should NOT be cached anymore due to prim modification
        dataset3 = await utils.get_input_dataset(
            operator_prim,
            "source",
            timeCode=Usd.TimeCode.EarliestTime(),
            device="cpu",
            required_fields={"Temp"},
        )
        self.assertIsNotNone(dataset3, "Third dataset should not be None")
        # This should be a new object since cache was invalidated
        self.assertIsNot(dataset1, dataset3, "Third call should return new dataset after cache invalidation")

        # Fourth call - should be cached again
        dataset4 = await utils.get_input_dataset(
            operator_prim,
            "source",
            timeCode=Usd.TimeCode.EarliestTime(),
            device="cpu",
            required_fields={"Temp"},
        )
        self.assertIs(dataset3, dataset4, "Fourth call should return cached dataset")

        # Test that different instance names use different cache entries
        # Note: Adding a new API to the prim will invalidate ALL caches for that prim
        # (as per the current implementation which uses prim-level invalidation)
        cae_viz.DatasetSelectionAPI.Apply(operator_prim, "other")
        dataset_selection_api_other = cae_viz.DatasetSelectionAPI(operator_prim, "other")
        dataset_selection_api_other.CreateTargetRel().SetTargets({self.datasets["disk_out_ref_tet_npz"]["dataset"]})
        await get_app().next_update_async()

        dataset_other = await utils.get_input_dataset(
            operator_prim,
            "other",
            timeCode=Usd.TimeCode.EarliestTime(),
            device="cpu",
            needs_fields=False,
        )
        self.assertIsNotNone(dataset_other, "Other instance dataset should not be None")
        self.assertIsNot(dataset3, dataset_other, "Different instance should have different cache entry")

        # Because we modified the prim by adding a new API, the cache for "source" was also invalidated
        # This is expected behavior with prim-level invalidation
        dataset5 = await utils.get_input_dataset(
            operator_prim,
            "source",
            timeCode=Usd.TimeCode.EarliestTime(),
            device="cpu",
            required_fields={"Temp"},
        )
        self.assertIsNot(dataset3, dataset5, "Adding new API invalidated all caches for this prim")

        # But now both should be cached
        dataset6 = await utils.get_input_dataset(
            operator_prim,
            "source",
            timeCode=Usd.TimeCode.EarliestTime(),
            device="cpu",
            required_fields={"Temp"},
        )
        self.assertIs(dataset5, dataset6, "Should be cached after re-fetch")

    async def test_utils_get_input_dataset_cache_invalidation_on_dataset_change(self):
        """Test that cache is invalidated when the underlying dataset prim changes."""
        operator_prim = self.stage.DefinePrim("/World/OperatorTestCacheDatasetChange")

        # Setup dataset selection
        cae_viz.DatasetSelectionAPI.Apply(operator_prim, "source")
        dataset_selection_api = cae_viz.DatasetSelectionAPI(operator_prim, "source")
        dataset_selection_api.CreateTargetRel().SetTargets({self.datasets["disk_out_ref_npz"]["dataset"]})

        # First call
        dataset1 = await utils.get_input_dataset(
            operator_prim,
            "source",
            timeCode=Usd.TimeCode.EarliestTime(),
            device="cpu",
            needs_fields=False,
        )
        self.assertIsNotNone(dataset1, "First dataset should not be None")
        self.assertEqual(
            dataset1.get_num_cells(),
            self.datasets["disk_out_ref_npz"]["nb_cells"],
            "Should have correct number of cells",
        )

        # Second call - should be cached
        dataset2 = await utils.get_input_dataset(
            operator_prim,
            "source",
            timeCode=Usd.TimeCode.EarliestTime(),
            device="cpu",
            needs_fields=False,
        )
        self.assertIs(dataset1, dataset2, "Should return cached dataset")

        # Modify the dataset prim (change the target to a different dataset)
        dataset_selection_api.GetTargetRel().SetTargets({self.datasets["disk_out_ref_tet_npz"]["dataset"]})
        await get_app().next_update_async()

        # Third call - should NOT be cached due to dataset change
        dataset3 = await utils.get_input_dataset(
            operator_prim,
            "source",
            timeCode=Usd.TimeCode.EarliestTime(),
            device="cpu",
            needs_fields=False,
        )
        self.assertIsNotNone(dataset3, "Third dataset should not be None")
        self.assertIsNot(dataset1, dataset3, "Should return new dataset after target change")
        self.assertEqual(
            dataset3.get_num_cells(),
            self.datasets["disk_out_ref_tet_npz"]["nb_cells"],
            "Should have correct number of cells from new dataset",
        )

    async def test_utils_get_input_dataset_cache_invalidation_on_roi_xform(self):
        """Cache must be invalidated when the ROI prim's xform changes.

        DatasetSubsetAPI/DatasetVoxelizationAPI derive their box from the ROI prim's world
        transform. Without a watch on the ROI prim itself, moving the ROI leaves the cached
        pre-move result in place and downstream operators run against stale data.
        """
        dataset_info = self.datasets["disk_out_ref_npz"]

        # Cube size 4.0 => extents (-2, 2) on each axis; strict subset of the dataset.
        roi_path = "/World/SubsetROIXform"
        roi_cube = UsdGeom.Cube.Define(self.stage, roi_path)
        roi_cube.CreateSizeAttr(4.0)
        translate_op = roi_cube.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(0.0, 0.0, 0.0))

        operator_prim = self.stage.DefinePrim("/World/OperatorSubsetROIXform")
        cae_viz.DatasetSelectionAPI.Apply(operator_prim, "source")
        cae_viz.DatasetSelectionAPI(operator_prim, "source").CreateTargetRel().SetTargets({dataset_info["dataset"]})
        cae_viz.DatasetSubsetAPI.Apply(operator_prim, "source")
        subset_api = cae_viz.DatasetSubsetAPI(operator_prim, "source")
        subset_api.CreateRoiRel().SetTargets({roi_path})
        subset_api.CreateModeAttr().Set("any")

        dataset1 = await utils.get_input_dataset(
            operator_prim, "source", timeCode=Usd.TimeCode.EarliestTime(), device="cuda:0", needs_fields=False
        )
        count1 = dataset1.get_num_cells()
        self.assertGreater(count1, 0, "Initial ROI should select at least one cell")

        # Sanity: repeated call hits the cache.
        dataset2 = await utils.get_input_dataset(
            operator_prim, "source", timeCode=Usd.TimeCode.EarliestTime(), device="cuda:0", needs_fields=False
        )
        self.assertIs(dataset1, dataset2, "Repeated call should return the cached dataset")

        # Shift the ROI along Z (dataset Z-extent is ~[-10, 10.16]) so the effective box moves to
        # [6, 10] in Z and selects a different set of cells.
        translate_op.Set(Gf.Vec3d(0.0, 0.0, 8.0))
        await get_app().next_update_async()

        dataset3 = await utils.get_input_dataset(
            operator_prim, "source", timeCode=Usd.TimeCode.EarliestTime(), device="cuda:0", needs_fields=False
        )
        self.assertIsNot(dataset1, dataset3, "Cache must be invalidated by ROI xform change")
        self.assertNotEqual(count1, dataset3.get_num_cells(), "Selected cell count must change when the ROI moves")


class TestRtSubPrimGuard(omni.kit.test.AsyncTestCase):
    """Tests for RtSubPrimGuard stage-transition correctness."""

    async def test_registry_cleared_on_stage_detach(self):
        """Registry is emptied when a stage is detached.

        Verifies that ``clear_all()`` is called via the stage-update subscription
        on stage close, so stale guards from the previous stage do not linger.
        """
        SOURCE_PATH = "/World/Source"
        RT_PATH = SdfRT.Path("/RT/SubPrim")

        async with new_stage() as stage:
            stage_id = omni.usd.get_context().get_stage_id()
            rt_stage = UsdRT.Stage.Attach(stage_id)

            source_prim = stage.DefinePrim(SOURCE_PATH, "Xform")
            rt_stage.DefinePrim(RT_PATH)
            RtSubPrimGuard.register(source_prim, rt_stage, [RT_PATH])

            self.assertIn(SOURCE_PATH, RtSubPrimGuard._registry, "Guard should be registered")

            del rt_stage  # release Fabric reference before stage closes

        # on_detach → clear_all() must have fired by the time close_stage_async() returns.
        self.assertEqual(len(RtSubPrimGuard._registry), 0, "Registry must be empty after stage detach")
