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
from omni.cae.data import settings, usd_utils
from omni.cae.importer.npz import import_to_stage
from omni.cae.schema import cae
from omni.cae.schema import viz as cae_viz
from omni.cae.testing import get_test_data_path
from omni.cae.viz import utils
from omni.kit.app import get_app
from omni.usd import get_context
from pxr import Usd


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
            len(dataset.get_field_names()), 2, "Expected 2 fields in the dataset, got {len(dataset.get_field_names())}"
        )
        self.assertEqual(dataset.get_field_names()[0], "Temp", "Mismatch in field name")
        self.assertEqual(dataset.get_field_names()[1], "V", "Mismatch in field name")

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
            len(dataset.get_field_names()), 1, "Expected 1 fields in the dataset, got {len(dataset.get_field_names())}"
        )
        self.assertEqual(dataset.get_field_names()[0], "Temp", "Mismatch in field name")

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
