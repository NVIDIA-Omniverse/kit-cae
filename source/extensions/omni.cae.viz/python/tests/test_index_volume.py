# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import dav
import numpy as np
import omni.kit.test
from omni.cae.data import cache
from omni.cae.data.commands import execute_command
from omni.cae.importer.cgns import import_to_stage
from omni.cae.schema import viz as cae_viz
from omni.cae.testing import get_test_data_path, new_stage, wait_for_update
from pxr import Usd, UsdShade, UsdVol


class TestIndeXVolume(omni.kit.test.AsyncTestCase):
    """
    These tests still don't verify that the IndeX components have executed since
    I am not sure how to trigger RTX rendering passes from the test. Until then, we will rely on the fact that
    the cached dataset is valid to verify that the IndeX components have executed.
    """

    tolerance = 1e-5

    def get_cached_dataset(
        self, prim: Usd.Prim, timeCode: Usd.TimeCode = Usd.TimeCode.EarliestTime(), must_exist: bool = True
    ) -> dav.Dataset:
        """
        Helper method to retrieve the dataset from cache for a given volume prim.

        Args:
            prim: The volume prim
            timeCode: The time code to retrieve (defaults to EarliestTime)

        Returns:
            The cached dav.Dataset
        """
        cache_key = f"[viz:index_volume]::{prim.GetPath()}"
        dataset = cache.get(cache_key, timeCode=timeCode)
        if must_exist:
            self.assertIsNotNone(dataset, f"Dataset should be in cache for key: {cache_key}")
        else:
            self.assertIsNone(dataset, f"Dataset should not be in cache for key: {cache_key}")
        return dataset

    def verify_cached_dataset(
        self,
        prim: Usd.Prim,
        expected_field_names: list[str],
        volume_type: str,
        expected_field_ranges: dict[str, tuple[float, float]] = None,
        timeCode: Usd.TimeCode = Usd.TimeCode.EarliestTime(),
    ):
        """
        Helper method to verify the cached dataset has the correct type, fields, and field ranges.

        Args:
            prim: The volume prim
            expected_field_names: List of expected field names
            volume_type: Either "irregular" or "vdb"
            expected_field_ranges: Optional dict mapping field names to (min, max) tuples for range verification
            timeCode: The time code to check (defaults to EarliestTime)
        """
        dataset = self.get_cached_dataset(prim, timeCode)

        # Verify dataset is valid
        self.assertIsInstance(dataset, dav.Dataset, "Cached object should be a dav.Dataset")

        # Verify field names
        field_names = dataset.get_field_names()
        self.assertEqual(
            len(field_names), len(expected_field_names), f"Dataset should have {len(expected_field_names)} field(s)"
        )
        for expected_field in expected_field_names:
            self.assertIn(expected_field, field_names, f"Field '{expected_field}' should be in dataset")

        # Verify field ranges if provided
        if expected_field_ranges:
            for field_name, (expected_min, expected_max) in expected_field_ranges.items():
                self.assertIn(field_name, field_names, f"Field '{field_name}' should be in dataset for range check")

                # Get the field from the dataset
                field = dataset.get_field(field_name)
                self.assertIsNotNone(field, f"Field '{field_name}' should be retrievable from dataset")

                # Get the field range
                field_range = field.get_range()
                self.assertIsNotNone(field_range, f"Field '{field_name}' should have a range")
                self.assertEqual(len(field_range), 2, f"Field '{field_name}' range should have min and max")

                actual_min, actual_max = field_range

                # Verify the range values
                np.testing.assert_allclose(
                    actual_min,
                    expected_min,
                    atol=self.tolerance,
                    err_msg=f"Field '{field_name}' min should be {expected_min}, got {actual_min}",
                )
                np.testing.assert_allclose(
                    actual_max,
                    expected_max,
                    atol=self.tolerance,
                    err_msg=f"Field '{field_name}' max should be {expected_max}, got {actual_max}",
                )

        # Verify bounds exist
        bounds = dataset.get_bounds()
        self.assertIsNotNone(bounds, "Dataset should have bounds")
        self.assertEqual(len(bounds), 2, "Bounds should have min and max")

        # Verify the data model type
        data_model = dataset.data_model
        self.assertIsNotNone(data_model, "Dataset should have a data model")

        # For vdb volumes, verify the dataset has been voxelized (would have different properties)
        if volume_type == "vdb":
            # VDB volumes should have voxelized data
            # The data model might be different after voxelization
            self.assertEqual(data_model, dav.data_models.vtk.image_data.DataModel)
        elif volume_type == "irregular":
            # Irregular volumes should preserve the original mesh structure
            # We can check for cells if it's an unstructured dataset
            pass  # Additional irregular-specific checks can be added here

        return dataset

    def verify_material_setup(self, stage: Usd.Stage, prim: Usd.Prim):
        """
        Helper method to verify that volume material is properly configured.

        Args:
            stage: The USD stage
            prim: The volume prim
        """
        viz_path = prim.GetPath()

        # Check material exists
        material_path = f"{viz_path}/Material"
        material_prim = stage.GetPrimAtPath(material_path)
        self.assertTrue(material_prim.IsValid(), "Material prim should exist")

        # Check colormap exists
        colormap_path = f"{material_path}/Colormap"
        colormap_prim = stage.GetPrimAtPath(colormap_path)
        self.assertTrue(colormap_prim.IsValid(), "Colormap prim should exist")

        # Check XAC shader exists
        shader_path = f"{material_path}/VolumeShader"
        shader_prim = stage.GetPrimAtPath(shader_path)
        self.assertTrue(shader_prim.IsValid(), "VolumeShader prim should exist")

        # Check data loader exists
        loader_path = f"{material_path}/DataLoader"
        loader_prim = stage.GetPrimAtPath(loader_path)
        self.assertTrue(loader_prim.IsValid(), "DataLoader prim should exist")

    async def create_volume(
        self, stage: Usd.Stage, dataset_path: str, prim_path: str, volume_type: str = "irregular"
    ) -> Usd.Prim:
        """Helper method to create a volume visualization."""
        ds_prim = stage.GetPrimAtPath(dataset_path)
        self.assertIsNotNone(ds_prim, "Dataset prim should be valid")

        await execute_command("CreateCaeVizVolume", dataset_path=dataset_path, prim_path=prim_path, type=volume_type)
        await wait_for_update()

        prim = stage.GetPrimAtPath(prim_path)
        self.assertIsNotNone(prim, "Volume prim should be valid")
        return prim

    def verify_importer_setup(self, prim: Usd.Prim, volume_type: str):
        """
        Verify that the importer is properly configured for the given volume type.

        Args:
            prim: The volume prim to check
            volume_type: Either "irregular" or "vdb"
        """
        importer = prim.GetChild("Importer")
        self.assertTrue(importer.IsValid(), "Importer prim should exist")
        self.assertEqual(importer.GetTypeName(), "FieldAsset", "Importer should be a FieldAsset")

        # Check importer custom data
        importer_settings = importer.GetCustomDataByKey("nvindex.importerSettings")
        self.assertIsNotNone(importer_settings, "Importer should have nvindex.importerSettings")

        if volume_type == "irregular":
            # Irregular volume uses PythonImporter
            self.assertEqual(
                importer_settings["importer"], "nv::omni::cae::index.PythonImporter", "Should use PythonImporter"
            )
            self.assertEqual(
                importer_settings["module_name"], "omni.cae.viz.index_volume", "Module name should be correct"
            )
            self.assertEqual(
                importer_settings["class_name"],
                "IndeXImporter_irregular_volume",
                "Class name should be IndeXImporter_irregular_volume",
            )
            self.assertEqual(
                str(importer_settings["params_prim_path"]), str(prim.GetPath()), "params_prim_path should match"
            )
            self.assertEqual(
                str(importer_settings["params_time_code"]),
                str(Usd.TimeCode.EarliestTime()),
                "params_time_code should be EarliestTime",
            )
            self.assertIn("params_cache_key", importer_settings, "Should have params_cache_key")
            self.assertIn("params_enable_field_interpolation", importer_settings, "Should have interpolation flag")

        elif volume_type == "vdb":
            # VDB/NanoVDB uses empty_init_importer
            self.assertEqual(
                str(importer_settings["importer"]),
                "nv::index::plugin::openvdb_integration.NanoVDB_empty_init_importer",
                "Should use NanoVDB_empty_init_importer",
            )
            self.assertIn("nb_attributes", importer_settings, "Should have nb_attributes")
            nb_attributes = importer_settings["nb_attributes"]
            self.assertGreater(nb_attributes, 0, "nb_attributes should be greater than 0")
        else:
            raise ValueError(f"Unknown volume_type: {volume_type}. Must be 'irregular' or 'vdb'")

        # Verify volume has field relationship to importer
        volume = UsdVol.Volume(prim)
        field_paths = volume.GetFieldPaths()
        self.assertGreater(len(field_paths), 0, "Volume should have at least one field relationship")
        self.assertIn(str(importer.GetPath()), field_paths.values(), "Importer should be in volume field paths")

    def verify_compute_task_setup(self, prim: Usd.Prim, volume_type: str):
        """
        Verify that the compute task (DataLoader) is properly configured for the given volume type.

        Args:
            prim: The volume prim to check
            volume_type: Either "irregular" or "vdb"
        """
        loader_prim = prim.GetChild("Material").GetChild("DataLoader")
        self.assertTrue(loader_prim.IsValid(), "DataLoader prim should exist")

        loader = UsdShade.Shader(loader_prim)
        self.assertTrue(loader, "DataLoader should be a shader")

        # Check shader attributes
        impl_source = loader.GetImplementationSourceAttr().Get()
        self.assertEqual(impl_source, UsdShade.Tokens.id, "Implementation source should be 'id'")

        shader_id = loader.GetIdAttr().Get()
        self.assertEqual(shader_id, "nv::omni::cae::index.PythonComputeTask", "Shader ID should be PythonComputeTask")

        # Check compute task inputs
        module_name_input = loader.GetInput("module_name")
        self.assertTrue(module_name_input, "Should have module_name input")
        self.assertEqual(module_name_input.Get(), "omni.cae.viz.index_volume", "module_name should be correct")

        class_name_input = loader.GetInput("class_name")
        self.assertTrue(class_name_input, "Should have class_name input")

        if volume_type == "irregular":
            expected_class_name = "IndeXComputeTask_irregular_volume"
            expected_is_gpu = False
        elif volume_type == "vdb":
            expected_class_name = "IndeXComputeTask_vdb"
            expected_is_gpu = True
        else:
            raise ValueError(f"Unknown volume_type: {volume_type}. Must be 'irregular' or 'vdb'")

        self.assertEqual(class_name_input.Get(), expected_class_name, f"class_name should be {expected_class_name}")

        enabled_input = loader.GetInput("enabled")
        self.assertTrue(enabled_input, "Should have enabled input")
        self.assertTrue(enabled_input.Get(), "enabled should be True")

        is_gpu_input = loader.GetInput("is_gpu_operation")
        self.assertTrue(is_gpu_input, "Should have is_gpu_operation input")
        self.assertEqual(
            is_gpu_input.Get(),
            expected_is_gpu,
            f"is_gpu_operation should be {expected_is_gpu} for {volume_type} volume",
        )

        prim_path_input = loader.GetInput("params_prim_path")
        self.assertTrue(prim_path_input, "Should have params_prim_path input")
        self.assertEqual(str(prim_path_input.Get()), str(prim.GetPath()), "params_prim_path should match")

        time_code_input = loader.GetInput("params_time_code")
        self.assertTrue(time_code_input, "Should have params_time_code input")
        self.assertIsNotNone(time_code_input.Get(), "params_time_code should be set")

        cache_key_input = loader.GetInput("params_cache_key")
        self.assertTrue(cache_key_input, "Should have params_cache_key input")
        self.assertIsNotNone(cache_key_input.Get(), "params_cache_key should be set")

    async def test_volume_creation_irregular(self):
        """Test basic irregular volume creation with field coloring and proper setup of importer and compute task."""
        async with new_stage() as stage:
            await import_to_stage(get_test_data_path("StaticMixer.cgns"), "/World/StaticMixer")
            base_path = "/World/StaticMixer/Base/StaticMixer"
            dataset_path = f"{base_path}/B1_P3"
            pressure_field_path = f"{base_path}/Flow_Solution/Pressure"
            viz_path = "/World/CAE/Volume_Irregular"

            prim = await self.create_volume(stage, dataset_path, viz_path, volume_type="irregular")

            # Verify the prim is a UsdVol.Volume
            self.assertTrue(prim.IsA(UsdVol.Volume), "Prim should be a UsdVol.Volume")

            # Verify the IndeXVolumeAPI is applied
            self.assertTrue(prim.HasAPI(cae_viz.IndeXVolumeAPI), "Should have IndeXVolumeAPI")

            # Verify nvindex:type is set correctly
            nvindex_type = prim.GetAttribute("nvindex:type").Get()
            self.assertEqual(nvindex_type, "irregular_volume", "nvindex:type should be 'irregular_volume'")

            # Verify DatasetSelectionAPI is present
            self.assertTrue(prim.HasAPI(cae_viz.DatasetSelectionAPI), "Should have DatasetSelectionAPI")

            # Set color field to Pressure; the operator doesn't really execute until the field is set.
            field_api = cae_viz.FieldSelectionAPI(prim, "colors")
            field_api.CreateTargetRel().SetTargets([pressure_field_path])
            await wait_for_update()

            # Verify extent attribute exists and has correct values
            volume = UsdVol.Volume(prim)
            extent_attr = volume.GetExtentAttr()
            self.assertTrue(extent_attr.IsValid(), "Extent attribute should be valid")
            extent = extent_attr.Get()
            self.assertEqual(len(extent), 2, "Extent should have 2 elements (min and max)")

            # Verify extent values match the dataset bounds
            expected_min = (-2, -3, -2)
            expected_max = (2, 3, 2)
            np.testing.assert_allclose(extent[0], expected_min, atol=self.tolerance, err_msg="Extent min should match")
            np.testing.assert_allclose(extent[1], expected_max, atol=self.tolerance, err_msg="Extent max should match")

            # Verify the field selection target is set
            targets = field_api.GetTargetRel().GetTargets()
            self.assertEqual(len(targets), 1, "Should have one field target")
            self.assertEqual(str(targets[0]), pressure_field_path, "Field target should match Pressure path")

            # Verify importer setup using utility method
            self.verify_importer_setup(prim, "irregular")

            # Verify compute task setup using utility method
            self.verify_compute_task_setup(prim, "irregular")

            # Verify material setup using utility method
            self.verify_material_setup(stage, prim)

            # Verify cached dataset contains the correct data
            self.verify_cached_dataset(
                prim,
                expected_field_names=["colors"],
                volume_type="irregular",
                expected_field_ranges={"colors": (-933.057556, 13029.029297)},
            )

    async def test_volume_creation_vdb(self):
        """Test basic VDB (NanoVDB) volume creation with field coloring and proper setup of importer and compute task."""
        async with new_stage() as stage:
            await import_to_stage(get_test_data_path("StaticMixer.cgns"), "/World/StaticMixer")
            base_path = "/World/StaticMixer/Base/StaticMixer"
            dataset_path = f"{base_path}/B1_P3"
            temp_field_path = f"{base_path}/Flow_Solution/Temperature"
            viz_path = "/World/CAE/Volume_VDB"

            prim = await self.create_volume(stage, dataset_path, viz_path, volume_type="vdb")

            # Verify the prim is a UsdVol.Volume
            self.assertTrue(prim.IsA(UsdVol.Volume), "Prim should be a UsdVol.Volume")

            # Verify the IndeXVolumeAPI is applied
            self.assertTrue(prim.HasAPI(cae_viz.IndeXVolumeAPI), "Should have IndeXVolumeAPI")

            # Verify nvindex:type is set correctly
            nvindex_type = prim.GetAttribute("nvindex:type").Get()
            self.assertEqual(nvindex_type, "vdb", "nvindex:type should be 'vdb'")

            # Verify DatasetVoxelizationAPI is present for VDB volumes
            self.assertTrue(
                prim.HasAPI(cae_viz.DatasetVoxelizationAPI), "VDB volume should have DatasetVoxelizationAPI"
            )

            # Verify DatasetSelectionAPI is present
            self.assertTrue(prim.HasAPI(cae_viz.DatasetSelectionAPI), "Should have DatasetSelectionAPI")

            # Set color field to Temperature; the operator doesn't really execute until the field is set.
            field_api = cae_viz.FieldSelectionAPI(prim, "colors")
            field_api.CreateTargetRel().SetTargets([temp_field_path])
            await wait_for_update()

            # Verify extent attribute exists and has correct values
            volume = UsdVol.Volume(prim)
            extent_attr = volume.GetExtentAttr()
            self.assertTrue(extent_attr.IsValid(), "Extent attribute should be valid")
            extent = extent_attr.Get()
            self.assertEqual(len(extent), 2, "Extent should have 2 elements (min and max)")

            # Verify extent values match the dataset bounds
            expected_min = (-2.019507, -3.02926, -2.019507)
            expected_max = (2.06759, 3.077344, 2.06759)
            np.testing.assert_allclose(extent[0], expected_min, atol=self.tolerance, err_msg="Extent min should match")
            np.testing.assert_allclose(extent[1], expected_max, atol=self.tolerance, err_msg="Extent max should match")

            # Verify the field selection target is set
            targets = field_api.GetTargetRel().GetTargets()
            self.assertEqual(len(targets), 1, "Should have one field target")
            self.assertEqual(str(targets[0]), temp_field_path, "Field target should match Temperature path")

            # Verify importer setup using utility method
            self.verify_importer_setup(prim, "vdb")

            # Verify compute task setup using utility method
            self.verify_compute_task_setup(prim, "vdb")

            # Verify material setup using utility method
            self.verify_material_setup(stage, prim)

            # Verify cached dataset contains the correct data
            self.verify_cached_dataset(
                prim,
                expected_field_names=["colors"],
                volume_type="vdb",
                expected_field_ranges={"colors": (285.0, 315.000458)},
            )

            # ===== Test sampling distance behavior =====
            # Verify initial sampling distance is set
            initial_sampling_distance = prim.GetCustomDataByKey("nvindex.renderSettings:samplingDistance")
            self.assertIsNotNone(initial_sampling_distance, "Sampling distance should be set initially")
            initial_voxel_size = prim.GetCustomDataByKey("cae:viz:last_voxel_size")
            self.assertIsNotNone(initial_voxel_size, "Voxel size should be tracked")

            # Block sampling distance updates
            prim.SetCustomDataByKey("cae:viz:block_sampling_distance_update", True)

            # Change maxResolution and verify sampling distance doesn't change
            voxel_api = cae_viz.DatasetVoxelizationAPI(prim, "source")
            voxel_api.CreateMaxResolutionAttr().Set(256)  # Change resolution
            await wait_for_update()

            blocked_sampling_distance = prim.GetCustomDataByKey("nvindex.renderSettings:samplingDistance")
            self.assertEqual(
                blocked_sampling_distance, initial_sampling_distance, "Sampling distance should NOT change when blocked"
            )

            # Unblock sampling distance updates
            prim.SetCustomDataByKey("cae:viz:block_sampling_distance_update", False)

            # Change maxResolution again and verify sampling distance DOES change
            voxel_api.GetMaxResolutionAttr().Set(32)  # Change to different resolution
            await wait_for_update()

            unblocked_sampling_distance = prim.GetCustomDataByKey("nvindex.renderSettings:samplingDistance")
            self.assertNotEqual(
                unblocked_sampling_distance,
                blocked_sampling_distance,
                "Sampling distance SHOULD change when unblocked and resolution changes",
            )

            # Verify voxel size was updated
            new_voxel_size = prim.GetCustomDataByKey("cae:viz:last_voxel_size")
            self.assertNotEqual(
                new_voxel_size, initial_voxel_size, "Voxel size should be updated after resolution change"
            )

    async def test_volume_with_field_selection(self):
        """Test volume with multi-field selection for coloring."""
        async with new_stage() as stage:
            await import_to_stage(get_test_data_path("StaticMixer.cgns"), "/World/StaticMixer")
            base_path = "/World/StaticMixer/Base/StaticMixer"
            dataset_path = f"{base_path}/B1_P3"
            temp_field_path = f"{base_path}/Flow_Solution/Temperature"
            pressure_field_path = f"{base_path}/Flow_Solution/Pressure"
            viz_path = "/World/CAE/Volume_WithField"

            prim = await self.create_volume(stage, dataset_path, viz_path, volume_type="irregular")

            # Verify FieldSelectionAPI is present
            self.assertTrue(prim.HasAPI(cae_viz.FieldSelectionAPI), "Should have FieldSelectionAPI")

            # Create multiple field selection instances with different names
            # First field: "colors" -> Temperature
            field_api_colors = cae_viz.FieldSelectionAPI(prim, "colors")
            field_api_colors.CreateTargetRel().SetTargets([temp_field_path])

            # Second field: "other_field" -> Pressure
            cae_viz.FieldSelectionAPI.Apply(prim, "other_field")
            field_api_other = cae_viz.FieldSelectionAPI(prim, "other_field")
            field_api_other.CreateTargetRel().SetTargets([pressure_field_path])

            await wait_for_update()

            # Verify extent attribute exists and has correct values (after fields are set)
            volume = UsdVol.Volume(prim)
            extent_attr = volume.GetExtentAttr()
            self.assertTrue(extent_attr.IsValid(), "Extent attribute should be valid")
            extent = extent_attr.Get()
            self.assertEqual(len(extent), 2, "Extent should have 2 elements (min and max)")

            # Verify extent values match the dataset bounds
            expected_min = (-2, -3, -2)
            expected_max = (2, 3, 2)
            np.testing.assert_allclose(extent[0], expected_min, atol=self.tolerance, err_msg="Extent min should match")
            np.testing.assert_allclose(extent[1], expected_max, atol=self.tolerance, err_msg="Extent max should match")

            # Verify importer setup using utility method
            self.verify_importer_setup(prim, "irregular")

            # Verify compute task setup using utility method
            self.verify_compute_task_setup(prim, "irregular")

            # Verify cached dataset contains both field instances with correct ranges
            # Note: The cached dataset uses the FieldSelectionAPI instance names ("colors" and "other_field")
            self.verify_cached_dataset(
                prim,
                expected_field_names=["colors", "other_field"],
                volume_type="irregular",
                expected_field_ranges={
                    "colors": (285.0, 315.000458),  # Temperature range (matching vdb test)
                    "other_field": (-933.057556, 13029.029297),  # Pressure range (matching irregular test)
                },
            )

    async def test_volume_temporal(self):
        """Test volume with temporal data at different time steps."""
        async with new_stage() as stage:
            from omni.timeline import get_timeline_interface

            # Import temporal dataset
            await import_to_stage(get_test_data_path("hex_timesteps.cgns"), "/World/hex_timesteps", time_scale=10.0)
            base_path = "/World/hex_timesteps/Base/Zone"
            dataset_path = f"{base_path}/ElementsUniform"
            field_path = f"{base_path}/SolutionVertex0001/PointSinusoid"
            other_field_path = f"{base_path}/SolutionCellCenter0001/CellSinusoid"
            viz_path = "/World/CAE/Volume_Temporal"

            # Set up timeline
            timeline = get_timeline_interface()
            timeline.set_time_codes_per_second(1.0)
            timeline.set_start_time(0.0)
            timeline.set_end_time(100.0)
            timeline.set_current_time(0.0)
            await wait_for_update()

            prim = await self.create_volume(stage, dataset_path, viz_path, volume_type="irregular")

            # Verify the prim is a UsdVol.Volume
            self.assertTrue(prim.IsA(UsdVol.Volume), "Prim should be a UsdVol.Volume")

            # Verify the IndeXVolumeAPI is applied
            self.assertTrue(prim.HasAPI(cae_viz.IndeXVolumeAPI), "Should have IndeXVolumeAPI")

            # Verify nvindex:type is set correctly
            nvindex_type = prim.GetAttribute("nvindex:type").Get()
            self.assertEqual(nvindex_type, "irregular_volume", "nvindex:type should be 'irregular_volume'")

            # Verify DatasetSelectionAPI is present
            self.assertTrue(prim.HasAPI(cae_viz.DatasetSelectionAPI), "Should have DatasetSelectionAPI")

            # Set color field to PointSinusoid; the operator doesn't really execute until the field is set.
            field_api = cae_viz.FieldSelectionAPI(prim, "colors")
            field_api.CreateTargetRel().SetTargets([field_path])
            await wait_for_update()

            # Verify extent attribute exists and has correct values at time 0
            volume = UsdVol.Volume(prim)
            extent_attr = volume.GetExtentAttr()
            self.assertTrue(extent_attr.IsValid(), "Extent attribute should be valid")
            extent_t0 = extent_attr.Get()
            self.assertEqual(len(extent_t0), 2, "Extent should have 2 elements (min and max)")

            # Verify importer setup using utility method
            self.verify_importer_setup(prim, "irregular")

            # Verify compute task setup using utility method
            self.verify_compute_task_setup(prim, "irregular")

            # Verify material setup using utility method
            self.verify_material_setup(stage, prim)

            # Verify cached dataset contains the correct data at time 0
            dataset_t0 = self.verify_cached_dataset(
                prim,
                expected_field_names=["colors"],
                volume_type="irregular",
                expected_field_ranges={"colors": (-1.0, 1.0)},  # PointSinusoid range
            )

            # ===== Test temporal behavior at different time steps =====

            # Move to time 10 and verify operator still works
            timeline.set_current_time(10.0)
            await wait_for_update()

            # Verify prim is still valid
            self.assertTrue(prim.IsValid(), "Volume should be valid at time 10")

            # Verify extent is still valid (geometry doesn't change in this dataset)
            extent_t10 = extent_attr.Get()
            self.assertEqual(len(extent_t10), 2, "Extent should have 2 elements at time 10")
            np.testing.assert_allclose(
                extent_t10[0], extent_t0[0], atol=self.tolerance, err_msg="Extent min should be consistent"
            )
            np.testing.assert_allclose(
                extent_t10[1], extent_t0[1], atol=self.tolerance, err_msg="Extent max should be consistent"
            )

            # Verify cached dataset at time 10 (field values may differ)
            dataset_t10 = self.get_cached_dataset(prim, timeCode=Usd.TimeCode(10.0))
            self.assertIsNotNone(dataset_t10, "Dataset should be cached at time 10")
            self.assertEqual(dataset_t10.get_field_names(), ["colors"], "Should have colors field at time 10")

            # Move to time 20 and verify operator still works
            timeline.set_current_time(20.0)
            await wait_for_update()

            # Verify prim is still valid
            self.assertTrue(prim.IsValid(), "Volume should be valid at time 20")

            # Verify cached dataset at time 20
            dataset_t20 = self.get_cached_dataset(prim, timeCode=Usd.TimeCode(20.0))
            self.assertIsNotNone(dataset_t20, "Dataset should be cached at time 20")
            self.assertEqual(dataset_t20.get_field_names(), ["colors"], "Should have colors field at time 20")

            # Verify compute task time parameters are updated
            loader_prim = prim.GetChild("Material").GetChild("DataLoader")
            loader = UsdShade.Shader(loader_prim)
            time_code_input = loader.GetInput("params_time_code")
            self.assertTrue(time_code_input, "Should have params_time_code input")
            # The time code should reflect the current timeline position
            self.assertIsNotNone(time_code_input.Get(), "params_time_code should be set at time 20")

            # modify something (say coloring field) on the prim, and verify the cached dataset is invalidated
            field_api = cae_viz.FieldSelectionAPI(prim, "colors")
            field_api.CreateTargetRel().SetTargets([other_field_path])
            await wait_for_update()

            # Verify cached dataset is invalidated
            self.get_cached_dataset(prim, timeCode=Usd.TimeCode(0.0), must_exist=False)
            self.get_cached_dataset(prim, timeCode=Usd.TimeCode(20.0), must_exist=True)

    async def test_volume_temporal_interpolation(self):
        """Test volume with temporal data and field interpolation enabled."""
        async with new_stage() as stage:
            from omni.timeline import get_timeline_interface

            # Import temporal dataset
            await import_to_stage(get_test_data_path("hex_timesteps.cgns"), "/World/hex_timesteps", time_scale=10.0)
            base_path = "/World/hex_timesteps/Base/Zone"
            dataset_path = f"{base_path}/ElementsUniform"
            field_path = f"{base_path}/SolutionVertex0001/PointSinusoid"
            viz_path = "/World/CAE/Volume_Temporal_Interp"

            # Set up timeline
            timeline = get_timeline_interface()
            timeline.set_time_codes_per_second(1.0)
            timeline.set_start_time(0.0)
            timeline.set_end_time(100.0)
            timeline.set_current_time(0.0)
            await wait_for_update()

            prim = await self.create_volume(stage, dataset_path, viz_path, volume_type="irregular")

            # Verify the prim is a UsdVol.Volume
            self.assertTrue(prim.IsA(UsdVol.Volume), "Prim should be a UsdVol.Volume")

            # Verify the IndeXVolumeAPI is applied
            self.assertTrue(prim.HasAPI(cae_viz.IndeXVolumeAPI), "Should have IndeXVolumeAPI")

            # Set color field to PointSinusoid
            field_api = cae_viz.FieldSelectionAPI(prim, "colors")
            field_api.CreateTargetRel().SetTargets([field_path])
            await wait_for_update()

            # Enable temporal interpolation
            cae_viz.OperatorTemporalAPI.Apply(prim)
            temporal_api = cae_viz.OperatorTemporalAPI(prim)
            temporal_api.CreateEnableFieldInterpolationAttr().Set(True)
            await wait_for_update()

            # Verify OperatorTemporalAPI is applied
            self.assertTrue(prim.HasAPI(cae_viz.OperatorTemporalAPI), "Should have OperatorTemporalAPI")
            self.assertTrue(
                temporal_api.GetEnableFieldInterpolationAttr().Get(), "Field interpolation should be enabled"
            )

            # Verify importer setup reflects interpolation settings
            importer = prim.GetChild("Importer")
            self.assertTrue(importer.IsValid(), "Importer prim should exist")
            importer_settings = importer.GetCustomDataByKey("nvindex.importerSettings")
            self.assertIsNotNone(importer_settings, "Importer should have nvindex.importerSettings")
            self.assertIn("params_enable_field_interpolation", importer_settings, "Should have interpolation flag")
            self.assertTrue(
                importer_settings["params_enable_field_interpolation"], "Importer interpolation flag should be True"
            )

            # Verify compute task has next_time_code parameters
            loader_prim = prim.GetChild("Material").GetChild("DataLoader")
            loader = UsdShade.Shader(loader_prim)

            time_code_input = loader.GetInput("params_time_code")
            self.assertTrue(time_code_input, "Should have params_time_code input")
            self.assertIsNotNone(time_code_input.Get(), "params_time_code should be set")

            next_time_code_input = loader.GetInput("params_next_time_code")
            self.assertTrue(next_time_code_input, "Should have params_next_time_code input")
            # At time 0, next_time_code might be None or set to next keyframe

            # Verify cached dataset at time 0
            self.verify_cached_dataset(
                prim,
                expected_field_names=["colors"],
                volume_type="irregular",
                expected_field_ranges={"colors": (-1.0, 1.0)},
                timeCode=Usd.TimeCode(0.0),
            )

            # since 0 doens't need next, we should not have a cached dataset for it
            self.get_cached_dataset(prim, timeCode=Usd.TimeCode(10.0), must_exist=False)

            # Move to time 5 (between keyframes 0 and 10) to test interpolation
            timeline.set_current_time(5.0)
            await wait_for_update()

            # Verify prim is still valid
            self.assertTrue(prim.IsValid(), "Volume should be valid at time 5 (interpolated)")

            # Verify compute task parameters include both current and next time codes
            time_code_str = time_code_input.Get()
            next_time_code_str = next_time_code_input.Get()
            self.assertIsNotNone(time_code_str, "params_time_code should be set at interpolated time")
            self.assertIsNotNone(next_time_code_str, "params_next_time_code should be set for interpolation")
            # next_time_code should not be "None" when interpolating
            self.assertNotEqual(
                str(next_time_code_str), "None", "params_next_time_code should not be None when interpolating"
            )

            # Verify cached datasets exist for both bracketing time codes
            # The operator should cache data at the bracketing times (0 and 10)
            dataset_t0 = self.get_cached_dataset(prim, timeCode=Usd.TimeCode(0.0))
            self.assertIsNotNone(dataset_t0, "Dataset should be cached at time 0 (bracket)")

            dataset_t10 = self.get_cached_dataset(prim, timeCode=Usd.TimeCode(10.0))
            self.assertIsNotNone(dataset_t10, "Dataset should be cached at time 10 (bracket)")

            # Move to an exact keyframe (time 10)
            timeline.set_current_time(10.0)
            await wait_for_update()

            # At exact keyframe, next_time_code might be None or next keyframe
            time_code_str_t10 = time_code_input.Get()
            self.assertIsNotNone(time_code_str_t10, "params_time_code should be set at keyframe")

            # Disable interpolation and verify behavior changes
            temporal_api.GetEnableFieldInterpolationAttr().Set(False)
            await wait_for_update()

            # Move to interpolated time again
            timeline.set_current_time(15.0)
            await wait_for_update()

            # With interpolation disabled, the time should snap to nearest keyframe
            # The operator should still work but without interpolation
            self.get_cached_dataset(prim, timeCode=Usd.TimeCode(10.0), must_exist=True)
            self.get_cached_dataset(prim, timeCode=Usd.TimeCode(20.0), must_exist=False)
            self.get_cached_dataset(prim, timeCode=Usd.TimeCode(0.0), must_exist=False)
