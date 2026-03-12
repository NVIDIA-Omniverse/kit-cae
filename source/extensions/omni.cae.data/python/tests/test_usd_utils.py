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

import omni.kit.test
from omni.cae.data import usd_utils
from omni.cae.schema import cae, sids
from omni.cae.testing import get_test_data_path
from omni.usd import get_context
from pxr import Sdf, Usd, UsdGeom, UsdUtils

logger = getLogger(__name__)


class TestUsdUtils(omni.kit.test.AsyncTestCase):

    async def test_with_hex_polyhedra(self):
        usd_context = get_context()
        await usd_context.open_stage_async(get_test_data_path("hex_polyhedra.cgns"))
        stage: Usd.Stage = usd_context.get_stage()

        gcPrim = stage.GetPrimAtPath("/World/hex_polyhedra_cgns/Base/Zone/GridCoordinates")
        self.assertTrue(gcPrim.IsValid())
        coords = usd_utils.get_target_paths(gcPrim, cae.Tokens.caePointCloudCoordinates)
        self.assertEqual(len(coords), 3)

        with self.assertRaises(usd_utils.QuietableException):
            _ = usd_utils.get_target_path(gcPrim, sids.Tokens.caeSidsGridCoordinates)

        none_field = usd_utils.get_target_path(gcPrim, sids.Tokens.caeSidsGridCoordinates, quiet=True)
        self.assertIsNone(none_field, "None should be returned on quiet=True")

        prim = stage.GetPrimAtPath("/World/hex_polyhedra_cgns/Base/Zone/ElementsNfaces")
        self.assertTrue(prim.IsValid())

        array = await usd_utils.get_vecN_from_relationship(
            prim, sids.Tokens.caeSidsGridCoordinates, 3, Usd.TimeCode.EarliestTime()
        )
        self.assertEqual(array.shape[1], 3)
        self.assertEqual(array.shape[0], 35937)

        field = await usd_utils.get_array_from_relationship(
            prim, "field:PointDistanceToCenter", Usd.TimeCode.EarliestTime()
        )
        self.assertEqual(field.shape[0], 35937)

        fields = await usd_utils.get_arrays_from_relationship(
            prim, "field:CellDistanceToCenter", Usd.TimeCode.EarliestTime()
        )
        self.assertEqual(len(fields), 1)
        self.assertEqual(fields[0].shape[0], 32768)

        cellfieldPath = "/World/hex_polyhedra_cgns/Base/Zone/SolutionCellCenter/CellDistanceToCenter"
        fname = usd_utils.get_field_name(prim, stage.GetPrimAtPath(cellfieldPath))
        self.assertEqual(fname, "CellDistanceToCenter")

        with self.assertRaises(usd_utils.QuietableException):
            _ = usd_utils.get_field_name(prim, gcPrim)

    async def test_get_bracketing_time_samples_for_prim(self):
        """Test getBracketingTimeSamplesForPrim C++ implementation"""
        # Create a stage with time samples
        stage = Usd.Stage.CreateInMemory()

        # Create a DataSet prim
        dataset = cae.DataSet.Define(stage, "/Root/DataSet")

        # Create FieldArray prims with time samples
        field1 = cae.FieldArray.Define(stage, "/Root/DataSet/Field1")
        field2 = cae.FieldArray.Define(stage, "/Root/DataSet/Field2")

        # Create attributes with time samples
        attr1 = field1.GetPrim().CreateAttribute("testAttr1", Sdf.ValueTypeNames.Float)
        attr2 = field2.GetPrim().CreateAttribute("testAttr2", Sdf.ValueTypeNames.Float)

        # Set time samples: 0.0, 1.0, 2.0, 3.0
        attr1.Set(10.0, Usd.TimeCode(0.0))
        attr1.Set(20.0, Usd.TimeCode(1.0))
        attr1.Set(30.0, Usd.TimeCode(2.0))
        attr1.Set(40.0, Usd.TimeCode(3.0))

        attr2.Set(100.0, Usd.TimeCode(1.5))
        attr2.Set(200.0, Usd.TimeCode(2.5))

        # Create relationship from dataset to fields
        rel = dataset.GetPrim().CreateRelationship("field:Field1")
        rel.AddTarget(field1.GetPrim().GetPath())

        rel2 = dataset.GetPrim().CreateRelationship("field:Field2")
        rel2.AddTarget(field2.GetPrim().GetPath())

        await omni.usd.get_context().attach_stage_async(stage)

        # Test 1: Exact match
        lower, upper, has_time_samples = usd_utils.get_bracketing_time_samples_for_prim(dataset.GetPrim(), 1.0)
        self.assertEqual(lower, 1.0)
        self.assertEqual(upper, 1.0)
        self.assertTrue(has_time_samples)

        # Test 2: Exact match (1.5 is a time sample)
        lower, upper, has_time_samples = usd_utils.get_bracketing_time_samples_for_prim(dataset.GetPrim(), 1.5)
        self.assertEqual(lower, 1.5)
        self.assertEqual(upper, 1.5)
        self.assertTrue(has_time_samples)

        # Test 3: Between samples (2.2 should bracket between 2.0 and 2.5)
        lower, upper, has_time_samples = usd_utils.get_bracketing_time_samples_for_prim(dataset.GetPrim(), 2.2)
        self.assertEqual(lower, 2.0)
        self.assertEqual(upper, 2.5)
        self.assertTrue(has_time_samples)

        # Test 4: Before first sample
        lower, upper, has_time_samples = usd_utils.get_bracketing_time_samples_for_prim(dataset.GetPrim(), -1.0)
        self.assertEqual(lower, 0.0)
        self.assertEqual(upper, 0.0)
        self.assertTrue(has_time_samples)

        # Test 5: After last sample
        lower, upper, has_time_samples = usd_utils.get_bracketing_time_samples_for_prim(dataset.GetPrim(), 5.0)
        self.assertEqual(lower, 3.0)
        self.assertEqual(upper, 3.0)
        self.assertTrue(has_time_samples)

        # Test 6: Prim with no time samples
        empty_prim = cae.DataSet.Define(stage, "/Root/EmptyDataSet")
        earliest_time = Usd.TimeCode.EarliestTime().GetValue()
        lower, upper, has_time_samples = usd_utils.get_bracketing_time_samples_for_prim(empty_prim.GetPrim(), 1.0)
        self.assertEqual(lower, earliest_time)
        self.assertEqual(upper, earliest_time)
        self.assertFalse(has_time_samples)

        # Test 7: Invalid prim
        invalid_prim = stage.GetPrimAtPath("/Root/NonExistent")
        lower, upper, has_time_samples = usd_utils.get_bracketing_time_samples_for_prim(invalid_prim, 1.0)
        self.assertEqual(lower, earliest_time)
        self.assertEqual(upper, earliest_time)
        self.assertFalse(has_time_samples)

    async def test_get_bracketing_time_samples_for_data_set_prim(self):
        """Test getBracketingTimeSamplesForDataSetPrim C++ implementation with field relationship control"""
        # Create a stage with time samples
        stage = Usd.Stage.CreateInMemory()

        # Create a DataSet prim
        dataset = cae.DataSet.Define(stage, "/Root/DataSet")

        # Create FieldArray prims with time samples
        field1 = cae.FieldArray.Define(stage, "/Root/DataSet/Field1")
        field2 = cae.FieldArray.Define(stage, "/Root/DataSet/Field2")

        # Create attributes with time samples
        attr1 = field1.GetPrim().CreateAttribute("testAttr1", Sdf.ValueTypeNames.Float)
        attr2 = field2.GetPrim().CreateAttribute("testAttr2", Sdf.ValueTypeNames.Float)

        # Set time samples: 0.0, 1.0, 2.0, 3.0
        attr1.Set(10.0, Usd.TimeCode(0.0))
        attr1.Set(20.0, Usd.TimeCode(1.0))
        attr1.Set(30.0, Usd.TimeCode(2.0))
        attr1.Set(40.0, Usd.TimeCode(3.0))

        attr2.Set(100.0, Usd.TimeCode(1.5))
        attr2.Set(200.0, Usd.TimeCode(2.5))

        # Create field relationships from dataset to fields
        rel = dataset.GetPrim().CreateRelationship("field:Field1")
        rel.AddTarget(field1.GetPrim().GetPath())

        rel2 = dataset.GetPrim().CreateRelationship("field:Field2")
        rel2.AddTarget(field2.GetPrim().GetPath())

        # Create a non-field relationship (e.g., coordinates)
        coords = cae.FieldArray.Define(stage, "/Root/DataSet/Coords")
        coords_attr = coords.GetPrim().CreateAttribute("testAttr", Sdf.ValueTypeNames.Float)
        coords_attr.Set(5.0, Usd.TimeCode(0.5))
        coords_attr.Set(6.0, Usd.TimeCode(1.5))

        coords_rel = dataset.GetPrim().CreateRelationship("coordinates")
        coords_rel.AddTarget(coords.GetPrim().GetPath())

        await omni.usd.get_context().attach_stage_async(stage)

        earliest_time = Usd.TimeCode.EarliestTime().GetValue()

        # Test 1: With traverse_field_relationships=True (should include field relationships)
        # Should find all time samples: 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0
        lower, upper, has_time_samples = usd_utils.get_bracketing_time_samples_for_data_set_prim(
            dataset.GetPrim(), 1.0, traverse_field_relationships=True
        )
        self.assertEqual(lower, 1.0)
        self.assertEqual(upper, 1.0)
        self.assertTrue(has_time_samples)

        # Test 2: With traverse_field_relationships=False (should exclude field relationships)
        # Should only find time samples from coordinates: 0.5, 1.5
        lower, upper, has_time_samples = usd_utils.get_bracketing_time_samples_for_data_set_prim(
            dataset.GetPrim(), 1.0, traverse_field_relationships=False
        )
        self.assertEqual(lower, 0.5)
        self.assertEqual(upper, 1.5)
        self.assertTrue(has_time_samples)

        # Test 3: With traverse_field_relationships=False, query time between coordinates samples
        lower, upper, has_time_samples = usd_utils.get_bracketing_time_samples_for_data_set_prim(
            dataset.GetPrim(), 1.0, traverse_field_relationships=False
        )
        self.assertEqual(lower, 0.5)
        self.assertEqual(upper, 1.5)
        self.assertTrue(has_time_samples)

        # Test 4: With traverse_field_relationships=False, query time before coordinates samples
        lower, upper, has_time_samples = usd_utils.get_bracketing_time_samples_for_data_set_prim(
            dataset.GetPrim(), 0.3, traverse_field_relationships=False
        )
        self.assertEqual(lower, 0.5)
        self.assertEqual(upper, 0.5)
        self.assertTrue(has_time_samples)

        # Test 5: With traverse_field_relationships=False, query time after coordinates samples
        lower, upper, has_time_samples = usd_utils.get_bracketing_time_samples_for_data_set_prim(
            dataset.GetPrim(), 2.0, traverse_field_relationships=False
        )
        self.assertEqual(lower, 1.5)
        self.assertEqual(upper, 1.5)
        self.assertTrue(has_time_samples)

        # Test 6: Dataset with only field relationships, traverse_field_relationships=False
        # Should return no time samples (EarliestTime)
        dataset_only_fields = cae.DataSet.Define(stage, "/Root/DataSetOnlyFields")
        field_only = cae.FieldArray.Define(stage, "/Root/DataSetOnlyFields/Field")
        field_attr = field_only.GetPrim().CreateAttribute("testAttr", Sdf.ValueTypeNames.Float)
        field_attr.Set(10.0, Usd.TimeCode(1.0))

        field_rel = dataset_only_fields.GetPrim().CreateRelationship("field:Field")
        field_rel.AddTarget(field_only.GetPrim().GetPath())

        lower, upper, has_time_samples = usd_utils.get_bracketing_time_samples_for_data_set_prim(
            dataset_only_fields.GetPrim(), 1.0, traverse_field_relationships=False
        )
        self.assertEqual(lower, earliest_time)
        self.assertEqual(upper, earliest_time)
        self.assertFalse(has_time_samples)

        # Test 7: Dataset with only field relationships, traverse_field_relationships=True
        # Should find time samples
        lower, upper, has_time_samples = usd_utils.get_bracketing_time_samples_for_data_set_prim(
            dataset_only_fields.GetPrim(), 1.0, traverse_field_relationships=True
        )
        self.assertEqual(lower, 1.0)
        self.assertEqual(upper, 1.0)
        self.assertTrue(has_time_samples)

        # Test 8: Invalid prim
        invalid_prim = stage.GetPrimAtPath("/Root/NonExistent")
        lower, upper, has_time_samples = usd_utils.get_bracketing_time_samples_for_data_set_prim(
            invalid_prim, 1.0, traverse_field_relationships=True
        )
        self.assertEqual(lower, earliest_time)
        self.assertEqual(upper, earliest_time)
        self.assertFalse(has_time_samples)

    async def test_snap_time_code_to_prim(self):
        """Test snap_time_code_to_prim function"""
        # Create a stage with time samples
        stage = Usd.Stage.CreateInMemory()

        # Create a DataSet prim
        dataset = cae.DataSet.Define(stage, "/Root/DataSet")

        # Create FieldArray prims with time samples
        field1 = cae.FieldArray.Define(stage, "/Root/DataSet/Field1")
        field2 = cae.FieldArray.Define(stage, "/Root/DataSet/Field2")

        # Create attributes with time samples
        attr1 = field1.GetPrim().CreateAttribute("testAttr1", Sdf.ValueTypeNames.Float)
        attr2 = field2.GetPrim().CreateAttribute("testAttr2", Sdf.ValueTypeNames.Float)

        # Set time samples: 0.0, 1.0, 2.0, 3.0
        attr1.Set(10.0, Usd.TimeCode(0.0))
        attr1.Set(20.0, Usd.TimeCode(1.0))
        attr1.Set(30.0, Usd.TimeCode(2.0))
        attr1.Set(40.0, Usd.TimeCode(3.0))

        attr2.Set(100.0, Usd.TimeCode(1.5))
        attr2.Set(200.0, Usd.TimeCode(2.5))

        # Create relationship from dataset to fields
        rel = dataset.GetPrim().CreateRelationship("field:Field1")
        rel.AddTarget(field1.GetPrim().GetPath())

        rel2 = dataset.GetPrim().CreateRelationship("field:Field2")
        rel2.AddTarget(field2.GetPrim().GetPath())

        await omni.usd.get_context().attach_stage_async(stage)

        earliest_time = Usd.TimeCode.EarliestTime()

        # Test 1: Exact match - should return that time
        snapped = usd_utils.snap_time_code_to_prim(dataset.GetPrim(), Usd.TimeCode(1.0))
        self.assertEqual(snapped, Usd.TimeCode(1.0))

        # Test 2: Between samples - should snap down to lower
        snapped = usd_utils.snap_time_code_to_prim(dataset.GetPrim(), Usd.TimeCode(2.2))
        self.assertEqual(snapped, Usd.TimeCode(2.0))

        # Test 3: Before first sample - should return first sample
        snapped = usd_utils.snap_time_code_to_prim(dataset.GetPrim(), Usd.TimeCode(-1.0))
        self.assertEqual(snapped, Usd.TimeCode(0.0))

        # Test 4: After last sample - should return last sample
        snapped = usd_utils.snap_time_code_to_prim(dataset.GetPrim(), Usd.TimeCode(5.0))
        self.assertEqual(snapped, Usd.TimeCode(3.0))

        # Test 5: Prim with no time samples - should return EarliestTime
        empty_prim = cae.DataSet.Define(stage, "/Root/EmptyDataSet")
        snapped = usd_utils.snap_time_code_to_prim(empty_prim.GetPrim(), Usd.TimeCode(1.0))
        self.assertEqual(snapped, earliest_time)

        # Test 6: Invalid prim - should return EarliestTime
        invalid_prim = stage.GetPrimAtPath("/Root/NonExistent")
        snapped = usd_utils.snap_time_code_to_prim(invalid_prim, Usd.TimeCode(1.0))
        self.assertEqual(snapped, earliest_time)

    async def test_snap_time_code_to_prims(self):
        """Test snap_time_code_to_prims function"""
        # Create a stage with multiple DataSet prims
        stage = Usd.Stage.CreateInMemory()

        # Create first DataSet with time samples: 0.0, 1.0, 2.0, 3.0
        dataset1 = cae.DataSet.Define(stage, "/Root/DataSet1")
        field1 = cae.FieldArray.Define(stage, "/Root/DataSet1/Field1")
        attr1 = field1.GetPrim().CreateAttribute("testAttr1", Sdf.ValueTypeNames.Float)
        attr1.Set(10.0, Usd.TimeCode(0.0))
        attr1.Set(20.0, Usd.TimeCode(1.0))
        attr1.Set(30.0, Usd.TimeCode(2.0))
        attr1.Set(40.0, Usd.TimeCode(3.0))
        rel1 = dataset1.GetPrim().CreateRelationship("field:Field1")
        rel1.AddTarget(field1.GetPrim().GetPath())

        # Create second DataSet with time samples: 0.5, 1.5, 2.5
        dataset2 = cae.DataSet.Define(stage, "/Root/DataSet2")
        field2 = cae.FieldArray.Define(stage, "/Root/DataSet2/Field2")
        attr2 = field2.GetPrim().CreateAttribute("testAttr2", Sdf.ValueTypeNames.Float)
        attr2.Set(100.0, Usd.TimeCode(0.5))
        attr2.Set(200.0, Usd.TimeCode(1.5))
        attr2.Set(300.0, Usd.TimeCode(2.5))
        rel2 = dataset2.GetPrim().CreateRelationship("field:Field2")
        rel2.AddTarget(field2.GetPrim().GetPath())

        # Create third DataSet with time samples: 1.2, 2.2
        dataset3 = cae.DataSet.Define(stage, "/Root/DataSet3")
        field3 = cae.FieldArray.Define(stage, "/Root/DataSet3/Field3")
        attr3 = field3.GetPrim().CreateAttribute("testAttr3", Sdf.ValueTypeNames.Float)
        attr3.Set(1000.0, Usd.TimeCode(1.2))
        attr3.Set(2000.0, Usd.TimeCode(2.2))
        rel3 = dataset3.GetPrim().CreateRelationship("field:Field3")
        rel3.AddTarget(field3.GetPrim().GetPath())

        await omni.usd.get_context().attach_stage_async(stage)

        earliest_time = Usd.TimeCode.EarliestTime()

        # Test 1: Query time 1.0 - dataset1 snaps to 1.0, dataset2 snaps to 0.5, dataset3 snaps to EarliestTime (no sample <= 1.0)
        # From snapped times [0.5, 1.0], closest lower to 1.0 is 1.0
        snapped = usd_utils.snap_time_code_to_prims(
            [dataset1.GetPrim(), dataset2.GetPrim(), dataset3.GetPrim()], Usd.TimeCode(1.0)
        )
        self.assertEqual(snapped, Usd.TimeCode(1.0))

        # Test 2: Query time 1.1 - dataset1 snaps to 1.0, dataset2 snaps to 0.5, dataset3 snaps to EarliestTime
        # From snapped times [0.5, 1.0], closest lower to 1.1 is 1.0
        snapped = usd_utils.snap_time_code_to_prims(
            [dataset1.GetPrim(), dataset2.GetPrim(), dataset3.GetPrim()], Usd.TimeCode(1.1)
        )
        self.assertEqual(snapped, Usd.TimeCode(1.0))

        # Test 3: Query time 1.3 - dataset1 snaps to 1.0, dataset2 snaps to 0.5, dataset3 snaps to 1.2
        # From snapped times [0.5, 1.0, 1.2], closest lower to 1.3 is 1.2
        snapped = usd_utils.snap_time_code_to_prims(
            [dataset1.GetPrim(), dataset2.GetPrim(), dataset3.GetPrim()], Usd.TimeCode(1.3)
        )
        self.assertEqual(snapped, Usd.TimeCode(1.2))

        # Test 4: Query time 0.3 - dataset1 snaps to 0.0, dataset2 snaps to EarliestTime (no sample <= 0.3), dataset3 snaps to EarliestTime
        # From snapped times [0.0], closest lower to 0.3 is 0.0
        snapped = usd_utils.snap_time_code_to_prims(
            [dataset1.GetPrim(), dataset2.GetPrim(), dataset3.GetPrim()], Usd.TimeCode(0.3)
        )
        self.assertEqual(snapped, Usd.TimeCode(0.0))

        # Test 5: Query time -1.0 (before all samples) - all snap to their first samples
        # dataset1 snaps to 0.0, dataset2 snaps to 0.5, dataset3 snaps to EarliestTime
        # From snapped times [0.0, 0.5], all are > -1.0, so return first (lowest): 0.0
        snapped = usd_utils.snap_time_code_to_prims(
            [dataset1.GetPrim(), dataset2.GetPrim(), dataset3.GetPrim()], Usd.TimeCode(-1.0)
        )
        self.assertEqual(snapped, Usd.TimeCode(0.0))

        # Test 6: Query time 5.0 (after all samples) - all snap to their last samples
        # dataset1 snaps to 3.0, dataset2 snaps to 2.5, dataset3 snaps to 2.2
        # From snapped times [2.2, 2.5, 3.0], closest lower to 5.0 is 3.0
        snapped = usd_utils.snap_time_code_to_prims(
            [dataset1.GetPrim(), dataset2.GetPrim(), dataset3.GetPrim()], Usd.TimeCode(5.0)
        )
        self.assertEqual(snapped, Usd.TimeCode(3.0))

        # Test 7: Empty list - should return EarliestTime
        snapped = usd_utils.snap_time_code_to_prims([], Usd.TimeCode(1.0))
        self.assertEqual(snapped, earliest_time)

        # Test 8: List with invalid prim - should skip invalid and use valid ones
        invalid_prim = stage.GetPrimAtPath("/Root/NonExistent")
        snapped = usd_utils.snap_time_code_to_prims([dataset1.GetPrim(), invalid_prim], Usd.TimeCode(1.0))
        self.assertEqual(snapped, Usd.TimeCode(1.0))

        # Test 9: List with prims that have no time samples - should return EarliestTime
        empty_prim = cae.DataSet.Define(stage, "/Root/EmptyDataSet")
        snapped = usd_utils.snap_time_code_to_prims([empty_prim.GetPrim()], Usd.TimeCode(1.0))
        self.assertEqual(snapped, earliest_time)

        # Test 10: Mixed valid and empty prims - should use valid ones
        snapped = usd_utils.snap_time_code_to_prims([dataset1.GetPrim(), empty_prim.GetPrim()], Usd.TimeCode(1.0))
        self.assertEqual(snapped, Usd.TimeCode(1.0))

        # Test 11: With traverse_field_relationships=False
        # Create a dataset with field and non-field relationships
        dataset_mixed = cae.DataSet.Define(stage, "/Root/DataSetMixed")
        field_field = cae.FieldArray.Define(stage, "/Root/DataSetMixed/FieldField")
        field_field_attr = field_field.GetPrim().CreateAttribute("testAttr", Sdf.ValueTypeNames.Float)
        field_field_attr.Set(10.0, Usd.TimeCode(1.0))

        coords = cae.FieldArray.Define(stage, "/Root/DataSetMixed/Coords")
        coords_attr = coords.GetPrim().CreateAttribute("testAttr", Sdf.ValueTypeNames.Float)
        coords_attr.Set(20.0, Usd.TimeCode(0.5))

        field_rel = dataset_mixed.GetPrim().CreateRelationship("field:FieldField")
        field_rel.AddTarget(field_field.GetPrim().GetPath())

        coords_rel = dataset_mixed.GetPrim().CreateRelationship("coordinates")
        coords_rel.AddTarget(coords.GetPrim().GetPath())

        # With traverse_field_relationships=False, should only use coordinates (0.5)
        snapped = usd_utils.snap_time_code_to_prims(
            [dataset_mixed.GetPrim()], Usd.TimeCode(1.0), traverse_field_relationships=False
        )
        self.assertEqual(snapped, Usd.TimeCode(0.5))

        # With traverse_field_relationships=True (default), should use both (0.5, 1.0)
        snapped = usd_utils.snap_time_code_to_prims(
            [dataset_mixed.GetPrim()], Usd.TimeCode(1.0), traverse_field_relationships=True
        )
        self.assertEqual(snapped, Usd.TimeCode(1.0))

    async def test_get_related_data_prims(self):
        """Test getRelatedDataPrims API"""
        # Create a stage with nested relationships
        stage = Usd.Stage.CreateInMemory()

        # Create a hierarchy: DataSet -> Field1 -> Field2
        dataset = cae.DataSet.Define(stage, "/Root/DataSet")
        field1 = cae.FieldArray.Define(stage, "/Root/DataSet/Field1")
        field2 = cae.FieldArray.Define(stage, "/Root/DataSet/Field2")
        field3 = cae.FieldArray.Define(stage, "/Root/DataSet/Field3")

        # Create relationships: DataSet -> Field1 -> Field2
        dataset.GetPrim().CreateRelationship("field:Field1").AddTarget(field1.GetPrim().GetPath())
        field1.GetPrim().CreateRelationship("coordinates").AddTarget(field2.GetPrim().GetPath())

        # Field3 is not related to anything
        dataset.GetPrim().CreateRelationship("field:Field3").AddTarget(field3.GetPrim().GetPath())

        await omni.usd.get_context().attach_stage_async(stage)

        # Test 1: transitive=True, includeSelf=True (default)
        related = usd_utils.get_related_data_prims(dataset.GetPrim())
        related_paths = {p.GetPath() for p in related}
        self.assertIn(dataset.GetPrim().GetPath(), related_paths)
        self.assertIn(field1.GetPrim().GetPath(), related_paths)
        self.assertIn(field2.GetPrim().GetPath(), related_paths)
        self.assertIn(field3.GetPrim().GetPath(), related_paths)

        # Test 2: transitive=False, includeSelf=True
        related = usd_utils.get_related_data_prims(dataset.GetPrim(), transitive=False, include_self=True)
        related_paths = {p.GetPath() for p in related}
        self.assertIn(dataset.GetPrim().GetPath(), related_paths)
        self.assertIn(field1.GetPrim().GetPath(), related_paths)
        self.assertIn(field3.GetPrim().GetPath(), related_paths)
        # Field2 should NOT be included (not transitive)
        self.assertNotIn(field2.GetPrim().GetPath(), related_paths)

        # Test 3: transitive=True, includeSelf=False
        related = usd_utils.get_related_data_prims(dataset.GetPrim(), transitive=True, include_self=False)
        related_paths = {p.GetPath() for p in related}
        self.assertNotIn(dataset.GetPrim().GetPath(), related_paths)
        self.assertIn(field1.GetPrim().GetPath(), related_paths)
        self.assertIn(field2.GetPrim().GetPath(), related_paths)
        self.assertIn(field3.GetPrim().GetPath(), related_paths)

        # Test 4: transitive=False, includeSelf=False
        related = usd_utils.get_related_data_prims(dataset.GetPrim(), transitive=False, include_self=False)
        related_paths = {p.GetPath() for p in related}
        self.assertNotIn(dataset.GetPrim().GetPath(), related_paths)
        self.assertIn(field1.GetPrim().GetPath(), related_paths)
        self.assertIn(field3.GetPrim().GetPath(), related_paths)
        self.assertNotIn(field2.GetPrim().GetPath(), related_paths)

        # Test 5: Non-DataSet/FieldArray prim with includeSelf=True
        # Should still include the prim itself even if it's not a DataSet or FieldArray
        xform = UsdGeom.Xform.Define(stage, "/Root/Xform")
        xform.GetPrim().CreateRelationship("targetDataSet").AddTarget(dataset.GetPrim().GetPath())

        related = usd_utils.get_related_data_prims(xform.GetPrim(), transitive=True, include_self=True)
        related_paths = {p.GetPath() for p in related}
        # The Xform itself should be included because includeSelf=True
        self.assertIn(xform.GetPrim().GetPath(), related_paths)
        # The DataSet and all its related prims should also be included
        self.assertIn(dataset.GetPrim().GetPath(), related_paths)
        self.assertIn(field1.GetPrim().GetPath(), related_paths)
        self.assertIn(field2.GetPrim().GetPath(), related_paths)
        self.assertIn(field3.GetPrim().GetPath(), related_paths)

        # Test 6: Non-DataSet/FieldArray prim with includeSelf=False
        related = usd_utils.get_related_data_prims(xform.GetPrim(), transitive=True, include_self=False)
        related_paths = {p.GetPath() for p in related}
        # The Xform itself should NOT be included because includeSelf=False
        self.assertNotIn(xform.GetPrim().GetPath(), related_paths)
        # But the related DataSet and FieldArray prims should be included
        self.assertIn(dataset.GetPrim().GetPath(), related_paths)
        self.assertIn(field1.GetPrim().GetPath(), related_paths)
        self.assertIn(field2.GetPrim().GetPath(), related_paths)
        self.assertIn(field3.GetPrim().GetPath(), related_paths)

        # Test 7: Starting from Field1 (transitive should find Field2)
        related = usd_utils.get_related_data_prims(field1.GetPrim(), transitive=True, include_self=False)
        related_paths = {p.GetPath() for p in related}
        self.assertIn(field2.GetPrim().GetPath(), related_paths)
        self.assertNotIn(field1.GetPrim().GetPath(), related_paths)

        # Test 8: Empty prim should return empty list
        related = usd_utils.get_related_data_prims(Usd.Prim())
        self.assertEqual(len(related), 0)

        # Test 9: Prim with no relationships
        standalone = cae.FieldArray.Define(stage, "/Root/Standalone")
        related = usd_utils.get_related_data_prims(standalone.GetPrim(), transitive=True, include_self=True)
        self.assertEqual(len(related), 1)
        self.assertEqual(related[0].GetPath(), standalone.GetPrim().GetPath())

        related = usd_utils.get_related_data_prims(standalone.GetPrim(), transitive=True, include_self=False)
        self.assertEqual(len(related), 0)

        omni.usd.get_context().close_stage()
        del stage
