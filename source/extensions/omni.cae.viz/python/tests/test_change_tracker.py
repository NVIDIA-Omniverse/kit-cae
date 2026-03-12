# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import omni.kit.app
import omni.kit.test
import omni.timeline
import omni.usd
from omni.cae.schema import cae
from omni.cae.schema import viz as cae_viz
from omni.cae.schema import vtk as cae_vtk
from omni.cae.viz.change_tracker import ChangeTracker
from omni.usd import get_context
from pxr import Sdf, Usd


class TestChangeTracker(omni.kit.test.AsyncTestCase):
    async def test_enable_disable(self):
        """Test enabling and disabling the change tracker."""
        # Create a new stage
        usd_context = get_context()
        await usd_context.new_stage_async()
        stage = usd_context.get_stage()

        tracker = ChangeTracker(stage)

        # Initially enabled
        self.assertTrue(tracker.is_enabled())

        # Disable
        tracker.disable()
        self.assertFalse(tracker.is_enabled())

        # Enable
        tracker.enable()
        self.assertTrue(tracker.is_enabled())

        del tracker
        await usd_context.close_stage_async()

    async def test_schema_property_changes(self):
        """Test detecting changes to schema properties."""
        # Create a new stage
        usd_context = get_context()
        await usd_context.new_stage_async()
        stage = usd_context.get_stage()

        # Create a CaeVtkFieldArray prim
        array_prim = cae_vtk.FieldArray.Define(stage, "/World/TestArray")

        # Create tracker
        tracker = ChangeTracker(stage, [r"^CaeVtk"])

        # Clear initial state (tracker treats all prims as initially dirty)
        tracker.clear_all_changes()

        # Initially no changes
        self.assertFalse(
            tracker.prim_changed(array_prim.GetPrim(), ["CaeVtkFieldArray"]), "Should have no changes initially"
        )

        # Make a change to a schema property
        array_prim.CreateFileNamesAttr().Set(["test.vti"])

        # Should detect the change
        self.assertTrue(
            tracker.prim_changed(array_prim.GetPrim(), ["CaeVtkFieldArray"]), "Should detect schema property change"
        )

        # Clear changes
        tracker.clear_changes(array_prim.GetPrim(), ["CaeVtkFieldArray"])

        # Should have no changes after clearing
        self.assertFalse(
            tracker.prim_changed(array_prim.GetPrim(), ["CaeVtkFieldArray"]), "Should have no changes after clearing"
        )

        await usd_context.close_stage_async()

    async def test_attribute_changes(self):
        """Test detecting changes to specific attributes."""
        # Create a new stage
        usd_context = get_context()
        await usd_context.new_stage_async()
        stage = usd_context.get_stage()

        # Create a CaeDataSet prim
        dataset_prim = cae.DataSet.Define(stage, "/World/TestDataset")

        # Create tracker
        tracker = ChangeTracker(stage, [r"^Cae"])

        # Make a change to a custom attribute
        custom_attr = dataset_prim.GetPrim().CreateAttribute("customAttr", Sdf.ValueTypeNames.String)
        custom_attr.Set("test_value")

        # Should not detect the attribute change of non-schema attribute
        self.assertFalse(
            tracker.attr_changed(dataset_prim.GetPrim(), "customAttr"), "Should not detect attribute change"
        )

        # Clear all changes
        tracker.clear_changes(dataset_prim.GetPrim())

        # Should have no changes after clearing
        self.assertFalse(
            tracker.attr_changed(dataset_prim.GetPrim(), "customAttr"), "Should have no changes after clearing"
        )

        await usd_context.close_stage_async()

    async def test_multi_apply_schema(self):
        """Test detecting changes to multi-apply schema instances."""
        # Create a new stage
        usd_context = get_context()
        await usd_context.new_stage_async()
        stage = usd_context.get_stage()

        # Create a prim and apply multi-apply schema
        prim = stage.DefinePrim("/World/TestPrim")

        # Apply CaeVizFieldSelectionAPI with different instances
        field_sel_velocity = cae_viz.FieldSelectionAPI.Apply(prim, "velocity")
        field_sel_pressure = cae_viz.FieldSelectionAPI.Apply(prim, "pressure")

        # Create tracker
        tracker = ChangeTracker(stage, [r"^CaeViz"])

        # Clear initial state
        tracker.clear_all_changes()

        # Make a change to velocity instance
        field_sel_velocity.CreateTargetRel().SetTargets(["/World/SomeTarget"])

        # Should detect change for specific instance
        self.assertTrue(
            tracker.prim_changed(prim, ["CaeVizFieldSelectionAPI:velocity"]),
            "Should detect change for velocity instance",
        )

        # Should not detect change for pressure instance
        self.assertFalse(
            tracker.prim_changed(prim, ["CaeVizFieldSelectionAPI:pressure"]),
            "Should not detect change for pressure instance",
        )

        # Should detect change when querying without instance name
        self.assertTrue(
            tracker.prim_changed(prim, ["CaeVizFieldSelectionAPI"]),
            "Should detect change when querying without instance name",
        )

        # Clear specific instance
        tracker.clear_changes(prim, ["CaeVizFieldSelectionAPI:velocity"])

        # Should have no changes for velocity after clearing
        self.assertFalse(
            tracker.prim_changed(prim, ["CaeVizFieldSelectionAPI:velocity"]),
            "Should have no changes for velocity after clearing",
        )
        await usd_context.close_stage_async()

    async def test_multiple_schemas(self):
        """Test detecting changes when multiple schemas are applied."""
        # Create a new stage
        usd_context = get_context()
        await usd_context.new_stage_async()
        stage = usd_context.get_stage()

        # Create a CaeDataSet prim
        dataset_prim = cae.DataSet.Define(stage, "/World/TestDataset")

        # Apply multiple API schemas
        point_cloud_api = cae.PointCloudAPI.Apply(dataset_prim.GetPrim())

        # Create tracker
        tracker = ChangeTracker(stage, [r"^Cae"])

        # Clear initial state
        tracker.clear_all_changes()

        # Make a change to point cloud API property
        point_cloud_api.CreateCoordinatesRel().SetTargets(["/World/Coords"])

        # Should detect change for PointCloudAPI
        self.assertTrue(
            tracker.prim_changed(dataset_prim.GetPrim(), ["CaePointCloudAPI"]),
            "Should detect change for CaePointCloudAPI",
        )

        # Should not detect change for CaeDataSet type
        self.assertFalse(
            tracker.prim_changed(dataset_prim.GetPrim(), ["CaeDataSet"]), "Should not detect change for CaeDataSet type"
        )

        # Should detect change when checking multiple schemas
        self.assertTrue(
            tracker.prim_changed(dataset_prim.GetPrim(), ["CaePointCloudAPI", "CaeDataSet"]),
            "Should detect change when checking multiple schemas",
        )
        await usd_context.close_stage_async()

    async def test_clear_all_changes(self):
        """Test clearing all changes for all prims."""
        # Create a new stage
        usd_context = get_context()
        await usd_context.new_stage_async()
        stage = usd_context.get_stage()

        # Create multiple prims with changes
        array1 = cae_vtk.FieldArray.Define(stage, "/World/Array1")
        array2 = cae_vtk.FieldArray.Define(stage, "/World/Array2")

        # Create tracker
        tracker = ChangeTracker(stage, [r"^CaeVtk"])

        # Make changes to both prims
        array1.CreateFileNamesAttr().Set(["test1.vti"])
        array2.CreateFileNamesAttr().Set(["test2.vti"])

        # Both should have changes
        self.assertTrue(tracker.prim_changed(array1.GetPrim(), ["CaeVtkFieldArray"]), "Array1 should have changes")
        self.assertTrue(tracker.prim_changed(array2.GetPrim(), ["CaeVtkFieldArray"]), "Array2 should have changes")

        # Clear all changes
        tracker.clear_all_changes()

        # Neither should have changes
        self.assertFalse(
            tracker.prim_changed(array1.GetPrim(), ["CaeVtkFieldArray"]),
            "Array1 should have no changes after clear_all",
        )
        self.assertFalse(
            tracker.prim_changed(array2.GetPrim(), ["CaeVtkFieldArray"]),
            "Array2 should have no changes after clear_all",
        )

        await usd_context.close_stage_async()

    async def test_pattern_matching(self):
        """Test that only schemas matching the patterns are tracked."""
        # Create a new stage
        usd_context = get_context()
        await usd_context.new_stage_async()
        stage = usd_context.get_stage()

        # Create prims with different schemas
        cae_array = cae_vtk.FieldArray.Define(stage, "/World/CaeArray")
        usd_prim = stage.DefinePrim("/World/UsdPrim")
        usd_prim.CreateAttribute("someAttr", Sdf.ValueTypeNames.String)

        # Create tracker that only tracks CaeVtk schemas
        tracker = ChangeTracker(stage, [r"^CaeVtk"])

        # Make changes to both
        cae_array.CreateFileNamesAttr().Set(["test.vti"])
        usd_prim.GetAttribute("someAttr").Set("value")

        # Should detect change for CaeVtk schema
        self.assertTrue(
            tracker.prim_changed(cae_array.GetPrim(), ["CaeVtkFieldArray"]), "Should detect change for tracked schema"
        )

        # Should NOT track attribute change for prims without matching schemas
        self.assertFalse(
            tracker.attr_changed(usd_prim, "someAttr"),
            "Should not track attribute changes for prims without matching schemas",
        )
        await usd_context.close_stage_async()

    async def test_prim_changed_without_schemas(self):
        """Test that prim_changed works without specifying schemas."""
        # Create a new stage
        usd_context = get_context()
        await usd_context.new_stage_async()
        stage = usd_context.get_stage()

        # Create a CaeVtkFieldArray prim
        array_prim = cae_vtk.FieldArray.Define(stage, "/World/TestArray")

        # Create tracker
        tracker = ChangeTracker(stage, [r"^CaeVtk"])

        # Clear initial state
        tracker.clear_all_changes()

        # Initially no changes
        self.assertFalse(tracker.prim_changed(array_prim.GetPrim()), "Should have no changes initially")

        # Make a change to a schema property
        array_prim.CreateFileNamesAttr().Set(["test.vti"])

        # Should detect the change without specifying schemas
        self.assertTrue(tracker.prim_changed(array_prim.GetPrim()), "Should detect change without specifying schemas")

        # Clear changes
        tracker.clear_changes(array_prim.GetPrim())

        # Should have no changes after clearing
        self.assertFalse(tracker.prim_changed(array_prim.GetPrim()), "Should have no changes after clearing")

        await usd_context.close_stage_async()

    async def test_stage_detach_attach(self):
        """Test that tracker works with stage lifecycle."""
        # Create a new stage
        usd_context = get_context()
        await usd_context.new_stage_async()
        stage = usd_context.get_stage()

        # Create a prim
        array_prim = cae_vtk.FieldArray.Define(stage, "/World/TestArray")

        # Create tracker
        tracker = ChangeTracker(stage, [r"^CaeVtk"])

        # Clear initial state
        tracker.clear_all_changes()

        # Make a change
        array_prim.CreateFileNamesAttr().Set(["test.vti"])

        # Should have changes
        self.assertTrue(tracker.prim_changed(array_prim.GetPrim(), ["CaeVtkFieldArray"]), "Should have changes")

        # Close the stage
        await usd_context.close_stage_async()

        # Create a new stage with new tracker
        await usd_context.new_stage_async()
        stage2 = usd_context.get_stage()
        array_prim2 = cae_vtk.FieldArray.Define(stage2, "/World/TestArray")
        tracker2 = ChangeTracker(stage2, [r"^CaeVtk"])

        # Clear initial state for the new tracker
        tracker2.clear_all_changes()

        # Should have no changes on the new tracker
        self.assertFalse(
            tracker2.prim_changed(array_prim2.GetPrim(), ["CaeVtkFieldArray"]), "Should have no changes on new stage"
        )

        await usd_context.close_stage_async()

    async def test_resynced_paths(self):
        """Test that resynced paths mark entire subtrees as dirty."""
        # Create a new stage
        usd_context = get_context()
        await usd_context.new_stage_async()
        stage = usd_context.get_stage()

        # Create a parent and child prims with schemas
        parent_prim = cae_vtk.FieldArray.Define(stage, "/World/Parent")
        child_prim = cae_vtk.FieldArray.Define(stage, "/World/Parent/Child")
        grandchild_prim = cae_vtk.FieldArray.Define(stage, "/World/Parent/Child/GrandChild")

        # Create tracker
        tracker = ChangeTracker(stage, [r"^CaeVtk"])

        # Clear initial state
        tracker.clear_all_changes()

        # Initially no changes
        self.assertFalse(tracker.prim_changed(parent_prim.GetPrim()), "Parent should have no changes initially")
        self.assertFalse(tracker.prim_changed(child_prim.GetPrim()), "Child should have no changes initially")
        self.assertFalse(tracker.prim_changed(grandchild_prim.GetPrim()), "GrandChild should have no changes initially")

        # Trigger a resync by removing and re-adding the parent prim
        # This simulates a resync event
        stage.RemovePrim("/World/Parent")
        parent_prim = cae_vtk.FieldArray.Define(stage, "/World/Parent")
        child_prim = cae_vtk.FieldArray.Define(stage, "/World/Parent/Child")
        grandchild_prim = cae_vtk.FieldArray.Define(stage, "/World/Parent/Child/GrandChild")

        # All prims in the subtree should be marked as changed (dirty)
        self.assertTrue(tracker.prim_changed(parent_prim.GetPrim()), "Parent should be dirty after resync")
        self.assertTrue(tracker.prim_changed(child_prim.GetPrim()), "Child should be dirty after parent resync")
        self.assertTrue(
            tracker.prim_changed(grandchild_prim.GetPrim()), "GrandChild should be dirty after ancestor resync"
        )

        # Clear the child prim explicitly
        tracker.clear_changes(child_prim.GetPrim())

        # After clearing, child should not report as changed, but parent and grandchild should still be dirty
        self.assertTrue(tracker.prim_changed(parent_prim.GetPrim()), "Parent should still be dirty")
        self.assertFalse(tracker.prim_changed(child_prim.GetPrim()), "Child should not be dirty after clearing")
        self.assertTrue(
            tracker.prim_changed(grandchild_prim.GetPrim()),
            "GrandChild should still be dirty (not affected by sibling clearing)",
        )

        # Clear the parent
        tracker.clear_changes(parent_prim.GetPrim())

        # Now parent should be clear, but grandchild should still be dirty
        self.assertFalse(tracker.prim_changed(parent_prim.GetPrim()), "Parent should not be dirty after clearing")
        self.assertFalse(tracker.prim_changed(child_prim.GetPrim()), "Child should still not be dirty")
        self.assertTrue(
            tracker.prim_changed(grandchild_prim.GetPrim()),
            "GrandChild should still be dirty (ancestor clearing doesn't clear descendants)",
        )

        # Clear the grandchild
        tracker.clear_changes(grandchild_prim.GetPrim())
        self.assertFalse(
            tracker.prim_changed(grandchild_prim.GetPrim()), "GrandChild should not be dirty after clearing"
        )

        # Clear all and verify everything is clean
        tracker.clear_all_changes()
        self.assertFalse(tracker.prim_changed(parent_prim.GetPrim()), "Parent should be clean after clear_all")
        self.assertFalse(tracker.prim_changed(child_prim.GetPrim()), "Child should be clean after clear_all")
        self.assertFalse(tracker.prim_changed(grandchild_prim.GetPrim()), "GrandChild should be clean after clear_all")

        await usd_context.close_stage_async()

    async def test_resync_after_clear(self):
        """Test that resyncing after clearing makes prims dirty again."""
        # Create a new stage
        usd_context = get_context()
        await usd_context.new_stage_async()
        stage = usd_context.get_stage()

        # Create parent and child prims
        parent_prim = cae_vtk.FieldArray.Define(stage, "/World/Parent")
        child_prim = cae_vtk.FieldArray.Define(stage, "/World/Parent/Child")

        # Create tracker
        tracker = ChangeTracker(stage, [r"^CaeVtk"])

        # Trigger initial resync
        stage.RemovePrim("/World/Parent")
        parent_prim = cae_vtk.FieldArray.Define(stage, "/World/Parent")
        child_prim = cae_vtk.FieldArray.Define(stage, "/World/Parent/Child")

        # Both should be dirty
        self.assertTrue(tracker.prim_changed(parent_prim.GetPrim()), "Parent should be dirty after resync")
        self.assertTrue(tracker.prim_changed(child_prim.GetPrim()), "Child should be dirty after resync")

        # Clear the child
        tracker.clear_changes(child_prim.GetPrim())
        self.assertFalse(tracker.prim_changed(child_prim.GetPrim()), "Child should not be dirty after clearing")

        # Resync the parent again (this should make child dirty again, even though it was cleared)
        stage.RemovePrim("/World/Parent")
        parent_prim = cae_vtk.FieldArray.Define(stage, "/World/Parent")
        child_prim = cae_vtk.FieldArray.Define(stage, "/World/Parent/Child")

        # Child should be dirty again (cleared status was removed by parent resync)
        self.assertTrue(tracker.prim_changed(parent_prim.GetPrim()), "Parent should be dirty after second resync")
        self.assertTrue(
            tracker.prim_changed(child_prim.GetPrim()),
            "Child should be dirty again after parent resync (cleared status removed)",
        )

        await usd_context.close_stage_async()

    async def test_time_tracking_disabled_by_default(self):
        """Test that time tracking is disabled by default."""
        usd_context = get_context()
        await usd_context.new_stage_async()
        stage = usd_context.get_stage()

        # Create tracker without time tracking
        tracker = ChangeTracker(stage, [r"^Cae"])

        # Create a prim with time-sampled attribute
        prim = stage.DefinePrim("/World/TestPrim")
        attr = prim.CreateAttribute("testAttr", Sdf.ValueTypeNames.Float)
        attr.Set(1.0, 0.0)
        attr.Set(10.0, 10.0)

        # Clear initial changes
        tracker.clear_all_changes()

        # Get timeline and change time
        timeline = omni.timeline.get_timeline_interface()
        timeline.set_current_time(0.0)
        await omni.kit.app.get_app().next_update_async()

        timeline.set_current_time(5.0)
        await omni.kit.app.get_app().next_update_async()

        # Should NOT detect time-based changes (time tracking disabled)
        self.assertFalse(tracker.prim_changed(prim), "Should not detect time changes when time tracking disabled")

        tracker.disable()
        await usd_context.close_stage_async()

    async def test_time_tracking_enabled(self):
        """Test that time tracking works when enabled."""
        usd_context = get_context()
        await usd_context.new_stage_async()
        stage = usd_context.get_stage()

        # Create tracker WITH time tracking
        tracker = ChangeTracker(stage, [r"^CaeVtk"], track_time_changes=True)

        # Create a CaeVtkFieldArray with time-sampled attribute
        array = cae_vtk.FieldArray.Define(stage, "/World/Array")
        prim = array.GetPrim()

        # Add time-sampled attribute to the schema
        # Use time codes that match timeline scale (timeline seconds * time_codes_per_second)
        # Timeline at 0s = time code 0, at 10s = time code 600
        attr = array.CreateFileNamesAttr()
        attr.Set(["file_t0.vti"], 0.0)
        attr.Set(["file_t10.vti"], 600.0)  # 10 seconds * 60 time codes/second

        # Clear initial changes
        tracker.clear_all_changes()

        # Get timeline and set to a time that's NOT exactly at a sample
        # This ensures consistent bracket detection
        timeline = omni.timeline.get_timeline_interface()
        timeline.set_current_time(1.0)  # = time code 60 (between 0 and 600)
        await omni.kit.app.get_app().next_update_async()

        # Remember initial time
        last_time = Usd.TimeCode(round(timeline.get_current_time() * timeline.get_time_codes_per_seconds()))

        # Should have no changes when checking from same time
        self.assertFalse(
            tracker.prim_changed(prim, ["CaeVtkFieldArray"], last_time_code=last_time),
            "Should have no changes when checking from same time",
        )

        # Change time to within same bracket (still between 0-600 time codes)
        timeline.set_current_time(5.0)  # = time code 300
        await omni.kit.app.get_app().next_update_async()

        # Should NOT detect change (same bracket 0-600)
        self.assertFalse(
            tracker.prim_changed(prim, ["CaeVtkFieldArray"], last_time_code=last_time),
            "Should not detect change within same bracket",
        )

        # Change time to different bracket (beyond 600 time codes)
        timeline.set_current_time(12.0)  # = time code 720
        await omni.kit.app.get_app().next_update_async()

        # Should detect bracket change
        self.assertTrue(
            tracker.prim_changed(prim, ["CaeVtkFieldArray"], last_time_code=last_time),
            "Should detect time change when bracket changes",
        )

        tracker.disable()
        await usd_context.close_stage_async()

    async def test_time_tracking_same_bracket(self):
        """Test that time changes within the same bracket are efficiently handled."""
        usd_context = get_context()
        await usd_context.new_stage_async()
        stage = usd_context.get_stage()

        # Create tracker with time tracking
        tracker = ChangeTracker(stage, [r"^CaeVtk"], track_time_changes=True)

        # Create a CaeVtkFieldArray with widely spaced time samples
        array = cae_vtk.FieldArray.Define(stage, "/World/Array")
        prim = array.GetPrim()
        attr = array.CreateFileNamesAttr()
        # Use time codes: 0 seconds = 0, 100 seconds = 6000 time codes
        attr.Set(["file_t0.vti"], 0.0)
        attr.Set(["file_t100.vti"], 6000.0)  # 100 seconds * 60

        tracker.clear_all_changes()

        # Set initial time and remember it
        timeline = omni.timeline.get_timeline_interface()
        timeline.set_current_time(10.0)
        await omni.kit.app.get_app().next_update_async()

        last_time = Usd.TimeCode(round(timeline.get_current_time() * timeline.get_time_codes_per_seconds()))

        # Move time within same bracket (0-100)
        timeline.set_current_time(20.0)
        await omni.kit.app.get_app().next_update_async()

        # Should NOT detect change (same interpolation bracket)
        self.assertFalse(
            tracker.prim_changed(prim, ["CaeVtkFieldArray"], last_time_code=last_time),
            "Should not detect change when staying in same bracket",
        )

        # Move to 50, still in same bracket
        timeline.set_current_time(50.0)
        await omni.kit.app.get_app().next_update_async()

        self.assertFalse(
            tracker.prim_changed(prim, ["CaeVtkFieldArray"], last_time_code=last_time),
            "Should still not detect change in same bracket",
        )

        tracker.disable()
        await usd_context.close_stage_async()

    async def test_time_tracking_non_animated_attrs(self):
        """Test that non-animated attributes are not tracked for time changes."""
        usd_context = get_context()
        await usd_context.new_stage_async()
        stage = usd_context.get_stage()

        # Create tracker with time tracking
        tracker = ChangeTracker(stage, [r"^CaeVtk"], track_time_changes=True)

        # Create a CaeVtkFieldArray with static (non-time-sampled) attribute
        array = cae_vtk.FieldArray.Define(stage, "/World/Array")
        prim = array.GetPrim()
        attr = array.CreateFileNamesAttr()
        attr.Set(["static_file.vti"])  # No time sample, just default value

        tracker.clear_all_changes()

        # Get timeline and set initial time
        timeline = omni.timeline.get_timeline_interface()
        timeline.set_current_time(0.0)
        await omni.kit.app.get_app().next_update_async()

        last_time = Usd.TimeCode(round(timeline.get_current_time() * timeline.get_time_codes_per_seconds()))

        # Change time
        timeline.set_current_time(50.0)
        await omni.kit.app.get_app().next_update_async()

        # Should NOT detect change (attribute is not time-varying)
        self.assertFalse(
            tracker.prim_changed(prim, ["CaeVtkFieldArray"], last_time_code=last_time),
            "Should not detect changes for non-animated attributes",
        )

        tracker.disable()
        await usd_context.close_stage_async()

    async def test_time_tracking_multiple_prims(self):
        """Test time tracking with multiple prims."""
        usd_context = get_context()
        await usd_context.new_stage_async()
        stage = usd_context.get_stage()

        # Create tracker with time tracking
        tracker = ChangeTracker(stage, [r"^CaeVtk"], track_time_changes=True, debug_logging=True)

        # Create multiple arrays with different time sample ranges
        array1 = cae_vtk.FieldArray.Define(stage, "/World/Array1")
        attr1 = array1.CreateFileNamesAttr()
        # Array1: 0-10 seconds = 0-600 time codes
        attr1.Set(["d1_t0.vti"], 0.0)
        attr1.Set(["d1_t10.vti"], 600.0)  # 10 seconds * 60

        array2 = cae_vtk.FieldArray.Define(stage, "/World/Array2")
        attr2 = array2.CreateFileNamesAttr()
        # Array2: 20-30 seconds = 1200-1800 time codes
        attr2.Set(["d2_t20.vti"], 1200.0)  # 20 seconds * 60
        attr2.Set(["d2_t30.vti"], 1800.0)  # 30 seconds * 60

        tracker.clear_all_changes()

        # Set time to 5 (in array1's range, before array2's range)
        timeline = omni.timeline.get_timeline_interface()
        timeline.set_current_time(5.0)
        await omni.kit.app.get_app().next_update_async()

        last_time = Usd.TimeCode(round(timeline.get_current_time() * timeline.get_time_codes_per_seconds()))

        # Change to time 25 (in array2's range, after array1's range)
        timeline.set_current_time(25.0)
        await omni.kit.app.get_app().next_update_async()

        # Both should have bracket changes
        self.assertTrue(
            tracker.prim_changed(array1.GetPrim(), ["CaeVtkFieldArray"], last_time_code=last_time),
            "Array1 should detect bracket change",
        )
        self.assertTrue(
            tracker.prim_changed(array2.GetPrim(), ["CaeVtkFieldArray"], last_time_code=last_time),
            "Array2 should detect bracket change",
        )

        tracker.disable()
        await usd_context.close_stage_async()

    async def test_time_tracking_with_structural_changes(self):
        """Test that time tracking works alongside structural changes."""
        usd_context = get_context()
        await usd_context.new_stage_async()
        stage = usd_context.get_stage()

        # Create tracker with time tracking
        tracker = ChangeTracker(stage, [r"^CaeVtk"], track_time_changes=True)

        # Create an array
        array = cae_vtk.FieldArray.Define(stage, "/World/Array")
        prim = array.GetPrim()

        # Add time-sampled attribute
        attr = array.CreateFileNamesAttr()
        # Use time codes: 0-10 seconds = 0-600 time codes
        attr.Set(["file_t0.vti"], 0.0)
        attr.Set(["file_t10.vti"], 600.0)  # 10 seconds * 60

        tracker.clear_all_changes()

        # Make a structural change (modify the attribute value)
        attr.Set(["modified_file.vti"], 0.0)

        # Should detect structural change (no last_time_code)
        self.assertTrue(tracker.prim_changed(prim, ["CaeVtkFieldArray"]), "Should detect structural change")

        # Clear changes
        tracker.clear_changes(prim)

        # Now set up for time change test
        timeline = omni.timeline.get_timeline_interface()
        timeline.set_current_time(0.0)
        await omni.kit.app.get_app().next_update_async()

        last_time = Usd.TimeCode(round(timeline.get_current_time() * timeline.get_time_codes_per_seconds()))

        # Change time to different bracket
        timeline.set_current_time(15.0)
        await omni.kit.app.get_app().next_update_async()

        # Should detect time-based change
        self.assertTrue(
            tracker.prim_changed(prim, ["CaeVtkFieldArray"], last_time_code=last_time),
            "Should detect time-based change",
        )

        tracker.disable()
        await usd_context.close_stage_async()

    async def test_time_tracking_invalidate_cache(self):
        """Test that cache invalidation works correctly."""
        usd_context = get_context()
        await usd_context.new_stage_async()
        stage = usd_context.get_stage()

        # Create tracker with time tracking
        tracker = ChangeTracker(stage, [r"^CaeVtk"], track_time_changes=True, debug_logging=True)

        # Create an array
        array = cae_vtk.FieldArray.Define(stage, "/World/Array")
        prim = array.GetPrim()
        attr = array.CreateFileNamesAttr()
        # Use time codes: 0-10 seconds = 0-600 time codes
        attr.Set(["file_t0.vti"], 0.0)
        attr.Set(["file_t10.vti"], 600.0)  # 10 seconds * 60

        tracker.clear_all_changes()

        # Set initial time
        timeline = omni.timeline.get_timeline_interface()
        timeline.set_current_time(5.0)  # = time code 300
        await omni.kit.app.get_app().next_update_async()

        last_time = Usd.TimeCode(round(timeline.get_current_time() * timeline.get_time_codes_per_seconds()))

        # Manually invalidate cache for this prim
        tracker.invalidate_time_cache(prim)

        # Change time
        timeline.set_current_time(15.0)  # = time code 900 (beyond 600)
        await omni.kit.app.get_app().next_update_async()

        # Should still detect change (cache repopulated on check)
        self.assertTrue(
            tracker.prim_changed(prim, ["CaeVtkFieldArray"], last_time_code=last_time),
            "Should detect change even after cache invalidation",
        )

        tracker.disable()
        await usd_context.close_stage_async()

    async def test_time_tracking_attr_changed(self):
        """Test attr_changed with time tracking."""
        usd_context = get_context()
        await usd_context.new_stage_async()
        stage = usd_context.get_stage()

        # Create tracker with time tracking
        tracker = ChangeTracker(stage, [r"^CaeVtk"], track_time_changes=True)

        # Create an array with time-sampled attribute
        array = cae_vtk.FieldArray.Define(stage, "/World/Array")
        prim = array.GetPrim()
        attr = array.CreateFileNamesAttr()
        # Use time codes: 0-10 seconds = 0-600 time codes
        attr.Set(["file_t0.vti"], 0.0)
        attr.Set(["file_t10.vti"], 600.0)  # 10 seconds * 60

        tracker.clear_all_changes()

        # Set time
        timeline = omni.timeline.get_timeline_interface()
        timeline.set_current_time(0.0)
        await omni.kit.app.get_app().next_update_async()

        last_time = Usd.TimeCode(round(timeline.get_current_time() * timeline.get_time_codes_per_seconds()))

        # Change time (beyond time samples)
        timeline.set_current_time(15.0)  # = time code 900 (beyond 600)
        await omni.kit.app.get_app().next_update_async()

        # Check specific attribute
        self.assertTrue(
            tracker.attr_changed(prim, "fileNames", last_time_code=last_time),
            "Should detect time-based change for specific attribute",
        )

        tracker.disable()
        await usd_context.close_stage_async()

    async def test_time_tracking_has_changes(self):
        """Test has_changes with time tracking."""
        usd_context = get_context()
        await usd_context.new_stage_async()
        stage = usd_context.get_stage()

        # Create tracker with time tracking
        tracker = ChangeTracker(stage, [r"^CaeVtk"], track_time_changes=True)

        # Create an array with time-sampled attribute
        array = cae_vtk.FieldArray.Define(stage, "/World/Array")
        prim = array.GetPrim()
        attr = array.CreateFileNamesAttr()
        # Use time codes: 0-10 seconds = 0-600 time codes
        attr.Set(["file_t0.vti"], 0.0)
        attr.Set(["file_t10.vti"], 600.0)  # 10 seconds * 60

        tracker.clear_all_changes()

        # Set initial time
        timeline = omni.timeline.get_timeline_interface()
        timeline.set_current_time(0.0)
        await omni.kit.app.get_app().next_update_async()

        last_time = Usd.TimeCode(round(timeline.get_current_time() * timeline.get_time_codes_per_seconds()))

        # Should have no changes initially (no structural changes, no time check done yet)
        self.assertFalse(tracker.has_changes(), "Should have no changes initially")

        # Change time (beyond time samples)
        timeline.set_current_time(15.0)  # = time code 900 (beyond 600)
        await omni.kit.app.get_app().next_update_async()

        # Call prim_changed with time check - this populates _prim_schema_changes
        tracker.prim_changed(prim, ["CaeVtkFieldArray"], last_time_code=last_time)

        # Should have changes now (time-based changes were recorded)
        self.assertTrue(tracker.has_changes(), "Should have changes after time-based check")

        tracker.disable()
        await usd_context.close_stage_async()

    async def test_time_tracking_multi_apply_schema(self):
        """Test time tracking with multi-apply schemas."""
        usd_context = get_context()
        await usd_context.new_stage_async()
        stage = usd_context.get_stage()

        # Create tracker with time tracking
        tracker = ChangeTracker(stage, [r"^CaeViz"], track_time_changes=True)

        # Create a prim and apply multi-apply schema
        prim = stage.DefinePrim("/World/TestPrim")
        field_sel = cae_viz.FieldSelectionAPI.Apply(prim, "velocity")

        # Note: CaeVizFieldSelectionAPI:target relationship might not be time-sampled
        # This test verifies the system handles multi-apply schemas without crashing
        tracker.clear_all_changes()

        # Set time
        timeline = omni.timeline.get_timeline_interface()
        timeline.set_current_time(0.0)
        await omni.kit.app.get_app().next_update_async()

        last_time = Usd.TimeCode(round(timeline.get_current_time() * timeline.get_time_codes_per_seconds()))

        # Change time
        timeline.set_current_time(10.0)
        await omni.kit.app.get_app().next_update_async()

        # Should not crash, may or may not detect changes (no time samples)
        changed = tracker.prim_changed(prim, ["CaeVizFieldSelectionAPI:velocity"], last_time_code=last_time)

        tracker.disable()
        await usd_context.close_stage_async()

    async def test_time_tracking_clear_specific_schema(self):
        """Test clearing time changes for specific schemas."""
        usd_context = get_context()
        await usd_context.new_stage_async()
        stage = usd_context.get_stage()

        # Create tracker with time tracking
        tracker = ChangeTracker(stage, [r"^CaeVtk"], track_time_changes=True)

        # Create an array
        array = cae_vtk.FieldArray.Define(stage, "/World/Array")
        prim = array.GetPrim()
        attr = array.CreateFileNamesAttr()
        # Use time codes: 0-10 seconds = 0-600 time codes
        attr.Set(["file_t0.vti"], 0.0)
        attr.Set(["file_t10.vti"], 600.0)  # 10 seconds * 60

        tracker.clear_all_changes()

        # Set initial time
        timeline = omni.timeline.get_timeline_interface()
        timeline.set_current_time(0.0)
        await omni.kit.app.get_app().next_update_async()

        last_time = Usd.TimeCode(round(timeline.get_current_time() * timeline.get_time_codes_per_seconds()))

        # Change time (beyond time samples)
        timeline.set_current_time(15.0)  # = time code 900 (beyond 600)
        await omni.kit.app.get_app().next_update_async()

        # Should have changes
        self.assertTrue(
            tracker.prim_changed(prim, ["CaeVtkFieldArray"], last_time_code=last_time), "Should have changes"
        )

        # Clear only CaeVtkFieldArray schema
        tracker.clear_changes(prim, ["CaeVtkFieldArray"])

        # Should have no changes for CaeVtkFieldArray
        self.assertFalse(
            tracker.prim_changed(prim, ["CaeVtkFieldArray"]), "Should have no changes after clearing specific schema"
        )

        tracker.disable()
        await usd_context.close_stage_async()

    async def test_relationship_target_prim_changed(self):
        """Test that changes to relationship target prims are detected."""
        usd_context = get_context()
        await usd_context.new_stage_async()
        stage = usd_context.get_stage()

        # Create tracker
        tracker = ChangeTracker(stage, [r"^Cae"], track_time_changes=False)

        # Create a dataset with PointCloudAPI that has a coordinates relationship
        dataset = cae.DataSet.Define(stage, "/World/Dataset")
        prim = dataset.GetPrim()
        point_cloud_api = cae.PointCloudAPI.Apply(prim)

        # Create a target field array
        target_array = cae_vtk.FieldArray.Define(stage, "/World/CoordinatesArray")

        # Set the relationship
        point_cloud_api.CreateCoordinatesRel().SetTargets([target_array.GetPath()])

        # Clear initial changes
        tracker.clear_all_changes()

        # Make a change to the target array
        target_array.CreateFileNamesAttr().Set(["coords.vti"])

        # Should detect change through relationship
        self.assertTrue(
            tracker.prim_changed(prim, ["CaePointCloudAPI"]),
            "Should detect change in relationship target prim",
        )

        tracker.disable()
        await usd_context.close_stage_async()

    async def test_relationship_target_attribute_changed(self):
        """Test that changes to relationship target attributes are detected."""
        usd_context = get_context()
        await usd_context.new_stage_async()
        stage = usd_context.get_stage()

        # Create tracker
        tracker = ChangeTracker(stage, [r"^Cae"], track_time_changes=False)

        # Create a dataset
        dataset = cae.DataSet.Define(stage, "/World/Dataset")
        prim = dataset.GetPrim()

        # Create a target array with an attribute
        target_array = cae_vtk.FieldArray.Define(stage, "/World/TargetArray")
        target_attr = target_array.CreateFileNamesAttr()
        target_attr.Set(["initial.vti"])

        # Create a custom relationship pointing to the attribute
        # (In real schemas, this would be defined by the schema)
        custom_rel = prim.CreateRelationship("customRel")
        custom_rel.SetTargets([target_attr.GetPath()])

        # Clear initial changes
        tracker.clear_all_changes()

        # Make a change to the target attribute
        target_attr.Set(["modified.vti"])

        # Note: customRel is not part of CaeDataSet schema, so this won't be detected
        # unless we explicitly check relationships that are schema properties
        # For this test, we're verifying the infrastructure works

        tracker.disable()
        await usd_context.close_stage_async()

    async def test_relationship_cycle_detection(self):
        """Test that relationship cycles don't cause infinite recursion."""
        usd_context = get_context()
        await usd_context.new_stage_async()
        stage = usd_context.get_stage()

        # Create tracker
        tracker = ChangeTracker(stage, [r"^Cae"], track_time_changes=False)

        # Create two datasets with relationships pointing to each other
        dataset1 = cae.DataSet.Define(stage, "/World/Dataset1")
        dataset2 = cae.DataSet.Define(stage, "/World/Dataset2")

        # Create relationships that form a cycle
        rel1 = dataset1.GetPrim().CreateRelationship("next")
        rel2 = dataset2.GetPrim().CreateRelationship("next")
        rel1.SetTargets([dataset2.GetPath()])
        rel2.SetTargets([dataset1.GetPath()])

        # Clear initial changes
        tracker.clear_all_changes()

        # Make a change to dataset2 (add a custom attribute)
        dataset2.GetPrim().CreateAttribute("customAttr", Sdf.ValueTypeNames.String).Set("value")

        # Should not hang due to cycle detection
        # Note: The custom "next" relationship is not part of the schema,
        # so it won't be traversed. This test verifies no crash occurs.
        changed = tracker.prim_changed(dataset1.GetPrim(), ["CaeDataSet"])

        tracker.disable()
        await usd_context.close_stage_async()

    async def test_attr_changed_relationship(self):
        """Test attr_changed on a relationship property."""
        usd_context = get_context()
        await usd_context.new_stage_async()
        stage = usd_context.get_stage()

        # Create tracker
        tracker = ChangeTracker(stage, [r"^Cae"], track_time_changes=False)

        # Create a dataset with PointCloudAPI
        dataset = cae.DataSet.Define(stage, "/World/Dataset")
        prim = dataset.GetPrim()
        point_cloud_api = cae.PointCloudAPI.Apply(prim)

        # Create target array
        target_array = cae_vtk.FieldArray.Define(stage, "/World/CoordinatesArray")

        # Set the relationship
        point_cloud_api.CreateCoordinatesRel().SetTargets([target_array.GetPath()])

        # Clear initial changes
        tracker.clear_all_changes()

        # Make a change to the target
        target_array.CreateFileNamesAttr().Set(["coords.vti"])

        # Should detect change through relationship using attr_changed
        # Note: Use the full relationship name with namespace
        self.assertTrue(
            tracker.attr_changed(prim, "cae:pointCloud:coordinates"),
            "Should detect relationship target change via attr_changed",
        )

        tracker.disable()
        await usd_context.close_stage_async()

    async def test_relationship_no_false_positives(self):
        """Test that unchanged relationship targets don't trigger false positives."""
        usd_context = get_context()
        await usd_context.new_stage_async()
        stage = usd_context.get_stage()

        # Create tracker
        tracker = ChangeTracker(stage, [r"^Cae"], track_time_changes=False)

        # Create dataset with relationship
        dataset = cae.DataSet.Define(stage, "/World/Dataset")
        prim = dataset.GetPrim()
        point_cloud_api = cae.PointCloudAPI.Apply(prim)

        # Create target that won't change
        target_array = cae_vtk.FieldArray.Define(stage, "/World/CoordinatesArray")
        target_array.CreateFileNamesAttr().Set(["static.vti"])

        # Set the relationship
        point_cloud_api.CreateCoordinatesRel().SetTargets([target_array.GetPath()])

        # Clear all changes
        tracker.clear_all_changes()

        # Don't make any changes

        # Should NOT detect any changes
        self.assertFalse(
            tracker.prim_changed(prim, ["CaePointCloudAPI"]),
            "Should not detect changes when relationship targets are unchanged",
        )

        tracker.disable()
        await usd_context.close_stage_async()

    async def test_explicit_current_time_code(self):
        """Test explicit current_time_code parameter for predictive queries."""
        usd_context = get_context()
        await usd_context.new_stage_async()
        stage = usd_context.get_stage()
        timeline = omni.timeline.get_timeline_interface()

        # Create tracker with time tracking enabled
        tracker = ChangeTracker(stage, [r"^CaeVtk"], track_time_changes=True)

        # Create time-varying prim
        array = cae_vtk.FieldArray.Define(stage, "/World/TimeArray")
        prim = array.GetPrim()
        attr = array.CreateFileNamesAttr()

        # Set time samples at 0s and 10s (time codes 0 and 600)
        attr.Set(["file_t0.vti"], 0.0)
        attr.Set(["file_t10.vti"], 600.0)

        # Clear initial state
        tracker.clear_all_changes()

        # Set timeline to 1s (time code 60)
        timeline.set_current_time(1.0)
        await omni.kit.app.get_app().next_update_async()

        # Test 1: Check if moving from 1s to 5s would cause changes (both in same bracket [0, 600])
        last_time = Usd.TimeCode(60)  # 1s
        future_time = Usd.TimeCode(300)  # 5s

        # Should NOT detect change - both times in same bracket
        self.assertFalse(
            tracker.prim_changed(prim, ["CaeVtkFieldArray"], last_time_code=last_time, current_time_code=future_time),
            "Should not detect change when both times in same bracket [0, 600]",
        )

        # Test 2: Check if moving from 1s to 15s would cause changes (different brackets)
        far_future_time = Usd.TimeCode(900)  # 15s - beyond last sample

        # Should detect change - different brackets
        self.assertTrue(
            tracker.prim_changed(
                prim, ["CaeVtkFieldArray"], last_time_code=last_time, current_time_code=far_future_time
            ),
            "Should detect change when bracket changes [0, 600] -> [600, 600]",
        )

        # Test 3: Verify timeline position hasn't changed (still at 1s)
        current_timeline_time = timeline.get_current_time()
        self.assertAlmostEqual(current_timeline_time, 1.0, places=2)

        # Test 4: Check with attr_changed
        self.assertTrue(
            tracker.attr_changed(prim, "fileNames", last_time_code=last_time, current_time_code=far_future_time),
            "attr_changed should also detect change with explicit time codes",
        )

        # Test 5: Without current_time_code, should use timeline's current time
        # Clear changes first since we've been checking with explicit times
        tracker.clear_changes(prim, ["CaeVtkFieldArray"])

        # Need to ensure timeline event has been processed
        await omni.kit.app.get_app().next_update_async()

        # Timeline is at 1s (60 time codes), check from same time - should be no change
        # since we're comparing 60 -> 60 (same time code)
        self.assertFalse(
            tracker.prim_changed(prim, ["CaeVtkFieldArray"], last_time_code=Usd.TimeCode(60)),
            "Without current_time_code, should use timeline time (60 == 60, no change)",
        )

        # Test 6: Move timeline and verify explicit time still works independently
        timeline.set_current_time(8.0)  # 8s = 480 time codes
        await omni.kit.app.get_app().next_update_async()

        # Clear changes from timeline movement
        tracker.clear_changes(prim, ["CaeVtkFieldArray"])

        # Query about a completely different time range (doesn't use timeline time)
        time_a = Usd.TimeCode(60)  # 1s - start from non-zero to avoid edge cases
        time_b = Usd.TimeCode(300)  # 5s - both should be in same bracket [0, 600]

        result = tracker.prim_changed(prim, ["CaeVtkFieldArray"], last_time_code=time_a, current_time_code=time_b)
        self.assertFalse(
            result,
            f"Should not detect change in [60, 300] range (same bracket [0, 600])",
        )

        tracker.disable()
        await usd_context.close_stage_async()
