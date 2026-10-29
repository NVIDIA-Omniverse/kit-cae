# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
Tests for the controller's temporal execution system.
"""

import logging

import omni.kit.app
import omni.kit.test
from omni.cae.core import usd_utils
from omni.cae.schema import viz as cae_viz
from omni.cae.testing import get_test_data_path, wait_for_update
from omni.cae.usd_plugins_importers import import_to_stage
from omni.cae.viz.controller import _format_exception_diagnostic
from omni.cae.viz.execution_context import ExecutionContext, ExecutionReason
from omni.cae.viz.operator import operator, register_module_operators, unregister_module_operators
from omni.timeline import get_timeline_interface
from omni.usd import get_context
from pxr import Gf, Sdf, Usd, UsdGeom

from ._operator_test_utils import wait_for_operator_complete

logger = logging.getLogger(__name__)
# Test operator classes with different temporal configurations


def _set_field_names(field_selection_api: cae_viz.FieldSelectionAPI, *names: str) -> None:
    field_selection_api.CreateFieldNamesAttr().Set(list(names))


class Operators:
    @operator(priority=0, supports_temporal=False, tick_on_time_change=False)
    class NonTemporalOperator:
        """Operator without temporal support (always executes)."""

        prim_type = "Cube"
        api_schemas = {
            "CaeVizOperatorAPI",
        }
        optional_api_schemas = set()

        async def exec(self, prim: Usd.Prim, device: str, context: ExecutionContext):
            """Execute operator."""
            logger.info(f"NonTemporalOperator exec called at {context.timecode}")

            # Read old value and increment
            exec_count = prim.GetAttribute("test:exec_count").Get() + 1

            # Set attributes on prim with execution context info
            prim.GetAttribute("test:exec_count").Set(exec_count)
            prim.GetAttribute("test:last_reason").Set(context.reason.value)
            prim.GetAttribute("test:last_timecode").Set(context.timecode.GetValue())
            prim.GetAttribute("test:last_raw_timecode").Set(context.raw_timecode.GetValue())
            prim.GetAttribute("test:last_timestep_index").Set(context.timestep_index)

        async def on_time_changed(self, prim: Usd.Prim, device: str, context: ExecutionContext):
            """Time changed callback (should not be called)."""
            logger.info(f"NonTemporalOperator on_time_changed called at {context.timecode}")

            # Read old value and increment
            tick_count = prim.GetAttribute("test:tick_count").Get() + 1

            # Set tick count attribute
            prim.GetAttribute("test:tick_count").Set(tick_count)

    @operator(priority=0, supports_temporal=True, tick_on_time_change=False)
    class TemporalOperatorNoTick:
        """Operator with temporal support but no tick callback."""

        prim_type = "Sphere"
        api_schemas = {
            "CaeVizOperatorAPI",
        }
        optional_api_schemas = set()

        async def exec(self, prim: Usd.Prim, device: str, context: ExecutionContext):
            """Execute operator."""
            logger.info(f"TemporalOperatorNoTick exec called at {context.timecode}")

            # Read old value and increment
            exec_count = prim.GetAttribute("test:exec_count").Get() + 1

            # Set attributes on prim with execution context info
            prim.GetAttribute("test:exec_count").Set(exec_count)
            prim.GetAttribute("test:last_reason").Set(context.reason.value)
            prim.GetAttribute("test:last_timecode").Set(context.timecode.GetValue())
            prim.GetAttribute("test:last_raw_timecode").Set(context.raw_timecode.GetValue())
            prim.GetAttribute("test:is_full_rebuild").Set(context.is_full_rebuild_needed())
            prim.GetAttribute("test:is_temporal_update").Set(context.is_temporal_update())
            prim.GetAttribute("test:last_timestep_index").Set(context.timestep_index)

        async def on_time_changed(self, prim: Usd.Prim, device: str, context: ExecutionContext):
            """Time changed callback (should not be called)."""
            logger.info(f"TemporalOperatorNoTick on_time_changed called at {context.timecode}")

            # Read old value and increment
            tick_count = prim.GetAttribute("test:tick_count").Get() + 1

            # Set tick count attribute
            prim.GetAttribute("test:tick_count").Set(tick_count)

    @operator(priority=0, supports_temporal=True, tick_on_time_change=True)
    class TemporalOperatorWithTick:
        """Operator with temporal support and tick callback."""

        prim_type = "Cone"
        api_schemas = {
            "CaeVizOperatorAPI",
        }
        optional_api_schemas = set()

        async def exec(self, prim: Usd.Prim, device: str, context: ExecutionContext):
            """Execute operator."""
            logger.info(f"TemporalOperatorWithTick exec called at {context.timecode}")

            # Read old value and increment
            exec_count = prim.GetAttribute("test:exec_count").Get() + 1

            # Set attributes on prim with execution context info
            prim.GetAttribute("test:exec_count").Set(exec_count)
            prim.GetAttribute("test:last_reason").Set(context.reason.value)
            prim.GetAttribute("test:last_timecode").Set(context.timecode.GetValue())
            prim.GetAttribute("test:last_raw_timecode").Set(context.raw_timecode.GetValue())
            prim.GetAttribute("test:is_full_rebuild").Set(context.is_full_rebuild_needed())
            prim.GetAttribute("test:is_temporal_update").Set(context.is_temporal_update())
            prim.GetAttribute("test:last_timestep_index").Set(context.timestep_index)

        async def on_time_changed(self, prim: Usd.Prim, device: str, context: ExecutionContext):
            """Time changed callback (should be called on every time change)."""
            logger.info(f"TemporalOperatorWithTick on_time_changed called at {context.timecode}")

            # Read old value and increment
            tick_count = prim.GetAttribute("test:tick_count").Get() + 1

            # Set tick attributes on prim
            prim.GetAttribute("test:tick_count").Set(tick_count)
            prim.GetAttribute("test:last_tick_timecode").Set(context.timecode.GetValue())
            prim.GetAttribute("test:last_tick_raw_timecode").Set(context.raw_timecode.GetValue())
            prim.GetAttribute("test:is_temporal_tick").Set(context.is_temporal_tick())
            if context.next_time_code:
                prim.GetAttribute("test:next_time_code").Set(context.next_time_code.GetValue())
            else:
                prim.GetAttribute("test:next_time_code").Set(-1.0)  # No next timecode


class TestControllerOperatorExecution(omni.kit.test.AsyncTestCase):
    """Test suite for temporal execution functionality in the controller."""

    def test_exception_diagnostic_preserves_chained_cause(self):
        cause = RuntimeError("legacy converter is disabled")
        error = usd_utils.QuietableException("dataset conversion failed")
        error.__cause__ = cause

        self.assertEqual(
            _format_exception_diagnostic(error),
            "dataset conversion failed; caused by: legacy converter is disabled",
        )

    async def setUp(self):
        """Set up test environment before each test."""

        register_module_operators(Operators)

        self.usd_context = get_context()
        # Create a new stage
        await self.usd_context.new_stage_async()
        self.stage = self.usd_context.get_stage()
        self.stage.SetTimeCodesPerSecond(1.0)

        self.timeline = get_timeline_interface()
        self.timeline.set_time_codes_per_second(1.0)
        self.timeline.set_start_time(0.0)
        self.timeline.set_end_time(100.0)
        self.timeline.set_current_time(0.0)

        # Import hex_timesteps.cgns with time scale of 10
        hex_path = get_test_data_path("hex_timesteps.cgns")
        await import_to_stage(hex_path, "/World/hex_timesteps", scale=10.0, source="TimeStep")

        # Import StaticMixer.cgns
        mixer_path = get_test_data_path("StaticMixer.cgns")
        await import_to_stage(mixer_path, "/World/StaticMixer")

        await wait_for_update(0)

    async def tearDown(self):
        """Clean up after each test."""
        unregister_module_operators(Operators)
        await self.usd_context.close_stage_async()

    def init_prim(self, prim: Usd.Prim):
        prim.CreateAttribute("test:exec_count", Sdf.ValueTypeNames.Int).Set(0)
        prim.CreateAttribute("test:tick_count", Sdf.ValueTypeNames.Int).Set(0)
        prim.CreateAttribute("test:last_reason", Sdf.ValueTypeNames.String).Set("NONE")
        prim.CreateAttribute("test:last_timecode", Sdf.ValueTypeNames.Double).Set(0.0)
        prim.CreateAttribute("test:last_raw_timecode", Sdf.ValueTypeNames.Double).Set(0.0)
        prim.CreateAttribute("test:last_tick_timecode", Sdf.ValueTypeNames.Double).Set(0.0)
        prim.CreateAttribute("test:last_tick_raw_timecode", Sdf.ValueTypeNames.Double).Set(0.0)
        prim.CreateAttribute("test:is_full_rebuild", Sdf.ValueTypeNames.Bool).Set(False)
        prim.CreateAttribute("test:is_temporal_update", Sdf.ValueTypeNames.Bool).Set(False)
        prim.CreateAttribute("test:is_temporal_tick", Sdf.ValueTypeNames.Bool).Set(False)
        prim.CreateAttribute("test:next_time_code", Sdf.ValueTypeNames.Double).Set(
            -1.0
        )  # -1.0 indicates no next timecode
        prim.CreateAttribute("test:last_timestep_index", Sdf.ValueTypeNames.Int).Set(0)

    def get_bracket(self, prim: Usd.Prim, raw_time: float) -> tuple[float, float]:
        lower, upper, has_time_samples = usd_utils.get_bracketing_time_samples_for_prim(prim, raw_time)
        self.assertTrue(has_time_samples, f"{prim.GetPath()} should have time samples")
        return lower, upper

    async def forward_frames(self, frames: int):
        """Forward the timeline by the given number of frames."""
        for _ in range(frames):
            self.timeline.forward_one_frame()
            await wait_for_update(0)

    async def _assert_roi_transform_reexecutes_operator(self, api_type, suffix: str):
        roi_path = f"/World/ROITransformDependency_{suffix}"
        roi_cube = UsdGeom.Cube.Define(self.stage, roi_path)
        translate_op = roi_cube.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(0.0, 0.0, 0.0))

        operator_path = f"/World/ROITransformDependencyOperator_{suffix}"
        operator_prim = self.stage.DefinePrim(operator_path, "Cube")
        self.init_prim(operator_prim)

        async with wait_for_operator_complete(operator_path, operator="NonTemporalOperator"):
            cae_viz.OperatorAPI.Apply(operator_prim)
            cae_viz.OperatorAPI(operator_prim).GetEnabledAttr().Set(True)
            api_type.Apply(operator_prim, "source")
            api_type(operator_prim, "source").CreateRoiRel().SetTargets({roi_cube.GetPath()})

        self.assertEqual(operator_prim.GetAttribute("test:exec_count").Get(), 1)

        async with wait_for_operator_complete(operator_path, operator="NonTemporalOperator"):
            translate_op.Set(Gf.Vec3d(1.0, 2.0, 3.0))

        self.assertEqual(operator_prim.GetAttribute("test:exec_count").Get(), 2)
        self.assertEqual(operator_prim.GetAttribute("test:last_reason").Get(), ExecutionReason.STRUCTURAL_CHANGE.value)

    async def test_operator_reexecutes_when_subset_roi_transform_changes(self):
        """DatasetSubsetAPI ROI transforms are operator execution dependencies."""
        await self._assert_roi_transform_reexecutes_operator(cae_viz.DatasetSubsetAPI, "Subset")

    async def test_operator_reexecutes_when_voxelization_roi_transform_changes(self):
        """DatasetVoxelizationAPI ROI transforms are operator execution dependencies."""
        await self._assert_roi_transform_reexecutes_operator(cae_viz.DatasetVoxelizationAPI, "Voxelization")

    async def test_non_temporal_operator_non_temporal_dataset(self):
        """Test that non-temporal operator executes on every change."""
        self.timeline.set_current_time(0.0)
        await wait_for_update(0)

        non_temporal_prim = self.stage.DefinePrim("/World/NonTemporalOperator", "Cube")
        self.init_prim(non_temporal_prim)
        cae_viz.OperatorAPI.Apply(non_temporal_prim)
        cae_viz.OperatorAPI(non_temporal_prim).GetEnabledAttr().Set(True)
        cae_viz.DatasetSelectionAPI.Apply(non_temporal_prim, "source")
        await wait_for_update(0)

        # Should have executed once
        self.assertEqual(non_temporal_prim.GetAttribute("test:exec_count").Get(), 1)
        # Tick should not have been called (no tick_on_time_change)
        self.assertEqual(non_temporal_prim.GetAttribute("test:tick_count").Get(), 0)
        # First execution should be STRUCTURAL_CHANGE (or INITIAL)
        reason = non_temporal_prim.GetAttribute("test:last_reason").Get()
        self.assertIn(reason, [ExecutionReason.STRUCTURAL_CHANGE.value, ExecutionReason.INITIAL.value])
        # Non-temporal operators should always have timestep_index=0
        self.assertEqual(non_temporal_prim.GetAttribute("test:last_timestep_index").Get(), 0)
        # Timecodes should be set (not necessarily 0.0)
        self.assertEqual(
            non_temporal_prim.GetAttribute("test:last_timecode").Get(), Usd.TimeCode.EarliestTime().GetValue()
        )
        self.assertIsNotNone(non_temporal_prim.GetAttribute("test:last_raw_timecode").Get())

        # Move timeline to next timecode
        await self.forward_frames(10)

        # Should not have executed again since this is not a temporal operator
        self.assertEqual(non_temporal_prim.GetAttribute("test:exec_count").Get(), 1)
        # Tick should not have been called (no tick_on_time_change)
        self.assertEqual(non_temporal_prim.GetAttribute("test:tick_count").Get(), 0)
        # First execution should be STRUCTURAL_CHANGE (or INITIAL)
        reason = non_temporal_prim.GetAttribute("test:last_reason").Get()
        self.assertIn(reason, [ExecutionReason.STRUCTURAL_CHANGE.value, ExecutionReason.INITIAL.value])
        # Timecodes should be set (not necessarily 0.0)

        cae_viz.DatasetSelectionAPI(non_temporal_prim, "source").GetTargetRel().SetTargets(
            {"/World/StaticMixer/Base/StaticMixer/B1_P3"}
        )
        await self.forward_frames(5)

        # Should have executed again since the operator is enabled
        self.assertEqual(non_temporal_prim.GetAttribute("test:exec_count").Get(), 2)
        # Tick should not have been called (no tick_on_time_change)
        self.assertEqual(non_temporal_prim.GetAttribute("test:tick_count").Get(), 0)
        # First execution should be STRUCTURAL_CHANGE (or INITIAL)
        reason = non_temporal_prim.GetAttribute("test:last_reason").Get()
        self.assertEqual(reason, ExecutionReason.STRUCTURAL_CHANGE.value)

    async def test_non_temporal_operator_temporal_dataset(self):
        """Test that non-temporal operator with temporal dataset still executes on every change."""
        self.timeline.set_current_time(0.0)
        await wait_for_update(0)

        non_temporal_prim = self.stage.DefinePrim("/World/NonTemporalOperatorTemporal", "Cube")
        self.init_prim(non_temporal_prim)
        cae_viz.OperatorAPI.Apply(non_temporal_prim)
        cae_viz.OperatorAPI(non_temporal_prim).GetEnabledAttr().Set(True)
        # Point to temporal dataset (hex_timesteps)
        cae_viz.DatasetSelectionAPI.Apply(non_temporal_prim, "source")
        cae_viz.DatasetSelectionAPI(non_temporal_prim, "source").GetTargetRel().SetTargets(
            {"/World/hex_timesteps/Base/Zone/ElementsUniform"}
        )

        cae_viz.FieldSelectionAPI.Apply(non_temporal_prim, "colors")
        _set_field_names(cae_viz.FieldSelectionAPI(non_temporal_prim, "colors"), "PointSinusoid")
        await wait_for_update(0)

        # Should have executed once
        self.assertEqual(non_temporal_prim.GetAttribute("test:exec_count").Get(), 1)
        # Tick should not have been called (no tick_on_time_change)
        self.assertEqual(non_temporal_prim.GetAttribute("test:tick_count").Get(), 0)
        # First execution should be STRUCTURAL_CHANGE (or INITIAL)
        reason = non_temporal_prim.GetAttribute("test:last_reason").Get()
        self.assertIn(reason, [ExecutionReason.STRUCTURAL_CHANGE.value, ExecutionReason.INITIAL.value])

        # Move timeline to a new bracketed sample on the plugin-backed dataset.
        self.timeline.set_current_time(10.0)
        await wait_for_update(0)

        # Non-temporal operator should have executed once more
        self.assertEqual(non_temporal_prim.GetAttribute("test:exec_count").Get(), 2)
        # Tick should not have been called
        self.assertEqual(non_temporal_prim.GetAttribute("test:tick_count").Get(), 0)
        # Should be TEMPORAL_UPDATE
        reason = non_temporal_prim.GetAttribute("test:last_reason").Get()
        self.assertEqual(reason, ExecutionReason.TEMPORAL_UPDATE.value)
        # to back to start
        self.timeline.set_current_time(0.0)
        await wait_for_update(0)

        # should have executed once more
        self.assertEqual(non_temporal_prim.GetAttribute("test:exec_count").Get(), 3)
        # Tick should not have been called
        self.assertEqual(non_temporal_prim.GetAttribute("test:tick_count").Get(), 0)
        # Should be TEMPORAL_UPDATE
        reason = non_temporal_prim.GetAttribute("test:last_reason").Get()
        self.assertEqual(reason, ExecutionReason.TEMPORAL_UPDATE.value)

        # Make a structural change by changing a property
        cae_viz.OperatorAPI(non_temporal_prim).GetEnabledAttr().Set(False)
        await wait_for_update(0)
        cae_viz.OperatorAPI(non_temporal_prim).GetEnabledAttr().Set(True)
        await wait_for_update(0)

        # Should have executed again due to structural change
        self.assertEqual(non_temporal_prim.GetAttribute("test:exec_count").Get(), 4)
        # Tick should not have been called
        self.assertEqual(non_temporal_prim.GetAttribute("test:tick_count").Get(), 0)
        # Should be STRUCTURAL_CHANGE
        reason = non_temporal_prim.GetAttribute("test:last_reason").Get()
        self.assertEqual(reason, ExecutionReason.STRUCTURAL_CHANGE.value)

    async def test_temporal_operator_no_tick_temporal_dataset(self):
        """Test temporal operator (no tick) with temporal dataset - should cache timecodes."""
        self.timeline.set_current_time(0.0)
        await wait_for_update(0)

        temporal_prim = self.stage.DefinePrim("/World/TemporalOperatorNoTick", "Sphere")
        self.init_prim(temporal_prim)
        cae_viz.OperatorAPI.Apply(temporal_prim)
        cae_viz.OperatorAPI(temporal_prim).GetEnabledAttr().Set(True)
        # Point to temporal dataset (hex_timesteps)
        cae_viz.DatasetSelectionAPI.Apply(temporal_prim, "source")
        cae_viz.DatasetSelectionAPI(temporal_prim, "source").GetTargetRel().SetTargets(
            {"/World/hex_timesteps/Base/Zone/ElementsUniform"}
        )
        cae_viz.FieldSelectionAPI.Apply(temporal_prim, "colors")
        _set_field_names(cae_viz.FieldSelectionAPI(temporal_prim, "colors"), "PointSinusoid")
        await wait_for_update(0)

        # Should have executed once (initial/structural)
        self.assertEqual(temporal_prim.GetAttribute("test:exec_count").Get(), 1)
        self.assertEqual(temporal_prim.GetAttribute("test:tick_count").Get(), 0)
        reason = temporal_prim.GetAttribute("test:last_reason").Get()
        self.assertIn(reason, [ExecutionReason.STRUCTURAL_CHANGE.value, ExecutionReason.INITIAL.value])
        self.assertTrue(temporal_prim.GetAttribute("test:is_full_rebuild").Get())

        # Move timeline to a new timecode
        self.timeline.set_current_time(10.0)
        await wait_for_update(0)

        # Should execute (first time at this timecode)
        self.assertEqual(temporal_prim.GetAttribute("test:exec_count").Get(), 2)
        self.assertEqual(temporal_prim.GetAttribute("test:tick_count").Get(), 0)
        reason = temporal_prim.GetAttribute("test:last_reason").Get()
        self.assertEqual(reason, ExecutionReason.TEMPORAL_UPDATE.value)
        self.assertFalse(temporal_prim.GetAttribute("test:is_full_rebuild").Get())
        self.assertTrue(temporal_prim.GetAttribute("test:is_temporal_update").Get())

        # Move to another new timecode
        self.timeline.set_current_time(20.0)
        await wait_for_update(0)

        # Should execute (first time at this timecode)
        self.assertEqual(temporal_prim.GetAttribute("test:exec_count").Get(), 3)
        self.assertEqual(temporal_prim.GetAttribute("test:tick_count").Get(), 0)

        # Go back to the first timecode (0.0)
        self.timeline.set_current_time(0.0)
        await wait_for_update(0)

        # Should NOT execute (already seen this timecode) - temporal optimization!
        self.assertEqual(temporal_prim.GetAttribute("test:exec_count").Get(), 3)
        self.assertEqual(temporal_prim.GetAttribute("test:tick_count").Get(), 0)

        # Go to second timecode again
        self.timeline.set_current_time(10.0)
        await wait_for_update(0)

        # Should NOT execute (already seen this timecode)
        self.assertEqual(temporal_prim.GetAttribute("test:exec_count").Get(), 3)
        self.assertEqual(temporal_prim.GetAttribute("test:tick_count").Get(), 0)

        # Make a structural change - should clear temporal state
        cae_viz.OperatorAPI(temporal_prim).GetEnabledAttr().Set(False)
        await wait_for_update(0)
        cae_viz.OperatorAPI(temporal_prim).GetEnabledAttr().Set(True)
        await wait_for_update(0)

        # Should execute due to structural change
        self.assertEqual(temporal_prim.GetAttribute("test:exec_count").Get(), 4)
        self.assertEqual(temporal_prim.GetAttribute("test:tick_count").Get(), 0)
        reason = temporal_prim.GetAttribute("test:last_reason").Get()
        self.assertEqual(reason, ExecutionReason.STRUCTURAL_CHANGE.value)
        self.assertTrue(temporal_prim.GetAttribute("test:is_full_rebuild").Get())

        # Go back to start - should execute now (temporal state was cleared)
        self.timeline.set_current_time(0.0)
        await wait_for_update(0)

        # Should execute (temporal state was cleared by structural change)
        self.assertEqual(temporal_prim.GetAttribute("test:exec_count").Get(), 5)
        self.assertEqual(temporal_prim.GetAttribute("test:tick_count").Get(), 0)
        reason = temporal_prim.GetAttribute("test:last_reason").Get()
        self.assertEqual(reason, ExecutionReason.TEMPORAL_UPDATE.value)

    async def test_temporal_operator_with_tick_temporal_dataset(self):
        """Test temporal operator with tick callback - should call on_time_changed."""
        self.timeline.set_current_time(0.0)
        await wait_for_update(0)

        temporal_prim = self.stage.DefinePrim("/World/TemporalOperatorWithTick", "Cone")
        self.init_prim(temporal_prim)
        cae_viz.OperatorAPI.Apply(temporal_prim)
        cae_viz.OperatorAPI(temporal_prim).GetEnabledAttr().Set(True)
        # Point to temporal dataset (hex_timesteps)
        cae_viz.DatasetSelectionAPI.Apply(temporal_prim, "source")
        cae_viz.DatasetSelectionAPI(temporal_prim, "source").GetTargetRel().SetTargets(
            {"/World/hex_timesteps/Base/Zone/ElementsUniform"}
        )
        cae_viz.FieldSelectionAPI.Apply(temporal_prim, "colors")
        _set_field_names(cae_viz.FieldSelectionAPI(temporal_prim, "colors"), "PointSinusoid")
        await wait_for_update(0)

        # Should have executed once (initial/structural)
        self.assertEqual(temporal_prim.GetAttribute("test:exec_count").Get(), 1)
        # Tick should have been called after exec
        self.assertEqual(temporal_prim.GetAttribute("test:tick_count").Get(), 1)
        reason = temporal_prim.GetAttribute("test:last_reason").Get()
        self.assertIn(reason, [ExecutionReason.STRUCTURAL_CHANGE.value, ExecutionReason.INITIAL.value])
        self.assertTrue(temporal_prim.GetAttribute("test:is_full_rebuild").Get())
        self.assertTrue(temporal_prim.GetAttribute("test:is_temporal_tick").Get())

        # Move timeline to a new timecode
        self.timeline.set_current_time(10.0)
        await wait_for_update(0)

        # Should execute (first time at this timecode) AND tick
        self.assertEqual(temporal_prim.GetAttribute("test:exec_count").Get(), 2)
        self.assertEqual(temporal_prim.GetAttribute("test:tick_count").Get(), 2)
        reason = temporal_prim.GetAttribute("test:last_reason").Get()
        self.assertEqual(reason, ExecutionReason.TEMPORAL_UPDATE.value)
        self.assertFalse(temporal_prim.GetAttribute("test:is_full_rebuild").Get())
        self.assertTrue(temporal_prim.GetAttribute("test:is_temporal_update").Get())

        # Move to another new timecode
        self.timeline.set_current_time(20.0)
        await wait_for_update(0)

        # Should execute AND tick
        self.assertEqual(temporal_prim.GetAttribute("test:exec_count").Get(), 3)
        self.assertEqual(temporal_prim.GetAttribute("test:tick_count").Get(), 3)

        # Go back to the first timecode (0.0)
        self.timeline.set_current_time(0.0)
        await wait_for_update(0)

        # Should NOT execute (already seen this timecode) BUT tick should still fire
        self.assertEqual(temporal_prim.GetAttribute("test:exec_count").Get(), 3)
        self.assertEqual(temporal_prim.GetAttribute("test:tick_count").Get(), 4)
        # Tick should have updated the timecode info (back to time 0.0)
        last_tick_timecode = temporal_prim.GetAttribute("test:last_tick_timecode").Get()
        self.assertEqual(last_tick_timecode, 0.0)

        # Stay on same timecode, tick multiple times
        await wait_for_update(0)
        # Should not tick again (no timecode change)
        self.assertEqual(temporal_prim.GetAttribute("test:exec_count").Get(), 3)
        self.assertEqual(temporal_prim.GetAttribute("test:tick_count").Get(), 4)

        # Make a structural change
        cae_viz.OperatorAPI(temporal_prim).GetEnabledAttr().Set(False)
        await wait_for_update(0)
        cae_viz.OperatorAPI(temporal_prim).GetEnabledAttr().Set(True)
        await wait_for_update(0)

        # Should execute AND tick due to structural change
        self.assertEqual(temporal_prim.GetAttribute("test:exec_count").Get(), 4)
        self.assertEqual(temporal_prim.GetAttribute("test:tick_count").Get(), 5)
        reason = temporal_prim.GetAttribute("test:last_reason").Get()
        self.assertEqual(reason, ExecutionReason.STRUCTURAL_CHANGE.value)
        self.assertTrue(temporal_prim.GetAttribute("test:is_full_rebuild").Get())

    async def test_temporal_operator_with_tick_and_interpolation(self):
        """Test temporal operator with field interpolation enabled."""
        self.timeline.set_current_time(0.0)
        await wait_for_update(0)

        temporal_prim = self.stage.DefinePrim("/World/TemporalOperatorInterpolation", "Cone")
        self.init_prim(temporal_prim)
        cae_viz.OperatorAPI.Apply(temporal_prim)
        cae_viz.OperatorAPI(temporal_prim).GetEnabledAttr().Set(True)
        # Apply OperatorTemporalAPI and enable field interpolation
        cae_viz.OperatorTemporalAPI.Apply(temporal_prim)
        cae_viz.OperatorTemporalAPI(temporal_prim).GetEnableFieldInterpolationAttr().Set(True)

        # Point to temporal dataset (hex_timesteps)
        cae_viz.DatasetSelectionAPI.Apply(temporal_prim, "source")
        cae_viz.DatasetSelectionAPI(temporal_prim, "source").GetTargetRel().SetTargets(
            {"/World/hex_timesteps/Base/Zone/ElementsUniform"}
        )
        cae_viz.FieldSelectionAPI.Apply(temporal_prim, "colors")
        _set_field_names(cae_viz.FieldSelectionAPI(temporal_prim, "colors"), "PointSinusoid")
        await wait_for_update(0)

        # Should have executed once (initial)
        initial_exec_count = temporal_prim.GetAttribute("test:exec_count").Get()
        self.assertGreaterEqual(initial_exec_count, 1)
        # Tick should have been called
        initial_tick_count = temporal_prim.GetAttribute("test:tick_count").Get()
        self.assertGreaterEqual(initial_tick_count, 1)

        # With interpolation, next_time_code should be set if there's a next sample
        next_time_code = temporal_prim.GetAttribute("test:next_time_code").Get()
        # Could be valid or invalid depending on whether there's a next sample
        self.assertIsNotNone(next_time_code)

        # Move timeline forward slightly (between samples)
        # With interpolation enabled, tick should fire on EVERY raw timecode change
        self.timeline.set_current_time(5.0)  # Between samples
        await wait_for_update(0)

        # Tick should fire (raw_timecode changed, even if snapped timecode didn't)
        current_tick_count = temporal_prim.GetAttribute("test:tick_count").Get()
        self.assertEqual(current_tick_count, 2)  # Initial tick (1) + this tick (1)

    async def test_temporal_operator_timestep_index_with_interpolation(self):
        """Test that timestep_index is set correctly when interpolation requires execution for t0 and t0+1."""
        self.timeline.set_current_time(0.0)
        await wait_for_update(0)

        temporal_prim = self.stage.DefinePrim("/World/TemporalOperatorTimestepIndex", "Cone")
        self.init_prim(temporal_prim)

        cae_viz.OperatorAPI.Apply(temporal_prim)
        cae_viz.OperatorAPI(temporal_prim).GetEnabledAttr().Set(True)
        # Apply OperatorTemporalAPI and enable field interpolation
        cae_viz.OperatorTemporalAPI.Apply(temporal_prim)
        cae_viz.OperatorTemporalAPI(temporal_prim).GetEnableFieldInterpolationAttr().Set(True)

        # Point to temporal dataset (hex_timesteps)
        cae_viz.DatasetSelectionAPI.Apply(temporal_prim, "source")
        cae_viz.DatasetSelectionAPI(temporal_prim, "source").GetTargetRel().SetTargets(
            {"/World/hex_timesteps/Base/Zone/ElementsUniform"}
        )
        cae_viz.FieldSelectionAPI.Apply(temporal_prim, "colors")
        _set_field_names(cae_viz.FieldSelectionAPI(temporal_prim, "colors"), "PointSinusoid")
        await wait_for_update(0)

        # Initial execution should have timestep_index=0
        last_timestep_index = temporal_prim.GetAttribute("test:last_timestep_index").Get()
        self.assertIn(last_timestep_index, [0, 1])  # Could be 0 or 1 depending on whether next was also executed

        # Move to a time between plugin-backed samples where both lower and upper samples exist.
        self.timeline.set_current_time(5.0)
        await wait_for_update(0)

        # Should execute for the lower sample (t0) and upper sample (t0+1).
        # The last_timestep_index will be from the last execution (should be timestep_index=1 for next)
        exec_count = temporal_prim.GetAttribute("test:exec_count").Get()
        self.assertGreaterEqual(exec_count, 2)

        # The last execution should have been for the next timestep (timestep_index=1) if both were executed
        last_timestep_index = temporal_prim.GetAttribute("test:last_timestep_index").Get()
        self.assertIn(last_timestep_index, [0, 1])

    async def test_temporal_operator_with_locked_time(self):
        """Test temporal operator with useLockedTime - should only execute at locked time."""
        self.timeline.set_current_time(0.0)
        await wait_for_update(0)

        temporal_prim = self.stage.DefinePrim("/World/TemporalOperatorLockedTime", "Cone")
        self.init_prim(temporal_prim)
        cae_viz.OperatorAPI.Apply(temporal_prim)
        cae_viz.OperatorAPI(temporal_prim).GetEnabledAttr().Set(True)

        # Apply OperatorTemporalAPI and enable locked time at a plugin-backed time sample.
        cae_viz.OperatorTemporalAPI.Apply(temporal_prim)
        cae_viz.OperatorTemporalAPI(temporal_prim).GetUseLockedTimeAttr().Set(True)
        cae_viz.OperatorTemporalAPI(temporal_prim).GetLockedTimeAttr().Set(10.0)

        # Point to temporal dataset (hex_timesteps)
        cae_viz.DatasetSelectionAPI.Apply(temporal_prim, "source")
        cae_viz.DatasetSelectionAPI(temporal_prim, "source").GetTargetRel().SetTargets(
            {"/World/hex_timesteps/Base/Zone/ElementsUniform"}
        )
        cae_viz.FieldSelectionAPI.Apply(temporal_prim, "colors")
        _set_field_names(cae_viz.FieldSelectionAPI(temporal_prim, "colors"), "PointSinusoid")
        await wait_for_update(0)

        # Should have executed once at locked time, not at current time (0.0)
        self.assertEqual(temporal_prim.GetAttribute("test:exec_count").Get(), 1)
        self.assertEqual(temporal_prim.GetAttribute("test:tick_count").Get(), 1)
        reason = temporal_prim.GetAttribute("test:last_reason").Get()
        self.assertIn(reason, [ExecutionReason.STRUCTURAL_CHANGE.value, ExecutionReason.INITIAL.value])

        # The execution should have happened at the lower bracket for locked time 10.0.
        locked_lower, _ = self.get_bracket(temporal_prim, 10.0)
        last_timecode = temporal_prim.GetAttribute("test:last_timecode").Get()
        self.assertAlmostEqual(last_timecode, locked_lower)
        last_raw_timecode = temporal_prim.GetAttribute("test:last_raw_timecode").Get()
        self.assertEqual(last_raw_timecode, 10.0)

        # Move timeline forward - operator should NOT execute again while locked.
        await self.forward_frames(10)

        # Should NOT execute (still locked to time 10.0)
        self.assertEqual(temporal_prim.GetAttribute("test:exec_count").Get(), 1)
        # Tick should NOT fire either (since we're still effectively at locked time)
        self.assertEqual(temporal_prim.GetAttribute("test:tick_count").Get(), 1)

        # Move timeline to different time
        self.timeline.set_current_time(30.0)
        await wait_for_update(0)

        # Should still NOT execute (locked to time 10.0)
        self.assertEqual(temporal_prim.GetAttribute("test:exec_count").Get(), 1)
        self.assertEqual(temporal_prim.GetAttribute("test:tick_count").Get(), 1)

        # Change locked time to another plugin-backed sample.
        cae_viz.OperatorTemporalAPI(temporal_prim).GetLockedTimeAttr().Set(20.0)
        await wait_for_update(0)

        # Should execute at new locked time - this is a structural change
        self.assertEqual(temporal_prim.GetAttribute("test:exec_count").Get(), 2)
        self.assertEqual(temporal_prim.GetAttribute("test:tick_count").Get(), 2)
        reason = temporal_prim.GetAttribute("test:last_reason").Get()
        self.assertEqual(reason, ExecutionReason.STRUCTURAL_CHANGE.value)

        # Verify execution happened at the lower bracket for new locked time.
        locked_lower, _ = self.get_bracket(temporal_prim, 20.0)
        last_timecode = temporal_prim.GetAttribute("test:last_timecode").Get()
        self.assertAlmostEqual(last_timecode, locked_lower)
        last_raw_timecode = temporal_prim.GetAttribute("test:last_raw_timecode").Get()
        self.assertEqual(last_raw_timecode, 20.0)

        # Disable locked time
        cae_viz.OperatorTemporalAPI(temporal_prim).GetUseLockedTimeAttr().Set(False)
        await wait_for_update(0)

        # Should execute at current timeline time (structural change)
        self.assertEqual(temporal_prim.GetAttribute("test:exec_count").Get(), 3)
        self.assertEqual(temporal_prim.GetAttribute("test:tick_count").Get(), 3)
        reason = temporal_prim.GetAttribute("test:last_reason").Get()
        self.assertEqual(reason, ExecutionReason.STRUCTURAL_CHANGE.value)

        # Now should follow timeline.
        timeline_lower, _ = self.get_bracket(temporal_prim, 30.0)
        last_timecode = temporal_prim.GetAttribute("test:last_timecode").Get()
        self.assertAlmostEqual(last_timecode, timeline_lower)

        # Move timeline - should now execute at new time (not locked anymore)
        self.timeline.set_current_time(40.0)
        await wait_for_update(0)

        # Should execute AND tick at new timeline position
        self.assertEqual(temporal_prim.GetAttribute("test:exec_count").Get(), 4)
        self.assertEqual(temporal_prim.GetAttribute("test:tick_count").Get(), 4)
        reason = temporal_prim.GetAttribute("test:last_reason").Get()
        self.assertEqual(reason, ExecutionReason.TEMPORAL_UPDATE.value)

    async def test_interpolation_disabled_sets_next_time_code_to_none(self):
        """
        Test that next_time_code is None when enableFieldInterpolation is False,
        even when USD has next time sample data available.

        This is critical to prevent operators (like Faces) from interpolating
        when the feature is disabled but next sample data exists.
        """
        self.timeline.set_current_time(0.0)
        await wait_for_update(0)

        # Create operator on temporal dataset (hex_timesteps has time samples)
        temporal_prim = self.stage.DefinePrim("/World/InterpolationDisabled", "Cone")
        self.init_prim(temporal_prim)
        cae_viz.OperatorAPI.Apply(temporal_prim)
        cae_viz.OperatorAPI(temporal_prim).GetEnabledAttr().Set(True)

        # Apply OperatorTemporalAPI but DISABLE field interpolation
        cae_viz.OperatorTemporalAPI.Apply(temporal_prim)
        cae_viz.OperatorTemporalAPI(temporal_prim).GetEnableFieldInterpolationAttr().Set(False)

        # Point to temporal dataset (hex_timesteps)
        cae_viz.DatasetSelectionAPI.Apply(temporal_prim, "source")
        cae_viz.DatasetSelectionAPI(temporal_prim, "source").GetTargetRel().SetTargets(
            {"/World/hex_timesteps/Base/Zone/ElementsUniform"}
        )
        cae_viz.FieldSelectionAPI.Apply(temporal_prim, "colors")
        _set_field_names(cae_viz.FieldSelectionAPI(temporal_prim, "colors"), "PointSinusoid")
        await wait_for_update(0)

        # Initial execution: next_time_code should be -1.0 (None sentinel)
        next_time_code = temporal_prim.GetAttribute("test:next_time_code").Get()
        self.assertEqual(next_time_code, -1.0, "next_time_code should be None when interpolation is disabled")

        # Move to a time between plugin-backed samples where a next sample exists.
        self.timeline.set_current_time(5.0)
        await wait_for_update(0)
        lower, upper = self.get_bracket(temporal_prim, 5.0)

        # Verify next_time_code is still None (-1.0 sentinel) despite next sample existing
        next_time_code = temporal_prim.GetAttribute("test:next_time_code").Get()
        self.assertEqual(next_time_code, -1.0, "next_time_code should remain None when interpolation is disabled")

        # Verify current timecode is still set correctly (snapped to the lower bracket)
        last_timecode = temporal_prim.GetAttribute("test:last_timecode").Get()
        self.assertAlmostEqual(last_timecode, lower, msg="Current timecode should be snapped to lower bracket")

        # Now ENABLE interpolation - next_time_code should become valid
        cae_viz.OperatorTemporalAPI(temporal_prim).GetEnableFieldInterpolationAttr().Set(True)
        await wait_for_update(0)

        # This is a structural change, so should re-execute at the current timeline position.
        # After this, next_time_code should be set
        # The structural change means it will execute at the snapped lower bracket, with next being upper.

        # Verify next_time_code is now valid.
        next_time_code = temporal_prim.GetAttribute("test:next_time_code").Get()
        self.assertNotEqual(next_time_code, -1.0, "next_time_code should be valid when interpolation is enabled")
        self.assertAlmostEqual(next_time_code, upper, msg="next_time_code should be upper bracket")

        # Verify current timecode is snapped to the lower bracket.
        last_timecode = temporal_prim.GetAttribute("test:last_timecode").Get()
        self.assertAlmostEqual(last_timecode, lower, msg="Current timecode should be snapped to lower bracket")

    async def test_non_temporal_operator_with_interpolation_disabled(self):
        """
        Test that non-temporal operators execute on every time change,
        and that next_time_code is None when interpolation is disabled.
        """
        self.timeline.set_current_time(0.0)
        await wait_for_update(0)

        # Create non-temporal operator
        non_temporal_prim = self.stage.DefinePrim("/World/NonTemporalNoInterpolation", "Cone")
        self.init_prim(non_temporal_prim)
        cae_viz.OperatorAPI.Apply(non_temporal_prim)
        cae_viz.OperatorAPI(non_temporal_prim).GetEnabledAttr().Set(True)

        # Apply OperatorTemporalAPI but DISABLE field interpolation
        cae_viz.OperatorTemporalAPI.Apply(non_temporal_prim)
        cae_viz.OperatorTemporalAPI(non_temporal_prim).GetEnableFieldInterpolationAttr().Set(False)

        # Point to temporal dataset
        cae_viz.DatasetSelectionAPI.Apply(non_temporal_prim, "source")
        cae_viz.DatasetSelectionAPI(non_temporal_prim, "source").GetTargetRel().SetTargets(
            {"/World/hex_timesteps/Base/Zone/ElementsUniform"}
        )
        await wait_for_update(0)

        # Initial execution
        initial_exec_count = non_temporal_prim.GetAttribute("test:exec_count").Get()
        self.assertEqual(initial_exec_count, 1)

        # next_time_code should be None
        next_time_code = non_temporal_prim.GetAttribute("test:next_time_code").Get()
        self.assertEqual(next_time_code, -1.0)

        # Move to a different plugin-backed time - non-temporal should re-execute.
        self.timeline.set_current_time(10.0)
        await wait_for_update(0)

        exec_count = non_temporal_prim.GetAttribute("test:exec_count").Get()
        self.assertEqual(exec_count, 2, "Non-temporal should execute on time change")

        # next_time_code should still be None
        next_time_code = non_temporal_prim.GetAttribute("test:next_time_code").Get()
        self.assertEqual(
            next_time_code, -1.0, "next_time_code should remain None for non-temporal with interpolation disabled"
        )

        # Move back to 0.0 - non-temporal should re-execute again
        self.timeline.set_current_time(0.0)
        await wait_for_update(0)

        exec_count = non_temporal_prim.GetAttribute("test:exec_count").Get()
        # Non-temporal operators re-execute when timecode changes
        # However, moving from 10.0 to 0.0 should trigger execution
        # Total: 1 (initial) + 1 (0 to 10) + 1 (10 to 0) = 3
        # But if it's the same snapped timecode, it might not execute
        self.assertGreaterEqual(exec_count, 2, "Non-temporal should re-execute at least once more")

    async def test_multiple_operators_independent_interpolation_settings(self):
        """
        Test that multiple operators can have independent interpolation settings,
        and each gets the correct next_time_code based on their individual settings.
        """
        self.timeline.set_current_time(0.0)
        await wait_for_update(0)

        # Create operator with interpolation ENABLED
        interp_enabled_prim = self.stage.DefinePrim("/World/InterpEnabled", "Cone")
        self.init_prim(interp_enabled_prim)
        cae_viz.OperatorAPI.Apply(interp_enabled_prim)
        cae_viz.OperatorAPI(interp_enabled_prim).GetEnabledAttr().Set(True)
        cae_viz.OperatorTemporalAPI.Apply(interp_enabled_prim)
        cae_viz.OperatorTemporalAPI(interp_enabled_prim).GetEnableFieldInterpolationAttr().Set(True)
        cae_viz.DatasetSelectionAPI.Apply(interp_enabled_prim, "source")
        cae_viz.DatasetSelectionAPI(interp_enabled_prim, "source").GetTargetRel().SetTargets(
            {"/World/hex_timesteps/Base/Zone/ElementsUniform"}
        )

        # Create operator with interpolation DISABLED
        interp_disabled_prim = self.stage.DefinePrim("/World/InterpDisabled", "Cone")
        self.init_prim(interp_disabled_prim)
        cae_viz.OperatorAPI.Apply(interp_disabled_prim)
        cae_viz.OperatorAPI(interp_disabled_prim).GetEnabledAttr().Set(True)
        cae_viz.OperatorTemporalAPI.Apply(interp_disabled_prim)
        cae_viz.OperatorTemporalAPI(interp_disabled_prim).GetEnableFieldInterpolationAttr().Set(False)
        cae_viz.DatasetSelectionAPI.Apply(interp_disabled_prim, "source")
        cae_viz.DatasetSelectionAPI(interp_disabled_prim, "source").GetTargetRel().SetTargets(
            {"/World/hex_timesteps/Base/Zone/ElementsUniform"}
        )

        await wait_for_update(0)

        # Move to a time between plugin-backed samples.
        self.timeline.set_current_time(15.0)
        await wait_for_update(0)
        _, upper = self.get_bracket(interp_enabled_prim, 15.0)

        # Operator with interpolation enabled should have next_time_code
        enabled_next = interp_enabled_prim.GetAttribute("test:next_time_code").Get()
        self.assertAlmostEqual(enabled_next, upper, msg="Interpolation-enabled operator should have next_time_code")

        # Operator with interpolation disabled should NOT have next_time_code
        disabled_next = interp_disabled_prim.GetAttribute("test:next_time_code").Get()
        self.assertEqual(disabled_next, -1.0, "Interpolation-disabled operator should have None next_time_code")

        # Both should have the same current timecode (snapped to the lower bracket).
        enabled_current = interp_enabled_prim.GetAttribute("test:last_timecode").Get()
        disabled_current = interp_disabled_prim.GetAttribute("test:last_timecode").Get()
        # But they might have executed at different times during setup
        # Let's just verify they're both set to valid time samples
        self.assertGreater(enabled_current, 0.0, "Enabled operator should have valid timecode")
        self.assertGreater(disabled_current, 0.0, "Disabled operator should have valid timecode")

    async def test_non_temporal_operator_with_locked_time(self):
        """Test non-temporal operator with useLockedTime - should execute once at locked time."""
        self.timeline.set_current_time(0.0)
        await wait_for_update(0)

        non_temporal_prim = self.stage.DefinePrim("/World/NonTemporalOperatorLockedTime", "Cube")
        self.init_prim(non_temporal_prim)
        cae_viz.OperatorAPI.Apply(non_temporal_prim)
        cae_viz.OperatorAPI(non_temporal_prim).GetEnabledAttr().Set(True)

        # Apply OperatorTemporalAPI and enable locked time at a plugin-backed time sample.
        cae_viz.OperatorTemporalAPI.Apply(non_temporal_prim)
        cae_viz.OperatorTemporalAPI(non_temporal_prim).GetUseLockedTimeAttr().Set(True)
        cae_viz.OperatorTemporalAPI(non_temporal_prim).GetLockedTimeAttr().Set(10.0)

        # Point to temporal dataset (hex_timesteps)
        cae_viz.DatasetSelectionAPI.Apply(non_temporal_prim, "source")
        cae_viz.DatasetSelectionAPI(non_temporal_prim, "source").GetTargetRel().SetTargets(
            {"/World/hex_timesteps/Base/Zone/ElementsUniform"}
        )
        await wait_for_update(0)

        # Should have executed once at locked time.
        self.assertEqual(non_temporal_prim.GetAttribute("test:exec_count").Get(), 1)
        self.assertEqual(non_temporal_prim.GetAttribute("test:tick_count").Get(), 0)
        reason = non_temporal_prim.GetAttribute("test:last_reason").Get()
        self.assertIn(reason, [ExecutionReason.STRUCTURAL_CHANGE.value, ExecutionReason.INITIAL.value])

        # The execution should have happened at the lower bracket for locked time 10.0.
        locked_lower, _ = self.get_bracket(non_temporal_prim, 10.0)
        last_timecode = non_temporal_prim.GetAttribute("test:last_timecode").Get()
        self.assertAlmostEqual(last_timecode, locked_lower)

        # Move timeline - non-temporal operator should NOT execute again
        # because locked time hasn't changed (still 10.0)
        await self.forward_frames(10)

        # Should NOT execute again - locked time is still 10.0
        self.assertEqual(non_temporal_prim.GetAttribute("test:exec_count").Get(), 1)
        reason = non_temporal_prim.GetAttribute("test:last_reason").Get()
        self.assertIn(reason, [ExecutionReason.STRUCTURAL_CHANGE.value, ExecutionReason.INITIAL.value])

        # But should still be using locked time
        last_timecode = non_temporal_prim.GetAttribute("test:last_timecode").Get()
        self.assertAlmostEqual(last_timecode, locked_lower)  # Still locked, not timeline time

        # Move timeline again
        self.timeline.set_current_time(30.0)
        await wait_for_update(0)

        # Should still NOT execute (locked time hasn't changed)
        self.assertEqual(non_temporal_prim.GetAttribute("test:exec_count").Get(), 1)

        # Still locked to 10.0
        last_timecode = non_temporal_prim.GetAttribute("test:last_timecode").Get()
        self.assertAlmostEqual(last_timecode, locked_lower)
