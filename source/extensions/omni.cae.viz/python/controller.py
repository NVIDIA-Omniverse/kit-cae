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
Controller for managing CAE visualization operators.

The controller is responsible for managing operator lifecycle, tracking changes,
and executing operators based on USD stage state.
"""

__all__ = ["Controller"]

import logging
import weakref
from typing import Any, ClassVar, Iterator

import dav
import omni.kit.app
import omni.timeline
import omni.usd
from omni.cae.data import usd_utils
from omni.cae.schema import viz as cae_viz
from pxr import Usd, UsdGeom, UsdUtils
from usdrt import Rt
from usdrt import Usd as UsdRT
from usdrt import UsdGeom as UsdGeomRT

from . import settings
from .change_tracker import ChangeTracker
from .execution_context import ExecutionContext, ExecutionReason
from .operator import get_operators

logger = logging.getLogger(__name__)

# Notification event names (base name — queue_event also dispatches "<name>:immediate")
EVT_SYNC_BEGIN = "omni.cae.viz@sync_begin"
EVT_SYNC_END = "omni.cae.viz@sync_end"
EVT_OPERATOR_BEGIN = "omni.cae.viz@operator_begin"
EVT_OPERATOR_END = "omni.cae.viz@operator_end"


class Controller:
    """
    Controller that manages CAE visualization operators based on the USD stage.

    The controller monitors prims with CAE schemas and executes appropriate
    visualization operators when changes are detected.

    Parameters
    ----------
    stageId : int
        The stage ID from the stage cache

    Attributes
    ----------
    _stage : Usd.Stage
        The USD stage being monitored
    _tracker : ChangeTracker
        Change tracker for monitoring schema property changes
    """

    _schema_regexs: ClassVar[list[str]] = [r"^Cae"]
    _instances: ClassVar[weakref.WeakValueDictionary] = weakref.WeakValueDictionary()

    def __new__(cls, stageId: int):
        existing = cls._instances.get(stageId)
        if existing is not None:
            return existing
        instance = super().__new__(cls)
        cls._instances[stageId] = instance
        return instance

    @classmethod
    def add_schema_regex(cls, regex: str) -> None:
        """Add a schema name regex pattern to track."""
        if regex not in cls._schema_regexs:
            cls._schema_regexs.append(regex)

    @classmethod
    def remove_schema_regex(cls, regex: str) -> None:
        """Remove a schema name regex pattern from tracking."""
        cls._schema_regexs = [r for r in cls._schema_regexs if r != regex]

    @classmethod
    async def sync_active_controller(cls) -> bool:
        """
        Convenience method to sync the controller for the currently active stage.

        Looks up the active stage ID, finds the corresponding Controller instance
        (if one exists), and calls sync() on it.

        Returns
        -------
        bool
            True if any operators were executed, False if no controller was found
            or no operators ran.
        """
        stage_id = omni.usd.get_context().get_stage_id()
        controller = cls._instances.get(stage_id)
        if controller is None:
            return False

        timeline = omni.timeline.get_timeline_interface()
        timecode = Usd.TimeCode(timeline.get_current_time() * timeline.get_time_codes_per_second())
        return await controller.sync(timecode)

    def __init__(self, stageId: int):
        """
        Initialize the controller for a specific stage.

        Parameters
        ----------
        stageId : int
            The stage ID from UsdUtils.StageCache
        """
        if hasattr(self, "_stage"):
            return  # Already initialized for this stageId

        # Get the stage from the cache
        cache = UsdUtils.StageCache.Get()
        stage_cache_id = cache.Id.FromLongInt(stageId)
        self._stage: Usd.Stage = cache.Find(stage_cache_id)
        self._stage_rt: UsdRT.Stage = UsdRT.Stage.Attach(stageId)
        self._xform_change_tracker = Rt.ChangeTracker(self._stage_rt)
        self._xform_change_tracker.TrackAttribute("omni:fabric:localMatrix")
        self._xform_change_tracker.TrackAttribute("omni:fabric:worldMatrix")
        self._last_execution_cache = {}

        if not self._stage:
            raise RuntimeError(f"Could not find stage with ID {stageId} in cache")

        # Create change tracker for this stage
        self._tracker = ChangeTracker(self._stage, schema_regexs=list(self._schema_regexs), track_time_changes=True)

        logger.info("Controller initialized for stage ID %s", stageId)

    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, "_tracker") and self._tracker:
            self._tracker.disable()

    async def sync(self, timecode: Usd.TimeCode) -> bool:
        """
        Synchronize the controller state with the USD stage.

        This method is called periodically to check for changes and execute
        operators as needed. Currently, this is a placeholder for future
        implementation.

        Parameters
        ----------
        timecode : Usd.TimeCode
            The current time code value

        Returns
        -------
        bool
            True if any operators were executed, False otherwise
        """
        # Find all prims with the CaeVizOperatorAPI schema
        assert self._stage_rt is not None
        operator_prim_paths = self._stage_rt.GetPrimsWithAppliedAPIName("CaeVizOperatorAPI")
        updated = False

        omni.kit.app.queue_event(EVT_SYNC_BEGIN, {"timecode": timecode.GetValue()})
        try:
            executed_paths = set()
            pass_count = 0
            # to support the use-case where operator A's execution triggers a change that causes operator B to become dirty and need execution
            # we loop through all operators up to 3 times or until no new operators are executed, whichever comes first.
            # this is necessary since we flush all changes after a sync cycle, so if we don't execute B in the same sync cycle as A,
            # then we may miss the change to B entirely!
            # the 3 limit along with the executed_paths set is a simple way to prevent infinite loops in case of operator
            # cycles (e.g. A triggers B which triggers A which triggers B...)
            while pass_count < 3 and operator_prim_paths:
                updated_this_pass = False
                for operator_prim_path in operator_prim_paths:
                    prim = self._stage.GetPrimAtPath(str(operator_prim_path))
                    cae_operator_api = cae_viz.OperatorAPI(prim)
                    if not cae_operator_api.GetEnabledAttr().Get(timecode):
                        continue

                    if str(prim.GetPath()) in executed_paths:
                        # skip already executed prims.
                        continue

                    device = cae_operator_api.GetDeviceAttr().Get()
                    if device == cae_viz.Tokens.auto_:
                        device = settings.get_default_device_for_auto()

                    if await self._execute_operator(prim, timecode, device):
                        updated = True
                        updated_this_pass = True
                        executed_paths.add(str(prim.GetPath()))

                if not updated_this_pass:
                    # If no operators were executed in this pass, we can stop early
                    break
                pass_count += 1
        finally:
            omni.kit.app.queue_event(EVT_SYNC_END, {"timecode": timecode.GetValue(), "updated": updated})

        # clear all changes post a sync cycle. Even if an operator fails,
        # unless user changes something again, we don't want to keep retrying since
        # the same error will likely occur again. So clearing all changes here
        # makes sense.
        self._tracker.clear_all_changes()
        self._xform_change_tracker.ClearChanges()
        return updated

    async def _execute_operator(self, prim: Usd.Prim, timecode: Usd.TimeCode, device: str):
        """
        Execute the operator for the given prim at the given timecode.
        """
        if not prim or not prim.IsActive():
            return False

        if (
            prim.IsA(UsdGeom.Imageable)
            and UsdGeom.Imageable(prim).ComputeVisibility(timecode) == UsdGeom.Tokens.invisible
        ):
            # if this prim is invisible, check if any of its dependents are visible.
            # This perhaps should be done transitively, but until we need it, let's not do it.
            should_execute = False
            if prim.HasAPI(cae_viz.OperatorDependenciesAPI):
                operator_dependencies_api = cae_viz.OperatorDependenciesAPI(prim)
                dependents = operator_dependencies_api.GetDependentsRel().GetForwardedTargets()
                for dependent in dependents:
                    dependent_prim = self._stage.GetPrimAtPath(str(dependent))
                    if (
                        dependent_prim
                        and dependent_prim.IsActive()
                        and UsdGeom.Imageable(dependent_prim).ComputeVisibility(timecode) != UsdGeom.Tokens.invisible
                    ):
                        should_execute = True
                        break
            if not should_execute:
                return False

        operator_class = self._get_operator_class(prim)
        if not operator_class:
            logger.info(f"No operator found for prim {prim.GetPath()}")
            return False

        prim_path_str = str(prim.GetPath())
        if prim_path_str not in self._last_execution_cache:
            self._last_execution_cache[prim_path_str] = {
                "timecode": Usd.TimeCode.EarliestTime(),
                "raw_timecode": None,  # Track raw timecode for interpolation
                "device": None,
                "operator_class": None,
                "temporal_state": None,
            }

        last_execution = self._last_execution_cache[prim_path_str]

        # Build execution contexts (generator yields one or more contexts)
        # Pass raw timeline timecode - will be snapped inside
        execution_contexts = list(self._build_execution_context(prim, operator_class, timecode, device, last_execution))

        if not execution_contexts:
            return False  # Skip execution

        dav_enable_timing = None
        if prim.HasAPI(cae_viz.OperatorDebuggingAPI):
            operator_debugging_api = cae_viz.OperatorDebuggingAPI(prim)
            if operator_debugging_api.GetEnableTimingAttr().Get():
                dav_enable_timing = True
            else:
                dav_enable_timing = False

        operator_instance = operator_class()

        if dav_enable_timing is not None:
            dav.config.enable_timing = dav_enable_timing

        last_successful = False

        # Execute each yielded context
        omni.kit.app.queue_event(EVT_OPERATOR_BEGIN, {"prim_path": prim_path_str})
        try:
            for execution_context in execution_contexts:
                try:
                    logger.info(
                        "Executing operator %s for prim %s at %s [device=%s, reason=%s]",
                        operator_class.__name__,
                        prim.GetPath(),
                        execution_context.timecode,
                        device,
                        execution_context.reason.value,
                    )

                    last_successful = False

                    # Check if this is a temporal tick (minimal update on repeat timecode)
                    if execution_context.reason == ExecutionReason.TEMPORAL_TICK:
                        # Call lightweight hook if available
                        if hasattr(operator_instance, "on_time_changed"):
                            await operator_instance.on_time_changed(prim, device, execution_context)
                        else:
                            logger.warning(
                                f"Operator {operator_class.__name__} has tick_on_time_change=True "
                                f"but does not implement on_time_changed() method"
                            )
                    else:
                        # Normal execution
                        await operator_instance.exec(prim, device, execution_context)

                    last_successful = True

                except NotImplementedError as e:
                    # Log the original error as warning.
                    # We don't log exc_info to avoid cluttering the output which makes it easier to miss the actual
                    # error. NotImplementedError is a reasonable exception since everything is not fully implemented in
                    # the sample.
                    logger.exception(f"{operator_class.__name__}: {e}", exc_info=False)
                    # Don't continue with remaining contexts after error
                    break

                except usd_utils.QuietableException as e:
                    # Log the original error as warning.
                    logger.warning(f"{operator_class.__name__}: {e}")
                    # Don't continue with remaining contexts after error
                    break

                except Exception as e:
                    # Log the original error based on its type
                    logger.exception(f"Error executing operator {operator_class.__name__}: {e}", exc_info=True)
                    # Don't continue with remaining contexts after error
                    break

                finally:
                    # Update cache after each context (successful or not)
                    self._last_execution_cache[prim_path_str] = {
                        "timecode": execution_context.timecode,
                        "raw_timecode": execution_context.raw_timecode,  # Track raw timecode for interpolation
                        "device": device,
                        "operator_class": operator_class,
                        "temporal_state": last_execution["temporal_state"],  # Preserve temporal state
                    }
        finally:
            omni.kit.app.queue_event(EVT_OPERATOR_END, {"prim_path": prim_path_str, "success": last_successful})

        # Cleanup after all contexts processed
        if dav_enable_timing is not None:
            dav.config.enable_timing = False

        if last_successful:
            # if operator succeeded, set prim visibility to inherited
            logger.info("post-exec: %s", operator_instance.__class__.__name__)
        else:
            # Deactivate the operator on error
            try:
                operator_instance.deactivate(prim) if hasattr(operator_instance, "deactivate") else None
            except Exception as deactivate_error:
                logger.warning(f"Error deactivating operator {operator_class.__name__}: {deactivate_error}")

        # Set prim visibility to inherited if operator succeeded, otherwise invisible
        if prim and prim.IsA(UsdGeom.Imageable):
            prim_rt = UsdGeomRT.Imageable(usd_utils.get_prim_rt(prim))
            prim_rt.CreateVisibilityAttr().Set(
                UsdGeomRT.Tokens.inherited if last_successful else UsdGeomRT.Tokens.invisible
            )

        del operator_instance
        self._tracker.clear_changes(prim)
        return last_successful

    def _build_execution_context(
        self,
        prim: Usd.Prim,
        operator_class: Any,
        raw_timecode: Usd.TimeCode,  # Original timeline timecode
        device: str,
        last_execution: dict,
    ) -> Iterator[ExecutionContext]:
        """
        Build execution context(s) and determine if execution should proceed.

        This generator yields one or more ExecutionContext objects based on temporal
        mode and interpolation settings. When interpolation is enabled, it may yield
        contexts for current and next timecodes, followed by a TEMPORAL_TICK context.

        Parameters
        ----------
        prim : Usd.Prim
            The prim to execute the operator on
        operator_class : Type
            The operator class to execute
        raw_timecode : Usd.TimeCode
            The original timeline timecode
        device : str
            The device to execute on
        last_execution : dict
            The last execution cache entry for this prim

        Yields
        ------
        ExecutionContext
            One or more execution contexts to process
        """
        # Check if useLockedTime is enabled
        use_locked_time = False
        locked_time_value = None
        if prim.HasAPI(cae_viz.OperatorTemporalAPI):
            temporal_api = cae_viz.OperatorTemporalAPI(prim)
            use_locked_time = temporal_api.GetUseLockedTimeAttr().Get()
            if use_locked_time:
                locked_time_value = temporal_api.GetLockedTimeAttr().Get()

            # If useLockedTime is enabled, override the timecode with lockedTime
            if use_locked_time and locked_time_value is not None:
                logger.debug(f"Using locked time {locked_time_value} for {prim.GetPath()}")
                raw_timecode = Usd.TimeCode(locked_time_value)

        # Get bracketing timecodes for interpolation support
        lower, upper, has_time_samples = usd_utils.get_bracketing_time_samples_for_prim(prim, raw_timecode.GetValue())
        if has_time_samples:
            current_time_code = Usd.TimeCode(lower)
            # Only set next_time_code if lower and upper differ (for interpolation)
            next_time_code = Usd.TimeCode(upper) if lower != upper else None
        else:
            current_time_code = Usd.TimeCode.EarliestTime()
            next_time_code = None

        # Check basic conditions that force execution
        structural_change = (
            last_execution["operator_class"] != operator_class
            or last_execution["device"] != device
            or self._has_data_selection_transform_changed(prim)
        )
        structural_change = structural_change or self._tracker.prim_changed(
            prim
        )  # if prim change for non-temporal reasons.

        supports_temporal = getattr(operator_class, "__supports_temporal__", False)
        logger.debug(f"[TEMPORAL DEBUG] {prim.GetPath()}: supports_temporal={supports_temporal}")

        enable_field_interpolation = (
            prim.HasAPI(cae_viz.OperatorTemporalAPI)
            and cae_viz.OperatorTemporalAPI(prim).GetEnableFieldInterpolationAttr().Get()
        )
        if not enable_field_interpolation:
            # if field interpolation is not enabled, we don't need to process next time code
            next_time_code = None

        executed = False

        if structural_change:
            last_execution["temporal_state"] = {"executed_timecodes": {current_time_code.GetValue()}}
            yield ExecutionContext(
                reason=ExecutionReason.STRUCTURAL_CHANGE,
                timecode=current_time_code,
                raw_timecode=raw_timecode,
                next_time_code=next_time_code,
                device=device,
                timestep_index=0,
            )
            executed = True
        elif supports_temporal:
            # Temporal operators: only execute if timecode hasn't been seen before
            temporal_state = last_execution["temporal_state"]
            assert temporal_state is not None

            timecode_to_execute = []

            if current_time_code.GetValue() not in temporal_state["executed_timecodes"]:
                timecode_to_execute.append((0, current_time_code))  # timestep_index=0 for t0

            if enable_field_interpolation and next_time_code:
                if next_time_code.GetValue() not in temporal_state["executed_timecodes"]:
                    timecode_to_execute.append((1, next_time_code))  # timestep_index=1 for t0+1

            for timestep_idx, timecode in timecode_to_execute:
                temporal_state["executed_timecodes"].add(timecode.GetValue())
                yield ExecutionContext(
                    reason=ExecutionReason.TEMPORAL_UPDATE,
                    timecode=timecode,
                    raw_timecode=raw_timecode,
                    next_time_code=next_time_code,
                    device=device,
                    timestep_index=timestep_idx,
                )
                executed = True
        else:
            # Non-temporal operators: always execute on every time change
            temporal_state = last_execution["temporal_state"]
            assert temporal_state is not None
            if current_time_code.GetValue() not in temporal_state["executed_timecodes"]:
                temporal_state["executed_timecodes"].clear()
                temporal_state["executed_timecodes"].add(current_time_code.GetValue())
                yield ExecutionContext(
                    reason=ExecutionReason.TEMPORAL_UPDATE,  # Still a time-based update
                    timecode=current_time_code,
                    raw_timecode=raw_timecode,
                    next_time_code=next_time_code,
                    device=device,
                    timestep_index=0,
                )
                executed = True

        # Check if we need to yield tick
        tick_on_time_change = getattr(operator_class, "__tick_on_time_change__", False)

        if supports_temporal and tick_on_time_change:
            # Determine if tick should fire
            if enable_field_interpolation:
                # With interpolation: check if raw_timecode changed
                last_raw_tc = last_execution.get("raw_timecode", None)
                needs_tick = executed or (last_raw_tc is not None and raw_timecode.GetValue() != last_raw_tc.GetValue())
            else:
                # Without interpolation: tick if we executed OR if current timecode changed
                needs_tick = executed or (current_time_code.GetValue() != last_execution["timecode"].GetValue())

            if needs_tick:
                logger.debug(f"Temporal tick for {prim.GetPath()} at raw={raw_timecode}, snapped={current_time_code}")
                yield ExecutionContext(
                    reason=ExecutionReason.TEMPORAL_TICK,
                    timecode=current_time_code,
                    raw_timecode=raw_timecode,
                    next_time_code=next_time_code,
                    device=device,
                    timestep_index=0,  # Tick always uses timestep_index=0 (current time)
                )

    def _get_operator_class(self, prim: Usd.Prim) -> Any:
        """
        Get the operator class for the given prim.
        """
        for operator_class in get_operators():
            if operator_class.prim_type == prim.GetTypeName():
                if operator_class.api_schemas.issubset(set(prim.GetAppliedSchemas())):
                    return operator_class
        return None

    def _has_data_selection_transform_changed(self, prim: Usd.Prim) -> bool:
        """
        If prim has "CaeVizDatasetSelectionAPI" schemas applied then
        for all instaces, we need to track transforms from the target dataset prims.
        Subsequently, the `prim` is considered dirty if any of the target dataset prims
        have changed transforms.
        """
        for instance_name in usd_utils.get_instances(prim, "CaeVizDatasetTransformingAPI"):
            # locate the DatasetSelectionAPI instance
            if not prim.HasAPI(cae_viz.DatasetSelectionAPI, instance_name):
                continue

            xforming_api = cae_viz.DatasetTransformingAPI(prim, instance_name)
            ds_api = cae_viz.DatasetSelectionAPI(prim, instance_name)
            targets = ds_api.GetTargetRel().GetForwardedTargets()
            attr_name = (
                "omni:fabric:worldMatrix"
                if xforming_api.GetUseGlobalTransformAttr().Get()
                else "omni:fabric:localMatrix"
            )
            for target in targets:
                attr_path = target.AppendProperty(attr_name)
                if self._xform_change_tracker.AttributeChanged(str(attr_path)):
                    return True
        return False
