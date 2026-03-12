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
Execution context for CAE visualization operators.

This module defines the ExecutionContext class and ExecutionReason enum that
provide operators with information about why they are being executed.
"""

__all__ = ["ExecutionContext", "ExecutionReason"]

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from pxr import Usd


class ExecutionReason(Enum):
    """
    Reason why an operator is being executed.

    Attributes
    ----------
    INITIAL : str
        First execution of the operator for this prim
    STRUCTURAL_CHANGE : str
        Prim properties or structure changed (time-independent changes)
    TEMPORAL_UPDATE : str
        New timecode in temporal mode (time-only change, first time at this timecode)
    TEMPORAL_TICK : str
        Time changed to already-seen timecode (minimal update only)
    """

    INITIAL = "initial"
    STRUCTURAL_CHANGE = "structural"
    TEMPORAL_UPDATE = "temporal"
    TEMPORAL_TICK = "temporal_tick"


@dataclass
class ExecutionContext:
    """
    Context information about why an operator is being executed.

    This class provides operators with information about the execution reason,
    allowing them to optimize their behavior. For example, operators can skip
    expensive rebuilds for temporal-only updates.

    Parameters
    ----------
    reason : ExecutionReason
        The reason why this execution is happening
    timecode : Usd.TimeCode
        The snapped timecode to use for execution (from get_bracketing_time_samples_for_prim)
    raw_timecode : Usd.TimeCode
        The original timeline timecode before any snapping
    next_time_code : Optional[Usd.TimeCode]
        The next bracketing timecode for interpolation (None if not available)
    device : str
        The device to execute on (e.g., "cpu", "cuda:0")
    timestep_index : int
        Index indicating which timestep is being executed (0=current/t0, 1=next/t0+1).
        Used when interpolation requires execution at multiple timesteps.

    Examples
    --------
    >>> async def exec(self, prim, device, context):
    ...     # Use context.timecode (always snapped to time samples)
    ...     logger.info(f"Executing at {context.timecode}, timestep_index={context.timestep_index}")
    ...
    ...     # Check if this is the next timestep (t0+1) for interpolation
    ...     if context.is_next_timestep():
    ...         logger.info(f"Executing for next timestep (t0+1)")
    ...
    ...     if context.is_temporal_update():
    ...         # Fast path: only update time-varying data
    ...         await self._update_fields(prim, context.timecode, device)
    ...     else:
    ...         # Full rebuild
    ...         await self._full_rebuild(prim, context.timecode, device)
    >>>
    >>> async def on_time_changed(self, prim, device, context):
    ...     # Check if field interpolation is enabled
    ...     if prim.HasAPI(cae_viz.OperatorTemporalAPI):
    ...         api = cae_viz.OperatorTemporalAPI(prim)
    ...         if api.GetEnableFieldInterpolationAttr().Get() and context.next_time_code:
    ...             # Interpolate between context.timecode and context.next_time_code
    ...             logger.debug(f"Interpolating: raw={context.raw_timecode}, "
    ...                         f"current={context.timecode}, next={context.next_time_code}")

    See Also
    --------
    ExecutionReason : Enum defining possible execution reasons
    """

    reason: ExecutionReason
    timecode: Usd.TimeCode  # Snapped timecode (from get_bracketing_time_samples_for_prim)
    raw_timecode: Usd.TimeCode  # Original timeline timecode
    next_time_code: Optional[Usd.TimeCode]  # Next bracketing timecode (if available)
    device: str
    timestep_index: int = 0  # 0=current/t0, 1=next/t0+1 (for interpolation)

    def is_full_rebuild_needed(self) -> bool:
        """
        Check if a full rebuild is needed.

        Returns True for INITIAL and STRUCTURAL_CHANGE reasons, indicating that
        the operator should perform a complete rebuild of its output. Returns
        False for TEMPORAL_UPDATE, where only time-varying data needs updating.

        Returns
        -------
        bool
            True if full rebuild is needed, False for temporal-only updates
        """
        return self.reason != ExecutionReason.TEMPORAL_UPDATE and self.reason != ExecutionReason.TEMPORAL_TICK

    def is_temporal_update(self) -> bool:
        """
        Check if this is a temporal-only update.

        Returns True only when the reason is TEMPORAL_UPDATE, indicating that
        only time-varying data has changed and the operator can skip expensive
        structural rebuilds.

        Returns
        -------
        bool
            True if this is a temporal-only update, False otherwise
        """
        return self.reason == ExecutionReason.TEMPORAL_UPDATE

    def is_temporal_tick(self) -> bool:
        """
        Check if this is a minimal time tick update.

        Returns True only when the reason is TEMPORAL_TICK, indicating the
        timeline has moved to an already-executed timecode. This is used for
        operators with tick_on_time_change=True to perform minimal updates
        on every time change.

        Returns
        -------
        bool
            True if this is a minimal time tick, False otherwise
        """
        return self.reason == ExecutionReason.TEMPORAL_TICK

    def is_next_timestep(self) -> bool:
        """
        Check if this execution is for the next timestep (t0+1).

        This is useful when field interpolation is enabled and the operator
        needs to execute for both current and next timesteps. The first
        execution will have timestep_index=0 (current), and the second will
        have timestep_index=1 (next).

        Returns
        -------
        bool
            True if timestep_index > 0, False otherwise
        """
        return self.timestep_index > 0
