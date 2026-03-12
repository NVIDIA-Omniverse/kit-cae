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
Stage event listener for CAE visualization.

This module provides a listener that responds to USD stage lifecycle events
(attach, detach, update) and manages a Controller instance for each stage.
"""

__all__ = ["Listener", "exclusive_with_sync"]

import asyncio
import functools
import logging
import threading

import carb.settings
import omni.kit.app
import omni.timeline
from omni.kit.async_engine import run_coroutine
from omni.stageupdate import get_stage_update_interface
from pxr import Usd

from .controller import Controller

logger = logging.getLogger(__name__)


def exclusive_with_sync(func):
    """Decorator that acquires Listener._sync_lock, making execution mutually exclusive with _sync.
    Concurrent calls while sync (or another exclusive_with_sync function) is running are dropped."""

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        if Listener._sync_lock.locked():
            return None
        async with Listener._sync_lock:
            return await func(*args, **kwargs)

    return wrapper


class Listener:
    """
    Listener that handles stage notification events from USD.

    This class responds to stage lifecycle events and manages a Controller
    instance for processing CAE visualization operators.

    Attributes
    ----------
    _controller : Controller
        The controller managing operators for the current stage
    _sync_task : asyncio.Task
        The current sync task (if any)
    _update_counter : int
        Counter for tracking update frequency
    _timeline : omni.timeline.Timeline
        Timeline interface for accessing time codes
    _last_timecode : Usd.TimeCode
        The last processed time code
    _stage_subscription : StageUpdateNode
        The stage update subscription
    """

    PAUSE_AFTER_MAX_UPDATES = 1
    _sync_lock = asyncio.Lock()

    def __init__(self):
        """Initialize the listener and subscribe to stage events."""
        self._controller: Controller = None
        self._sync_task: asyncio.Task = None
        self._update_counter: int = 0
        self._timeline: omni.timeline.Timeline = omni.timeline.get_timeline_interface()
        self._last_timecode: Usd.TimeCode = Usd.TimeCode.EarliestTime()

        # Subscribe to stage update events
        stage_update_iface = get_stage_update_interface()
        self._stage_subscription = stage_update_iface.create_stage_update_node(
            "cae.viz.listener",
            on_attach_fn=self.on_attach,
            on_detach_fn=self.on_detach,
            on_update_fn=self.on_update,
            on_prim_add_fn=self.on_prim_add,
            on_prim_remove_fn=self.on_prim_remove,
            on_prim_or_property_change_fn=self.on_prim_or_property_change,
        )

        logger.info("Listener initialized and subscribed to stage events")

    def __del__(self):
        """Cleanup resources."""
        if self._sync_task is not None:
            self._sync_task.cancel()

        self.finalize()

    def finalize(self):
        """Finalize the listener."""
        if self._stage_subscription:
            del self._stage_subscription
            self._stage_subscription = None

        if self._controller:
            del self._controller
            self._controller = None

    @property
    def timecode(self) -> Usd.TimeCode:
        """Get the current time code from the timeline."""
        return Usd.TimeCode(round(self._timeline.get_current_time() * self._timeline.get_time_codes_per_seconds()))

    def on_prim_add(self, path):
        """
        Called when a prim is added to the stage.

        Parameters
        ----------
        path : str or Sdf.Path
            The path of the added prim
        """
        logger.debug("Prim added: %s", path)
        self._update_counter = 0

    def on_prim_remove(self, path):
        """
        Called when a prim is removed from the stage.

        Parameters
        ----------
        path : str or Sdf.Path
            The path of the removed prim
        """
        logger.debug("Prim removed: %s", path)
        self._update_counter = 0

    def on_prim_or_property_change(self, path, *args, **kwargs):
        """
        Called when a prim or its properties change.

        Parameters
        ----------
        path : str or Sdf.Path
            The path of the changed prim or property
        """
        logger.debug("Prim or property changed: %s", path)
        self._update_counter = 0

    def on_attach(self, stageId: int, metersPerUnit: float):
        """
        Called when a stage is attached.

        Parameters
        ----------
        stageId : int
            The stage ID from the stage cache
        metersPerUnit : float
            The meters per unit setting for the stage
        """
        logger.info("Stage attached: ID=%s, metersPerUnit=%s", stageId, metersPerUnit)

        try:
            self._controller = Controller(stageId)
            self._update_counter = 0
            logger.info("Controller created for stage")
        except Exception as e:
            logger.error("Failed to create controller for stage %s: %s", stageId, e, exc_info=True)
            self._controller = None

    def on_detach(self):
        """Called when the stage is detached."""
        logger.info("Stage detached")

        if self._controller:
            del self._controller
            self._controller = None

    def on_update(self, _0, _1):
        """
        Called periodically when the stage updates.

        This method checks if the time code has changed and enqueues a sync
        operation if needed.
        """
        # If timecode changed, restart the update counter
        timecode = self.timecode
        if self._last_timecode != timecode:
            logger.debug("Timecode changed %s -> %s", self._last_timecode, timecode)
            self._last_timecode = timecode
            self._update_counter = 0

        # Limit update frequency to avoid excessive processing
        # if self._controller and self._update_counter < self.PAUSE_AFTER_MAX_UPDATES:
        if self._controller:
            if self._update_counter < self.PAUSE_AFTER_MAX_UPDATES or self._controller._tracker.has_changes():
                self._update_counter += 1
                self.enqueue_sync()

    def enqueue_sync(self):
        """
        Enqueue a sync operation to be executed asynchronously.

        This ensures sync operations don't block the main thread and prevents
        multiple concurrent sync operations.

        When async rendering is disabled, this will block until the sync completes.
        """
        assert threading.current_thread() is threading.main_thread()

        # Check if we need to block based on async rendering setting
        settings = carb.settings.get_settings()
        async_rendering = settings.get("/app/asyncRendering")

        if not async_rendering:
            logger.debug("Running sync synchronously")
            # Run synchronously - block until complete
            loop = asyncio.get_event_loop()
            try:
                loop.run_until_complete(self._sync())
            except Exception as e:
                logger.error("Sync failed: %s", e, exc_info=True)
        else:
            # Normal async behavior
            if self._sync_task is None or self._sync_task.done():
                self._sync_task = run_coroutine(self._sync())

    async def _sync(self):
        """
        Asynchronously synchronize the controller with the stage.

        This method ensures only one sync operation runs at a time using a lock.
        """
        assert threading.current_thread() is threading.main_thread()

        if self._controller:
            # Prevent concurrent sync operations
            async with Listener._sync_lock:
                # Get current time code (may have changed since enqueue)
                if await self._controller.sync(self.timecode):
                    # Something changed, restart the counter
                    self._update_counter = 0
