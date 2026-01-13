# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

__all__ = ["Listener"]

import asyncio
import logging
import threading

import omni.timeline
from omni.kit.async_engine import run_coroutine
from omni.stageupdate import get_stage_update_interface
from pxr import Usd

from .controller import UsdRtController
from .playback import set_controller

logger = logging.getLogger(__name__)


class Listener:
    """Listener that stage/prim notification events from pxr.Usd."""

    PAUSE_AFTER_MAX_UPDATES = 10
    _sync_lock = asyncio.Lock()

    def __init__(self):
        self._controller: UsdRtController = None
        self._sync_task: asyncio.Task = None
        self._update_counter: int = 0
        self._timeline: omni.timeline.Timeline = omni.timeline.get_timeline_interface()
        self._last_timecode: Usd.TimeCode = Usd.TimeCode.EarliestTime()

        stage_update_iface = get_stage_update_interface()
        self._stage_subscription = stage_update_iface.create_stage_update_node(
            "cae.algorithms.core",
            on_attach_fn=self.on_attach,
            on_detach_fn=self.on_detach,
            on_update_fn=self.on_update,
            on_prim_add_fn=self.on_prim_add,
            on_prim_remove_fn=self.on_prim_remove,
            on_prim_or_property_change_fn=self.on_prim_or_property_change,
        )

    def __del__(self):
        if self._sync_task_or_future is not None:
            self._sync_task_or_future.cancel()

        del self._stage_subscription
        del self._controller

    @property
    def timecode(self) -> Usd.TimeCode:
        return Usd.TimeCode(round(self._timeline.get_current_time() * self._timeline.get_time_codes_per_seconds()))

    def on_prim_add(self, path):
        # logger.warning(f"notify in tid={get_thread_id()}")
        self._update_counter = 0

    def on_prim_remove(self, path):
        # logger.error("prim removed %s" % path)
        self._update_counter = 0

    def on_prim_or_property_change(self, path, *args, **kwargs):
        self._update_counter = 0

    def on_attach(self, stageId, metersPerUnit):
        logger.info("on_attach %s (%s) %s", str(stageId), type(stageId), str(metersPerUnit))
        self._controller = UsdRtController(stageId)
        set_controller(self._controller)
        self._update_counter = 0

    def on_detach(self):
        logger.info("on_detach")
        self._controller = None
        set_controller(None)

    def on_update(self, _0, _1):
        # if timecode changed, restart the update counter.
        timecode = self.timecode
        if self._last_timecode != timecode:
            logger.info("timecode changed %s -> %s", self._last_timecode, timecode)
            self._last_timecode = timecode
            self._update_counter = 0

        # TODO: I am not entirely sure PAUSE_AFTER_MAX_UPDATES > 1 make sense anymore.
        # It was necessary with USDRT since USDRT state could lag behind PXR changes, but
        # that's not the case anymore so we can perhaps remove this
        if self._controller and self._update_counter < self.PAUSE_AFTER_MAX_UPDATES:
            # logger.warning("update")
            self._update_counter += 1
            self.enqueue_sync()

    def enqueue_sync(self):
        assert threading.current_thread() is threading.main_thread()
        if self._sync_task is None or self._sync_task.done():
            self._sync_task = run_coroutine(self._sync())

    async def _sync(self):
        assert threading.current_thread() is threading.main_thread()
        if self._controller:
            # we need to ensure that we don't end up calling sync while
            # an earlier sync is still running. this can happen easily happening
            # during progress events even though this is only called from the main thread.
            async with Listener._sync_lock:
                # we use current time code for sync. note, this may have changed
                # since the `on_update` call that enqueued this sync.
                if await self._controller.sync(self.timecode.GetValue()):
                    # since something changed, restarted the counter.
                    self._update_counter = 0
