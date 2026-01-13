# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

__all__ = ["UsdRtController"]

import logging

import omni.kit.notification_manager as nm
import pxr.Usd
from omni.cae.data import progress, usd_utils
from usdrt import Rt, Sdf, Usd

from .algorithm import Algorithm
from .factory import Factory, get_factory

logger = logging.getLogger(__name__)


class UsdRtController:
    """
    The controller class that manages creation / deletion of algorithms based on the stage.
    As the name suggests, it operates on the usdrt.Usd.Stage.

    Algorithms, originally also operated on UsdRT prims. However, eventually, due to our desire
    to not rely on Fabric Scene Delegate (FSD), we refactored algorithms to primarily work with pxr.Usd
    prims.

    Change tracking, however, is still handled by usdrt (Rt.ChangeTracker).
    """

    def __init__(self, stageId: int):
        self._stage: Usd.Stage = Usd.Stage.Attach(stageId)
        self._tracker = Rt.ChangeTracker(self._stage)
        self._factory: Factory = get_factory()
        self._algorithms: dict[Sdf.Path, Algorithm] = {}
        self._tracker.EnablePrimCreateTracking()

        # ensure we check for prims created before the tracker was created.
        self._needs_init: bool = True

        # notifications
        self._abortingNotification = None
        self._executingNotification = None

    async def _create_new(self, paths: list[Sdf.Path]) -> bool:
        something_created = False
        for path in paths:
            if prim := self._stage.GetPrimAtPath(path):
                if path in self._algorithms:
                    # with 106.4, we get repeated create notifications quite consistently
                    # so we check.
                    # logger.warning("not recreating algo for %s", path)
                    pass
                elif algo := self._factory.create(usd_utils.get_prim_pxr(prim), quiet=True):
                    logger.info("Created Algorithm (%s) for '%s'", algo, path)
                    self._algorithms[path] = algo
                    await algo.initialize()
                    something_created = True
                else:
                    logger.info("failed to create Algorithm for '%s'", path)
        return something_created

    async def _create_algorithms_for_new_prims(self) -> bool:
        # if this is the first time this is called, the tracker will miss prims created before the
        # tracker was created. So we need to skip the tracker and directly check for all prims.
        check_all_prims = self._needs_init
        self._needs_init = False

        if not check_all_prims:
            return await self._create_new(self._tracker.GetAddedPrims())
        else:
            something_created: bool = False
            for primType, apiSchemas in self._factory.get_supported_prim_types():
                if paths := self._stage.GetPrimsWithTypeAndAppliedAPIName(primType, apiSchemas):
                    something_created = (
                        await self._create_new(filter(lambda x: x not in self._algorithms, paths)) or something_created
                    )
            return something_created

    def _remove_algorithms_for_old_prims(self) -> bool:
        obsolete_paths = list(filter(lambda path: not self._stage.HasPrimAtPath(path), self._algorithms.keys()))
        for path in obsolete_paths:
            logger.info("Removing algorithm for %s", path)
            self._algorithms[path].release()
            del self._algorithms[path]
        return len(obsolete_paths) > 0

    async def sync(self, timecode) -> bool:
        """
        Called to requested update of the algorithms state based on USD.
        This does several things:

            1. Check if new prims have been added for which we need to create Algorithm instances for them
            2. Check if prims have been removed, if so we discard the corresponding Algorithms.
            3. Check if any algorithms need update and if so execute them.

        Returns:
            - True if something was updated else False.
        """
        # to avoid confusion between pxr.Usd.TimeCode or usdrt.Usd.TimeCode, we convert to float
        timecode_value = timecode.GetValue() if hasattr(timecode, "GetValue") else float(timecode)
        pxr_timecode: pxr.Usd.TimeCode = pxr.Usd.TimeCode(timecode_value)
        # logger.warning("sync %s", pxr_timecode)

        # remove algorithms for deleted prims
        changed = self._remove_algorithms_for_old_prims()

        # for newly added prims of interest, we create algos for them.
        changed = await self._create_algorithms_for_new_prims() or changed

        # reset tracked changes
        # since this tracker only tracks prim additions and removals, we reset it before
        # we start processing the algorithms. That way, if as the algorithms are executing, the user adds
        # a new prim, we won't miss it.
        self._tracker.ClearChanges()

        deps = {}
        for algo in self._algorithms.values():
            for req in algo.get_requirements():
                req = str(req)
                if req not in deps:
                    deps[req] = set()
                deps[req].add(algo)

        def needed_by_deps(algo: Algorithm, timeCode: pxr.Usd.TimeCode) -> bool:
            path = str(algo.prim.GetPath())
            for dep_algo in deps.get(path, []):
                if dep_algo.is_enabled(timeCode):
                    return True
            return False

        def get_dependency_count(algo: Algorithm) -> int:
            return len(deps.get(str(algo.prim.GetPath()), []))

        # check if any algo needs update
        algos_to_update = []
        for algo in self._algorithms.values():
            if not algo.is_enabled(pxr_timecode) and not needed_by_deps(algo, pxr_timecode):
                pass
            elif algo.needs_update(pxr_timecode):
                algos_to_update.append((algo, True))
            elif algo.needs_update_for_time(pxr_timecode):
                algos_to_update.append((algo, False))

        # sort algos to update to ensure that the ones with dependencies are executed first.
        algos_to_update.sort(key=lambda x: get_dependency_count(x[0]), reverse=True)

        if algos_to_update:
            cancel_button = nm.NotificationButtonInfo("Abort", on_complete=self.abort)
            logger.info("Executing %d algorithms", len(algos_to_update))
            for algo, force in algos_to_update:
                self._executingNotification = nm.post_notification(
                    f"Processing {algo.prim.GetName()}", hide_after_timeout=False, button_infos=[cancel_button]
                )
                await algo.execute(pxr_timecode, force)
                self._executingNotification.dismiss()
                self._executingNotification = None
                changed = True
                if self._abortingNotification:
                    self._abortingNotification.dismiss()
                    break

        # Some algorithms, viz IndeX-basedm may need to update attributes on the prims
        # to indicate which timestep to render. This is done after all algorithms have executed.
        for algo in self._algorithms.values():
            algo.sync_time(pxr_timecode)
        return changed

    def abort(self):
        logger.warning("Abort requested by user")
        progress.interrupt()
        self._abortingNotification = nm.post_notification("Aborting...", hide_after_timeout=False)
