# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
#  its affiliates is strictly prohibited.

__all__ = ["Algorithm"]

import asyncio
import traceback
from abc import ABC, abstractmethod
from logging import getLogger
from typing import Union

from omni.cae.data import progress, usd_utils
from omni.cae.data.usd_utils import ChangeTracker, QuietableException
from omni.cae.schema import cae
from omni.timeline import get_timeline_interface
from pxr import Gf, Usd, UsdGeom, UsdShade
from usdrt import Usd as UsdRt

logger = getLogger(__name__)


class Algorithm(ABC):

    @classmethod
    def get_current_timecode(cls) -> Usd.TimeCode:
        if not hasattr(cls, "_timeline"):
            cls._timeline = get_timeline_interface()
        return Usd.TimeCode(round(cls._timeline.get_current_time() * cls._timeline.get_time_codes_per_seconds()))

    def __init__(self, prim: Usd.Prim, schemaTypeNames: list[str]):
        self._prim: Usd.Prim = prim
        self._needs_init: bool = True
        self._tracker: ChangeTracker = ChangeTracker(self._prim.GetStage())
        self._prim_rt: UsdRt.Prim = self._tracker._rt_stage.GetPrimAtPath(str(self._prim.GetPath()))
        self._execute_lock: asyncio.Lock = asyncio.Lock()
        self._executed_time_codes: set[Usd.TimeCode] = set()

        for schemaTypeName in schemaTypeNames:
            self._tracker.TrackSchemaProperties(schemaTypeName)

        # # track all register field array properties; this is necessary to catch changes to field arrays
        # self._tracker.TrackCaeFieldArrayProperties()

        # since USDRT prim is reset when material binding changes, we need to re-execute
        self._tracker.TrackAttribute("material:binding")

    def release(self):
        pass

    def get_requirements(self) -> set[str]:
        """
        Returns paths to Prims for other algorithms that this algorithm depends on.
        This is poor man's pipeline management since we don't really support pipelines yet,
        but is needed for algos like Slice that depend on Volume algorithms.
        """
        return set()

    def uses_temporal_cache(self) -> bool:
        """
        Returns True if the algorithm uses a temporal cache.
        This is used to determine if the algorithm needs to be executed
        at every time step.
        """
        return False

    @property
    def prim(self) -> Usd.Prim:
        return self._prim

    @property
    def prim_rt(self) -> UsdRt.Prim:
        """Returns the prim that is used to execute the algorithm."""
        return self._prim_rt

    @property
    def stage(self) -> Usd.Stage:
        return self._prim.GetStage()

    def is_enabled(self, timeCode: Usd.TimeCode) -> bool:
        if not self._prim:
            logger.warning("Algorithm needs to be created with a valid prim")
            return False

        if not self._prim.IsActive():
            logger.debug(f"{id(self)} inactive {self._prim}")
            return False

        if (
            self._prim.IsA(UsdGeom.Imageable)
            and UsdGeom.Imageable(self._prim).ComputeVisibility(timeCode) == UsdGeom.Tokens.invisible
        ):
            logger.debug(f"{id(self)} invisible {self._prim}")
            return False

        return True

    def needs_update(self, timeCode: Usd.TimeCode) -> bool:
        if self._needs_init:
            logger.info(f"{id(self)} need_init")
            return True

        if self._tracker.PrimChanged(self._prim):
            logger.info(f"{id(self)} prim_or_targets_changed {self._prim}")
            return True

        # logger.warning(f"{id(self)} nothing changed {self.prim}")
        return False

    def needs_update_for_time(self, timeCode: Usd.TimeCode) -> bool:
        if times := self.get_bracketing_time_codes(timeCode):
            if times[0] not in self._executed_time_codes:
                logger.info(f"{id(self)} timecode_changed {self._prim} {timeCode} -> {times[0]}")
                return True

        # logger.warning(f"{id(self)} nothing changed {self.prim}")
        return False

    def get_bracketing_time_codes(self, timeCode: Usd.TimeCode) -> tuple[Usd.TimeCode, Usd.TimeCode]:
        """
        Returns the bracketing time samples for the given timecode.
        This is used to determine if the algorithm needs to be executed.
        """
        tcs = usd_utils.get_bracketing_time_codes(self._prim, timeCode)
        # logger.warning("bracketing time codes (%s, %s)", tcs[0], tcs[1])
        return tcs

    def get_time_code(self, timeCode: Usd.TimeCode) -> Usd.TimeCode:
        """
        Returns the time code for the given timecode. This is used to determine which timecode should
        the algorithm execute at.
        """
        if bracketing_time_codes := self.get_bracketing_time_codes(timeCode):
            return bracketing_time_codes[0]
        else:
            return Usd.TimeCode.EarliestTime()

    async def initialize(self) -> None:
        pass

    async def execute(self, timeCode: Union[Usd.TimeCode, None] = None, force: bool = True) -> None:
        async with self._execute_lock:
            if force or not self.uses_temporal_cache():
                self._executed_time_codes.clear()

            timeCode = self.get_time_code(timeCode or Algorithm.get_current_timecode())
            self._executed_time_codes.add(timeCode)

            logger.info("[algorithm] executing %s at %s ", self.__class__.__name__, timeCode)
            with progress.ProgressContext("Executing %s" % self.__class__.__name__):
                try:
                    await self.execute_impl(timeCode)
                except QuietableException as e:
                    logger.warning("%s", e)
                    # we put trackback on info channel to assist in debugging.
                    logger.debug("%s\n%s", e, traceback.format_exc())
                except InterruptedError:
                    logger.info("Execution interrupted")
                except Exception as e:
                    logger.error("%s", e)
                    logger.error("Execution failed with %s\n%s", e, traceback.format_exc())

            self._tracker.ClearChanges()
            self._needs_init = False

    @abstractmethod
    async def execute_impl(self, timecode: Usd.TimeCode) -> None:
        raise NotImplementedError("Missing execute_impl")

    def sync_time(self, timecode: Usd.TimeCode) -> None:
        """
        Called to sync the algorithm to the given timecode.
        """
        pass

    @property
    def edit_context(self) -> Usd.EditContext:
        """
        Returns the edit context for the stage under which all USD operations
        by subclasses must be performed.
        """
        return Usd.EditContext(self.stage, self.stage.GetEditTargetForLocalLayer(self.stage.GetSessionLayer()))
        # return Usd.EditContext(self.stage, self.stage.GetSessionLayer())

    def bind_material(self, material_name) -> Usd.Prim:
        import omni.kit.commands

        material_prim = self.prim.GetChild("Materials").GetChild(material_name)
        if not material_prim:
            raise RuntimeError(f"Material {material_name} not found in {self.prim.GetPath()}")

        rel = self.prim.GetRelationship("material:binding")
        if rel and material_prim.GetPath() in rel.GetTargets():
            logger.info("Material %s already bound to %s", material_name, self.prim.GetPath())
            return material_prim

        omni.kit.commands.execute(
            "BindMaterial",
            prim_path=str(self.prim.GetPath()),
            material_path=str(material_prim.GetPath()),
            strength=UsdShade.Tokens.weakerThanDescendants,
        )
        logger.info("Bound material %s to %s", material_name, self.prim.GetPath())
        return material_prim

    def get_material(self, name: str) -> Usd.Prim:
        """
        Returns the material with the given name. If it does not exist, it creates it.
        """
        materials = self.prim.GetChild("Materials")
        material_prim = materials.GetChild(name) if materials else None
        return material_prim

    def set_extent(self, extent: Gf.Range3d) -> None:
        """
        Sets the extent of the algorithm. This is used to determine the bounding box of the algorithm.
        """
        if extent and not extent.IsEmpty():
            if not self._prim.IsA(UsdGeom.Boundable):
                raise RuntimeError("Extent can only be set on UsdGeom.Boundable prims")
            UsdGeom.Boundable(self._prim).CreateExtentAttr().Set([extent.min, extent.max])
