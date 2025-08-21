# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
#  its affiliates is strictly prohibited.

__all__ = ["DataSetEmitter"]

import asyncio
from logging import getLogger

import numpy as np
import warp as wp
from omni.cae.algorithms.core import Algorithm
from omni.cae.data import array_utils, progress, usd_utils
from omni.cae.data.commands import ComputeIJKExtents, Voxelize
from omni.cae.schema import cae
from omni.kit.async_engine import run_coroutine
from pxr import Sdf, Usd, Vt

logger = getLogger(__name__)


class DataSetEmitter(Algorithm):
    _xform_ops = [
        "omni:fabric:localMatrix",
    ]

    def __init__(self, prim: Usd.Prim):
        super().__init__(prim, ["CaeFlowDataSetEmitterAPI"])
        self._ns = "omni:cae:flow:emitter"
        self._rel_tracker: usd_utils.ChangeTracker = usd_utils.ChangeTracker(self.stage)
        self._force_update = None

        # xform ops
        for p in self._xform_ops:
            self._rel_tracker.TrackAttribute(p)

    def is_enabled(self, timeCode: Usd.TimeCode) -> bool:
        if not super().is_enabled(timeCode):
            return False

        if self._prim.HasAttribute("enabled"):
            # Flow dataset emitters are not Imageable, so we respect "enabled"
            # attribute on them
            return self._prim.GetAttribute("enabled").Get(timeCode)

        return True

    def needs_update(self, timeCode: Usd.TimeCode) -> bool:
        if super().needs_update(timeCode):
            return True

        # if roi was transformed, we need to reexecute.
        # If ROI changes, we need to revoxelize; we avoid doing that too frequently
        # It's best to delay the update until the interaction has stabilized. we achieve this
        # using a coroutine that will force an update after 0.1 seconds (unless it gets canceled due
        # to a new interaction event).
        roi_targets = self.prim.GetRelationship(f"{self._ns}:roi").GetForwardedTargets()
        for t in roi_targets:
            if self._rel_tracker.PrimChanged(t):
                if self._force_update is None or self._force_update.done():
                    # logger.error("enqueue force update")
                    self._force_update = run_coroutine(self.force_update())
                self._rel_tracker.ClearChanges()
        return False

    async def force_update(self):
        # logger.error("start sleep force update")
        await asyncio.sleep(0.5)
        # logger.error("end sleep force update")
        run_coroutine(self.execute())

    async def execute_impl(self, timeCode: Usd.TimeCode) -> None:
        if self._force_update:
            self._force_update.cancel()
            self._force_update = None

        prim: Usd.Prim = self.prim
        dataset_prim = usd_utils.get_target_prim(prim, f"{self._ns}:dataset")
        v_field_prims = usd_utils.get_target_prims(prim, f"{self._ns}:velocity")
        if len(v_field_prims) != 1 and len(v_field_prims) != 3:
            raise usd_utils.QuietableException("Invalid number of velocity fields specified!")

        v_assoc = cae.FieldArray(v_field_prims[0]).GetFieldAssociationAttr().Get()
        if v_assoc != cae.Tokens.vertex and v_assoc != cae.Tokens.cell:
            raise usd_utils.QuietableException("Invalid field association '%s' for velocity" % v_assoc)

        t_field_prim = usd_utils.get_target_prim(prim, f"{self._ns}:colors", quiet=True)
        if t_field_prim:
            t_assoc = cae.FieldArray(t_field_prim).GetFieldAssociationAttr().Get()
            if t_assoc != v_assoc:
                raise usd_utils.QuietableException("Invalid field association '%s' for colors" % t_assoc)

        v_field_names = [usd_utils.get_field_name(dataset_prim, t) for t in v_field_prims]

        # given max resolution and dataset bounds, let's determine the voxel grid size and extents.
        maxResolution: int = usd_utils.get_attribute(prim, f"{self._ns}:maxResolution")
        roi_prim: Usd.Prim = usd_utils.get_target_prim(prim, f"{self._ns}:roi", quiet=True)
        roi = usd_utils.get_bounds(roi_prim, Usd.TimeCode.EarliestTime(), quiet=True)
        logger.info("Using ROI: %s", roi)

        with progress.ProgressContext(scale=0.3):
            ijk_extents = await ComputeIJKExtents.invoke(
                dataset_prim, [maxResolution] * 3, roi=roi, timeCode=Usd.TimeCode.EarliestTime()
            )
        voxel_size = ijk_extents.spacing[0]

        with progress.ProgressContext("Voxelizing velocity", shift=0.3, scale=0.4):
            volume: wp.Volume = await Voxelize.invoke(
                dataset_prim,
                v_field_names,
                bbox=ijk_extents.getRange(),
                voxel_size=ijk_extents.spacing[0],
                device_ordinal=0,
                timeCode=Usd.TimeCode.EarliestTime(),
            )
        assert volume is not None
        vdb_data: np.ndarray = array_utils.get_nanovdb(volume).numpy().view(dtype=np.uint32)

        prim.CreateAttribute("nanoVdbVelocities", Sdf.ValueTypeNames.UIntArray, custom=True).Set(
            Vt.UIntArray.FromNumpy(vdb_data)
        )

        if t_field_prim:
            # Since Warp doesn't support vec4 for volumes, for now we will separately voxelize color
            # regardless of which voxelization implementation is used.
            t_field_names = [usd_utils.get_field_name(dataset_prim, t_field_prim)] if t_field_prim else []
            with progress.ProgressContext("Voxelizing color", shift=0.7, scale=0.3):
                volume_temperature: wp.Volume = await Voxelize.invoke(
                    dataset_prim,
                    t_field_names,
                    bbox=ijk_extents.getRange(),
                    voxel_size=voxel_size,
                    device_ordinal=0,
                    timeCode=Usd.TimeCode.EarliestTime(),
                )
            assert volume_temperature is not None
            tdb_data: np.ndarray = array_utils.get_nanovdb(volume_temperature).numpy().view(dtype=np.uint32)
            prim.CreateAttribute("nanoVdbTemperatures", Sdf.ValueTypeNames.UIntArray, custom=True).Set(
                Vt.UIntArray.FromNumpy(tdb_data)
            )

        self._rel_tracker.ClearChanges()
        return True


# Note to self:
# * /World/CAE/DataSetEmitter.allocationScale == 0.0 # makes smoke move
# * /World/FlowEnvironment/flowOffscreen/debugVolume.enableSpeedAsTemperature == true # to use velocity as temperature aka coloring
