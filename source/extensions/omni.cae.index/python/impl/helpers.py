# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
#  its affiliates is strictly prohibited.

import threading
from logging import getLogger

from omni.cae.algorithms.core.listener import Listener
from omni.cae.data import IFieldArray, array_utils, cache, usd_utils
from omni.cae.data.commands import ComputeBounds
from omni.cae.data.typing import FieldArrayLike
from omni.kit.async_engine import run_coroutine
from omni.usd import get_context
from pxr import Usd

from .._omni_cae_index import Bbox_float32, IIrregular_volume_subset
from .command_types import CreateIrregularVolumeSubset

logger = getLogger(__name__)


class CaeDataSetImporter:

    def __init__(self, dataset_path: str, field_names: list[str]):
        stage = get_context().get_stage()
        self._dataset: Usd.Prim = usd_utils.get_prim_at_path(stage, dataset_path)
        self._fields = field_names

    @property
    def dataset(self):
        return self._dataset

    @property
    def fields(self):
        return self._fields

    @staticmethod
    async def compute_bounds(*args, **kwargs):
        # this is temporary workaround; real fix requires we
        # avoid nesting `await` calls in `with wp.ScopedDevice()` blocks.
        async with Listener._sync_lock:
            return await ComputeBounds.invoke(*args, **kwargs)

    @staticmethod
    async def create_irregular_volume_subset(*args, **kwargs):
        # this is temporary workaround; real fix requires we
        # avoid nesting `await` calls in `with wp.ScopedDevice()` blocks.
        async with Listener._sync_lock:
            return await CreateIrregularVolumeSubset.invoke(*args, **kwargs)

    def get_bounds(self) -> Bbox_float32:
        assert threading.current_thread() is not threading.main_thread(), "should not be called from the main thread"

        timeCode = Usd.TimeCode.EarliestTime()  # TODO
        bbox = run_coroutine(CaeDataSetImporter.compute_bounds(self.dataset, timeCode)).result()
        result = Bbox_float32()
        result.min = bbox.min
        result.max = bbox.max
        logger.debug("bbox=%s, %s", bbox.min, bbox.max)
        return result

    def create(self, bbox: Bbox_float32, subset: IIrregular_volume_subset, timeCode: float):
        assert threading.current_thread() is not threading.main_thread(), "should not be called from the main thread"
        timeCode = Usd.TimeCode(timeCode)
        return run_coroutine(
            CaeDataSetImporter.create_irregular_volume_subset(self.dataset, self.fields, timeCode, bbox, subset)
        ).result()


class CaeDataSetVoxelizedNanoVdbFetchTechnique:

    def __init__(self, prim_path: str, cache_key: str):
        stage = get_context().get_stage()
        self._prim: Usd.Prim = usd_utils.get_prim_at_path(stage, prim_path)
        self._cache_key = cache_key

    @property
    def dataset(self):
        return self._dataset

    @property
    def fields(self):
        return self._fields

    def fetch_nanovdb(self, device_id: int, attrib_index: int) -> tuple[FieldArrayLike, object]:
        assert threading.current_thread() is not threading.main_thread(), "should not be called from the main thread"
        # timeCode = Usd.TimeCode(usd_utils.get_attribute(self._prim, "nvindex:timestep"))
        timeCode = Usd.TimeCode(float(self._prim.GetCustomDataByKey(f"omni.cae.index:timestep_{attrib_index}")))
        logger.info(f"for attrib {attrib_index} using timecode {timeCode}")
        volume = cache.get(self._cache_key, timeCode=timeCode)
        if volume is None:
            logger.error("failed to get voxelized data %s!!!", timeCode)
            return None, None

        wp_array = array_utils.get_nanovdb(volume)

        # we return `volume` too since it needs to be preserved till we're done with the field array
        # this helps us avoid creating any extra copies
        return IFieldArray.from_array(wp_array), volume
