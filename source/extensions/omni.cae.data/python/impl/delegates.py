# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import asyncio

from pxr import Usd

from .bindings import IDataDelegate, IDataDelegateRegistry, IFieldArray
from .typing import FieldArrayLike


def _get_stage(id) -> Usd.Stage:
    from pxr import UsdUtils

    cache = UsdUtils.StageCache.Get()
    return cache.Find(Usd.StageCache.Id.FromLongInt(id))


def _get_stage_id(stage: Usd.Stage) -> Usd.StageCache.Id:
    from pxr import UsdUtils

    cache = UsdUtils.StageCache.Get()
    return cache.GetId(stage)


class DataDelegateBase(IDataDelegate):
    """
    This is intended to be used as the superclass for Python-based implementations of
    DataDelegates.

    Subclasses must implement the `get_field_array` and `can_provide` methods.
    """

    def __init__(self, extensionId: str):
        super().__init__(extensionId)

    def _get_field_array(self, stageId: int, primPath: str, time: float) -> FieldArrayLike:
        """internal method used by C++ API to call Python implementation by passing stageId and primPath
        instead of the Prim itself. Do not use this method directly."""
        stage = _get_stage(stageId)
        return self.get_field_array(stage.GetPrimAtPath(primPath), Usd.TimeCode(time))

    def _can_provide(self, stageId: int, primPath: str) -> bool:
        """internal method used by C++ API to call Python implementation by passing stageId and primPath
        instead of the Prim itself. Do not use this method directly."""
        stage = _get_stage(stageId)
        return self.can_provide(stage.GetPrimAtPath(primPath))

    def get_field_array(self, prim: Usd.Prim, time: Usd.TimeCode) -> FieldArrayLike:
        """
        This method must be implemented by subclasses.  It should return an
        array-like object containing the field data for the given prim and
        time.
        """
        raise NotImplementedError("DataDelegateBase.get_field_array() not implemented")

    def can_provide(self, prim: Usd.Prim) -> bool:
        """
        This method must be implemented by subclasses.  It should return
        True if the delegate can provide data for the given prim.
        """
        return False


def _idatadelegateregistry_get_field_array(
    me: IDataDelegateRegistry, prim: Usd.Prim, time=Usd.TimeCode.EarliestTime()
) -> IFieldArray:
    """
    This function gets called when IDataDelegateRegistry.get_field_array is called.

    Note, we return an IFieldArray here which can support CUDA Array Interface or Numpy Array Interface.
    We cannot return `warp.array` here due to warp array max-length limitation.
    """
    assert isinstance(prim, Usd.Prim)
    prim_path = str(prim.GetPath())
    stage_id = _get_stage_id(prim.GetStage()).ToLongInt()
    return me._get_field_array(stage_id, prim_path, time.GetValue())


def _idatadelegateregistry_is_field_array_cached(
    me: IDataDelegateRegistry, prim: Usd.Prim, time=Usd.TimeCode.EarliestTime()
) -> bool:
    """
    This function gets called when IDataDelegateRegistry.is_field_array_cached is called.
    Forwards to _is_field_array_cached after translating stage to long and time to double.
    """
    assert isinstance(prim, Usd.Prim)
    prim_path = str(prim.GetPath())
    stage_id = _get_stage_id(prim.GetStage()).ToLongInt()
    return me._is_field_array_cached(stage_id, prim_path, time.GetValue())


def _idatadelegateregistry_get_field_array_async(
    me: IDataDelegateRegistry, prim: Usd.Prim, time=Usd.TimeCode.EarliestTime()
) -> asyncio.Future:
    """
    AsyncIO version of `get_field_array` that returns an awaitable.
    """
    assert isinstance(prim, Usd.Prim)
    prim_path = str(prim.GetPath())
    stage_id = _get_stage_id(prim.GetStage()).ToLongInt()
    return me._get_field_array_async(stage_id, prim_path, time.GetValue())


IDataDelegateRegistry.get_field_array = _idatadelegateregistry_get_field_array
IDataDelegateRegistry.is_field_array_cached = _idatadelegateregistry_is_field_array_cached
IDataDelegateRegistry.get_field_array_async = _idatadelegateregistry_get_field_array_async
