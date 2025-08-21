# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
#  its affiliates is strictly prohibited.

from logging import getLogger
from typing import Any

from omni.kit import app
from omni.stageupdate import get_stage_update_interface
from omni.usd import get_context, get_context_from_stage_id
from pxr import Tf, Usd

from . import settings

logger = getLogger(__name__)

_cache: dict[Any, Any] = {}
_sourcePrims: dict[Usd.Prim, set[Any]] = {}
_consumerPrims: dict[Usd.Prim, set[Any]] = {}


class Listener:
    def __init__(self):
        stage_update_iface = get_stage_update_interface()
        self._stage_subscription = stage_update_iface.create_stage_update_node(
            "cae.data.cache", on_attach_fn=self.on_attach, on_detach_fn=self.on_detach
        )
        self._stage = get_context().get_stage()
        self._listener = None
        self._subscriptions = []
        self._subscriptions.append(
            app.SettingChangeSubscription(
                settings.SettingsKeys.ENABLE_INTERMEDIATE_CACHE, lambda item, event_type: clear()
            )
        )

    def on_attach(self, stageId, metersPerUnit):
        ctx = get_context_from_stage_id(stageId)
        self._stage = ctx.get_stage()
        self._listener = Tf.Notice.Register(Usd.Notice.ObjectsChanged, self.on_objects_changed, self._stage)

    def on_detach(self):
        # clear cache on detach
        clear()
        self._stage = None
        self._listener = None

    def on_objects_changed(self, notice: Usd.Notice.ObjectsChanged, stage):
        if stage != self._stage or not isinstance(stage, Usd.Stage):
            return
        to_remove = set()

        # Gather non structural changes affecting prims' fields
        updatedpaths = {x.GetPrimPath() for x in notice.GetChangedInfoOnlyPaths() if x.IsPropertyPath()}
        # Gather structural changes that can affect cached prims
        resyncedpaths = notice.GetResyncedPaths()

        for primset in (_sourcePrims, _consumerPrims):
            for prim, keys in primset.items():
                if prim.GetPath() in updatedpaths:
                    logger.info("resynced prim: %s, # keys: %d", prim, len(keys))
                    to_remove.update(keys)
                    continue
                primpath = prim.GetPath()
                for prefix in resyncedpaths:
                    if primpath.HasPrefix(prefix):
                        logger.info("resynced prim: %s, # keys: %d", prim, len(keys))
                        to_remove.update(keys)
                        break

        for key in to_remove:
            remove(key)


_listener: Listener = None


def _initialize():
    global _listener
    _listener = Listener()


def _finalize():
    global _cache
    global _sourcePrims
    global _consumerPrims
    global _listener
    _cache = {}
    _sourcePrims = {}
    _consumerPrims = {}
    _listener = None


def remove(key: Any) -> bool:
    if key in _cache:
        logger.info("[py-cache]:remove(%s)", key)
        del _cache[key]
        for prim, keys in _sourcePrims.items():
            if key in keys:
                keys.remove(key)
        for prim, keys in _consumerPrims.items():
            if key in keys:
                keys.remove(key)
        return True
    return False


def put(
    key: Any,
    data: Any,
    state: Any = None,
    sourcePrims: list[Usd.Prim] = [],
    consumerPrims: list[Usd.Prim] = [],
    *,
    force: bool = False,
    timeCode: Usd.TimeCode = Usd.TimeCode.Default(),
):
    """
    Function to store data in the cache. The `key` and `data` are pretty typical for caching.

    `sourcePrims` and `consumerPrims` are a list of prims that is used to determine when cached data should be discarded.

    The cached data will be automatically discarded if:
      - any of the prims in `sourcePrims` is deleted
      - any of the prims in `sourcePrims` is modified
      - any of the prims in `consumerPrims` is deleted # FIXME: we should make this is **all** instead of any. If a cached
        item has no alive consumers, then it will be removed (unless of course, no consumers were specified in the first place).

    `state` acts as additional metadata that can be used to determine when data is still applicable.
    In `get`, `state` will be passed in as the second argument. The cached data is considered obsolete if the
    `state` doesn't match the state used when storing the data in cache.

    If `force` is True, then the cache will be updated regardless of the whether intermediate cache is enabled or not.
    This is useful when caching is a critical part of the algorithm to maintain intermediate results.
    """
    if not force and not settings.get_enable_intermediate_cache():
        logger.info("[py-cache]:put(%s, %s) -> skipped", key, state)
        return

    logger.info("[py-cache]:put(%s, %s, ..., %s) (force=%s)", key, state, force, timeCode)
    if key not in _cache:
        _cache[key] = {}
    _cache[key][timeCode.GetValue()] = (data, state)
    for prim in sourcePrims:
        keys = _sourcePrims.get(prim, set())
        keys.add(key)
        _sourcePrims[prim] = keys
    for prim in consumerPrims:
        keys = _consumerPrims.get(prim, set())
        keys.add(key)
        _consumerPrims[prim] = keys


def get(key: Any, state: Any = None, default: Any = None, *, timeCode: Usd.TimeCode = Usd.TimeCode.Default()) -> Any:
    """
    Function to retrieve data from the cache. See `put` for more information.
    If state is None, then state check is skipped.

    Returns `default` if no data is found.
    """
    if key in _cache and timeCode.GetValue() in _cache[key]:
        d, s = _cache[key][timeCode.GetValue()]
        if state is None or s == state:
            logger.info("[py-cache][**HIT**]:get(%s, %s, ..., tc=%s)", key, state, timeCode)
            return d
        logger.info(
            "[py-cache]:[**MISMATCH**]:get(%s, %s, ..., tc=%s) .. mismatched state(%s)* -> default ",
            key,
            state,
            s,
            timeCode,
        )
        return default
    logger.info("[py-cache]:[**MISS**]:get(%s, %s, ..., tc=%s) *miss* -> default", key, state, timeCode)
    return default


def clear():
    logger.info("[py-cache]:clear()")
    _cache.clear()
    _sourcePrims.clear()
    _consumerPrims.clear()
