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
CAE Data Python Cache
=====================

Time-code–aware, USD-change-driven in-memory cache for computed data such as
DAV datasets, meshes, fields, and other derived results.

Overview
--------
Data is stored under an arbitrary *key* and optionally scoped to a
``Usd.TimeCode``.  Each entry may carry optional *state* metadata so callers
can detect stale entries without a full recompute.

Cached entries are automatically discarded when USD change notices arrive for
*watched* prims.  Watch rules are expressed via :class:`PrimWatch` objects
passed to :func:`put_ex` (or implicitly through the legacy :func:`put`).

Invalidation Modes
------------------
Each :class:`PrimWatch` carries an ``on`` mode that controls which USD change
events trigger invalidation:

``"any"`` *(default)*
    Invalidate on **property update** (value set, or attribute added/removed)
    *or* **structural resync** (composition change, prim type change, etc.).
    This is the most sensitive mode and is equivalent to the former
    ``sourcePrims`` behaviour in :func:`put`.

``"update"``
    Invalidate on **property update** only.  Structural resyncs (e.g. an API
    schema applied to the prim) do *not* trigger invalidation.  Combine with
    ``schemas`` to restrict which property changes count.

``"resync"``
    Invalidate on **structural resync** only.  Property value changes do *not*
    trigger invalidation.  Equivalent to the former ``consumerPrims`` behaviour
    in :func:`put`.

``"delete"``
    Invalidate only when the prim is **deleted** from the stage — that is, a
    structural resync after which ``prim.IsValid()`` returns ``False``.

Schema Filtering
----------------
Pass a list of USD schema types (Python classes or schema type-name strings)
via ``PrimWatch.schemas`` to restrict property-update checks to properties
declared by those schemas::

    PrimWatch(prim, on="update", schemas=[cae.FieldArray])

With the above, only a change to a property listed in the ``CaeFieldArray``
schema definition (e.g. ``fileNames``, ``fieldAssociation``) will trigger
invalidation.  Changes to other attributes on the same prim are ignored.

Schema filtering has **no effect** for ``on="resync"`` or ``on="delete"``
because structural resyncs do not carry property-level information.

For ``on="any"`` with ``schemas``, property updates are filtered by schema but
structural resyncs still trigger invalidation unconditionally.

Property name sets are pre-computed **once** at :func:`put_ex` call time via
``Usd.SchemaRegistry`` and stored on the :class:`PrimWatch` object, keeping
:meth:`Listener.on_objects_changed` as fast as possible.

API
---
:func:`put_ex`
    Primary cache-insertion function with full :class:`PrimWatch` control.
:func:`put`
    Legacy API.  Translates ``sourcePrims`` → ``PrimWatch(on="any")`` and
    ``consumerPrims`` → ``PrimWatch(on="resync")``, then calls
    :func:`put_ex`.  All existing call sites continue to work unchanged.
:func:`get`
    Retrieve a cached entry, optionally validating state metadata.
:func:`remove`
    Explicitly evict a single cache entry.
:func:`clear`
    Discard all entries and watches.

Source-Prim Expansion
---------------------
When :func:`put` translates ``sourcePrims`` it first expands each prim to
include all transitively related ``CaeDataSet`` and ``CaeFieldArray`` prims
(via field relationships) using :func:`_expand_source_prims`.  This expansion
does **not** occur for watches registered directly via :func:`put_ex` — callers
have explicit control over exactly which prims they watch.

Internal Architecture
---------------------
State is kept in two module-level structures:

``_cache``
    ``dict[key, dict[TimeCode, (data, state)]]`` — the actual cached data.

``_watches``
    ``dict[Usd.Prim, dict[key, list[PrimWatch]]]`` — maps each watched prim to
    the set of cache keys that depend on it, together with the
    :class:`PrimWatch` that describes *when* to invalidate.  A single
    (prim, key) pair may have multiple watches (e.g. when :func:`put` registers
    the same prim as both source and consumer with different modes).

``_has_schema_watches``
    Boolean flag set to ``True`` as soon as any schema-filtered watch is
    registered.  Used by :meth:`Listener.on_objects_changed` to skip building
    the per-prim changed-property map when no schema filtering is in effect,
    keeping the common case fast.
"""

from dataclasses import dataclass, field
from logging import getLogger
from typing import Any, Literal, NamedTuple, Optional

from omni.kit import app
from omni.stageupdate import get_stage_update_interface
from omni.usd import get_context, get_context_from_stage_id
from pxr import Sdf, Tf, Usd

from . import settings, usd_utils

logger = getLogger(__name__)

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

InvalidateOn = Literal["any", "update", "resync", "delete"]
"""Trigger mode for :class:`PrimWatch`.

Controls which USD change events cause a cache entry to be discarded.

Values
------
``"any"``
    Property update **or** structural resync.  Replaces the former
    ``sourcePrims`` argument to :func:`put`.
``"update"``
    Property update only; structural resyncs are ignored.
``"resync"``
    Structural resync only; property value changes are ignored.  Replaces the
    former ``consumerPrims`` argument to :func:`put`.
``"delete"``
    Structural resync that results in the prim no longer being valid on the
    stage (i.e. the prim was deleted).
"""

# Sentinel used as a cheap default value for dict.get() calls in the hot path.
_EMPTY_FROZENSET: frozenset = frozenset()


@dataclass
class PrimWatch:
    """Describes a USD prim to observe and the conditions that trigger cache invalidation.

    Pass a list of :class:`PrimWatch` instances to :func:`put_ex` to control
    precisely when a cached entry should be discarded.

    Attributes
    ----------
    prim :
        The USD prim to watch.
    on :
        Invalidation trigger mode (see :data:`InvalidateOn`).  Defaults to
        ``"any"``.
    schemas :
        Optional list of USD schema types (Python classes such as
        ``cae.FieldArray``) or schema type-name strings (e.g.
        ``"CaeFieldArray"``).  When provided, property-update triggers (modes
        ``"any"`` and ``"update"``) are restricted to properties declared by
        those schemas.  Ignored for modes ``"resync"`` and ``"delete"``.

    Notes
    -----
    The ``_schema_props`` attribute is **not** part of ``__init__``.  It is
    populated by :func:`put_ex` the first time a watch is registered, and
    contains the pre-resolved frozenset of property names derived from
    ``schemas``.

    Examples
    --------
    ::

        # Drop on any change — equivalent to old sourcePrims
        PrimWatch(prim)

        # Drop on structural resync only — equivalent to old consumerPrims
        PrimWatch(prim, on="resync")

        # Drop only when the prim is deleted
        PrimWatch(prim, on="delete")

        # Drop only when a CaeFieldArray schema property changes
        PrimWatch(field_prim, on="update", schemas=[cae.FieldArray])

        # Drop on CaeFieldArray property change OR structural resync
        PrimWatch(field_prim, on="any", schemas=[cae.FieldArray])
    """

    prim: Usd.Prim
    on: InvalidateOn = "any"
    schemas: Optional[list] = None  # list[type | str]

    # Pre-resolved property-name set; populated by put_ex(), not part of __init__.
    _schema_props: Optional[frozenset] = field(default=None, init=False, repr=False, compare=False)


# ---------------------------------------------------------------------------
# Schema resolution helpers (called at put_ex() time, not in the hot path)
# ---------------------------------------------------------------------------


def _get_schema_type_name(schema) -> Optional[str]:
    """Return the USD schema type-name string for a schema class or string.

    For string inputs the value is returned as-is.  For USD schema Python
    classes the type name is extracted via ``_GetStaticTfType()`` and the
    schema registry.
    """
    if isinstance(schema, str):
        return schema
    if hasattr(schema, "_GetStaticTfType"):
        tf_type = schema._GetStaticTfType()
        registry = Usd.SchemaRegistry()
        name = registry.GetConcreteSchemaTypeName(tf_type) or registry.GetAPISchemaTypeName(tf_type)
        if name:
            return name
        logger.warning("[py-cache]: could not determine USD schema type name for %r", schema)
        return None
    logger.warning("[py-cache]: %r is not a string or a USD schema class, skipping", schema)
    return None


def _resolve_schema_props(schemas: list) -> Optional[frozenset]:
    """Pre-compute the set of USD property names declared by the given schemas.

    Called once per :class:`PrimWatch` at :func:`put_ex` time so that
    :meth:`Listener.on_objects_changed` can use a fast set intersection rather
    than performing registry lookups on every notice.

    Parameters
    ----------
    schemas:
        List of USD schema Python classes or schema type-name strings.

    Returns
    -------
    frozenset[str] | None
        Frozenset of property names, or ``None`` if ``schemas`` is empty or
        no properties could be resolved.
    """
    if not schemas:
        return None

    registry = Usd.SchemaRegistry()
    props: set[str] = set()

    for schema in schemas:
        schema_name = _get_schema_type_name(schema)
        if schema_name is None:
            continue

        # Try API schema first (matches change_tracker.py convention), then concrete.
        defn = registry.FindAppliedAPIPrimDefinition(schema_name) or registry.FindConcretePrimDefinition(schema_name)
        if defn is None:
            logger.warning("[py-cache]: schema %r not found in registry, skipping", schema_name)
            continue

        props.update(defn.GetPropertyNames())

    return frozenset(props) if props else None


# ---------------------------------------------------------------------------
# Internal types
# ---------------------------------------------------------------------------


class _CacheEntry(NamedTuple):
    data: Any
    state: Any


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_cache: dict[Any, dict[Usd.TimeCode, _CacheEntry]] = {}

# prim -> key -> list[PrimWatch]
# A list is used per (prim, key) pair because the same prim may legitimately
# appear with different modes (e.g. put() registers it as both "any" and
# "resync" when it appears in both sourcePrims and consumerPrims).
_watches: dict[Usd.Prim, dict[Any, list[PrimWatch]]] = {}

# Set to True the first time a schema-filtered watch is registered.
# Allows on_objects_changed() to skip building the per-prim property map when
# no schema filtering is in use.
_has_schema_watches: bool = False


# ---------------------------------------------------------------------------
# USD change listener
# ---------------------------------------------------------------------------


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
                settings.SettingsKeys.CACHE_MODE,
                lambda item, event_type: clear() if settings.get_cache_mode() == "disabled" else None,
            )
        )

    def on_attach(self, stageId, metersPerUnit):
        ctx = get_context_from_stage_id(stageId)
        self._stage = ctx.get_stage()
        self._listener = Tf.Notice.Register(Usd.Notice.ObjectsChanged, self.on_objects_changed, self._stage)

    def on_detach(self):
        clear()
        self._stage = None
        self._listener = None

    def on_objects_changed(self, notice: Usd.Notice.ObjectsChanged, stage):
        if stage != self._stage or not isinstance(stage, Usd.Stage):
            return

        to_remove: set[Any] = set()

        # --- Property-update paths ---
        # Prim paths whose properties have changed value, or where a property
        # was added/removed (property-level resync).
        updatedpaths: set[Sdf.Path] = {x.GetPrimPath() for x in notice.GetChangedInfoOnlyPaths() if x.IsPropertyPath()}
        updatedpaths.update(x.GetPrimPath() for x in notice.GetResyncedPaths() if x.IsPropertyPath())

        # --- Structural resync paths ---
        # Prefix paths covering prim-level structural changes: composition
        # changes, prim creation/deletion, type changes, etc.
        resyncedpaths = notice.GetResyncedPaths()

        logger.info(
            "[py-cache]:on_objects_changed: updated-prims=%d resynced-paths=%d watched-prims=%d",
            len(updatedpaths),
            len(resyncedpaths),
            len(_watches),
        )

        # --- Per-prim property-name map (built lazily) ---
        # Only constructed when schema-filtered watches exist AND there are
        # property updates to inspect.  Maps prim path → set of changed
        # property names (the last path element, e.g. "fileNames").
        changed_props: Optional[dict[Sdf.Path, set[str]]] = None
        if _has_schema_watches and updatedpaths:
            changed_props = {}
            for x in notice.GetChangedInfoOnlyPaths():
                if x.IsPropertyPath():
                    changed_props.setdefault(x.GetPrimPath(), set()).add(x.name)
            for x in notice.GetResyncedPaths():
                if x.IsPropertyPath():
                    changed_props.setdefault(x.GetPrimPath(), set()).add(x.name)

        # --- Main invalidation loop ---
        for prim, key_watches in _watches.items():
            primpath = prim.GetPath()

            in_updated = primpath in updatedpaths
            resync_hit = any(primpath.HasPrefix(p) for p in resyncedpaths)
            # Computed lazily: only call IsValid() if we actually have a
            # "delete"-mode watch and a resync did hit this prim.
            was_deleted: Optional[bool] = None

            for key, watch_list in key_watches.items():
                if key in to_remove:
                    continue  # already flagged; skip remaining watches for this key

                for watch in watch_list:
                    invalidate = False

                    # Property-update check (modes: "any", "update")
                    if in_updated and watch.on in ("any", "update"):
                        if watch._schema_props is None:
                            # No schema filter: any property change triggers.
                            invalidate = True
                            logger.info(
                                "[py-cache][**INVALIDATE**]:key=%s <- property-update prim=%s mode=%r",
                                key,
                                primpath,
                                watch.on,
                            )
                        elif changed_props is not None:
                            prim_changed = changed_props.get(primpath, _EMPTY_FROZENSET)
                            matched = prim_changed & watch._schema_props
                            if matched:
                                invalidate = True
                                logger.info(
                                    "[py-cache][**INVALIDATE**]:key=%s <- schema-update prim=%s matched-props=%s mode=%r",
                                    key,
                                    primpath,
                                    matched,
                                    watch.on,
                                )

                    # Structural resync check (modes: "any", "resync", "delete")
                    if not invalidate and resync_hit:
                        if watch.on in ("any", "resync"):
                            invalidate = True
                            logger.info(
                                "[py-cache][**INVALIDATE**]:key=%s <- resync prim=%s mode=%r",
                                key,
                                primpath,
                                watch.on,
                            )
                        elif watch.on == "delete":
                            if was_deleted is None:
                                was_deleted = not prim.IsValid()
                            if was_deleted:
                                invalidate = True
                                logger.info(
                                    "[py-cache][**INVALIDATE**]:key=%s <- delete prim=%s",
                                    key,
                                    primpath,
                                )

                    if invalidate:
                        to_remove.add(key)
                        break  # no need to check other watches for this key

        if to_remove:
            logger.info("[py-cache]:on_objects_changed -> invalidating %d key(s): %s", len(to_remove), to_remove)

        for key in to_remove:
            remove(key)


_listener: Listener = None


def _initialize():
    global _listener
    _listener = Listener()


def _finalize():
    global _cache, _watches, _has_schema_watches, _listener
    _cache = {}
    _watches = {}
    _has_schema_watches = False
    _listener = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def remove(key: Any) -> bool:
    """Explicitly evict a single cache entry and its watch registrations.

    Parameters
    ----------
    key:
        The cache key to remove.

    Returns
    -------
    bool
        ``True`` if the key was present and removed, ``False`` otherwise.
    """
    if key not in _cache:
        return False

    logger.info("[py-cache]:remove(%s)", key)
    del _cache[key]

    prims_to_remove = []
    for prim, key_watches in _watches.items():
        if key in key_watches:
            del key_watches[key]
        if not key_watches:
            prims_to_remove.append(prim)
    for prim in prims_to_remove:
        del _watches[prim]

    # Note: _has_schema_watches is NOT reset here for efficiency.  It is a
    # conservative flag: once True it stays True until clear() or _finalize().
    return True


def _expand_source_prims(prims: list[Usd.Prim]) -> list[Usd.Prim]:
    """Expand source prims to include all transitively related DataSet and FieldArray prims.

    For each prim in the input list, if it's a CaeDataSet or CaeFieldArray,
    this function uses get_related_data_prims to find all transitively related
    prims via relationships.  This ensures that changes to any related prim
    (e.g. a FieldArray targeted by a field relationship from a DataSet) will
    invalidate the cache.

    Parameters
    ----------
    prims:
        List of source prims to expand.

    Returns
    -------
    list[Usd.Prim]
        All prims including the originals and their transitively related prims.
    """
    expanded = []
    seen_paths: set[Sdf.Path] = set()

    for prim in prims:
        if not prim or not prim.IsValid():
            continue

        related_prims = usd_utils.get_related_data_prims(prim, transitive=True, include_self=True)

        for related_prim in related_prims:
            prim_path = related_prim.GetPath()
            if prim_path not in seen_paths:
                seen_paths.add(prim_path)
                expanded.append(related_prim)

    return expanded


def put_ex(
    key: Any,
    data: Any,
    state: Any = None,
    prims: list[PrimWatch] = [],
    *,
    force: bool = False,
    timeCode: Usd.TimeCode = Usd.TimeCode.Default(),
) -> None:
    """Store data in the cache with explicit per-prim watch control.

    This is the primary cache-insertion function.  Use :func:`put` if you only
    need the legacy ``sourcePrims`` / ``consumerPrims`` behaviour.

    Parameters
    ----------
    key:
        Arbitrary hashable cache key.
    data:
        The data to cache.
    state:
        Optional metadata snapshot.  :func:`get` will return the cached value
        only when the supplied state matches the stored state (unless state is
        ``None``, which skips the check).
    prims:
        List of :class:`PrimWatch` instances that describe which prims to
        observe and under which conditions to discard this entry.  Schema
        property-name sets (``PrimWatch.schemas``) are resolved once here via
        ``Usd.SchemaRegistry`` and stored on the watch object so that the
        notice callback stays as fast as possible.
    force:
        If ``True`` the entry is stored regardless of the current cache mode
        setting.  Use this when caching is essential to the algorithm (e.g.
        passing intermediate datasets between pipeline stages).
    timeCode:
        Scopes the cached data to a specific USD time code.  Multiple time
        codes can be stored under the same key.

    Notes
    -----
    - If the same (prim, key) pair appears more than once in ``prims`` (with
      different modes), all watches are stored and any one of them triggering
      will invalidate the entry.
    - Invalid prims (``prim.IsValid() == False``) are silently skipped.
    - Schema props are only resolved once per watch instance; subsequent calls
      with the same :class:`PrimWatch` object are therefore cheap.

    Examples
    --------
    ::

        # Equivalent to put(..., sourcePrims=[dataset])
        cache.put_ex(key, data, prims=[cache.PrimWatch(dataset)])

        # Equivalent to put(..., consumerPrims=[prim])
        cache.put_ex(key, data, prims=[cache.PrimWatch(prim, on="resync")])

        # Drop only when the consumer prim is deleted
        cache.put_ex(key, data, prims=[cache.PrimWatch(prim, on="delete")])

        # Drop when a CaeFieldArray property changes on the source
        cache.put_ex(
            key, data,
            prims=[cache.PrimWatch(field_prim, on="update", schemas=[cae.FieldArray])],
        )
    """
    global _has_schema_watches

    if not force:
        cache_mode = settings.get_cache_mode()
        if cache_mode == "disabled":
            logger.info("[py-cache]:put_ex(%s, %s) -> skipped (cache mode is disabled)", key, state)
            return
        if cache_mode == "static-fields" and timeCode != Usd.TimeCode.EarliestTime():
            logger.info(
                "[py-cache]:put_ex(%s, %s) -> skipped (static-fields mode, non-earliest timeCode)",
                key,
                state,
            )
            return

    logger.info("[py-cache]:put_ex(%s, state=%s, force=%s, tc=%s)", key, state, force, timeCode)

    if key not in _cache:
        _cache[key] = {}
    _cache[key][timeCode] = _CacheEntry(data, state)

    for watch in prims:
        if not watch.prim or not watch.prim.IsValid():
            continue

        # Resolve schema property names once per watch instance.
        if watch.schemas is not None and watch._schema_props is None:
            watch._schema_props = _resolve_schema_props(watch.schemas)
            if watch._schema_props:
                _has_schema_watches = True

        _watches.setdefault(watch.prim, {}).setdefault(key, []).append(watch)


def put(
    key: Any,
    data: Any,
    state: Any = None,
    sourcePrims: list[Usd.Prim] = [],
    consumerPrims: list[Usd.Prim] = [],
    *,
    force: bool = False,
    timeCode: Usd.TimeCode = Usd.TimeCode.Default(),
) -> None:
    """Store data in the cache. (Legacy API — prefer :func:`put_ex`.)

    The ``key`` and ``data`` arguments work like any cache.  ``sourcePrims``
    and ``consumerPrims`` control automatic eviction.

    The cached data is automatically discarded if:

    - any prim in ``sourcePrims`` has a property updated or is structurally
      resynced (equivalent to ``PrimWatch(p, on="any")``);
    - any transitively related DataSet / FieldArray prim of a source prim is
      updated or resynced (see :func:`_expand_source_prims`);
    - any prim in ``consumerPrims`` is structurally resynced or deleted
      (equivalent to ``PrimWatch(p, on="resync")``).

    For finer control — including the new ``"update"`` and ``"delete"`` modes
    and schema-filtered invalidation — use :func:`put_ex` directly.

    Parameters
    ----------
    key:
        Arbitrary hashable cache key.
    data:
        The data to cache.
    state:
        Optional metadata snapshot validated by :func:`get`.
    sourcePrims:
        Prims whose changes (property updates *or* structural resyncs) should
        invalidate this entry.  Automatically expanded to include all
        transitively related DataSet and FieldArray prims.
    consumerPrims:
        Prims whose structural resyncs (including deletion) should invalidate
        this entry.  Property-only changes on these prims do *not* trigger
        eviction.
    force:
        Store regardless of the current cache mode setting.
    timeCode:
        Scopes the cached data to a specific USD time code.
    """
    expanded = _expand_source_prims(sourcePrims)
    prims = [PrimWatch(p, on="any") for p in expanded] + [PrimWatch(p, on="resync") for p in consumerPrims]
    put_ex(key, data, state, prims, force=force, timeCode=timeCode)


def get(key: Any, state: Any = None, default: Any = None, *, timeCode: Usd.TimeCode = Usd.TimeCode.Default()) -> Any:
    """Retrieve a cached entry.

    Parameters
    ----------
    key:
        The cache key to look up.
    state:
        If not ``None``, the cached entry is returned only when its stored
        state equals this value.  A mismatch returns ``default``.
    default:
        Value returned on a cache miss or state mismatch.
    timeCode:
        Time code scope to look up.

    Returns
    -------
    Any
        The cached data, or ``default`` on a miss / state mismatch.
    """
    if key in _cache and timeCode in _cache[key]:
        entry = _cache[key][timeCode]
        if state is None or entry.state == state:
            logger.info("[py-cache][**HIT**]:get(%s, %s, ..., tc=%s)", key, state, timeCode)
            return entry.data
        logger.info(
            "[py-cache]:[**MISMATCH**]:get(%s, %s, ..., tc=%s) .. mismatched state(%s)* -> default ",
            key,
            state,
            entry.state,
            timeCode,
        )
        return default
    logger.info("[py-cache]:[**MISS**]:get(%s, %s, ..., tc=%s) *miss* -> default", key, state, timeCode)
    return default


def clear() -> None:
    """Discard all cached entries and watch registrations."""
    global _has_schema_watches
    logger.info("[py-cache]:clear()")
    _cache.clear()
    _watches.clear()
    _has_schema_watches = False
