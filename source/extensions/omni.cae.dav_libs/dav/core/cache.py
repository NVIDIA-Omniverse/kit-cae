# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Caching utilities for DAV."""

import inspect
from functools import wraps


def _make_cache_key_element(obj):
    """Convert an object to a cache key element, using id() for types."""
    if isinstance(obj, type):
        # Use id for types to avoid holding strong references
        return id(obj)
    elif isinstance(obj, (tuple, list)):
        # Recursively convert elements of tuples and lists
        return tuple(_make_cache_key_element(e) for e in obj)
    else:
        # Use hash for everything else, will fail naturally if not hashable
        return hash(obj)


def cached(func=None, *, aot=None, aot_roles=None):
    """Decorator to cache the result of a function based on input arguments.

    Can be used as a plain decorator or as a decorator factory with keyword
    arguments:

    .. code-block:: python

        @dav.cached                               # plain decorator
        def get_kernel(data_model): ...

        @dav.cached(aot="operators.centroid")     # decorator factory
        def get_kernel(data_model): ...

    The ``aot`` keyword opts the function into the two-level recording system
    described in :mod:`dav.core.recorder`:

    - **Level 1** (always active): every cache miss whose result is a Python
      ``type`` is registered in the process-global
      ``dav.core.recorder._class_registry`` so that any class object can later
      be reverse-mapped to the factory function and original arguments that
      created it.
    - **Level 2** (opt-in via ``aot=``): cache misses are forwarded to the
      currently active :class:`~dav.core.recorder.Recorder` (if any) and
      appended to ``wrapper._aot_observations`` for post-hoc inspection via
      :func:`~dav.core.recorder.build_config_from_cache`.

    Args:
        func: The function to cache.  Provided automatically when used as
            ``@cached`` (plain); omit when calling with kwargs.
        aot: Dot-notation path (e.g. ``"operators.centroid"``) that identifies
            this function's slot in the AOT configuration tree.
        aot_roles: Optional mapping of parameter name → sub-role name for
            operators that accept more than one data model.  For example,
            ``{"data_model": "dataset", "positions_data_model": "positions"}``
            causes the recorder to emit separate
            ``operators.probe.dataset.data_models`` and
            ``operators.probe.positions.data_models`` sections instead of a
            single flat ``operators.probe.data_models`` section.  Parameters
            absent from this dict are routed using the default (no sub-key)
            behaviour.  Only meaningful when ``aot`` is also set.

    Returns:
        A wrapped function with ``cache``, ``cache_clear()``, and
        ``cache_info()`` attributes.

    Note:
        For *type* arguments the cache key uses ``id()`` rather than ``hash()``
        because types are typically module-level singletons and ``id()`` avoids
        holding strong references.
    """
    if func is None:
        # Called as @cached(aot=...) — return a single-argument decorator.
        def decorator(f):
            return _make_cached(f, aot=aot, aot_roles=aot_roles)

        return decorator
    # Called as @cached — apply directly.
    return _make_cached(func, aot=None, aot_roles=None)


def _make_cached(func, aot=None, aot_roles=None):
    """Internal factory that builds the caching wrapper."""
    cache = {}
    sig = inspect.signature(func)
    # Pre-compute the full qualified name once; used by Level-1 registry.
    factory_name = func.__module__ + "." + func.__qualname__

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Bind all args/kwargs to parameter names, then sort by name so that
        # f(1, 2) and f(a=1, b=2) produce the same cache key.
        try:
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            cache_key = tuple(sorted((k, _make_cache_key_element(v)) for k, v in bound.arguments.items()))
        except TypeError as e:
            raise TypeError(f"Cannot cache function '{func.__name__}' with unhashable arguments. All arguments must be hashable. Original error: {e}") from e

        if cache_key not in cache:
            result = func(*args, **kwargs)
            cache[cache_key] = result

            # ------------------------------------------------------------------
            # Level 1: register result in the class registry if it is a type.
            # This lets build_config_from_cache() and Recorder.get_config()
            # reverse-map any data-model or field-model class back to the
            # factory call that produced it (factory name + original args).
            # ------------------------------------------------------------------
            if isinstance(result, type):
                import dav.core.recorder as _rec

                _rec._class_registry[id(result)] = (factory_name, dict(bound.arguments))

            # ------------------------------------------------------------------
            # Level 2: notify recorder and persist observation (aot= only).
            # ------------------------------------------------------------------
            if aot is not None:
                bound_copy = dict(bound.arguments)
                wrapper._aot_observations.append(bound_copy)

                import dav.core.recorder as _rec

                if _rec._active_recorder is not None:
                    _rec._active_recorder.observe(aot, bound_copy, aot_roles or {})

                # Continuous file output when dav.config.aot_record_path is set.
                import dav.core.config as _cfg

                if _cfg.aot_record_path is not None:
                    import json

                    config_dict = _rec.build_config_from_cache()
                    with open(_cfg.aot_record_path, "w") as f:
                        json.dump(config_dict, f, indent=2)

        return cache[cache_key]

    # Expose cache for introspection and testing
    wrapper.cache = cache
    wrapper.cache_clear = lambda: cache.clear()
    wrapper.cache_info = lambda: {"size": len(cache), "keys": list(cache.keys())}

    if aot is not None:
        wrapper._aot_path = aot
        wrapper._aot_roles = aot_roles or {}
        wrapper._aot_observations = []
        # Register this wrapper so build_config_from_cache() can find it later.
        import dav.core.recorder as _rec

        _rec._aot_functions.append(wrapper)

    return wrapper
