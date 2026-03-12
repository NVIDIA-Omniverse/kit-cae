# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Automatic AOT configuration generation via kernel specialization recording.

Overview
--------
Rather than hand-authoring ``dav/core/aot.py``, this module lets users run any
script or application session and then produce an AOT configuration dict that
exactly covers every kernel specialization that was actually exercised.  The
resulting dict has the same structure as ``dav.core.aot.configuration`` and can
be used directly as its replacement, or merged into an existing one.

Usage
-----
**Context manager** (recommended for scripts)::

    import dav

    with dav.Recorder() as rec:
        result = dav.operators.probe.compute(dataset, "velocity", positions)
        result = dav.operators.centroid.compute(mesh)

    config = rec.get_config()
    rec.save("aot_config.json")

**Standalone** (resume or inspect at any point)::

    rec = dav.Recorder()
    rec.start()
    # ... run operations ...
    rec.stop()
    config = rec.get_config()

**Post-hoc cache inspection** (zero setup — works after any normal session)::

    # After running arbitrary DAV operations (which JIT-compile kernels):
    config = dav.recorder.build_config_from_cache()

**Continuous file output** (long-running apps)::

    import dav
    dav.config.aot_record_path = "/tmp/dav_aot.json"
    # File is updated on every new kernel specialization encountered.

Architecture
------------
Recording is implemented as two co-operating levels inside ``@dav.cached``.

**Level 1 — Global class registry**

Every ``@dav.cached`` cache miss, regardless of decorator arguments, stores the
returned object in a process-global registry::

    _class_registry[id(result)] = (func.__qualname__, bound_args)

``bound_args`` is the ``dict`` produced by ``inspect.BoundArguments.arguments``
after applying defaults — it contains the original Python objects (dtype, length,
cell-type lists, etc.) exactly as passed by the caller.  This lets the recorder
resolve any class object back to the factory and arguments that produced it,
without requiring any ``__dav_spec__`` attribute on the class itself.

**Level 2 — Kernel factory interception**

Kernel factory functions are decorated with ``@dav.cached(aot="operators.<name>")``
(the ``aot`` argument is a dot-notation path into the AOT configuration tree).
On each cache miss in such a function the recorder:

1. Iterates the bound arguments of the ``get_kernel`` call.
2. For each argument that is a ``type``, looks it up in ``_class_registry`` to
   recover the factory qualname and original construction arguments.
3. Classifies the argument as a data model or field model based on the factory
   qualname (``"...data_models..."`` vs ``"...fields..."``).
4. Appends the resolved specialization to an internal observations list, keyed
   by the ``aot`` path (e.g. ``"operators.probe"``).

When no ``Recorder`` is active, Level 2 is a no-op.  Level 1 always runs because
the registry is needed by ``build_config_from_cache()`` even without a context
manager.

Config output format
--------------------
``Recorder.get_config()`` aggregates observations into a dict matching the
structure of ``dav.core.aot.configuration``:

- **field_models**: union of all field models seen across all operators.
  Per-operator ``field_models`` sections are also emitted under each operator
  entry, matching ``get_field_models("operators.<name>")``.
- **data_models**: top-level section left empty; per-operator sections are used
  instead (see below).
- **operators**: each operator gets an explicit ``data_models`` section listing
  only the data models actually used for that operator, with their full
  specialization arguments (e.g. cell-type lists for unstructured grids).  This
  avoids relying on the implicit top-level fallback and keeps the generated config
  unambiguous.
- **field_models**: uses the same list-of-specs format as ``dav.core.aot.configuration``
  (``scalar_types`` / ``vector_element_types`` are gone; types are embedded directly in
  each spec).
- **devices**: intentionally omitted — specify target devices via ``--devices``
  when running ``python -m dav.aot_compile``.

Multiple ``get_kernel`` functions within the same operator module (e.g.
``cell_faces``) are distinguished by their ``aot`` path
(``"operators.cell_faces.pass1"``, ``"operators.cell_faces.pass2"``).

Class registry lifetime
-----------------------
``_class_registry`` is process-global and grows monotonically; entries are never
evicted.  This is intentional: the ``@dav.cached`` functions already hold strong
references to the same objects, so the registry adds no meaningful memory
overhead beyond what the caches themselves hold.

Relationship to ``@dav.cached``
--------------------------------
``cached`` gains an optional ``aot`` keyword argument::

    @dav.cached(aot="operators.centroid")
    def get_kernel(data_model: dav.DataModel): ...

When ``aot`` is absent, behaviour is identical to the original ``@dav.cached``.
When present, the decorator opts the function into Level-2 recording in addition
to Level-1 registration.  The argument doubles as self-documentation: readers
can see at a glance which slot in the AOT config this kernel factory fills.
"""

import json
from typing import Optional

import warp as wp

__all__ = ["Recorder", "get_active_recorder", "build_config_from_cache", "save_config_from_cache"]

# ---------------------------------------------------------------------------
# Process-global state
# ---------------------------------------------------------------------------

# Maps id(result) -> (factory_qualname, bound_args) for every @dav.cached miss
# whose result is a Python type (class).
# factory_qualname = func.__module__ + "." + func.__qualname__
_class_registry: dict[int, tuple[str, dict]] = {}

# The currently active Recorder, or None.
_active_recorder: Optional["Recorder"] = None

# All @dav.cached(aot=...) wrappers — populated at decoration time.
# Used by build_config_from_cache() to iterate all aot observations.
_aot_functions: list = []


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _freeze(d: dict) -> tuple:
    """Make a dict hashable for use as a dict or set key."""
    items = []
    for k, v in sorted(d.items()):
        if isinstance(v, list):
            items.append((k, tuple(v)))
        elif isinstance(v, type):
            items.append((k, id(v)))
        else:
            try:
                hash(v)
                items.append((k, v))
            except TypeError:
                items.append((k, repr(v)))
    return tuple(items)


def _warp_type_name(dtype) -> str:
    """Return the warp attribute name for *dtype* (e.g. ``wp.float32`` → ``"float32"``)."""
    name = getattr(dtype, "__name__", None)
    if name and getattr(wp, name, None) is dtype:
        return name
    raise ValueError(f"Unrecognized Warp dtype: {dtype}")


def _classify_arg(arg_val):
    """Classify a type argument into one of three forms:

    - ``("data_model", package, module, cell_types)`` — a DAV data model class.
      *cell_types* is a list of element-type strings, or ``None`` for no-arg factories.
    - ``("field_model", factory_qualname, factory_bound_args)`` — a DAV field model class.
    - ``None`` — not a recognised DAV class (interpolator, unregistered type, etc.).
    """
    if not isinstance(arg_val, type):
        return None

    # --- Primary path: look up the class registry (populated by Level-1) ---
    if id(arg_val) in _class_registry:
        factory_qualname, factory_bound_args = _class_registry[id(arg_val)]
        parts = factory_qualname.split(".")

        if "data_models" in parts:
            dm_idx = parts.index("data_models")
            package = parts[dm_idx + 1]
            module = parts[dm_idx + 2]
            # Construction args: the first list-valued argument is the cell-type list.
            cell_types = None
            for val in factory_bound_args.values():
                if isinstance(val, list):
                    cell_types = val
                    break
            return ("data_model", package, module, cell_types)

        if "fields" in parts:
            return ("field_model", factory_qualname, factory_bound_args)

        # Registered but not a data model or field model (e.g. interpolator).
        return None

    # --- Fallback: no-arg data model factories that aren't @dav.cached ---
    # Such factories return a module-level class whose __module__ encodes
    # the dav.data_models.{package}.{module} path.
    mod_path = getattr(arg_val, "__module__", None)
    if mod_path:
        mod_parts = mod_path.split(".")
        if "data_models" in mod_parts:
            dm_idx = mod_parts.index("data_models")
            if dm_idx + 2 < len(mod_parts):
                package = mod_parts[dm_idx + 1]
                module = mod_parts[dm_idx + 2]
                return ("data_model", package, module, None)

    return None


def _factory_to_inner_spec(factory_qualname: str, factory_bound_args: dict) -> list | None:
    """Convert a factory qualname + bound args to an inner spec list.

    Returns ``["array", layout, dtype_str, length]`` for array models,
    ``["nanovdb", dtype_str]`` for nanovdb models, or ``None`` if unrecognised.
    """
    parts = factory_qualname.split(".")
    if "fields" not in parts:
        return None
    fields_idx = parts.index("fields")
    field_type = parts[fields_idx + 1] if fields_idx + 1 < len(parts) else None
    func_name = parts[-1]

    if field_type == "array":
        dtype = factory_bound_args.get("dtype")
        length = factory_bound_args.get("length")
        if dtype is not None and length is not None:
            layout = "SoA" if "SoA" in func_name else "AoS"
            return ["array", layout, _warp_type_name(dtype), length]

    elif field_type == "nanovdb":
        dtype = factory_bound_args.get("dtype")
        if dtype is not None:
            return ["nanovdb", _warp_type_name(dtype)]

    return None


def _build_field_models_config(factory_infos: list[tuple[str, dict]]) -> dict:
    """Build the ``field_models`` section from observed ``(factory_qualname, bound_args)`` pairs.

    Outputs the new list-of-specs format:
      ``array``         → list of ``[layout, dtype_str, length]``
      ``nanovdb``       → list of ``[dtype_str]``
      ``collection``    → list of inner specs ``["array"|"nanovdb", ...]``
      ``vector_reduced``→ list of ``[inner_spec, "component"|"magnitude", *args]``
    """
    array_specs: list = []
    nanovdb_specs: list = []
    collection_specs: list = []
    vr_entries: list = []

    for factory_qualname, factory_bound_args in factory_infos:
        parts = factory_qualname.split(".")
        if "fields" not in parts:
            continue
        fields_idx = parts.index("fields")
        field_type = parts[fields_idx + 1] if fields_idx + 1 < len(parts) else None
        func_name = parts[-1]

        if field_type == "array":
            dtype = factory_bound_args.get("dtype")
            length = factory_bound_args.get("length")
            if dtype is not None and length is not None:
                layout = "SoA" if "SoA" in func_name else "AoS"
                spec = [layout, _warp_type_name(dtype), length]
                if spec not in array_specs:
                    array_specs.append(spec)

        elif field_type == "nanovdb":
            dtype = factory_bound_args.get("dtype")
            if dtype is not None:
                spec = [_warp_type_name(dtype)]
                if spec not in nanovdb_specs:
                    nanovdb_specs.append(spec)

        elif field_type == "collection":
            base_model = factory_bound_args.get("field_model")
            if base_model is not None and id(base_model) in _class_registry:
                inner_qualname, inner_args = _class_registry[id(base_model)]
                inner_spec = _factory_to_inner_spec(inner_qualname, inner_args)
                if inner_spec is not None and inner_spec not in collection_specs:
                    collection_specs.append(inner_spec)

        elif field_type == "vector_reduced":
            inner_model = factory_bound_args.get("inner_field_model")
            component = factory_bound_args.get("component")
            magnitude = factory_bound_args.get("magnitude", False)
            if inner_model is not None and id(inner_model) in _class_registry:
                inner_qualname, inner_args = _class_registry[id(inner_model)]
                inner_spec = _factory_to_inner_spec(inner_qualname, inner_args)
                if inner_spec is None:
                    continue
                if component is not None:
                    entry = [inner_spec, "component", component]
                elif magnitude:
                    entry = [inner_spec, "magnitude"]
                else:
                    continue
                if entry not in vr_entries:
                    vr_entries.append(entry)

    config: dict = {}
    if array_specs:
        config["array"] = array_specs
    if nanovdb_specs:
        config["nanovdb"] = nanovdb_specs
    if collection_specs:
        config["collection"] = collection_specs
    if vr_entries:
        config["vector_reduced"] = vr_entries
    return config


def _build_data_models_config(data_models_seen: dict) -> dict:
    """Build a ``data_models`` config sub-dict from a ``{(package, module): set[cell_types]}`` mapping."""
    config: dict = {}
    for (package, module), cell_type_tuples in data_models_seen.items():
        if package not in config:
            config[package] = {}
        non_none = [list(t) for t in cell_type_tuples if t is not None]
        config[package][module] = non_none if non_none else {}
    return config


def _build_config_from_observations(observations: dict, roles: dict = None) -> dict:
    """Build an AOT configuration dict from a ``{aot_path: [bound_args, ...]}`` mapping.

    Args:
        observations: Mapping from aot paths (e.g. ``"operators.centroid"``) to
            lists of ``bound_args`` dicts captured on each kernel factory cache miss.
        roles: Optional mapping from aot paths to ``aot_roles`` dicts
            (parameter name → sub-role name).  When a parameter appears in the
            roles dict for its aot path its data model is recorded under a
            sub-key (e.g. ``operators.probe.dataset``) instead of the default
            flat ``operators.probe.data_models`` location.

    Returns:
        A dict with the same structure as ``dav.core.aot.configuration``.
    """
    if roles is None:
        roles = {}

    # Global union of all field models seen across every operator.
    all_field_models_seen: dict = {}
    operators_config: dict = {}

    for aot_path, obs_list in observations.items():
        if not obs_list:
            continue

        parts = aot_path.split(".")
        if not parts or parts[0] != "operators" or len(parts) < 2:
            continue

        operator_name = parts[1]
        sub_path = parts[2:]  # e.g. [] for simple operators, ["seeds"] for advection seeds

        # aot_roles for this path: param_name -> sub-role name (e.g. "dataset", "positions").
        # Parameters absent from this dict get role=None (default: flat data_models section).
        aot_roles = roles.get(aot_path, {})

        # role (str | None) -> {(package, module) -> set of frozen cell-type tuples}
        # role=None means the default flat "data_models" location.
        role_data_models: dict = {}
        # (factory_qualname, frozen_bound_args) -> (factory_qualname, bound_args)
        field_models_seen: dict = {}

        for bound_args in obs_list:
            for param_name, arg_val in bound_args.items():
                classification = _classify_arg(arg_val)
                if classification is None:
                    continue

                kind = classification[0]

                if kind == "data_model":
                    _, package, module, cell_types = classification
                    role = aot_roles.get(param_name)  # None = default location
                    if role not in role_data_models:
                        role_data_models[role] = {}
                    key = (package, module)
                    if key not in role_data_models[role]:
                        role_data_models[role][key] = set()
                    role_data_models[role][key].add(tuple(cell_types) if cell_types is not None else None)

                elif kind == "field_model":
                    _, factory_qualname, factory_bound_args = classification
                    fm_key = (factory_qualname, _freeze(factory_bound_args))
                    field_models_seen[fm_key] = (factory_qualname, factory_bound_args)
                    all_field_models_seen[fm_key] = (factory_qualname, factory_bound_args)

        # Convert field_models_seen -> config sub-dict.
        field_models_config = _build_field_models_config(list(field_models_seen.values()))

        # Navigate / create the nested operator entry.
        if operator_name not in operators_config:
            operators_config[operator_name] = {}
        target = operators_config[operator_name]
        for sub in sub_path:
            if sub not in target:
                target[sub] = {}
            target = target[sub]

        # Attach data_models, split by role.
        # role=None → target["data_models"]
        # role="dataset" → target["dataset"]["data_models"]
        for role, dm_seen in role_data_models.items():
            dm_config = _build_data_models_config(dm_seen)
            if not dm_config:
                continue
            if role is None:
                target["data_models"] = dm_config
            else:
                if role not in target:
                    target[role] = {}
                target[role]["data_models"] = dm_config

        if field_models_config:
            target["field_models"] = field_models_config

    # Also pick up field models that were created (and thus registered in
    # _class_registry by Level-1) but never passed as an argument to any
    # get_kernel call.  This covers field models created standalone without any
    # operator call in the session.
    for factory_qualname, factory_bound_args in _class_registry.values():
        if "fields" in factory_qualname.split("."):
            fm_key = (factory_qualname, _freeze(factory_bound_args))
            all_field_models_seen.setdefault(fm_key, (factory_qualname, factory_bound_args))

    all_fm_infos = list(all_field_models_seen.values())

    return {"field_models": _build_field_models_config(all_fm_infos), "data_models": {}, "operators": operators_config}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class Recorder:
    """Records kernel specializations during a DAV session for AOT config generation.

    Acts as a context manager or can be started/stopped manually.  Call
    ``get_config()`` to obtain the aggregated AOT configuration dict.
    """

    def __init__(self):
        # observations: aot_path -> list of bound_args dicts
        self._observations: dict[str, list] = {}
        # roles: aot_path -> aot_roles dict (param_name -> sub-role name)
        self._roles: dict[str, dict] = {}
        self._active = False

    def __enter__(self) -> "Recorder":
        """Activate this recorder as the process-global active recorder."""
        self.start()
        return self

    def __exit__(self, *_) -> None:
        """Deactivate this recorder."""
        self.stop()

    def start(self) -> None:
        """Activate this recorder (non-context-manager usage)."""
        global _active_recorder
        _active_recorder = self
        self._active = True

    def stop(self) -> None:
        """Deactivate this recorder (non-context-manager usage)."""
        global _active_recorder
        if _active_recorder is self:
            _active_recorder = None
        self._active = False

    def reset(self) -> None:
        """Clear all recorded observations without deactivating."""
        self._observations.clear()
        self._roles.clear()

    def observe(self, aot_path: str, bound_args: dict, aot_roles: dict = None) -> None:
        """Record one kernel factory cache miss.

        Called automatically by ``@dav.cached(aot=...)`` on cache miss when
        this recorder is active.  Not intended for direct use.

        Args:
            aot_path: The ``aot`` value from the decorator, e.g.
                ``"operators.centroid"``.
            bound_args: The ``BoundArguments.arguments`` dict for the
                ``get_kernel`` call, containing the actual class objects
                (data models, field models, etc.) as values.
            aot_roles: The ``aot_roles`` dict from the decorator, mapping
                parameter names to sub-role names (e.g.
                ``{"data_model": "dataset", "positions_data_model": "positions"}``).
        """
        if aot_path not in self._observations:
            self._observations[aot_path] = []
        self._observations[aot_path].append(dict(bound_args))
        if aot_roles:
            self._roles[aot_path] = aot_roles  # same dict for all calls at this path

    def get_config(self) -> dict:
        """Return an AOT configuration dict covering all observed specializations.

        The returned dict has the same structure as ``dav.core.aot.configuration``
        and can be used as a drop-in replacement or merged into an existing one.
        """
        return _build_config_from_observations(self._observations, self._roles)

    def save(self, path: str) -> None:
        """Write ``get_config()`` as JSON to *path*.

        Args:
            path: Destination file path.  Parent directories must exist.
        """
        with open(path, "w") as f:
            json.dump(self.get_config(), f, indent=2)


def get_active_recorder() -> Recorder | None:
    """Return the currently active ``Recorder``, or ``None``."""
    return _active_recorder


def build_config_from_cache() -> dict:
    """Build an AOT config from the current state of all ``@dav.cached`` caches.

    A post-hoc alternative to the ``Recorder`` context manager.  After running
    any DAV operations (which JIT-compile kernels and populate caches), call this
    function to produce a config dict covering all compiled specializations.

    Internally this inspects ``_class_registry`` (populated continuously) and
    all ``@dav.cached(aot=...)`` function caches to reconstruct what was used.

    Returns:
        A dict with the same structure as ``dav.core.aot.configuration``.
    """
    observations: dict = {}
    roles: dict = {}
    for wrapper in _aot_functions:
        path = wrapper._aot_path
        if path not in observations:
            observations[path] = []
        observations[path].extend(wrapper._aot_observations)
        if wrapper._aot_roles:
            roles[path] = wrapper._aot_roles
    return _build_config_from_observations(observations, roles)


def save_config_from_cache(path: str) -> None:
    """Write the AOT config derived from all cached specializations to a JSON file.

    Equivalent to calling :func:`build_config_from_cache` and writing the result
    as JSON to *path*.  Useful as a one-liner after any DAV session:

    .. code-block:: python

        import dav
        # ... run DAV operations ...
        dav.recorder.save_config_from_cache("aot_config.json")

    Args:
        path: Destination file path.  Parent directories must exist.
    """
    with open(path, "w") as f:
        json.dump(build_config_from_cache(), f, indent=2)
