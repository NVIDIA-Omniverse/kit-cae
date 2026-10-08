# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""In-memory Data Source."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from threading import Lock
from typing import Any
from uuid import uuid4

from ._data import DataSource
from ._errors import DatasetResolutionError
from ._usd_time import DatasetSample


class MemoryDataSource(DataSource):
    """Warp SimData datasets held in memory, published under the keys recipes select.

    A recipe selects a dataset by the path of the prim its ``target`` relationship
    points at, the **key**, plus an optional **identifier** string. Register a
    dataset under a key through the constructor mapping or :meth:`publish`. A
    ``(key, identifier)`` tuple in the mapping, or ``identifier=`` on ``publish``,
    registers a qualified dataset for recipes that set an identifier.

    Publishing again under the same key replaces the dataset and marks it changed:
    the next Update reruns the operators that read it and reuses everything else.
    Frame Data created by :meth:`~DataSource.at_time` keeps the datasets it saw
    until it is released, so an Update never observes a replacement midway.

    Pass ``time`` to publish one sample of a time-varying dataset. ``at_time(t)``
    then serves the latest sample at or before ``t``, or the first sample when
    ``t`` precedes the series. Without ``time`` a dataset is static and serves
    every time. Publishing a static dataset discards any samples under that key,
    and publishing a sample discards a static dataset.

    A dataset is served on the device it was built on; a recipe evaluated on a
    different device fails to resolve. Fields are attached under the recipe's role
    names at resolution time, so the dataset's own field names are what the
    recipe's field selections must name. An empty source is valid for discovery
    or cleanup runs.
    """

    def __init__(self, datasets: Mapping[Any, Any] | None = None):
        super().__init__()
        self._namespace = uuid4().hex
        self._registry_lock = Lock()
        self._generation = 0
        self._entries: dict[tuple[str, str], _Entry] = {}
        for item, dataset in (datasets or {}).items():
            key, identifier = _split_key(item)
            self.publish(key, dataset, identifier=identifier)

    def publish(self, key: str, dataset, *, identifier: str = "", time: float | None = None) -> None:
        """Register ``dataset`` under ``key`` (and ``identifier``), replacing what was there.

        ``dataset`` must be a Warp SimData dataset. With ``time`` in seconds it becomes
        one sample of a time-varying series under that key; without it the dataset is
        static. Each call marks the key changed for the next Update.
        """
        _require_dataset(dataset)
        key, identifier = str(key), str(identifier)
        sample_time = None if time is None else float(time)
        with self._registry_lock:
            if self._closed:
                raise DatasetResolutionError("cannot publish to a closed Data Source")
            self._generation += 1
            version = _Version(dataset, self._generation, sample_time)
            entry = self._entries.setdefault((key, identifier), _Entry())
            if sample_time is None:
                entry.static = version
                entry.samples.clear()
            else:
                entry.static = None
                entry.samples[sample_time] = version

    def remove(self, key: str, *, identifier: str = "") -> None:
        """Forget every dataset published under ``key`` and ``identifier``."""
        with self._registry_lock:
            try:
                del self._entries[(str(key), str(identifier))]
            except KeyError as exc:
                raise KeyError(f"no dataset published under {(str(key), str(identifier))!r}") from exc

    def published(self) -> tuple[tuple[str, str], ...]:
        """Return the ``(key, identifier)`` pairs that currently have a dataset, sorted."""
        with self._registry_lock:
            return tuple(sorted(self._entries))

    def resolves(self, key: str) -> bool:
        with self._registry_lock:
            return any(entry_key == str(key) for entry_key, _identifier in self._entries)

    def frame_at_time(self, time_seconds: float):
        with self._registry_lock:
            selected = {
                item: version
                for item, entry in self._entries.items()
                if (version := entry.select(time_seconds)) is not None
            }
        return _MemoryFrame(self._namespace, selected)

    def release_resources(self) -> None:
        with self._registry_lock:
            self._entries = {}


@dataclass(frozen=True)
class _Version:
    """One published dataset; frames hold these so replacements never reach an in-flight Update."""

    dataset: Any
    generation: int
    time: float | None


@dataclass
class _Entry:
    """Everything published under one ``(key, identifier)``: a static dataset or timed samples."""

    static: _Version | None = None
    samples: dict[float, _Version] = field(default_factory=dict)

    def select(self, time_seconds: float) -> _Version | None:
        if self.static is not None:
            return self.static
        if not self.samples:
            return None
        times = sorted(self.samples)
        return self.samples[times[max(0, bisect_right(times, time_seconds) - 1)]]


class _MemoryFrame:
    """The versions one Frame Data lease sees; both description and resolution read this map."""

    def __init__(self, namespace: str, selected: Mapping[tuple[str, str], _Version]):
        self._namespace = namespace
        self._selected = dict(selected)

    def _version(self, key: str, identifier: str) -> _Version:
        try:
            return self._selected[(str(key), str(identifier))]
        except KeyError as exc:
            available = ", ".join(repr(item) for item in sorted(self._selected)) or "<none>"
            raise DatasetResolutionError(
                f"dataset {(str(key), str(identifier))!r} was not found; published datasets: {available}"
            ) from exc

    def describe_component(self, key: str, identifier: str, component: str):
        """Identify a component by its published generation so unchanged data is reused."""
        version = self._version(key, identifier)
        identity = (self._namespace, str(key), str(identifier), component, version.generation)
        return identity, DatasetSample(version.time)

    def resolve_dataset(self, key: str, *, identifier: str, device: str, fields: Mapping[str, Sequence[str]]):
        """Return a copy of the published dataset carrying the selected fields under their role names."""
        version = self._version(key, identifier)
        dataset = version.dataset
        if str(dataset.device) != str(device):
            raise DatasetResolutionError(
                f"dataset {(str(key), str(identifier))!r} lives on {dataset.device!s} but the recipe evaluates on "
                f"{device}; publish it on {device} or create the Session with device={dataset.device!s}"
            )
        resolved = dataset.shallow_copy(pass_fields=False)
        for role, names in fields.items():
            resolved.add_field(role, _select_field(dataset, key, identifier, role, names), warn_if_exists=False)
        return resolved


def _split_key(item) -> tuple[str, str]:
    if isinstance(item, str):
        return item, ""
    if isinstance(item, tuple) and len(item) == 2 and all(isinstance(part, str) for part in item):
        return item
    raise TypeError(f"dataset keys are a path string or a (path, identifier) tuple of strings; got {item!r}")


def _require_dataset(dataset) -> None:
    for name in ("device", "shallow_copy", "get_field", "has_field", "get_field_names"):
        if not hasattr(dataset, name):
            raise TypeError(
                f"MemoryDataSource holds Warp SimData datasets; got {type(dataset).__name__}. Build one with a "
                "warp_simdata data model (for example data_models.vtk.unstructured_grid.create_dataset) or "
                "ovcae.data.dataset_from_arrays, and attach fields with Dataset.add_field."
            )


def _select_field(dataset, key: str, identifier: str, role: str, names: Sequence[str]):
    """Return the dataset field for ``names``: one field as is, several scalars combined into a vector."""
    names = tuple(str(name) for name in names)
    missing = [name for name in names if not dataset.has_field(name)]
    if missing or not names:
        available = ", ".join(sorted(dataset.get_field_names())) or "<none>"
        raise DatasetResolutionError(
            f"field selection {role!r} names {list(names)} but dataset {(str(key), str(identifier))!r} has: {available}"
        )
    if len(names) == 1:
        return dataset.get_field(names[0])
    try:
        import warp_simdata as simdata

        components = [dataset.get_field(name) for name in names]
        associations = {component.association for component in components}
        if len(associations) != 1:
            raise ValueError("component fields have different associations")
        if any(simdata.utils.is_vector_dtype(component.dtype) for component in components):
            raise ValueError("component fields must be scalars")
        return simdata.Field.from_arrays([component.to_array() for component in components], associations.pop())
    except Exception as exc:
        raise DatasetResolutionError(f"could not combine {list(names)} into the {role!r} vector field: {exc}") from exc
