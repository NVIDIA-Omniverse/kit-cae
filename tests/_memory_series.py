# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Test-only in-memory discrete samples for change-detection prototyping.

Publish copies producer-owned NumPy data. Existing Frame Data leases keep the
selected versions even after replacement. This is not a mutable solver adapter.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from threading import Lock
from types import MappingProxyType
from uuid import uuid4

from ovcae import DataSource
from ovcae._errors import DatasetResolutionError


@dataclass(frozen=True)
class Sample:
    """One immutable static or discrete native sample description."""

    time: float | None
    generation: int


@dataclass(frozen=True)
class _Version:
    """Immutable association between a sample description and fixture-owned payload.

    Active frames keep this object alive after the series replaces its registry entry.
    """

    sample: Sample
    value: object


class MemorySeries:
    """Publish independent geometry, topology, and ``field:<name>`` samples.

    Geometry is an N x 3 NumPy array. Topology is a mapping containing
    cell_types, connectivity_offsets, and connectivity, or an empty mapping for
    a point cloud. A field is a mapping with values and association. Native times
    are seconds; None denotes static content. Each component is either static or
    discrete. Discrete selection holds the preceding sample, clamping to the
    first sample before the series begins. Publish again to replace a sample.
    """

    def __init__(self):
        """Create an isolated source namespace and synchronized component-sample registry."""
        self._namespace = uuid4().hex
        self._lock = Lock()
        self._generation = 0
        self._samples = {}

    def publish(self, key: str, *, component: str, value, time: float | None = None, identifier: str = ""):
        """Copy one component into immutable storage and assign a replacement generation.

        Validate the component name and finite native time, freeze caller-owned data, then
        replace the sample under the registry lock. Static and discrete schedules cannot
        mix within a component. Existing frames retain their old version objects. Multiple
        component publications are separate operations, not an atomic dataset transaction.
        """
        if component not in {"geometry", "topology"} and not (
            component.startswith("field:") and component.removeprefix("field:")
        ):
            raise ValueError("component must be geometry, topology, or field:<name>")
        if time is not None:
            time = float(time)
            if not math.isfinite(time):
                raise ValueError("native sample time must be finite")
        frozen = _freeze(value)
        index = (str(key), str(identifier), component)
        with self._lock:
            samples = self._samples.setdefault(index, {})
            if samples and ((None in samples) != (time is None)):
                raise ValueError("a component cannot mix static and discrete samples")
            self._generation += 1
            samples[time] = _Version(Sample(time, self._generation), frozen)

    def _frame_at_time(self, time):
        """Select and retain one version per component under the registry lock.

        Hold the preceding discrete sample, clamping to the first before the series begins;
        static components always select their static version. Copy only the selection map,
        not payloads, so later publications cannot change this frame's selected versions.
        """
        if not math.isfinite(time):
            raise ValueError("requested time must be finite")
        with self._lock:
            selected = {}
            for key, samples in self._samples.items():
                if None in samples:
                    selected[key] = samples[None]
                else:
                    preceding = [t for t in samples if t <= time]
                    selected[key] = samples[max(preceding) if preceding else min(samples)]
        return _SeriesFrame(self._namespace, selected)


class _SeriesSource(DataSource):
    """Test-only Data Source over MemorySeries.

    It subclasses the public base so the fixture exercises the real lease lifecycle;
    created frames retain their selected versions independently. Not shipped.
    """

    def __init__(self, series):
        """Keep the producer series available for subsequent frame selection."""
        super().__init__()
        self._series = series

    def resolves(self, key):
        """Claim any key with at least one published component."""
        with self._series._lock:
            return any(index[0] == str(key) for index in self._series._samples)

    def frame_at_time(self, time):
        """Create a stable frame selection using the series' held-sample policy."""
        return self._series._frame_at_time(time)

    def release_resources(self):
        """Release the producer reference; existing frames retain their own selected versions."""
        self._series = None


class _SeriesFrame:
    """Stable scientific sample selection used by one Frame Data lease.

    Both metadata description and loading consult the same retained version map.
    This prevents a replacement between description and resolution from pairing old
    identities with new payloads. No operator state or execution history belongs here.
    """

    def __init__(self, namespace, selected):
        """Retain the source namespace and independently selected immutable versions."""
        self._namespace = namespace
        self._selected = selected

    def _version(self, key, identifier, component):
        """Find a selected component or raise DatasetResolutionError with its dataset identity."""
        try:
            return self._selected[(str(key), identifier, component)]
        except KeyError as exc:
            raise DatasetResolutionError(f"missing {component} sample for dataset {(key, identifier)!r}") from exc

    def describe_component(self, key, identifier, component):
        """Return opaque comparison identity and native sample metadata without loading arrays.

        Namespace, dataset/qualifier, component, and sample generation distinguish sources
        and same-time replacements. Unrelated publications do not alter existing versions.
        """
        version = self._version(key, identifier, component)
        identity = (self._namespace, str(key), identifier, component, version.sample)
        return identity, version.sample

    def resolve_dataset(self, key, *, identifier, device, fields):
        """Build a fresh Warp dataset from the versions described by this frame.

        Combine retained topology/geometry with the requested source fields using the
        existing memory-array conversion helpers. Fresh device storage allows downstream
        field interpretation without mutating the fixture's immutable source payloads.
        """
        from ovcae.data import add_fields, dataset_from_arrays

        topology = self._version(key, identifier, "topology").value
        geometry = self._version(key, identifier, "geometry").value
        selected = {name for names in fields.values() for name in names}
        value = {
            **topology,
            "points": geometry,
            "fields": {name: self._version(key, identifier, f"field:{name}").value for name in selected},
        }
        dataset = dataset_from_arrays(value, device)
        add_fields(dataset, value, fields, device)
        return dataset


def _freeze(value):
    """Recursively copy mapping/array values into immutable fixture-owned storage.

    Mappings become read-only views, strings remain strings, and numeric arrays use a
    bytes-backed NumPy view whose write flag cannot be re-enabled. Reject object arrays
    rather than retaining mutable Python objects inside a supposedly stable sample.
    """
    import numpy as np

    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, str):
        return value
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise TypeError("sample arrays must not contain Python objects")
    # An immutable bytes owner prevents callers from re-enabling NumPy writes.
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def open_series(series: MemorySeries) -> DataSource:
    """Wrap a series in the test-only Data Source subclass."""
    return _SeriesSource(series)
