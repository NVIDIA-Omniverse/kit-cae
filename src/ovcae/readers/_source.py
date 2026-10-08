# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Data Source over a usd-core stage whose datasets the CAE file-format plug-ins compose."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from pathlib import Path
from threading import RLock
from uuid import uuid4

from pxr import Usd

from ovcae import DatasetResolutionError, DataSourceOpenError
from ovcae.data import DatasetSample, DataSource, held_time

from ._timeline import dataset_time_samples


def _open_held_stage(path: str):
    """Open a private held stage without borrowing from an application stage cache."""
    with Usd.StageCacheContext(Usd.BlockStageCaches):
        stage = Usd.Stage.Open(path)
    if stage:
        stage.SetInterpolationType(Usd.InterpolationTypeHeld)
    return stage


class UsdDataSource(DataSource):
    """Resolve Dataset Keys against a usd-core stage.

    The stage must compose the same layers as the Execution Stage. ``open`` opens
    a private held-interpolation stage from a file; the constructor accepts a
    stage the application already owns and uses it as-is. Held selection always
    reads at an authored sample time, so the application's interpolation setting
    does not affect loaded values. Composed layers and backing assets must remain
    unchanged for this source's lifetime; open a new source after replacing them
    to discard cached timelines, identity evidence, and loaded arrays. There is
    no external-file monitoring or live reload.

    Loaded datasets are retained per dataset prim and device for the most recent
    selected sample, so acceleration structures and cached fields that operators
    build on them survive across Updates. Each resolution hands out a clone with
    its own field mapping; loaded field arrays are shared by name.
    """

    def __init__(self, stage):
        super().__init__()
        if not stage:
            raise DataSourceOpenError("UsdDataSource requires an open pxr.Usd.Stage")
        self._stage = stage
        self._namespace = uuid4().hex
        self._lock = RLock()
        self._timelines = {}
        self._claims = {}
        self._loaded = {}

    @classmethod
    def open(cls, path: str | Path, **options) -> UsdDataSource:
        if options:
            names = ", ".join(sorted(options))
            raise TypeError(f"unsupported UsdDataSource options: {names}")
        path = Path(path).resolve()
        if not path.is_file():
            raise DataSourceOpenError(f"Data Source asset {path} is not a file")
        stage = _open_held_stage(str(path))
        if not stage:
            raise DataSourceOpenError(f"could not open Data Source asset {path}")
        return cls(stage)

    def resolves(self, key: str) -> bool:
        """Claim keys that are, or compose, an OmniSciDataset; other prims belong to other sources."""
        key = str(key)
        with self._lock:
            if key not in self._claims:
                prim = self._stage.GetPrimAtPath(key) if self._stage else None
                self._claims[key] = bool(prim) and _composes_dataset(prim)
            return self._claims[key]

    def frame_at_time(self, time_seconds: float):
        """Create a frame whose selections are shared by description and loading."""
        if not isfinite(time_seconds):
            raise DatasetResolutionError("USD source time must be finite")
        time_code = Usd.TimeCode(float(time_seconds) * self._stage.GetTimeCodesPerSecond())
        return UsdFrameProvider(self, self._stage, float(time_seconds), time_code)

    def release_resources(self) -> None:
        self._timelines = {}
        self._claims = {}
        self._loaded = {}
        self._stage = None

    def loaded(self, identity, device: str, load):
        """Return the retained load for one dataset identity, replacing a stale sample.

        The key is the dataset prim and device; the value is the identity it was
        loaded for, the base Dataset, and the field arrays loaded from it. A frame
        that selects another sample of the same prim evicts the previous load, so
        stepping through time never accumulates device memory. Loading runs outside
        the lock; concurrent first loads of one prim may both complete and the
        later one wins, which is safe because loads are immutable.
        """
        _namespace, dataset_key, _sample = identity
        slot = (dataset_key, device)
        with self._lock:
            entry = self._loaded.get(slot)
            if entry is not None and entry[0] == identity:
                return entry[1], entry[2]
        dataset = load()
        with self._lock:
            entry = self._loaded.get(slot)
            if entry is not None and entry[0] == identity:
                return entry[1], entry[2]
            entry = (identity, dataset, {})
            self._loaded[slot] = entry
            return entry[1], entry[2]

    def timeline(self, prim):
        """Discover each resolved dataset's timeline once for this stable source."""
        key = (prim.GetStage().GetRootLayer().identifier, str(prim.GetPath()))
        with self._lock:
            if key not in self._timelines:
                self._timelines[key] = dataset_time_samples(prim)
            return key, self._timelines[key]


class UsdFrameProvider:
    """Resolve dataset samples shared by all components of a frame request.

    A selection stores the resolved prim, held loading time, and metadata evidence.
    Description never materializes scientific arrays. Loads go through the
    source's retained datasets, so an unchanged sample is materialized once.
    """

    def __init__(self, source, stage, time_seconds, time_code):
        self._source = source
        self._stage = stage
        self._time_seconds = time_seconds
        self._time_code = time_code
        self._selections = {}

    def _selection(self, key, identifier):
        """Bind a request to one dataset prim and held time for the frame lease."""
        request = (key, identifier)
        with self._source._lock:
            if request in self._selections:
                return self._selections[request]
            root = self._stage.GetPrimAtPath(key)
            if not root:
                raise DatasetResolutionError(f"dataset key {key!r} was not found in the Data Source")
            prim = _resolve_dataset_prim(root, identifier)
            dataset_key, samples = self._source.timeline(prim)
            rate = self._stage.GetTimeCodesPerSecond()
            selected = held_time(samples, self._time_seconds * rate)
            time_code = Usd.TimeCode(selected) if selected is not None else Usd.TimeCode.Default()
            identity = (self._source._namespace, dataset_key, selected)
            sample = DatasetSample(selected / rate if selected is not None else None)
            selection = (prim, time_code, identity, sample)
            self._selections[request] = selection
            return selection

    def describe_component(self, key, identifier, component):
        """Describe every component with the same dataset-wide identity and sample.

        Even unselected field samples contribute to this first-pass timeline.
        Time is the selected union time in seconds, not each array's native time.
        """
        _prim, _time_code, identity, sample = self._selection(key, identifier)
        return identity, sample

    def resolve_dataset(
        self,
        key: str,
        *,
        identifier: str,
        device: str,
        fields: Mapping[str, Sequence[str]],
    ):
        """Return a clone of the retained Dataset with the selected fields attached.

        The clone shares the base handle, acceleration structures, and cached
        fields, so a locator built by one operator serves every later Update at
        the same sample; its field mapping is private, so per-operator field
        interpretation never leaks between recipes.
        """
        try:
            import warp_simdata.usd as simdata_usd

            prim, time_code, identity, _sample = self._selection(key, identifier)
            base, loaded_fields = self._source.loaded(
                identity,
                device,
                lambda: simdata_usd.dataset_from_prim(prim, device=device, time_code=time_code),
            )
            dataset = base.shallow_copy(pass_fields=False)
            for output_name, field_names in fields.items():
                names = tuple(str(name) for name in field_names)
                field = loaded_fields.get(names)
                if field is None:
                    field = simdata_usd.field_from_prim(prim, list(names), device=device, time_code=time_code)
                    with self._source._lock:
                        field = loaded_fields.setdefault(names, field)
                dataset.add_field(output_name, field, warn_if_exists=False)
            return dataset
        except DatasetResolutionError:
            raise
        except Exception as exc:
            raise DatasetResolutionError(f"could not resolve dataset {key!r}: {exc}") from exc


def _composes_dataset(prim) -> bool:
    return any(child.GetTypeName() == "OmniSciDataset" for child in Usd.PrimRange(prim))


def _resolve_dataset_prim(root, identifier: str):
    """Select one OmniSciDataset at or beneath root, qualified by a relative identifier."""
    identifier = str(identifier or "")
    if identifier:
        if ".." in identifier.split("/"):
            raise DatasetResolutionError(f"dataset identifier cannot contain '..': {identifier!r}")
        prim = root.GetStage().GetPrimAtPath(f"{root.GetPath()}/{identifier.lstrip('/')}")
        if not prim:
            raise DatasetResolutionError(f"dataset identifier {identifier!r} did not resolve beneath {root.GetPath()}")
        return prim

    if root.GetTypeName() == "OmniSciDataset":
        return root
    candidates = [prim for prim in Usd.PrimRange(root) if prim.GetTypeName() == "OmniSciDataset"]
    if len(candidates) == 1:
        return candidates[0]
    available = ", ".join(str(prim.GetPath()).removeprefix(str(root.GetPath())) or "/" for prim in candidates)
    raise DatasetResolutionError(
        f"dataset identifier is required because {root.GetPath()} exposes {len(candidates)} datasets: "
        f"{available or '<none>'}"
    )
