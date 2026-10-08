# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Data Source base class, opaque Frame Data lease, and the composite source."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from threading import Lock

from ._errors import ActiveFrameDataError, DatasetResolutionError, DataSourceClosedError, FrameDataError

_FRAME_TOKEN = object()


class FrameData:
    """Opaque, time-bound data lease created only by :meth:`DataSource.at_time`."""

    def __init__(self, token: object, source: DataSource, provider):
        if token is not _FRAME_TOKEN:
            raise TypeError("FrameData instances are created by DataSource.at_time()")
        self._source = source
        self._provider = provider
        self._application_active = True
        self._update_references = 0

    def _retain_for_update(self) -> None:
        if not self._application_active:
            raise FrameDataError("Frame Data is outside its application lease")
        self._source._retain_update(self)
        self._update_references += 1

    def _release_from_update(self) -> None:
        self._update_references -= 1
        self._source._release_update(self)

    def _end_application_lease(self) -> None:
        self._application_active = False
        self._source._release_application(self)

    def _resolve_dataset(self, key: str, *, identifier: str, device: str, fields):
        return self._provider.resolve_dataset(key, identifier=identifier, device=device, fields=fields)


class DataSource:
    """Base class for sources of scientific Frame Data.

    The base owns lease accounting: ``at_time`` creates an opaque Frame Data lease,
    ``close`` waits for in-flight Updates before releasing resources, and the
    context manager closes on exit. A concrete source overrides three hooks:

    - ``resolves(key)`` claims a Dataset Key without loading anything;
    - ``frame_at_time(time_seconds)`` returns a frame provider bound to one
      selection of native samples. The provider exposes
      ``resolve_dataset(key, *, identifier, device, fields)`` and optionally
      ``describe_component(key, identifier, component)`` returning an opaque
      comparison identity and native sample metadata;
    - ``release_resources()`` runs once after close when no Update still holds a
      lease.

    Operators never see a source; they receive datasets resolved through the
    lease. Applications construct the concrete source they own.
    """

    def __init__(self):
        self._closed = False
        self._application_leases = 0
        self._update_leases = 0
        self._lock = Lock()

    def resolves(self, key: str) -> bool:
        raise NotImplementedError

    def frame_at_time(self, time_seconds: float):
        raise NotImplementedError

    def release_resources(self) -> None:
        pass

    @contextmanager
    def at_time(self, time_seconds: float) -> Iterator[FrameData]:
        self._begin_application_lease()
        try:
            provider = self.frame_at_time(float(time_seconds))
        except Exception:
            self._release_application(None)
            raise
        frame = FrameData(_FRAME_TOKEN, self, provider)
        try:
            yield frame
        finally:
            frame._end_application_lease()

    def _begin_application_lease(self) -> None:
        with self._lock:
            if self._closed:
                raise DataSourceClosedError("Data Source is closed")
            self._application_leases += 1

    def _retain_update(self, frame: FrameData) -> None:
        with self._lock:
            if self._closed or frame._source is not self:
                raise FrameDataError("Frame Data belongs to a closed or different Data Source")
            self._update_leases += 1

    def _release_update(self, _frame: FrameData) -> None:
        with self._lock:
            self._update_leases -= 1
            if self._closed and self._update_leases == 0:
                self.release_resources()

    def _release_application(self, _frame: FrameData | None) -> None:
        with self._lock:
            self._application_leases -= 1

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self._application_leases:
                raise ActiveFrameDataError("cannot close Data Source while Frame Data application leases are active")
            self._closed = True
            if self._update_leases == 0:
                self.release_resources()

    def __enter__(self) -> DataSource:
        if self._closed:
            raise DataSourceClosedError("Data Source is closed")
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.close()


class CompositeDataSource(DataSource):
    """Route each Dataset Key to the one child source that claims it.

    The composite owns its children: closing it closes them. Every lease it
    creates leases every child, and in-flight Updates retain the child leases, so
    no child can release resources under an Update. A key claimed by more than one
    child is an error at resolution time, never a first-wins choice.
    """

    def __init__(self, *sources: DataSource):
        super().__init__()
        if not sources:
            raise ValueError("CompositeDataSource requires at least one Data Source")
        for source in sources:
            if not isinstance(source, DataSource):
                raise TypeError(f"CompositeDataSource accepts DataSource instances, got {type(source).__name__}")
        self._sources = tuple(sources)

    def resolves(self, key: str) -> bool:
        return any(source.resolves(key) for source in self._sources)

    @contextmanager
    def at_time(self, time_seconds: float) -> Iterator[FrameData]:
        self._begin_application_lease()
        frame = None
        try:
            with ExitStack() as stack:
                frames = tuple(stack.enter_context(source.at_time(time_seconds)) for source in self._sources)
                frame = FrameData(_FRAME_TOKEN, self, _CompositeFrameProvider(self._sources, frames))
                yield frame
        finally:
            if frame is None:
                self._release_application(None)
            else:
                frame._end_application_lease()

    def _retain_update(self, frame: FrameData) -> None:
        super()._retain_update(frame)
        retained = []
        try:
            for child in frame._provider._frames:
                child._retain_for_update()
                retained.append(child)
        except Exception:
            for child in retained:
                child._release_from_update()
            super()._release_update(frame)
            raise

    def _release_update(self, frame: FrameData) -> None:
        for child in frame._provider._frames:
            child._release_from_update()
        super()._release_update(frame)

    def release_resources(self) -> None:
        for source in self._sources:
            source.close()


class _CompositeFrameProvider:
    def __init__(self, sources, frames):
        self._sources = sources
        self._frames = frames

    def _provider(self, key: str):
        claims = [frame for source, frame in zip(self._sources, self._frames, strict=True) if source.resolves(key)]
        if not claims:
            raise DatasetResolutionError(f"no Data Source resolves dataset key {key!r}")
        if len(claims) > 1:
            raise DatasetResolutionError(
                f"dataset key {key!r} is claimed by {len(claims)} Data Sources; composite keys must be unambiguous"
            )
        return claims[0]._provider

    def describe_component(self, key, identifier, component):
        describe = getattr(self._provider(key), "describe_component", None)
        if describe is None:
            return None, None
        return describe(key, identifier, component)

    def resolve_dataset(self, key: str, *, identifier: str, device: str, fields):
        return self._provider(key).resolve_dataset(key, identifier=identifier, device=device, fields=fields)
