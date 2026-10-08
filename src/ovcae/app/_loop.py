# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""The edit-update-commit loop every interactive OVCAE application runs."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from typing import Any

from .._errors import UpdateError, UpdateInFlightError
from .._session import Session, Update
from .._stage_io import read_value
from ._writes import committed_type, write_attribute

LOGGER = logging.getLogger(__name__)

EditKey = tuple[str, str]

# Attributes whose USD type no schema states because OVStage or the runtime defines them.
_KNOWN_TYPES = {"omni:xform": "matrix4d", "visibility": "token", "points": "point3f[]"}
_LOG_LEVELS = {"info": logging.INFO, "warning": logging.WARNING, "error": logging.ERROR}


class UpdateLoop:
    """Own a Session and the ordinal bookkeeping between an application's edits and its renderer.

    The application owns the Stage and the Data Source and populates the
    Stage before creating the loop; the loop creates the Session and closes it
    with :meth:`close`. Edits are ``{(prim path, property): value}`` batches
    of two kinds, and the application says which: :meth:`request` takes
    recipe edits, anything an operator reads, which are written together at
    the next ordinal by :meth:`submit` or :meth:`pump` and followed by an
    update; :meth:`write` takes presentation edits such as a camera move,
    which go to the stage at once with no update. The committed ordinal, the
    one a renderer steps at, is :attr:`ordinal`.

    ``time_code`` is the scientific time an update evaluates at: a number, or
    a callable returning one, asked on every submit. Everything here runs on
    the caller's thread; a renderer loop calls :meth:`pump` once per frame and
    other threads hand their edits to that loop rather than calling in.
    """

    def __init__(
        self,
        stage,
        data_source,
        *,
        ordinal: int,
        device: str = "auto",
        operators=None,
        time_code: float | Callable[[], float] = 0.0,
    ):
        self.stage = stage
        self.data_source = data_source
        self.ordinal = ordinal
        self.last_report = None
        self._time_code = time_code if callable(time_code) else (lambda: time_code)
        self._pending: dict[EditKey, Any] = {}
        self._types: dict[EditKey, str] = {}
        self._known_types: dict[EditKey, str] = {}
        self._update: Update | None = None
        self._submitted_at: float | None = None
        self._dirty = False
        self.session = Session(stage, device=device, operators=operators)

    # -- edits -----------------------------------------------------------------------

    def request(self, edits: Mapping[EditKey, Any], *, types: Mapping[EditKey, str] | None = None) -> None:
        """Buffer recipe edits; the next submit writes them at one ordinal and runs an update.

        Requests coalesce, latest value per key. Nothing is written here.
        ``types`` overrides the USD type an edit is written as; otherwise
        :meth:`property_type` answers, and failing that the type is inferred
        from the value (``bool``, ``int``, ``float``, ``str`` as token,
        sixteen numbers as ``matrix4d``, two to four as ``int2``,
        ``float3``, and so on). A batch with an edit whose type cannot be
        determined is rejected whole with ``TypeError``.

        A type is looked up on the stage once per property and remembered.
        While an update is in flight the stage is not read at all, because
        a read on a prim the update is publishing to fails that publication;
        a property seen for the first time then is resolved by the submit
        that writes it, which raises the ``TypeError`` instead.
        """
        resolved = self._resolve_types(edits, types, defer_unknown=self._update is not None)
        self._pending.update(edits)
        self._types.update(resolved)
        LOGGER.debug(
            "request %d edit(s) [%s]; pending=%d evaluating=%s",
            len(edits),
            ", ".join(f"{path.rsplit('/', 1)[-1]}.{name}" for path, name in edits),
            len(self._pending),
            self._update is not None,
        )

    def write(self, edits: Mapping[EditKey, Any], *, types: Mapping[EditKey, str] | None = None) -> None:
        """Write presentation edits at the next ordinal and advance the floor; no update follows.

        For camera moves, viewport sizes, material bindings, and anything
        else no operator reads. Raises ``UpdateInFlightError`` while an update
        is running, because its output ordinal is the next one; call
        :meth:`flush` first, or present the previous frame and try again after
        :meth:`pump` commits. Types resolve as for :meth:`request`.
        """
        if self._update is not None:
            raise UpdateInFlightError("cannot write while an update is in flight; flush or pump first")
        if not edits:
            return
        resolved = self._resolve_types(edits, types)
        self._write(edits, resolved, "presentation only")

    @property
    def pending(self) -> dict[EditKey, Any]:
        """The buffered recipe edits, latest value per key, that the next submit writes."""
        return dict(self._pending)

    def property_type(self, path: str, name: str) -> str | None:
        """The USD type name an edit to ``name`` on ``path`` is written as, or None when nothing declares it.

        ``omni:xform``, ``visibility``, and ``points`` are known; a committed
        value answers with the type it already has, since OVStage does not
        change an attribute's type in place; and any other attribute is looked
        up on the prim's typed schema (``focalLength`` on a ``Camera``).
        """
        if name in _KNOWN_TYPES:
            return _KNOWN_TYPES[name]
        committed = committed_type(self.stage, path, name, self.ordinal)
        if committed is not None:
            return committed
        type_name = read_value(self.stage, path, "usd-prim-type", self.ordinal)
        if type_name:
            from pxr import Usd

            definition = Usd.SchemaRegistry().FindConcretePrimDefinition(str(type_name))
            spec = definition.GetSchemaAttributeSpec(name) if definition is not None else None
            if spec is not None:
                return str(spec.typeName)
        return None

    def value(self, path: str, name: str, *, type_name: str | None = None):
        """The value an update would see: the pending edit if there is one, else the committed value or None."""
        if (path, name) in self._pending:
            return self._pending[(path, name)]
        return read_value(self.stage, path, name, self.ordinal, type_name=type_name)

    def invalidate(self) -> None:
        """Make the next submit run an update even without recipe edits, such as after new data was published."""
        self._dirty = True

    def _resolve_types(self, edits, types, *, defer_unknown: bool = False) -> dict[EditKey, str]:
        """The USD type of every edit: declared, remembered, or looked up; ``defer_unknown`` skips the lookups."""
        resolved = {}
        for key in edits:
            path, name = key
            type_name = (types or {}).get(key) or self._known_types.get(key)
            if type_name is None:
                if defer_unknown:
                    continue
                type_name = self.property_type(path, name) or _infer_type(edits[key])
                if type_name is None:
                    raise TypeError(f"cannot determine the USD type of {name!r} on {path!r}; pass types=")
            self._known_types[key] = resolved[key] = type_name
        return resolved

    def _write(self, edits, types, why: str) -> None:
        ordinal = self.ordinal + 1
        started = time.perf_counter()
        for (path, name), value in edits.items():
            write_attribute(self.stage, path, name, value, ordinal, type_name=types[(path, name)])
        self.stage.advance_write_floor(ordinal).wait()
        self.ordinal = ordinal
        LOGGER.debug(
            "wrote %d edit(s) at ordinal %d in %.1f ms: %s",
            len(edits),
            ordinal,
            (time.perf_counter() - started) * 1000,
            why,
        )

    # -- updates ---------------------------------------------------------------------

    @property
    def evaluating(self) -> bool:
        """True while a submitted update has not been committed."""
        return self._update is not None

    def await_evaluation(self, budget: float) -> bool:
        """Block up to ``budget`` seconds for the in-flight update; True once it is done or none is in flight.

        A UI loop that never idles must cede the GIL to the worker this way,
        or evaluations only progress between its own frames.
        """
        return True if self._update is None else self._update.done(timeout=budget)

    def pump(self):
        """Advance without blocking; return a report when an update commits.

        A finished update is committed and returned before anything else is
        submitted, so the caller renders the committed products first; a
        renderer cannot step while an update writes above the floor. When
        nothing is in flight, the pending edits are submitted as one batch.
        """
        if self._update is not None:
            if not self._update.done():
                return None
            update, self._update = self._update, None
            return self._commit(update)
        if self._pending or self._dirty:
            self.submit(wait=False)
        return None

    def submit(self, *, wait: bool = False):
        """Write the pending recipe edits at the next ordinal and start an update; None if there is nothing to do.

        With ``wait`` the update is committed before returning and its report
        returned; otherwise it runs in the background until :meth:`pump` or
        :meth:`flush` commits it. Raises ``UpdateInFlightError`` before
        touching the batch or the stage when an update is already running.
        """
        if self._update is not None:
            raise UpdateInFlightError("cannot submit while an update is in flight; flush or pump first")
        edits, types, self._pending, self._types = self._pending, self._types, {}, {}
        if not edits and not self._dirty:
            return None
        if edits:
            self._write(edits, self._resolve_types(edits, types), "update follows")
        self._dirty = False
        update = self._start(self.ordinal)
        if wait:
            return self._commit(update)
        self._update = update
        return None

    def update(self, edits: Mapping[EditKey, Any] | None = None, *, types=None):
        """Synchronously request, submit, and commit; the report.

        Without ``edits`` an update runs regardless, which is how an
        application evaluates the scene it just populated.
        """
        self.flush()
        if edits:
            self.request(edits, types=types)
        self.invalidate()
        return self.submit(wait=True)

    def flush(self):
        """Wait for the in-flight update, if any, commit it, and return its report."""
        if self._update is None:
            return None
        update, self._update = self._update, None
        return self._commit(update)

    def _start(self, input_ordinal: int) -> Update:
        with self.data_source.at_time(float(self._time_code())) as frame_data:
            update = self.session.update(
                input_ordinal=input_ordinal, output_ordinal=input_ordinal + 1, frame_data=frame_data
            )
        self._submitted_at = time.perf_counter()
        return update

    def _commit(self, update: Update):
        started = time.perf_counter()
        report = update.wait()
        LOGGER.debug(
            "commit %d -> %d: waited %.1f ms, %.0f ms since submit: %s",
            update.input_ordinal,
            update.output_ordinal,
            (time.perf_counter() - started) * 1000,
            (time.perf_counter() - self._submitted_at) * 1000 if self._submitted_at else float("nan"),
            ", ".join(f"{item.operator_path.rsplit('/', 1)[-1]}={item.status}" for item in report.operators),
        )
        self._submitted_at = None
        for diagnostic in report.diagnostics:
            LOGGER.log(
                _LOG_LEVELS[diagnostic.severity],
                "OVCAE %s at %s: %s",
                diagnostic.code,
                diagnostic.operator_path or "<stage>",
                diagnostic.message,
            )
        if not report.committable:
            raise UpdateError(
                f"update {update.input_ordinal} -> {update.output_ordinal} is not committable: {report.diagnostics}",
                report,
            )
        self.stage.advance_write_floor(update.output_ordinal).wait()
        self.ordinal, self.last_report = update.output_ordinal, report
        return report

    # -- lifecycle -------------------------------------------------------------------

    def close(self) -> None:
        """Drop pending edits and close the Session; the Stage and Data Source stay with the application."""
        self._pending, self._types = {}, {}
        self._update = None
        self.session.close()

    def __enter__(self) -> UpdateLoop:
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.close()


def _infer_type(value) -> str | None:
    import numpy as np

    if isinstance(value, bool | np.bool_):
        return "bool"
    if isinstance(value, int | np.integer):
        return "int"
    if isinstance(value, float | np.floating):
        return "float"
    if isinstance(value, str):
        return "token"
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return None
    if array.dtype == object or array.dtype.kind not in "biuf":
        return None
    base = "int" if array.dtype.kind in "biu" else "float"
    return {16: "matrix4d", 2: f"{base}2", 3: f"{base}3", 4: f"{base}4"}.get(array.size)


__all__ = ["EditKey", "UpdateLoop"]
