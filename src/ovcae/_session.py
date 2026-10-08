# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Core Session lifecycle, discovery, and product ownership."""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress
from dataclasses import replace
from typing import Any

from ._data import FrameData
from ._errors import (
    FrameDataError,
    OrdinalError,
    SessionClosedError,
    StageSessionConflictError,
    UpdateError,
    UpdateInFlightError,
)
from ._inputs import InputPlan
from ._products import CurvesProduct, EmptyProduct, InstancerProduct, MeshProduct, PointsProduct
from ._results import Diagnostic, OperatorResult, UpdateReport
from ._reuse import Evaluation, History, Recipe, oldest_preserved_ordinal
from ._stage_io import (
    PRODUCT_ATTRIBUTES,
    TEST_OPERATOR_SCHEMA,
    TEST_VALUE,
    clear_instancer_product,
    delete_product_attributes,
    discover_operators,
    read_scalar,
    read_value,
    write_curves_product,
    write_floor,
    write_instancer_product,
    write_mesh_product,
    write_points_product,
    write_test_product,
)
from .operators import AmbiguousOperatorError, Changes, InvalidRecipe, OperatorContext, OperatorRegistry
from .scene._plan import plan_operator

LOGGER = logging.getLogger(__name__)

_STAGE_LOCK = threading.Lock()
_RESERVED_STAGES: set[int] = set()
# PointInstancer arrays Session keeps alive across clears; see clear_instancer_product.
_INSTANCER_ARRAYS = frozenset({"positions", "protoIndices", "scales", "orientations"})


class Update:
    """Handle for one asynchronous evaluation and its terminal report.

    Session owns the worker and scientific lease. wait() drains the evaluation and
    releases the Session's in-flight slot, but never commits its Output Ordinal.
    Repeated waits return the same report after successful completion.
    """

    def __init__(self, session: Session, input_ordinal: int, output_ordinal: int):
        self._session = session
        self._input_ordinal = input_ordinal
        self._output_ordinal = output_ordinal
        self._future: Future[UpdateReport] | None = None
        self._report: UpdateReport | None = None

    @property
    def input_ordinal(self) -> int:
        return self._input_ordinal

    @property
    def output_ordinal(self) -> int:
        return self._output_ordinal

    def _attach(self, future: Future[UpdateReport]) -> None:
        self._future = future

    def done(self, timeout: float = 0.0) -> bool:
        """Report whether wait() would return immediately, blocking at most ``timeout`` seconds.

        Frame loops poll this to keep running while the worker evaluates. A
        positive timeout cedes the GIL to the worker for up to that long, which
        a UI loop that otherwise never idles must do for evaluation to progress.
        A completed Update still holds the in-flight slot until wait() runs.
        """
        if self._report is not None:
            return True
        assert self._future is not None
        if timeout <= 0.0:
            return self._future.done()
        try:
            self._future.exception(timeout=timeout)
        except TimeoutError:
            return False
        return True

    def wait(self) -> UpdateReport:
        """Wait for evaluation/publication completion and release the in-flight slot.

        Cache the terminal report for repeat calls. Unexpected worker errors become
        UpdateError; expected per-operator failures are represented in the report.
        The caller still decides whether to commit a committable output.
        """
        if self._report is not None:
            return self._report
        assert self._future is not None
        try:
            self._report = self._future.result()
        except Exception as exc:
            self._session._finish(self)
            if isinstance(exc, UpdateError):
                raise
            raise UpdateError(f"Update failed before a reliable report was available: {exc}") from exc
        self._session._finish(self)
        return self._report


class Session:
    """Long-lived evaluator and exclusive product publisher for a borrowed Stage.

    The application owns Stage lifetime, input stability, and ordinal commitment.
    Session snapshots the operator registry, binds instances by recipe path, plans
    inputs, and publishes products through one worker. Per-binding histories are
    disposable reuse evidence, never an alternative authority for committed state.
    Only one Update may remain outstanding until its wait() completes.
    """

    def __init__(
        self,
        stage,
        *,
        device: str = "auto",
        operators: OperatorRegistry | None = None,
    ):
        registry = operators if operators is not None else OperatorRegistry.with_builtins()
        if not isinstance(registry, OperatorRegistry):
            raise TypeError("operators must be an ovcae.operators.OperatorRegistry")
        stage_key = id(stage)
        with _STAGE_LOCK:
            if stage_key in _RESERVED_STAGES:
                raise StageSessionConflictError("A live ovcae Session already publishes to this Execution Stage")
            _RESERVED_STAGES.add(stage_key)
        self._stage = stage
        self._stage_key = stage_key
        self._device = device
        self._closed = False
        self._inflight: Update | None = None
        self._known_operators: set[str] = set()
        self._owned_attributes: dict[str, frozenset[str]] = {}
        # Bindings whose owned product is already the neutral empty PointInstancer
        # product, so repeated clears need no stage writes.
        self._neutral_products: set[str] = set()
        self._operator_registry = registry.snapshot()
        self._operator_instances: dict[str, tuple[type, object]] = {}
        self._scene_paths = None
        self._histories: dict[str, History] = {}
        self._pending_evaluations: dict[str, Evaluation] = {}
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ovcae")

    def update(self, *, input_ordinal: int, output_ordinal: int, frame_data: FrameData) -> Update:
        """Evaluate through at least a minimum committed input publication.

        The caller must keep recipe and referenced scene inputs stable until
        Update.wait() completes, and wait before committing the output ordinal.
        input_ordinal does not select historical values. Session does not lock
        input writers or advance the Stage write floor.
        """
        if self._closed:
            raise SessionClosedError("Session is closed")
        if self._inflight is not None:
            raise UpdateInFlightError("Session already has an Update awaiting wait()")
        if not isinstance(frame_data, FrameData):
            raise FrameDataError("frame_data must be a lease returned by DataSource.at_time()")
        floor = write_floor(self._stage)
        if input_ordinal > floor:
            raise OrdinalError(f"input_ordinal {input_ordinal} is not committed; write floor is {floor}")
        if output_ordinal <= floor:
            raise OrdinalError(f"output_ordinal {output_ordinal} must be above write floor {floor}")
        if output_ordinal <= input_ordinal:
            raise OrdinalError("output_ordinal must be greater than input_ordinal")

        frame_data._retain_for_update()
        update = Update(self, input_ordinal, output_ordinal)
        self._inflight = update
        try:
            future = self._executor.submit(self._evaluate, input_ordinal, output_ordinal, frame_data)
        except Exception:
            self._inflight = None
            frame_data._release_from_update()
            raise
        update._attach(future)
        return update

    def _evaluate(self, input_ordinal: int, output_ordinal: int, frame_data: FrameData) -> UpdateReport:
        """Run an Update while retaining Frame Data, releasing its lease even on failure."""
        started = time.perf_counter()
        LOGGER.debug("update %d -> %d started", input_ordinal, output_ordinal)
        try:
            return self._evaluate_retained(input_ordinal, output_ordinal, frame_data)
        finally:
            frame_data._release_from_update()
            LOGGER.debug(
                "update %d -> %d finished in %.1f ms",
                input_ordinal,
                output_ordinal,
                (time.perf_counter() - started) * 1000,
            )

    def _evaluate_retained(self, input_ordinal: int, output_ordinal: int, frame_data: FrameData) -> UpdateReport:
        """Evaluate with an explicit publication list and drain it on every exit.

        Already-enqueued writes may still borrow array storage if a later operator or
        Stage operation fails. Waiting in finally preserves those lifetimes before the
        outer evaluation releases its Frame Data reference.
        """
        pending_publications = []
        self._pending_evaluations.clear()
        try:
            return self._evaluate_with_pending_publications(
                input_ordinal,
                output_ordinal,
                frame_data,
                pending_publications,
            )
        finally:
            # Preserve OVStage's borrowed-input lifetime even if evaluation
            # aborts after an earlier product was enqueued.
            for _operator, publication, _result_index in pending_publications:
                with suppress(Exception):
                    publication.wait()

    def _evaluate_with_pending_publications(
        self,
        input_ordinal: int,
        output_ordinal: int,
        frame_data: FrameData,
        pending_publications: list,
    ) -> UpdateReport:
        """Discover recipes, validate baselines, evaluate bindings, and finish writes.

        Validate all committed products before authoring anything: writes to shared native
        columns can invalidate subsequent reads even for different recipe prims. Handle
        removed/disabled recipes with exact owned-attribute cleanup. Hidden recipes
        retain products without executing. Dispatch visible registered
        operators, then wait for their publications before staging successful candidates.
        Collect isolated failures in the report; committable depends on coherent cleanup,
        not on every operator succeeding.
        """
        # Checkpoint before discovery so all input reads occur at or after it.
        floor = write_floor(self._stage)
        oldest = oldest_preserved_ordinal(self._stage)
        discovered = discover_operators(self._stage, input_ordinal)
        # Validate products and recipe ranges before any writes to shared columns.
        baselines = {
            path: history.committed(self._stage, path, floor, oldest=oldest)
            for path, history in self._histories.items()
        }
        recipe_changes = {
            path: history.recipe_changed(self._stage, floor, oldest=oldest) for path, history in self._histories.items()
        }
        results: list[OperatorResult] = []
        diagnostics: list[Diagnostic] = []
        committable = True
        visibility: dict[str, bool] = {}

        removed = self._known_operators - discovered.keys()
        for operator in sorted(removed):
            self._operator_instances.pop(operator, None)
            self._histories.pop(operator, None)
            had_product = bool(self._owned_attributes.get(operator))
            if self._clear_operator(operator, output_ordinal, diagnostics):
                paths = (operator,) if had_product else ()
                results.append(OperatorResult(operator, "removed", "recipe removed", paths))
            else:
                committable = False
                results.append(OperatorResult(operator, "failed", "cleanup failed", (operator,)))
            # Forget the binding entirely; a recipe re-authored at this path recovers
            # ownership from the Stage manifest instead of this Session's memory.
            self._owned_attributes.pop(operator, None)
            self._neutral_products.discard(operator)

        for operator, schemas in sorted(discovered.items()):
            self._load_owned_attributes(operator, input_ordinal)
            enabled = self._resolve(operator, "cae:viz:operator:enabled", input_ordinal, False)

            if not bool(enabled):
                cleared = self._clear_operator(operator, output_ordinal, diagnostics)
                committable &= cleared
                results.append(OperatorResult(operator, "skipped", "operator disabled", (operator,)))
                continue

            if not self._is_visible(operator, input_ordinal, visibility):
                # Visibility suspends matching and recipe/input validation too.
                # Report those diagnostics when shown; retain hidden products.
                results.append(OperatorResult(operator, "skipped", "operator invisible"))
                continue

            try:
                declaration = self._operator_registry.match(schemas)
            except AmbiguousOperatorError as exc:
                self._operator_instances.pop(operator, None)
                cleared = self._clear_operator(operator, output_ordinal, diagnostics)
                committable &= cleared
                diagnostics.append(_diagnostic("AMBIGUOUS_OPERATOR", exc, operator))
                results.append(OperatorResult(operator, "failed", "operator match is ambiguous", (operator,)))
                continue

            if declaration is not None:
                succeeded = self._execute_registered(
                    operator=operator,
                    schemas=schemas,
                    declaration=declaration,
                    baselines=baselines,
                    recipe_changes=recipe_changes,
                    observed_ordinal=floor,
                    input_ordinal=input_ordinal,
                    output_ordinal=output_ordinal,
                    frame_data=frame_data,
                    diagnostics=diagnostics,
                    results=results,
                    pending_publications=pending_publications,
                )
                committable &= succeeded
                continue

            if TEST_OPERATOR_SCHEMA not in schemas:
                cleared = self._clear_operator(operator, output_ordinal, diagnostics)
                committable &= cleared
                diagnostics.append(
                    Diagnostic(
                        "warning",
                        "UNSUPPORTED_OPERATOR",
                        f"no installed implementation accepts applied schemas: {', '.join(sorted(schemas))}",
                        operator,
                    )
                )
                results.append(OperatorResult(operator, "skipped", "unsupported operator", (operator,)))
                continue

            value = self._resolve(operator, TEST_VALUE, input_ordinal, None)
            if value is None:
                cleared = self._clear_operator(operator, output_ordinal, diagnostics)
                committable &= cleared
                diagnostics.append(
                    Diagnostic("error", "MISSING_TEST_VALUE", f"missing required {TEST_VALUE}", operator)
                )
                results.append(OperatorResult(operator, "failed", "missing required value", (operator,)))
                continue

            try:
                previous = self._owned_attributes.get(operator, frozenset())
                publication = write_test_product(self._stage, operator, float(value), output_ordinal, previous)
                self._owned_attributes[operator] = publication.attributes
                result_index = len(results)
                results.append(OperatorResult(operator, "succeeded", product_paths=(operator,)))
                pending_publications.append((operator, publication, result_index))
            except Exception as exc:
                cleared = self._clear_operator(operator, output_ordinal, diagnostics)
                committable &= cleared
                diagnostics.append(_diagnostic("PRODUCT_PUBLICATION_FAILED", exc, operator))
                results.append(OperatorResult(operator, "failed", "publication failed", (operator,)))

        for operator, publication, result_index in pending_publications:
            try:
                publication.wait()
                evaluation = self._pending_evaluations.pop(operator, None)
                if evaluation is not None:
                    self._histories[operator].published(evaluation, output_ordinal, publication.attributes)
            except Exception as exc:
                cleared = self._clear_operator(operator, output_ordinal, diagnostics)
                committable &= cleared
                diagnostics.append(_diagnostic("PRODUCT_PUBLICATION_FAILED", exc, operator))
                results[result_index] = OperatorResult(operator, "failed", "publication failed", (operator,))

        self._known_operators = set(discovered)
        return UpdateReport(input_ordinal, output_ordinal, tuple(results), tuple(diagnostics), committable)

    def _execute_registered(
        self,
        *,
        operator: str,
        schemas: frozenset[str],
        declaration,
        baselines,
        recipe_changes,
        observed_ordinal: int,
        input_ordinal: int,
        output_ordinal: int,
        frame_data: FrameData,
        diagnostics: list[Diagnostic],
        results: list[OperatorResult],
        pending_publications: list,
    ) -> bool:
        """Capture one bound recipe, gate execution, and enqueue its returned product.

        For annotated declarations, build shared requests and compare with a prevalidated
        baseline of the same implementation/device. A hit returns without executing or
        writing. Otherwise run the operator against its read-through scene view, expire that view,
        validate the product type, and enqueue publication. The enclosing Update waits
        for writes before creating a candidate. Execution/publication failures clear
        product state and discard binding history rather than recording successful reuse.
        """
        view = None
        try:
            if self._scene_paths is None:
                import ovstage

                self._scene_paths = ovstage.PathDictionary(self._stage)
            view = plan_operator(
                self._stage,
                operator_path=operator,
                applied_api_schemas=schemas,
                paths=self._scene_paths,
                input_ordinal=input_ordinal,
            )
            prim = view.get_prim_at_path(operator)
            if view.unknown_schemas:
                names = ", ".join(view.unknown_schemas)
                diagnostics.append(
                    Diagnostic(
                        "warning",
                        "SCHEMA_NOT_CAPTURED",
                        f"usd-core has no schema definition for: {names}; register the plug-in with "
                        "ovcae.register_schema_paths before the process creates any stage",
                        operator,
                    )
                )

            device_attribute = prim.get_attribute("cae:viz:operator:device")
            device = _resolve_device(self._device, device_attribute.get())
            context = OperatorContext(
                stage=view,
                prim=prim,
                input_ordinal=input_ordinal,
                output_ordinal=output_ordinal,
                device=device,
                _frame_data=frame_data,
            )
            if declaration.dataset_components is not None:
                inputs = InputPlan(context, declaration)
                history = self._histories.setdefault(operator, History())
                signature = (declaration.implementation, device)
                baseline = baselines.get(operator)
                if baseline is not None and baseline.signature != signature:
                    history.baseline = baseline = None
                recipe = None if view.unknown_schemas else Recipe(observed_ordinal, view.dependencies)
                evaluation = Evaluation(inputs, recipe, signature, baseline, recipe_changes.get(operator, True))
                if evaluation.unchanged():
                    history.verified_unchanged(evaluation)
                    results.append(OperatorResult(operator, "unchanged", "inputs and committed product unchanged"))
                    return True
                context = replace(context, _inputs=inputs, _changes=Changes(context.stage, evaluation))
                self._pending_evaluations[operator] = evaluation
            else:
                self._histories.pop(operator, None)
            instance = self._operator_instance(operator, declaration.implementation)
            started = time.perf_counter()
            result = instance.execute(context)
            LOGGER.debug("%s executed in %.1f ms", operator, (time.perf_counter() - started) * 1000)
            prim_type = prim.type_name
            view._expire()
            view = None
            _validate_product_type(result, prim_type)
        except InvalidRecipe as exc:
            cleared = self._clear_operator(operator, output_ordinal, diagnostics)
            diagnostics.append(_diagnostic("INVALID_RECIPE", exc, operator))
            results.append(OperatorResult(operator, "failed", "invalid operator recipe", (operator,)))
            return cleared
        except Exception as exc:
            cleared = self._clear_operator(operator, output_ordinal, diagnostics)
            diagnostics.append(_diagnostic("OPERATOR_EXECUTION_FAILED", exc, operator))
            results.append(OperatorResult(operator, "failed", "operator execution failed", (operator,)))
            return cleared
        finally:
            if view is not None:
                view._expire()

        try:
            previous = self._owned_attributes.get(operator, frozenset())
            if isinstance(result, MeshProduct):
                publication = write_mesh_product(
                    self._stage,
                    operator,
                    result,
                    output_ordinal,
                    previous=previous,
                )
            elif isinstance(result, CurvesProduct):
                publication = write_curves_product(
                    self._stage,
                    operator,
                    result,
                    output_ordinal,
                    previous=previous,
                )
            elif isinstance(result, PointsProduct):
                publication = write_points_product(
                    self._stage,
                    operator,
                    result,
                    output_ordinal,
                    previous=previous,
                )
            elif isinstance(result, InstancerProduct):
                publication = write_instancer_product(self._stage, operator, result, output_ordinal, previous=previous)
            elif isinstance(result, EmptyProduct):
                if not self._clear_operator(operator, output_ordinal, diagnostics):
                    results.append(OperatorResult(operator, "failed", "product cleanup failed", (operator,)))
                    return False
                results.append(OperatorResult(operator, "succeeded", result.reason))
                return True
            else:  # pragma: no cover - validated before publication
                raise AssertionError(f"unsupported validated product {type(result).__name__}")
            self._owned_attributes[operator] = publication.attributes
            self._neutral_products.discard(operator)
            result_index = len(results)
            results.append(OperatorResult(operator, "succeeded", product_paths=(operator,)))
            pending_publications.append((operator, publication, result_index))
            return True
        except Exception as exc:
            cleared = self._clear_operator(operator, output_ordinal, diagnostics)
            diagnostics.append(_diagnostic("PRODUCT_PUBLICATION_FAILED", exc, operator))
            results.append(OperatorResult(operator, "failed", "publication failed", (operator,)))
            return cleared

    def _is_visible(self, operator: str, input_ordinal: int, visibility: dict[str, bool]) -> bool:
        """Check the operator and its ancestors for an authored invisible opinion.

        Missing visibility and ``inherited`` defer to ancestors; neither overrides
        an invisible parent. Cache local opinions only within this Update, while
        application inputs are stable. Do not inspect scientific input targets:
        invisible seed geometry may still be required by a visible operator.
        """
        path = operator
        while path and path != "/":
            if path not in visibility:
                value = read_value(self._stage, path, "visibility", input_ordinal, type_name="token")
                visibility[path] = value != "invisible"
            if not visibility[path]:
                return False
            path = path.rpartition("/")[0]
        return True

    def _operator_instance(self, operator: str, implementation: type):
        """Reuse a binding's instance, replacing it when the selected implementation changes."""
        current = self._operator_instances.get(operator)
        if current is None or current[0] is not implementation:
            current = (implementation, implementation())
            self._operator_instances[operator] = current
        return current[1]

    def _resolve(
        self,
        operator: str,
        attribute: str,
        input_ordinal: int,
        default: Any,
    ) -> Any:
        value = read_scalar(self._stage, operator, attribute, input_ordinal)
        return default if value is None else value

    def _clear_operator(self, operator: str, ordinal: int, diagnostics: list[Diagnostic]) -> bool:
        """Discard reuse state and clear only this binding's owned product attributes.

        PointInstancers keep owned empty transform arrays because deleting those
        attributes can leave stale rendered instances. Never recreate an application-
        deleted prim; all other products retain exact attribute deletion.

        Failures become cleanup diagnostics and return False so the Update cannot claim
        coherent output. Dropping history first prevents reuse even when cleanup fails.
        A binding already holding the neutral empty product writes nothing, so a
        recipe that stays disabled costs no stage traffic on later Updates.
        """
        self._histories.pop(operator, None)
        self._pending_evaluations.pop(operator, None)
        attributes = self._owned_attributes.pop(operator, frozenset())
        if operator in self._neutral_products and attributes >= _INSTANCER_ARRAYS:
            self._owned_attributes[operator] = attributes
            return True
        self._neutral_products.discard(operator)
        try:
            if (
                attributes >= _INSTANCER_ARRAYS
                and read_value(self._stage, operator, "usd-prim-type", ordinal, type_name="token") == "PointInstancer"
            ):
                publication = clear_instancer_product(self._stage, operator, ordinal, previous=attributes)
                publication.wait()
                # Record ownership only once the neutral product exists on the Stage;
                # after a failure the next Update recovers ownership from the manifest.
                self._owned_attributes[operator] = publication.attributes
                self._neutral_products.add(operator)
            else:
                delete_product_attributes(self._stage, operator, attributes, ordinal)
            return True
        except Exception as exc:
            diagnostics.append(_diagnostic("PRODUCT_CLEANUP_FAILED", exc, operator))
            return False

    def _load_owned_attributes(self, operator: str, input_ordinal: int) -> None:
        """Recover exact product ownership from the Stage manifest when a binding is first seen.

        This supports cleanup after Session restart without sweeping attributes by prefix.
        The in-memory ownership set is then maintained as publications are enqueued.
        """
        if operator in self._owned_attributes:
            return
        manifest = read_scalar(self._stage, operator, PRODUCT_ATTRIBUTES, input_ordinal)
        if manifest is None:
            self._owned_attributes[operator] = frozenset()
            return
        if isinstance(manifest, str):
            manifest = (manifest,)
        self._owned_attributes[operator] = frozenset(str(item) for item in manifest) | {PRODUCT_ATTRIBUTES}

    def _finish(self, update: Update) -> None:
        if self._inflight is update:
            self._inflight = None

    def close(self) -> None:
        """Drain outstanding work, release retained histories, and relinquish Stage ownership.

        Leave committed products on Stage: removing them would require a caller-owned
        output ordinal. Cleanup of Session resources still happens if waiting raises;
        that failure is propagated after shutdown. Repeated close calls are harmless.
        """
        if self._closed:
            return
        failure: Exception | None = None
        try:
            if self._inflight is not None:
                self._inflight.wait()
        except Exception as exc:
            failure = exc
        finally:
            self._executor.shutdown(wait=True)
            if self._scene_paths is not None:
                self._scene_paths.destroy()
                self._scene_paths = None
            self._histories.clear()
            self._pending_evaluations.clear()
            with _STAGE_LOCK:
                _RESERVED_STAGES.discard(self._stage_key)
            self._closed = True
        if failure is not None:
            raise failure

    def __enter__(self) -> Session:
        if self._closed:
            raise SessionClosedError("Session is closed")
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.close()


def _diagnostic(code: str, exc: Exception, operator: str) -> Diagnostic:
    return Diagnostic("error", code, str(exc), operator)


def _resolve_device(session_device: str, recipe_device) -> str:
    if session_device != "auto":
        return session_device
    recipe_device = str(recipe_device)
    if recipe_device != "auto":
        return recipe_device
    import warp as wp

    return "cuda:0" if wp.is_cuda_available() else "cpu"


def _validate_product_type(result, prim_type: str | None) -> None:
    expected = None
    if isinstance(result, MeshProduct):
        expected = "Mesh"
    elif isinstance(result, CurvesProduct):
        expected = "BasisCurves"
    elif isinstance(result, PointsProduct):
        expected = "Points"
    elif isinstance(result, InstancerProduct):
        expected = "PointInstancer"
    elif not isinstance(result, EmptyProduct):
        raise TypeError(
            f"operator execute() must return a product created by ovcae.operators.utils; got {type(result).__name__}"
        )
    if expected is not None and prim_type != expected:
        raise InvalidRecipe(f"{type(result).__name__} requires recipe prim type {expected!r}; got {prim_type!r}")
