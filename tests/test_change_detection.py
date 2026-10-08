# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from contextlib import contextmanager

import numpy as np
import ovstage
import pytest

import ovcae
from ovcae.operators._builtins.faces import Faces

from ._memory_series import MemorySeries, open_series
from ._ovstage_helpers import (
    path_query,
    read_array,
    release_path_query,
    write_relationship,
    write_scalar,
    write_schemas,
    write_token,
)

RECIPE = "/World/Faces"
KEY = "/data"


def publish_field(series, name, value, time=0):
    series.publish(
        KEY,
        component=f"field:{name}",
        time=time,
        value={"association": "node", "values": np.full(4, value, np.float32)},
    )


def setup_recipe(stage, path=RECIPE):
    write_schemas(
        stage,
        path,
        [
            "CaeVizOperatorAPI",
            "CaeVizFacesAPI",
            "CaeVizDatasetSelectionAPI:source",
            "CaeVizFieldSelectionAPI:temperature",
        ],
        1,
    )
    write_token(stage, path, "usd-prim-type", "Mesh", 1)
    write_scalar(stage, path, "cae:viz:operator:enabled", True, 1)
    write_relationship(stage, path, "cae:viz:dataset_selection:source:target", [KEY], 1)
    paths, plist, query = path_query(stage, path)
    try:
        stage.write_attribute(
            query,
            "cae:viz:field_selection:temperature:fieldNames",
            1,
            [np.asarray([paths.intern_token("T")], np.uint64)],
            is_array=True,
            semantic=ovstage.AttributeSemantic.TOKEN_ID,
        ).wait()
    finally:
        release_path_query(paths, plist, query)


@contextmanager
def pipeline(monkeypatch, bindings=(RECIPE,), *, operator_type=Faces, recipe_setup=setup_recipe):
    from warp_simdata.operators import element_faces

    from ._memory_series import _SeriesFrame

    counts = dict(execute=0, extract=0, resolve=0, writes=0)

    def count(obj, name, counter):
        original = getattr(obj, name)

        def wrapped(*args, **kwargs):
            counts[counter] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(obj, name, wrapped)

    count(operator_type, "execute", "execute")
    count(element_faces, "compute", "extract")
    count(_SeriesFrame, "resolve_dataset", "resolve")
    stage = ovstage.Stage("ovcae.tests.change-detection")
    series = MemorySeries()
    series.publish(
        KEY, component="geometry", value=np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float32)
    )
    series.publish(
        KEY,
        component="topology",
        value={
            "cell_types": np.asarray([10], np.uint8),
            "connectivity_offsets": np.asarray([0, 4], np.int32),
            "connectivity": np.asarray([0, 1, 2, 3], np.int32),
        },
    )
    publish_field(series, "T", 1)
    publish_field(series, "unused", 0)
    try:
        for path in bindings:
            recipe_setup(stage, path)
        stage.advance_write_floor(1).wait()
        count(stage, "write_attributes", "writes")
        count(stage, "delete_attributes", "writes")
        with open_series(series) as source, ovcae.Session(stage, device="cpu") as session:
            ordinal = 1

            def update(time=0, commit=True, minimum=None):
                nonlocal ordinal
                from ovcae._stage_io import write_floor

                floor = write_floor(stage)
                ordinal = max(ordinal, floor) + 1
                with source.at_time(time) as frame:
                    report = session.update(
                        input_ordinal=floor if minimum is None else minimum, output_ordinal=ordinal, frame_data=frame
                    ).wait()
                if commit:
                    assert report.committable, report.diagnostics
                    stage.advance_write_floor(ordinal).wait()
                return report

            yield stage, series, session, update, counts
    finally:
        stage.destroy()


def test_unchanged_skips_execution_materialization_and_all_prim_writes(monkeypatch):
    with pipeline(monkeypatch) as (stage, series, _session, update, counts):
        first = update()
        assert first.operators[0].status == "succeeded", first.diagnostics
        assert read_array(stage, RECIPE, "faceVertexCounts", first.output_ordinal).tolist() == [3, 3, 3, 3]
        before = counts.copy()
        for time in [0, 1, 2]:
            report = update(time, minimum=1)
            assert report.operators[0].status == "unchanged", report.diagnostics
            assert report.operators[0].product_paths == ()
        publish_field(series, "unused", 9, time=1)
        series.publish("/other", component="geometry", value=np.zeros((1, 3)), time=1)
        assert update(2).operators[0].status == "unchanged"
        assert counts == before


def test_selected_sample_and_same_time_replacement_execute(monkeypatch):
    with pipeline(monkeypatch) as (stage, series, _session, update, counts):
        assert update().operators[0].status == "succeeded"
        publish_field(series, "T", 5, time=1)
        report = update(1)
        assert report.operators[0].status == "succeeded", report.diagnostics
        np.testing.assert_array_equal(read_array(stage, RECIPE, "primvars:temperature", report.output_ordinal), [5] * 4)
        publish_field(series, "T", 7, time=1)
        report = update(1)
        assert report.operators[0].status == "succeeded"
        np.testing.assert_array_equal(read_array(stage, RECIPE, "primvars:temperature", report.output_ordinal), [7] * 4)
        assert counts["execute"] == counts["extract"] == 3


def test_uncommitted_publication_is_not_a_successful_baseline(monkeypatch, observed_changes):
    with pipeline(monkeypatch) as (_stage, _series, session, update, counts):
        report = update(commit=False)
        assert report.operators[0].status == "succeeded"
        assert session._histories[RECIPE].baseline is None
        second = update()
        assert second.operators[0].status == "succeeded"
        assert counts["execute"] == 2
        assert observed_changes == [(True, True, True, True)] * 2
        assert update().operators[0].status == "unchanged"


def test_product_overwrite_with_intact_manifest_forces_execution(monkeypatch):
    with pipeline(monkeypatch) as (stage, _series, _session, update, counts):
        first = update()
        write_token(stage, RECIPE, "subdivisionScheme", "catmullClark", first.output_ordinal + 1)
        stage.advance_write_floor(first.output_ordinal + 1).wait()
        report = update()
        assert report.operators[0].status == "succeeded", report.diagnostics
        assert counts["execute"] == 2


def test_recipe_setting_invalidates(monkeypatch):
    with pipeline(monkeypatch) as (stage, _series, _session, update, counts):
        first = update()
        write_scalar(stage, RECIPE, "cae:viz:faces:externalOnly", False, first.output_ordinal + 1)
        stage.advance_write_floor(first.output_ordinal + 1).wait()
        report = update()
        assert report.operators[0].status == "succeeded", report.diagnostics
        assert counts["execute"] == 2


def test_unknown_source_evidence_always_executes(monkeypatch, observed_changes):
    from ._memory_series import _SeriesFrame

    monkeypatch.setattr(_SeriesFrame, "describe_component", None)
    with pipeline(monkeypatch) as (_stage, _series, _session, update, counts):
        assert update().operators[0].status == "succeeded"
        assert update().operators[0].status == "succeeded"
        assert counts["execute"] == 2
        assert observed_changes == [(True, True, True, True), (False, True, True, True)]


@pytest.mark.parametrize("failure", ["execution", "publication", "completion"])
def test_failed_candidate_is_not_promoted_and_retries(monkeypatch, failure, observed_changes):
    from ovcae import _session

    with pipeline(monkeypatch) as (stage, series, session, update, _counts):
        assert update().operators[0].status == "succeeded"
        # Promote the first candidate before attempting a changed evaluation.
        assert update().operators[0].status == "unchanged"
        publish_field(series, "T", 3, time=1)
        from ovcae._stage_io import ProductPublication

        target, name = {
            "execution": (Faces, "_pass_field"),
            "publication": (_session, "write_mesh_product"),
            "completion": (ProductPublication, "wait"),
        }[failure]
        original = getattr(target, name)

        def fail(*args, **kwargs):
            if failure == "completion":
                original(*args, **kwargs)
            raise RuntimeError("injected failure")

        with monkeypatch.context() as patch:
            patch.setattr(target, name, fail)
            report = update(1)
            assert report.operators[0].status == "failed"
            assert RECIPE not in session._histories
        assert observed_changes[-1] == (False, False, False, True)
        assert getattr(target, name) is original
        report = update(1)
        assert report.operators[0].status == "succeeded", report.diagnostics
        np.testing.assert_array_equal(read_array(stage, RECIPE, "primvars:temperature", report.output_ordinal), [3] * 4)
        assert observed_changes[-1] == (True, True, True, True)
        assert update(1).operators[0].status == "unchanged"


def test_field_only_changes_rebuild_surface_and_expose_current_samples(monkeypatch):
    from ovcae.operators import utils

    observed = []
    with pipeline(monkeypatch) as (stage, series, _session, update, counts):
        original = Faces.execute

        def execute(self, context):
            observed.append(utils.get_input_samples(context, role="source"))
            return original(self, context)

        monkeypatch.setattr(Faces, "execute", execute)
        first = update()
        points = read_array(stage, RECIPE, "points", first.output_ordinal)
        publish_field(series, "T", 8, time=2)
        second = update(2)
        assert second.operators[0].status == "succeeded", second.diagnostics
        assert counts["execute"] == 2
        assert counts["extract"] == 2
        np.testing.assert_array_equal(read_array(stage, RECIPE, "points", second.output_ordinal), points)
        np.testing.assert_array_equal(read_array(stage, RECIPE, "primvars:temperature", second.output_ordinal), [8] * 4)
        previous, current = observed
        assert current.geometry == previous.geometry
        assert current.topology == previous.topology
        assert current.fields["temperature"].time == 2
        assert previous.fields["temperature"].time == 0


def test_geometry_topology_and_recipe_changes_rebuild_surface(monkeypatch):
    with pipeline(monkeypatch) as (stage, series, _session, update, counts):
        first = update()
        points = read_array(stage, RECIPE, "points", first.output_ordinal).reshape(-1, 3)
        series.publish(
            KEY, component="geometry", value=np.asarray([[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2]], np.float32)
        )
        second = update()
        np.testing.assert_array_equal(
            read_array(stage, RECIPE, "points", second.output_ordinal).reshape(-1, 3), points * 2
        )
        assert counts["extract"] == 2
        series.publish(
            KEY,
            component="topology",
            value={
                "cell_types": np.asarray([10], np.uint8),
                "connectivity_offsets": np.asarray([0, 4], np.int32),
                "connectivity": np.asarray([1, 0, 2, 3], np.int32),
            },
        )
        third = update()
        assert third.operators[0].status == "succeeded"
        assert counts["extract"] == 3
        write_scalar(stage, RECIPE, "cae:viz:faces:externalOnly", False, third.output_ordinal + 1)
        stage.advance_write_floor(third.output_ordinal + 1).wait()
        assert update().operators[0].status == "succeeded"
        assert counts["extract"] == 4


def test_one_binding_can_change_while_another_stays_unchanged(monkeypatch):
    other = "/World/OtherFaces"
    with pipeline(monkeypatch, bindings=(RECIPE, other)) as (stage, series, _session, update, counts):
        first = update()
        series.publish(
            "/other", component="geometry", value=np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float32)
        )
        series.publish(
            "/other",
            component="topology",
            value={
                "cell_types": np.asarray([10], np.uint8),
                "connectivity_offsets": np.asarray([0, 4], np.int32),
                "connectivity": np.asarray([0, 1, 2, 3], np.int32),
            },
        )
        series.publish(
            "/other", component="field:T", time=0, value={"association": "node", "values": np.ones(4, np.float32)}
        )
        write_relationship(
            stage, other, "cae:viz:dataset_selection:source:target", ["/other"], first.output_ordinal + 1
        )
        stage.advance_write_floor(first.output_ordinal + 1).wait()
        update()
        publish_field(series, "T", 9, time=1)
        report = update(1)
        assert {r.operator_path: r.status for r in report.operators} == {RECIPE: "succeeded", other: "unchanged"}
        before = counts.copy()
        assert all(r.status == "unchanged" for r in update(1).operators)
        assert counts == before


def test_field_input_must_refer_to_a_selected_dataset_role(monkeypatch):
    with pipeline(monkeypatch) as (stage, _series, _session, update, counts):
        first = update()
        write_token(stage, RECIPE, "cae:viz:field_selection:temperature:input", "missing", first.output_ordinal + 1)
        stage.advance_write_floor(first.output_ordinal + 1).wait()
        before = counts["resolve"]
        report = update()
        assert report.operators[0].status == "failed"
        assert report.diagnostics[0].code == "INVALID_RECIPE"
        assert "unavailable input" in report.diagnostics[0].message
        assert counts["resolve"] == before


@pytest.fixture
def observed_changes(monkeypatch):
    """Record public query results during real executions, before any materialization."""
    observed = []
    original = Faces.execute

    def execute(self, context):
        changes = context.changes
        observed.append(
            (
                changes.recipe_changed(),
                changes.input_changed("source", components={"topology"}),
                changes.input_changed("source", components={"geometry"}),
                changes.input_changed("source", fields={"temperature"}),
            )
        )
        return original(self, context)

    monkeypatch.setattr(Faces, "execute", execute)
    return observed


def test_change_queries_distinguish_field_geometry_and_recipe(monkeypatch, observed_changes):
    with pipeline(monkeypatch) as (stage, series, _session, update, _counts):
        update()
        assert observed_changes[-1] == (True, True, True, True)
        # Held samples and irrelevant changes never enter execute.
        publish_field(series, "unused", 9, time=1)
        assert update(1).operators[0].status == "unchanged"
        assert len(observed_changes) == 1
        for value in (5, 7):
            publish_field(series, "T", value, time=1)
            assert update(1).operators[0].status == "succeeded"
            assert observed_changes[-1] == (False, False, False, True)
        series.publish(
            KEY, component="geometry", value=np.asarray([[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2]], np.float32)
        )
        report = update(1)
        assert report.operators[0].status == "succeeded", report.diagnostics
        assert observed_changes[-1] == (False, False, True, False)
        write_scalar(stage, RECIPE, "cae:viz:faces:externalOnly", False, report.output_ordinal + 1)
        stage.advance_write_floor(report.output_ordinal + 1).wait()
        assert update(1).operators[0].status == "succeeded"
        # Recipe edits remain independent of held scientific identities even
        # when the original publication has aged out of retained history.
        assert observed_changes[-1] == (True, False, False, False)


def test_change_queries_validate_selections_and_expire(monkeypatch):
    from dataclasses import FrozenInstanceError, replace

    from ovcae.operators import InvalidRecipe
    from ovcae.scene import ExpiredViewError

    saved = []
    with pipeline(monkeypatch) as (_stage, _series, _session, update, counts):
        original = Faces.execute

        def execute(self, context):
            changes = context.changes
            before = counts.copy()
            for role, kwargs in [
                ("missing", {"components": {"geometry"}}),
                ("source", {}),
                ("source", {"components": {"field:temperature"}}),
                ("source", {"fields": {"T"}}),
                ("source", {"fields": "temperature"}),
                ("source", {"components": "geometry"}),
                ("source", {"fields": {1}}),
            ]:
                with pytest.raises(InvalidRecipe):
                    changes.input_changed(role, **kwargs)
            assert changes.input_changed("source", components=["geometry"], fields=["temperature"])
            with pytest.raises(FrozenInstanceError):
                changes._evaluation = None
            with pytest.raises(InvalidRecipe, match="dataset_components"):
                _ = replace(context, _changes=None).changes
            assert counts == before
            saved.append((context, changes))
            return original(self, context)

        monkeypatch.setattr(Faces, "execute", execute)
        report = update()
        assert report.operators[0].status == "succeeded", report.diagnostics
        context, changes = saved[0]
        with pytest.raises(ExpiredViewError):
            _ = context.changes
        with pytest.raises(ExpiredViewError):
            changes.recipe_changed()
        with pytest.raises(ExpiredViewError):
            changes.input_changed("source", components={"geometry"})


def test_recipe_checkpoint_uses_observed_floor_not_requested_minimum(monkeypatch):
    with pipeline(monkeypatch) as (stage, _series, session, update, _counts):
        stage.advance_write_floor(10).wait()
        assert update(minimum=1).operators[0].status == "succeeded"
        candidate = session._histories[RECIPE].candidate
        assert candidate.recipe.ordinal == 10
        assert candidate.ordinal == 11
        assert update(minimum=1).operators[0].status == "unchanged"
        assert session._histories[RECIPE].baseline is candidate


def test_identical_recipe_write_invalidates_committed_result(monkeypatch):
    with pipeline(monkeypatch) as (stage, _series, _session, update, counts):
        # Author the default explicitly; subsequent identical writes must still
        # invalidate, unlike the removed value fingerprint.
        first = update()
        ordinal = first.output_ordinal + 1
        write_scalar(stage, RECIPE, "cae:viz:faces:externalOnly", True, ordinal)
        stage.advance_write_floor(ordinal).wait()
        assert update().operators[0].status == "succeeded"
        assert counts["execute"] == 2
        assert update().operators[0].status == "unchanged"


def test_recipe_checkpoint_precedes_discovery(monkeypatch):
    from ovcae import _session

    with pipeline(monkeypatch) as (stage, _series, session, update, _counts):
        original = _session.discover_operators

        def discover(*args):
            discovered = original(*args)
            # Deliberately violate input stability to verify that the checkpoint
            # cannot consume a write made after discovery observed old schemas.
            write_schemas(stage, RECIPE, sorted(discovered[RECIPE]), 2)
            stage.advance_write_floor(2).wait()
            return discovered

        with monkeypatch.context() as patch:
            patch.setattr(_session, "discover_operators", discover)
            with open_series(_series) as source, source.at_time(0) as frame:
                report = session.update(input_ordinal=1, output_ordinal=3, frame_data=frame).wait()
            assert report.operators[0].status == "succeeded", report.diagnostics
        assert session._histories[RECIPE].candidate.recipe.ordinal == 1
        stage.advance_write_floor(3).wait()
        from ovcae._reuse import oldest_preserved_ordinal

        # The injected write must invalidate if its history is still retained.
        # If it already expired, the explicit assume-unchanged policy applies.
        expected = "unchanged" if oldest_preserved_ordinal(stage) > 2 else "succeeded"
        assert update().operators[0].status == expected


def test_unchanged_update_does_not_read_execution_only_recipe_values(monkeypatch):
    from ovcae.scene import _views

    with pipeline(monkeypatch) as (_stage, _series, _session, update, _counts):
        assert update().operators[0].status == "succeeded"
        original = _views.read_value

        def read(*args, **kwargs):
            assert args[2] != "cae:viz:faces:externalOnly"
            return original(*args, **kwargs)

        monkeypatch.setattr(_views, "read_value", read)
        assert update().operators[0].status == "unchanged"


def test_update_reads_history_frontier_once(monkeypatch):
    from unittest.mock import Mock

    with pipeline(monkeypatch) as (stage, _series, _session, update, _counts):
        assert update().operators[0].status == "succeeded"
        frontier = Mock(wraps=stage.get_oldest_preserved_ordinal)
        monkeypatch.setattr(stage, "get_oldest_preserved_ordinal", frontier)
        assert update().operators[0].status == "unchanged"
        frontier.assert_called_once()


def advance_camera_commits(stage, count=16):
    """Age native history using unrelated presentation writes, without Session."""
    from ovcae._stage_io import write_floor

    for value in range(count):
        ordinal = write_floor(stage) + 1
        write_scalar(stage, "/Camera", "focalLength", np.float32(30 + value), ordinal)
        stage.advance_write_floor(ordinal).wait()


def test_repeated_hidden_iso_edits_keep_surface_unchanged(monkeypatch):
    from ovcae._stage_io import write_floor

    hidden = "/World/IsoSurface"
    with pipeline(monkeypatch) as (stage, _series, session, update, counts):
        write_schemas(stage, hidden, ["CaeVizOperatorAPI", "CaeVizIsoSurfaceAPI"], 2)
        write_token(stage, hidden, "usd-prim-type", "Mesh", 2)
        write_scalar(stage, hidden, "cae:viz:operator:enabled", True, 2)
        write_token(stage, hidden, "visibility", "invisible", 2)
        stage.advance_write_floor(2).wait()
        first = update()
        before = counts.copy()
        for value in range(20):
            ordinal = write_floor(stage) + 1
            write_scalar(stage, hidden, "cae:viz:isoSurface:isoValue", np.float32(value), ordinal)
            stage.advance_write_floor(ordinal).wait()
            report = update()
            assert {r.operator_path: r.status for r in report.operators} == {RECIPE: "unchanged", hidden: "skipped"}
            record = session._histories[RECIPE].baseline
            assert record.ordinal == first.output_ordinal
            assert record.recipe.ordinal == record.product_checked_ordinal == report.input_ordinal
        assert counts == before


@pytest.mark.parametrize("promote_before_gap", [False, True])
def test_camera_gap_preserves_committed_product_and_refreshes_checkpoints(monkeypatch, promote_before_gap):
    from ovcae._reuse import oldest_preserved_ordinal

    with pipeline(monkeypatch) as (stage, _series, session, update, counts):
        first = update()
        if promote_before_gap:
            assert update().operators[0].status == "unchanged"
        advance_camera_commits(stage)
        assert oldest_preserved_ordinal(stage) > first.output_ordinal
        before = counts.copy()
        report = update(minimum=1)
        assert report.operators[0].status == "unchanged"
        assert counts == before
        record = session._histories[RECIPE].baseline
        assert record.ordinal == first.output_ordinal
        from ovcae._stage_io import write_floor

        assert record.recipe.ordinal == record.product_checked_ordinal == write_floor(stage) - 1


@pytest.mark.parametrize("change", ["recipe", "product"])
def test_lapsed_history_still_detects_changes_in_retained_suffix(monkeypatch, change):
    from ovcae._stage_io import write_floor

    with pipeline(monkeypatch) as (stage, _series, _session, update, counts):
        update()
        advance_camera_commits(stage)
        ordinal = write_floor(stage) + 1
        if change == "recipe":
            write_scalar(stage, RECIPE, "cae:viz:faces:externalOnly", False, ordinal)
        else:
            write_token(stage, RECIPE, "subdivisionScheme", "catmullClark", ordinal)
        stage.advance_write_floor(ordinal).wait()
        assert update().operators[0].status == "succeeded"
        assert counts["execute"] == 2


def test_product_deletion_before_expired_interval_still_invalidates(monkeypatch):
    with pipeline(monkeypatch) as (stage, _series, _session, update, counts):
        first = update()
        paths, plist, query = path_query(stage, RECIPE)
        try:
            stage.delete_attributes(query, ["points"], first.output_ordinal + 1).wait()
        finally:
            release_path_query(paths, plist, query)
        stage.advance_write_floor(first.output_ordinal + 1).wait()
        advance_camera_commits(stage)
        assert update().operators[0].status == "succeeded"
        assert counts["execute"] == 2


@pytest.mark.parametrize("change", ["recipe", "product"])
def test_hidden_observed_change_is_not_forgotten_after_expiry(monkeypatch, change):
    from ovcae._stage_io import write_floor

    with pipeline(monkeypatch) as (stage, _series, _session, update, counts):
        first = update()
        ordinal = first.output_ordinal + 1
        write_token(stage, RECIPE, "visibility", "invisible", ordinal)
        if change == "recipe":
            write_scalar(stage, RECIPE, "cae:viz:faces:externalOnly", False, ordinal)
        else:
            write_token(stage, RECIPE, "subdivisionScheme", "catmullClark", ordinal)
        stage.advance_write_floor(ordinal).wait()
        assert update().operators[0].status == "skipped"
        advance_camera_commits(stage)
        assert update().operators[0].status == "skipped"
        ordinal = write_floor(stage) + 1
        write_token(stage, RECIPE, "visibility", "inherited", ordinal)
        stage.advance_write_floor(ordinal).wait()
        assert update().operators[0].status == "succeeded"
        assert counts["execute"] == 2
        assert update().operators[0].status == "unchanged"


def test_expiry_does_not_mask_scientific_identity_change(monkeypatch, observed_changes):
    with pipeline(monkeypatch) as (stage, series, _session, update, counts):
        update()
        advance_camera_commits(stage)
        publish_field(series, "T", 12)
        assert update().operators[0].status == "succeeded"
        assert counts["execute"] == 2
        assert observed_changes[-1] == (False, False, False, True)


@pytest.mark.parametrize("change", ["recipe", "product"])
def test_unobserved_change_in_expired_interval_is_assumed_unchanged(monkeypatch, change):
    """Document the accepted risk when the application misses its check window."""
    with pipeline(monkeypatch) as (stage, _series, _session, update, counts):
        first = update()
        ordinal = first.output_ordinal + 1
        if change == "recipe":
            write_scalar(stage, RECIPE, "cae:viz:faces:externalOnly", False, ordinal)
        else:
            write_token(stage, RECIPE, "subdivisionScheme", "catmullClark", ordinal)
        stage.advance_write_floor(ordinal).wait()
        advance_camera_commits(stage)
        before = counts.copy()
        assert update().operators[0].status == "unchanged"
        assert counts == before


def test_product_read_failure_is_not_treated_as_history_expiry(monkeypatch):
    from ovcae._reuse import product_intact

    with pipeline(monkeypatch) as (stage, _series, session, update, _counts):
        first = update()
        candidate = session._histories[RECIPE].candidate

        def fail(*args, **kwargs):
            raise ovstage.OvstageError(1, "injected product read failure")

        monkeypatch.setattr(stage, "read_attributes", fail)
        assert not product_intact(stage, RECIPE, candidate, first.output_ordinal, oldest=first.output_ordinal + 1)


def test_uncommitted_replacement_does_not_reuse_old_baseline_at_same_floor(monkeypatch):
    with pipeline(monkeypatch) as (_stage, series, _session, update, counts):
        update()
        assert update().operators[0].status == "unchanged"
        publish_field(series, "T", 7, time=1)
        assert update(1, commit=False).operators[0].status == "succeeded"
        # Returning to the baseline's scientific sample is not enough: native
        # product storage now contains a newer, still-uncommitted publication.
        assert update(0).operators[0].status == "succeeded"
        assert counts["execute"] == 3
