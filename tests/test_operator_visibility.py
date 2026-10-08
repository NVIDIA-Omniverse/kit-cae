# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Visibility suspends operator work, not product ownership or input availability."""

import numpy as np
import pytest

import ovcae

from ._file_scenarios import open_scenario
from ._ovstage_helpers import delete_prim, read_array, write_relationship, write_scalar, write_schemas, write_token
from .test_change_detection import KEY, RECIPE, pipeline, publish_field


def set_visibility(stage, path, value, ordinal):
    write_token(stage, path, "visibility", value, ordinal)
    stage.advance_write_floor(ordinal).wait()


@pytest.mark.parametrize("hidden_path", [RECIPE, "/World"])
def test_initially_hidden_operator_does_not_plan_load_execute_or_publish(monkeypatch, hidden_path):
    def unexpected_plan(*args, **kwargs):
        pytest.fail("hidden operator planned scientific inputs")

    monkeypatch.setattr("ovcae._session.plan_operator", unexpected_plan)
    with pipeline(monkeypatch) as (stage, _series, session, update, counts):
        write_token(stage, RECIPE, "visibility", "inherited", 2)
        set_visibility(stage, hidden_path, "invisible", 2)
        before = counts.copy()
        report = update(minimum=1)
        assert report.operators[0].status == "skipped"
        assert report.operators[0].reason == "operator invisible"
        assert report.operators[0].product_paths == ()
        assert not report.diagnostics
        assert counts == before
        assert session._operator_instances == {}
        assert read_array(stage, RECIPE, "points", report.output_ordinal) is None


@pytest.mark.parametrize("hidden_path", [RECIPE, "/World"])
def test_hide_retains_product_and_show_observes_edits_made_while_hidden(monkeypatch, hidden_path):
    with pipeline(monkeypatch) as (stage, series, _session, update, counts):
        first = update()
        assert first.operators[0].status == "succeeded"
        points = read_array(stage, RECIPE, "points", first.output_ordinal)
        set_visibility(stage, hidden_path, "invisible", first.output_ordinal + 1)
        publish_field(series, "T", 9)
        before = counts.copy()
        hidden = update()
        assert hidden.operators[0].status == "skipped"
        assert counts == before
        np.testing.assert_array_equal(read_array(stage, RECIPE, "points", hidden.output_ordinal), points)
        np.testing.assert_array_equal(read_array(stage, RECIPE, "primvars:temperature", hidden.output_ordinal), [1] * 4)

        set_visibility(stage, hidden_path, "inherited", hidden.output_ordinal + 1)
        shown = update()
        assert shown.operators[0].status == "succeeded", shown.diagnostics
        assert counts["execute"] == before["execute"] + 1
        np.testing.assert_array_equal(read_array(stage, RECIPE, "primvars:temperature", shown.output_ordinal), [9] * 4)


@pytest.mark.parametrize("hidden_path", [RECIPE, "/World"])
@pytest.mark.parametrize("change_input", [False, True])
def test_hidden_unchanged_recipe_rolls_checkpoint_without_revalidating_inputs(monkeypatch, hidden_path, change_input):
    from ovcae._reuse import Recipe

    checks = []
    original = Recipe.changed

    def changed(recipe, stage, floor, **kwargs):
        checks.append((recipe.ordinal, floor))
        return original(recipe, stage, floor, **kwargs)

    monkeypatch.setattr(Recipe, "changed", changed)
    with pipeline(monkeypatch) as (stage, series, session, update, counts):
        first = update()
        assert update().operators[0].status == "unchanged"
        record = session._histories[RECIPE].baseline
        previous = record.recipe.ordinal
        inputs = record.inputs
        set_visibility(stage, hidden_path, "invisible", first.output_ordinal + 2)
        if change_input:
            publish_field(series, "T", 9)
        before = counts.copy()
        for _ in range(20):
            report = update()
            assert report.operators[0].reason == "operator invisible"
            assert report.operators[0].product_paths == ()
            assert not report.diagnostics
            assert checks[-1] == (previous, report.input_ordinal)
            assert record.recipe.ordinal == record.product_checked_ordinal == report.input_ordinal
            assert record.ordinal == first.output_ordinal
            assert record.inputs is inputs
            assert counts == before
            previous = report.input_ordinal

        set_visibility(stage, hidden_path, "inherited", report.output_ordinal + 1)
        shown = update()
        assert shown.operators[0].status == ("succeeded" if change_input else "unchanged")
        assert counts["execute"] == before["execute"] + int(change_input)


@pytest.mark.parametrize("diagnostic", ["AMBIGUOUS_OPERATOR", "UNSUPPORTED_OPERATOR", "INVALID_RECIPE"])
def test_hidden_recipe_diagnostics_are_deferred_until_shown(monkeypatch, diagnostic):
    with pipeline(monkeypatch) as (stage, _series, _session, update, counts):
        first = update()
        points = read_array(stage, RECIPE, "points", first.output_ordinal).copy()
        ordinal = first.output_ordinal + 1
        schemas = ["CaeVizOperatorAPI", "CaeVizDatasetSelectionAPI:source", "CaeVizFieldSelectionAPI:temperature"]
        if diagnostic != "UNSUPPORTED_OPERATOR":
            schemas.append("CaeVizFacesAPI")
        if diagnostic == "AMBIGUOUS_OPERATOR":
            schemas += ["CaeVizIsoSurfaceAPI", "CaeVizFieldSelectionAPI:contour", "CaeVizFieldSelectionAPI:colors"]
        if diagnostic == "INVALID_RECIPE":
            write_relationship(stage, RECIPE, "cae:viz:dataset_selection:source:target", [], ordinal)
        write_schemas(stage, RECIPE, schemas, ordinal)
        set_visibility(stage, RECIPE, "invisible", ordinal)
        before = counts.copy()
        hidden = update()
        assert hidden.operators[0].reason == "operator invisible"
        assert not hidden.diagnostics
        assert counts == before
        np.testing.assert_array_equal(read_array(stage, RECIPE, "points", hidden.output_ordinal), points)

        set_visibility(stage, RECIPE, "inherited", hidden.output_ordinal + 1)
        shown = update()
        assert [item.code for item in shown.diagnostics] == [diagnostic]
        assert counts["execute"] == before["execute"]


def test_show_local_opinion_does_not_override_hidden_ancestor(monkeypatch):
    with pipeline(monkeypatch) as (stage, _series, _session, update, counts):
        write_token(stage, "/World", "visibility", "invisible", 2)
        set_visibility(stage, RECIPE, "invisible", 2)
        assert update().operators[0].status == "skipped"
        set_visibility(stage, RECIPE, "inherited", 4)
        assert update().operators[0].status == "skipped"
        assert counts["execute"] == 0
        set_visibility(stage, "/World", "inherited", 6)
        assert update().operators[0].status == "succeeded"


@pytest.mark.parametrize("action", ["disable", "remove"])
def test_hidden_product_still_cleans_up_when_disabled_or_removed(monkeypatch, action):
    with pipeline(monkeypatch) as (stage, _series, _session, update, counts):
        first = update()
        set_visibility(stage, RECIPE, "invisible", first.output_ordinal + 1)
        hidden = update()
        assert read_array(stage, RECIPE, "points", hidden.output_ordinal) is not None
        ordinal = hidden.output_ordinal + 1
        if action == "disable":
            write_scalar(stage, RECIPE, "cae:viz:operator:enabled", False, ordinal)
        else:
            delete_prim(stage, RECIPE, ordinal)
        stage.advance_write_floor(ordinal).wait()
        report = update()
        assert report.operators[0].status == ("skipped" if action == "disable" else "removed")
        assert counts["execute"] == 1
        assert read_array(stage, RECIPE, "points", report.output_ordinal) is None


def test_hidden_input_does_not_suspend_visible_operator(monkeypatch):
    with pipeline(monkeypatch) as (stage, _series, _session, update, counts):
        set_visibility(stage, KEY, "invisible", 2)
        report = update()
        assert report.operators[0].status == "succeeded", report.diagnostics
        assert counts["execute"] == 1


def test_hidden_seed_points_still_feed_visible_streamlines():
    with (
        open_scenario("streamlines", stage_name="ovcae.tests.visibility.seeds") as (stage, source),
        ovcae.Session(stage, device="cpu") as session,
    ):
        set_visibility(stage, "/World/Controls/Seeds", "invisible", 2)
        with source.at_time(0) as frame:
            report = session.update(input_ordinal=2, output_ordinal=3, frame_data=frame).wait()
        assert report.committable
        assert report.operators[0].status == "succeeded", report.diagnostics
        stage.advance_write_floor(3).wait()
        assert read_array(stage, "/World/Recipes/Streamlines", "points", 3).size > 0
