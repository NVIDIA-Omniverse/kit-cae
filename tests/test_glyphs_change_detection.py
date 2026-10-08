# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Glyphs uses the shared gate while publishing current selections and fields."""

import numpy as np

from ovcae.operators._builtins.glyphs import Glyphs

from ._ovstage_helpers import read_array, write_scalar, write_schemas, write_token
from .test_change_detection import KEY, RECIPE, pipeline, publish_field, setup_recipe


def setup_glyphs(stage, path):
    """Use the shared tetrahedron recipe with a Glyphs output and selected temperature."""
    setup_recipe(stage, path)
    write_schemas(
        stage,
        path,
        [
            "CaeVizOperatorAPI",
            "CaeVizGlyphsAPI",
            "CaeVizDatasetSelectionAPI:source",
            "CaeVizFieldSelectionAPI:temperature",
        ],
        1,
    )
    write_token(stage, path, "usd-prim-type", "PointInstancer", 1)
    write_scalar(stage, path, "cae:viz:glyphs:maxCount", 0, 1)
    write_scalar(stage, path, "cae:viz:glyphs:scale", 0.1, 1)
    write_scalar(stage, path, "cae:viz:glyphs:useCellPoints", False, 1)


def test_glyphs_skips_held_and_unselected_samples_and_refreshes_fields(monkeypatch):
    with pipeline(monkeypatch, operator_type=Glyphs, recipe_setup=setup_glyphs) as (
        stage,
        series,
        _session,
        update,
        counts,
    ):
        first = update()
        assert first.operators[0].status == "succeeded", first.diagnostics
        glyphs = read_array(stage, RECIPE, "positions", first.output_ordinal).copy()
        assert len(glyphs.reshape(-1, 3)) == 4
        before = counts.copy()
        assert update(1).operators[0].status == "unchanged"
        publish_field(series, "unused", 9, time=1)
        assert update(1).operators[0].status == "unchanged"
        assert counts == before  # Includes execute, materialization, and all prim writes.
        for value in (5, 7):
            publish_field(series, "T", value, time=1)
            report = update(1)
            assert report.operators[0].status == "succeeded", report.diagnostics
            np.testing.assert_array_equal(read_array(stage, RECIPE, "positions", report.output_ordinal), glyphs)
            np.testing.assert_array_equal(
                read_array(stage, RECIPE, "primvars:temperature", report.output_ordinal), [value] * 4
            )
        assert counts["execute"] == counts["resolve"] == 3


def test_glyphs_updates_geometry_topology_and_recipe(monkeypatch):
    with pipeline(monkeypatch, operator_type=Glyphs, recipe_setup=setup_glyphs) as (
        stage,
        series,
        _session,
        update,
        counts,
    ):
        report = update()
        # Add an isolated node. Cell-connected selection must subsequently exclude it.
        glyphs = np.asarray([[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2], [9, 9, 9]], np.float32)
        series.publish(KEY, component="geometry", value=glyphs)
        series.publish(
            KEY, component="field:T", time=0, value={"association": "node", "values": np.arange(5, dtype=np.float32)}
        )
        report = update()
        assert report.operators[0].status == "succeeded", report.diagnostics
        np.testing.assert_array_equal(
            read_array(stage, RECIPE, "positions", report.output_ordinal).reshape(-1, 3), glyphs
        )
        ordinal = report.output_ordinal + 1
        write_scalar(stage, RECIPE, "cae:viz:glyphs:useCellPoints", True, ordinal)
        stage.advance_write_floor(ordinal).wait()
        report = update()
        assert report.operators[0].status == "succeeded", report.diagnostics
        assert len(read_array(stage, RECIPE, "positions", report.output_ordinal).reshape(-1, 3)) == 4
        series.publish(
            KEY,
            component="topology",
            value={
                "cell_types": np.asarray([10], np.uint8),
                "connectivity_offsets": np.asarray([0, 4], np.int32),
                "connectivity": np.asarray([0, 1, 2, 4], np.int32),
            },
        )
        report = update()
        assert report.operators[0].status == "succeeded", report.diagnostics
        np.testing.assert_array_equal(
            read_array(stage, RECIPE, "positions", report.output_ordinal).reshape(-1, 3), glyphs[[0, 1, 2, 4]]
        )
        ordinal = report.output_ordinal + 1
        write_scalar(stage, RECIPE, "cae:viz:glyphs:maxCount", 2, ordinal)
        write_scalar(stage, RECIPE, "cae:viz:glyphs:scale", 0.3, ordinal)
        stage.advance_write_floor(ordinal).wait()
        report = update()
        assert report.operators[0].status == "succeeded", report.diagnostics
        assert len(read_array(stage, RECIPE, "positions", report.output_ordinal).reshape(-1, 3)) == 2
        np.testing.assert_allclose(read_array(stage, RECIPE, "scales", report.output_ordinal), np.full((2, 3), 0.3))
        assert counts["execute"] == 5
        before = counts.copy()
        assert update().operators[0].status == "unchanged"
        assert counts == before


def test_glyph_transform_fields_follow_selected_samples_forward_and_backward(monkeypatch):
    from ._ovstage_helpers import path_query, release_path_query

    def setup(stage, path):
        setup_glyphs(stage, path)
        write_schemas(
            stage,
            path,
            [
                "CaeVizOperatorAPI",
                "CaeVizGlyphsAPI",
                "CaeVizDatasetSelectionAPI:source",
                "CaeVizFieldSelectionAPI:scales",
                "CaeVizFieldSelectionAPI:orientations",
            ],
            1,
        )
        paths, plist, query = path_query(stage, path)
        try:
            import ovstage

            for role, name in (("scales", "T"), ("orientations", "V")):
                stage.write_attribute(
                    query,
                    f"cae:viz:field_selection:{role}:fieldNames",
                    1,
                    np.asarray([paths.intern_token(name)], np.uint64),
                    is_array=True,
                    semantic=ovstage.AttributeSemantic.TOKEN_ID,
                ).wait()
        finally:
            release_path_query(paths, plist, query)

    with pipeline(monkeypatch, operator_type=Glyphs, recipe_setup=setup) as (stage, series, _, update, counts):
        for time, direction in ((0, [1.0, 0, 0]), (1, [0.0, 1, 0])):
            series.publish(
                KEY, component="field:V", time=time, value={"association": "node", "values": np.tile(direction, (4, 1))}
            )
            publish_field(series, "T", time + 1, time=time)
        for time in (0, 1, 0):
            report = update(time)
            assert report.operators[0].status == "succeeded", report.diagnostics
            n = report.output_ordinal
            np.testing.assert_allclose(read_array(stage, RECIPE, "scales", n), time + 1)
            expected = [0, 0, 0, 1] if time == 0 else [0, 0, 2**-0.5, 2**-0.5]
            np.testing.assert_allclose(
                read_array(stage, RECIPE, "orientations", n), np.tile(expected, (4, 1)), atol=3e-4
            )
        before = counts.copy()
        assert update(0).operators[0].status == "unchanged"
        assert counts == before


def test_removal_clears_instances_without_recreating_a_deleted_prim(monkeypatch):
    from ._ovstage_helpers import delete_prim

    with pipeline(monkeypatch, operator_type=Glyphs, recipe_setup=setup_glyphs) as (stage, _, _, update, _):
        first = update()
        n = first.output_ordinal + 1
        # Removing the operator API leaves the application-owned prim intact.
        write_schemas(stage, RECIPE, [], n)
        stage.advance_write_floor(n).wait()
        removed = update()
        assert removed.operators[0].status == "removed"
        assert read_array(stage, RECIPE, "positions", removed.output_ordinal).size == 0
        # Recreate the recipe, then delete the entire application-owned prim.
        write_schemas(
            stage,
            RECIPE,
            ["CaeVizOperatorAPI", "CaeVizGlyphsAPI", "CaeVizDatasetSelectionAPI:source"],
            removed.output_ordinal + 1,
        )
        stage.advance_write_floor(removed.output_ordinal + 1).wait()
        again = update()
        assert again.operators[0].status == "succeeded", again.diagnostics
        delete_prim(stage, RECIPE, again.output_ordinal + 1)
        stage.advance_write_floor(again.output_ordinal + 1).wait()
        removed = update()
        assert removed.operators[0].status == "removed"
        assert read_array(stage, RECIPE, "positions", removed.output_ordinal) is None
