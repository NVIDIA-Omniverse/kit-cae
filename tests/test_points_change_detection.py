# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Points uses the shared gate while publishing current selections and fields."""

import numpy as np

from ovcae.operators._builtins.points import Points

from ._ovstage_helpers import read_array, write_scalar, write_schemas, write_token
from .test_change_detection import KEY, RECIPE, pipeline, publish_field, setup_recipe


def setup_points(stage, path):
    """Use the shared tetrahedron recipe with a Points output and selected temperature."""
    setup_recipe(stage, path)
    write_schemas(
        stage,
        path,
        [
            "CaeVizOperatorAPI",
            "CaeVizPointsAPI",
            "CaeVizDatasetSelectionAPI:source",
            "CaeVizFieldSelectionAPI:temperature",
        ],
        1,
    )
    write_token(stage, path, "usd-prim-type", "Points", 1)
    write_scalar(stage, path, "cae:viz:points:maxCount", 0, 1)
    write_scalar(stage, path, "cae:viz:points:width", 0.1, 1)
    write_scalar(stage, path, "cae:viz:points:useCellPoints", False, 1)


def test_points_skips_held_and_unselected_samples_and_refreshes_fields(monkeypatch):
    with pipeline(monkeypatch, operator_type=Points, recipe_setup=setup_points) as (
        stage,
        series,
        _session,
        update,
        counts,
    ):
        first = update()
        assert first.operators[0].status == "succeeded", first.diagnostics
        points = read_array(stage, RECIPE, "points", first.output_ordinal).copy()
        assert len(points.reshape(-1, 3)) == 4
        before = counts.copy()
        assert update(1).operators[0].status == "unchanged"
        publish_field(series, "unused", 9, time=1)
        assert update(1).operators[0].status == "unchanged"
        assert counts == before  # Includes execute, materialization, and all prim writes.
        for value in (5, 7):
            publish_field(series, "T", value, time=1)
            report = update(1)
            assert report.operators[0].status == "succeeded", report.diagnostics
            np.testing.assert_array_equal(read_array(stage, RECIPE, "points", report.output_ordinal), points)
            np.testing.assert_array_equal(
                read_array(stage, RECIPE, "primvars:temperature", report.output_ordinal), [value] * 4
            )
        assert counts["execute"] == counts["resolve"] == 3


def test_points_updates_geometry_topology_and_recipe(monkeypatch):
    with pipeline(monkeypatch, operator_type=Points, recipe_setup=setup_points) as (
        stage,
        series,
        _session,
        update,
        counts,
    ):
        report = update()
        # Add an isolated node. Cell-connected selection must subsequently exclude it.
        points = np.asarray([[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2], [9, 9, 9]], np.float32)
        series.publish(KEY, component="geometry", value=points)
        series.publish(
            KEY, component="field:T", time=0, value={"association": "node", "values": np.arange(5, dtype=np.float32)}
        )
        report = update()
        assert report.operators[0].status == "succeeded", report.diagnostics
        np.testing.assert_array_equal(read_array(stage, RECIPE, "points", report.output_ordinal).reshape(-1, 3), points)
        ordinal = report.output_ordinal + 1
        write_scalar(stage, RECIPE, "cae:viz:points:useCellPoints", True, ordinal)
        stage.advance_write_floor(ordinal).wait()
        report = update()
        assert report.operators[0].status == "succeeded", report.diagnostics
        assert len(read_array(stage, RECIPE, "points", report.output_ordinal).reshape(-1, 3)) == 4
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
            read_array(stage, RECIPE, "points", report.output_ordinal).reshape(-1, 3), points[[0, 1, 2, 4]]
        )
        ordinal = report.output_ordinal + 1
        write_scalar(stage, RECIPE, "cae:viz:points:maxCount", 2, ordinal)
        write_scalar(stage, RECIPE, "cae:viz:points:width", 0.3, ordinal)
        stage.advance_write_floor(ordinal).wait()
        report = update()
        assert report.operators[0].status == "succeeded", report.diagnostics
        assert len(read_array(stage, RECIPE, "points", report.output_ordinal).reshape(-1, 3)) == 2
        np.testing.assert_allclose(read_array(stage, RECIPE, "widths", report.output_ordinal), [0.3])
        assert counts["execute"] == 5
        before = counts.copy()
        assert update().operators[0].status == "unchanged"
        assert counts == before
