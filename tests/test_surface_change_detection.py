# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""The migrated surface operators share precise input gating and recipe invalidation."""

from functools import partial

import numpy as np
import ovstage
import pytest

from ovcae.operators._builtins.iso_surface import IsoSurface
from ovcae.operators._builtins.planar_slice import PlanarSlice

from ._ovstage_helpers import path_query, read_array, release_path_query, write_scalar, write_schemas, write_transform
from .test_change_detection import KEY, RECIPE, pipeline, publish_field, setup_recipe

CASES = [(IsoSurface, "CaeVizIsoSurfaceAPI"), (PlanarSlice, "CaeVizPlanarSliceAPI")]


def setup_surface(stage, path, *, schema):
    """Select a nonconstant contour and an independently changing color field."""
    setup_recipe(stage, path)
    fields = ["colors", "contour"] if schema == "CaeVizIsoSurfaceAPI" else ["colors"]
    write_schemas(
        stage,
        path,
        [
            "CaeVizOperatorAPI",
            schema,
            "CaeVizDatasetSelectionAPI:source",
            *[f"CaeVizFieldSelectionAPI:{role}" for role in fields],
        ],
        1,
    )
    paths, plist, query = path_query(stage, path)
    try:
        for role in fields:
            field = "C" if role == "contour" else "T"
            stage.write_attribute(
                query,
                f"cae:viz:field_selection:{role}:fieldNames",
                1,
                [np.asarray([paths.intern_token(field)], np.uint64)],
                is_array=True,
                semantic=ovstage.AttributeSemantic.TOKEN_ID,
            ).wait()
    finally:
        release_path_query(paths, plist, query)
    if schema == "CaeVizIsoSurfaceAPI":
        write_scalar(stage, path, "cae:viz:isoSurface:isoValue", np.float32(0.25), 1)
    else:
        # A free plane uses the native local +Y axis.
        write_transform(stage, path, (0.0, 0.25, 0.0), 1)


@pytest.mark.parametrize("operator_type,schema", CASES)
def test_surface_gate_skips_and_refreshes_scientific_fields(monkeypatch, operator_type, schema):
    with pipeline(monkeypatch, operator_type=operator_type, recipe_setup=partial(setup_surface, schema=schema)) as (
        stage,
        series,
        _session,
        update,
        counts,
    ):
        series.publish(
            KEY,
            component="field:C",
            time=0,
            value={"association": "node", "values": np.asarray([0, 1, 0, 0], np.float32)},
        )
        report = update()
        assert report.operators[0].status == "succeeded", report.diagnostics
        points = read_array(stage, RECIPE, "points", report.output_ordinal).copy()
        assert points.size > 0
        before = counts.copy()
        assert update(1).operators[0].status == "unchanged"
        publish_field(series, "unused", 9, time=1)
        assert update(1).operators[0].status == "unchanged"
        assert counts == before
        for value in (5, 7):
            publish_field(series, "T", value, time=1)
            report = update(1)
            assert report.operators[0].status == "succeeded", report.diagnostics
            np.testing.assert_allclose(read_array(stage, RECIPE, "points", report.output_ordinal), points)
            np.testing.assert_allclose(read_array(stage, RECIPE, "primvars:colors", report.output_ordinal), value)
        assert counts["execute"] == counts["resolve"] == 3
        before = counts.copy()
        assert update(1).operators[0].status == "unchanged"
        assert counts == before


@pytest.mark.parametrize("operator_type,schema", CASES)
def test_surface_recipe_and_geometry_changes_rebuild(monkeypatch, operator_type, schema):
    with pipeline(monkeypatch, operator_type=operator_type, recipe_setup=partial(setup_surface, schema=schema)) as (
        stage,
        series,
        _session,
        update,
        counts,
    ):
        series.publish(
            KEY,
            component="field:C",
            time=0,
            value={"association": "node", "values": np.asarray([0, 1, 0, 0], np.float32)},
        )
        report = update()
        assert report.operators[0].status == "succeeded", report.diagnostics
        original = read_array(stage, RECIPE, "points", report.output_ordinal).copy()
        ordinal = report.output_ordinal + 1
        if operator_type is IsoSurface:
            write_scalar(stage, RECIPE, "cae:viz:isoSurface:isoValue", np.float32(0.5), ordinal)
        else:
            write_transform(stage, RECIPE, (0.0, 0.5, 0.0), ordinal)
        stage.advance_write_floor(ordinal).wait()
        report = update()
        assert report.operators[0].status == "succeeded", report.diagnostics
        changed = read_array(stage, RECIPE, "points", report.output_ordinal).copy()
        assert not np.array_equal(original, changed)
        series.publish(
            KEY, component="geometry", value=np.asarray([[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2]], np.float32)
        )
        report = update()
        assert report.operators[0].status == "succeeded", report.diagnostics
        assert not np.array_equal(changed, read_array(stage, RECIPE, "points", report.output_ordinal))
        assert counts["execute"] == counts["resolve"] == 3
