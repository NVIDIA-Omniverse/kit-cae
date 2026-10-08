# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

import numpy as np
import pytest

import ovcae

from ._file_scenarios import open_scenario
from ._ovstage_helpers import read_array, write_scalar, write_schemas, write_transform

wp = pytest.importorskip("warp")
pytest.importorskip("warp_simdata")

OPERATOR = "/World/Recipes/Streamlines"
SEEDS = "/World/Controls/Seeds"
SEED_POINTS = np.asarray(
    [
        (1.0, 1.7320508, 0.5),
        (1.6558933, 1.1216137, 1.1121716),
        (-0.25716954, -1.0663241, 2.0),
        (-0.75494856, -2.6578617, 0.5641677),
        (-0.8992655, -2.0279686, 0.74843544),
        (1.3887061, -1.0787921, 1.4502406),
        (-1.0772537, -2.0021725, 0.7974226),
        (-1.0389479, -1.5137408, 0.64242834),
        (-0.3080468, -1.2898262, 1.047858),
        (0.92161566, 0.9685264, 0.6376164),
        (0.3725212, -1.2179179, 1.0938346),
        (0.5581898, 0.60221547, 0.9858344),
    ],
    dtype=np.float32,
)


def _run_once(device: str):
    with open_scenario("streamlines", stage_name=f"ovcae.tests.streamlines.{device}") as (stage, source):
        with (
            ovcae.Session(stage, device=device) as session,
            source.at_time(0.0) as frame,
        ):
            report = session.update(input_ordinal=1, output_ordinal=2, frame_data=frame).wait()
        assert report.operators[0].status == "succeeded", report.diagnostics
        stage.advance_write_floor(2).wait()
        product = report.operators[0].product_paths[0]
        assert product == OPERATOR
        points = read_array(stage, product, "points", 2).reshape(-1, 3)
        counts = read_array(stage, product, "curveVertexCounts", 2)
        assert len(points) > 0 and len(counts) > 0
        assert int(counts.sum()) == len(points)
        return points, counts


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_streamlines_uses_default_termination_and_ignores_schema_threshold(monkeypatch, device) -> None:
    """Match Kit-CAE without generating a custom termination model on each edit."""
    from warp_simdata.operators import streamlines

    if device != "cpu" and not wp.is_cuda_available():
        pytest.skip("CUDA device unavailable")
    original = streamlines.compute
    calls = []

    def compute(*args, **kwargs):
        assert "tc_model" not in kwargs
        assert "tc_handle" not in kwargs
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(streamlines, "compute", compute)
    with (
        open_scenario("streamlines", stage_name=f"ovcae.tests.streamlines.default-termination.{device}") as (
            stage,
            source,
        ),
        ovcae.Session(stage, device=device) as session,
    ):
        products = []
        for ordinal, threshold in ((2, 0.0), (4, 1e10)):
            write_scalar(stage, OPERATOR, "cae:viz:streamlines:threshold", np.float32(threshold), ordinal)
            stage.advance_write_floor(ordinal).wait()
            with source.at_time(0.0) as frame:
                report = session.update(input_ordinal=ordinal, output_ordinal=ordinal + 1, frame_data=frame).wait()
            assert report.operators[0].status == "succeeded", report.diagnostics
            stage.advance_write_floor(ordinal + 1).wait()
            products.append(
                tuple(
                    read_array(stage, OPERATOR, name, ordinal + 1)
                    for name in ("points", "curveVertexCounts", "primvars:rnd")
                )
            )
        # The animation material needs distinct, repeatable phases per curve.
        phases = products[0][2]
        assert phases.dtype == np.float32
        assert len(phases) == len(products[0][1])
        assert len(phases) > 1 and np.unique(phases).size > 1
        assert np.all((phases >= 0.0) & (phases < 1.0))
        assert len(calls) == 2
        assert products[0][0].size > 0
        for first, second in zip(*products, strict=True):
            np.testing.assert_array_equal(first, second)


def test_streamlines_transform_scene_seed_points() -> None:
    with open_scenario("streamlines", stage_name="ovcae.tests.streamlines.moving-seeds") as (stage, source):
        with ovcae.Session(stage, device="cpu") as session:
            with source.at_time(0.0) as frame:
                first = session.update(input_ordinal=1, output_ordinal=2, frame_data=frame).wait()
            assert first.operators[0].status == "succeeded", first.diagnostics
            product = first.operators[0].product_paths[0]
            stage.advance_write_floor(2).wait()
            first_points = read_array(stage, product, "points", 2).reshape(-1, 3)
            first_counts = read_array(stage, product, "curveVertexCounts", 2)

            write_transform(stage, SEEDS, (0.0, 0.001, 0.0), 3)
            stage.advance_write_floor(3).wait()
            with source.at_time(0.0) as frame:
                second = session.update(input_ordinal=3, output_ordinal=4, frame_data=frame).wait()
            assert second.operators[0].status == "succeeded", second.diagnostics
            stage.advance_write_floor(4).wait()
            second_points = read_array(stage, product, "points", 4).reshape(-1, 3)
            second_counts = read_array(stage, product, "curveVertexCounts", 4)

        # The mixer field is nonlinear, so translating seeds can also change
        # which trajectories remain inside the volume. Every curve must still
        # start at one of the captured Points, with its local transform applied.
        assert len(second_points) > 0 and len(second_counts) > 0
        _assert_curve_starts_are_seeds(first_points, first_counts, translation=(0.0, 0.0, 0.0))
        _assert_curve_starts_are_seeds(second_points, second_counts, translation=(0.0, 0.001, 0.0))
        assert read_array(stage, product, "widths", 4).tolist() == pytest.approx([0.02])
        # Kit-CAE materials read primvars:widths; publishing it also overwrites Kit placeholders.
        assert read_array(stage, product, "primvars:widths", 4).tolist() == pytest.approx([0.02])
        assert read_array(stage, product, "primvars:times", 4) is not None


def test_streamlines_publish_and_replace_selected_fields() -> None:
    with (
        open_scenario("streamlines", stage_name="ovcae.tests.streamlines.fields") as (stage, source),
        ovcae.Session(stage, device="cpu") as session,
    ):
        with source.at_time(0.0) as frame:
            first = session.update(input_ordinal=1, output_ordinal=2, frame_data=frame).wait()
        assert first.operators[0].status == "succeeded", first.diagnostics
        product = first.operators[0].product_paths[0]
        stage.advance_write_floor(2).wait()
        points = read_array(stage, product, "points", 2).reshape(-1, 3)
        colors = read_array(stage, product, "primvars:colors", 2)
        assert colors is not None and len(colors) == len(points)

        write_schemas(
            stage,
            OPERATOR,
            [
                "CaeVizOperatorAPI",
                "CaeVizStreamlinesAPI",
                "CaeVizDatasetSelectionAPI:source",
                "CaeVizDatasetSelectionAPI:seeds",
                "CaeVizFieldSelectionAPI:velocities",
            ],
            3,
        )
        stage.advance_write_floor(3).wait()
        with source.at_time(0.0) as frame:
            second = session.update(input_ordinal=3, output_ordinal=4, frame_data=frame).wait()
        assert second.operators[0].status == "succeeded", second.diagnostics
        stage.advance_write_floor(4).wait()
        assert read_array(stage, product, "primvars:colors", 4) is None
        assert read_array(stage, product, "widths", 4).tolist() == pytest.approx([0.02])


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA device unavailable")
def test_streamlines_cuda_matches_cpu() -> None:
    cpu_points, cpu_counts = _run_once("cpu")
    gpu_points, gpu_counts = _run_once("cuda:0")
    # CPU and CUDA use different element-locator BVHs. At shared cell
    # boundaries they may choose different valid cells, causing adaptive
    # integration to sample the same trajectory at slightly different points.
    # Compare the polyline geometry within its discrete sampling resolution.
    _assert_curves_geometrically_close(
        cpu_points[: int(cpu_counts[0])],
        gpu_points[: int(gpu_counts[0])],
    )
    assert abs(len(gpu_counts) - len(cpu_counts)) / len(cpu_counts) < 0.01
    assert abs(len(gpu_points) - len(cpu_points)) / len(cpu_points) < 0.01


def _assert_curves_geometrically_close(cpu_curve, gpu_curve) -> None:
    np.testing.assert_allclose(gpu_curve[0], cpu_curve[0], rtol=1e-5, atol=1e-6)

    cpu_segments = np.linalg.norm(np.diff(cpu_curve, axis=0), axis=1)
    gpu_segments = np.linalg.norm(np.diff(gpu_curve, axis=0), axis=1)
    cpu_length = float(cpu_segments.sum())
    gpu_length = float(gpu_segments.sum())
    assert cpu_length > 0.0 and gpu_length > 0.0
    assert gpu_length == pytest.approx(cpu_length, rel=0.1)

    sample_count = max(len(cpu_curve), len(gpu_curve), 64)
    cpu_samples = _resample_curve(cpu_curve, cpu_segments, sample_count)
    gpu_samples = _resample_curve(gpu_curve, gpu_segments, sample_count)
    max_deviation = float(np.linalg.norm(gpu_samples - cpu_samples, axis=1).max())
    sampling_resolution = float(max(cpu_segments.max(), gpu_segments.max()))
    assert max_deviation <= sampling_resolution


def _resample_curve(points, segment_lengths, sample_count):
    arclength = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    keep = np.concatenate(([True], np.diff(arclength) > 0.0))
    progress = arclength[keep] / arclength[-1]
    samples = np.linspace(0.0, 1.0, sample_count)
    return np.column_stack([np.interp(samples, progress, points[keep, axis]) for axis in range(points.shape[1])])


def test_streamlines_seed_from_any_point_based_prim() -> None:
    with open_scenario("streamlines_mesh_seeds", stage_name="ovcae.tests.streamlines.mesh-seeds") as (stage, source):
        with ovcae.Session(stage, device="cpu") as session, source.at_time(0.0) as frame:
            report = session.update(input_ordinal=1, output_ordinal=2, frame_data=frame).wait()
        assert report.operators[0].status == "succeeded", report.diagnostics
        stage.advance_write_floor(2).wait()
        points = read_array(stage, OPERATOR, "points", 2).reshape(-1, 3)
        counts = read_array(stage, OPERATOR, "curveVertexCounts", 2)
        assert len(counts) > 0
        starts = points[np.concatenate(([0], np.cumsum(counts[:-1])))]
        expected = SEED_POINTS[[0, 1, 2, 9]] + np.asarray((0.0, 0.001, 0.0), dtype=np.float32)
        distances = np.linalg.norm(starts[:, None, :] - expected[None, :, :], axis=2)
        assert np.all(np.min(distances, axis=1) < 1e-5)


def _assert_curve_starts_are_seeds(points, counts, *, translation) -> None:
    starts = np.concatenate(([0], np.cumsum(counts[:-1])))
    expected = SEED_POINTS + np.asarray(translation, dtype=np.float32)
    distances = np.linalg.norm(points[starts, None, :] - expected[None, :, :], axis=2)
    assert np.all(np.min(distances, axis=1) < 1e-5)


def test_declared_scene_seeds_conservatively_reexecute(monkeypatch):
    """Unknown scene-array evidence must never produce a false whole-operator hit."""
    from ovcae.operators import utils
    from ovcae.operators._builtins.streamlines import Streamlines

    observed = []
    original = Streamlines.execute

    def execute(self, context):
        observed.append(
            (
                context.changes.input_changed("seeds", components={"geometry"}),
                utils.get_input_samples(context, role="seeds").geometry,
            )
        )
        return original(self, context)

    monkeypatch.setattr(Streamlines, "execute", execute)
    with (
        open_scenario("streamlines", stage_name="ovcae.tests.streamlines.declared") as (stage, source),
        ovcae.Session(stage, device="cpu") as session,
    ):
        products = []
        for ordinal in (1, 2):
            with source.at_time(0) as frame:
                report = session.update(input_ordinal=ordinal, output_ordinal=ordinal + 1, frame_data=frame).wait()
            assert report.operators[0].status == "succeeded", report.diagnostics
            stage.advance_write_floor(ordinal + 1).wait()
            products.append(read_array(stage, OPERATOR, "points", ordinal + 1).copy())
        assert observed == [(True, None), (True, None)]
        np.testing.assert_allclose(products[0], products[1])


def test_streamlines_rejects_fields_bound_to_scene_seeds_before_loading(monkeypatch):
    from ovcae.readers._source import UsdFrameProvider

    from ._ovstage_helpers import write_token

    def unexpected_load(*args, **kwargs):
        pytest.fail("invalid input binding loaded scientific data")

    monkeypatch.setattr(UsdFrameProvider, "resolve_dataset", unexpected_load)
    with (
        open_scenario("streamlines", stage_name="ovcae.tests.streamlines.invalid-role") as (stage, source),
        ovcae.Session(stage, device="cpu") as session,
    ):
        write_token(stage, OPERATOR, "cae:viz:field_selection:velocities:input", "seeds", 2)
        stage.advance_write_floor(2).wait()
        with source.at_time(0) as frame:
            report = session.update(input_ordinal=2, output_ordinal=3, frame_data=frame).wait()
        assert report.operators[0].status == "failed"
        assert any(d.code == "INVALID_RECIPE" and "scientific fields" in d.message for d in report.diagnostics)
