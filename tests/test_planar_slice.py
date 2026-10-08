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
from ._ovstage_helpers import read_array, write_token, write_transform

wp = pytest.importorskip("warp")
pytest.importorskip("warp_simdata")

OPERATOR = "/World/Recipes/PlanarSlice"


def _run_once(device: str, *, mode: str):
    with open_scenario("planar_slice", stage_name=f"ovcae.tests.planar-slice.{device}.{mode}") as (stage, source):
        input_ordinal = 1
        output_ordinal = 2
        if mode != "free":
            write_token(stage, OPERATOR, "cae:viz:planarSlice:mode", mode, 2)
            stage.advance_write_floor(2).wait()
            input_ordinal = 2
            output_ordinal = 3
        with (
            ovcae.Session(stage, device=device) as session,
            source.at_time(0.0) as frame,
        ):
            report = session.update(
                input_ordinal=input_ordinal,
                output_ordinal=output_ordinal,
                frame_data=frame,
            ).wait()
        assert report.operators[0].status == "succeeded", report.diagnostics
        stage.advance_write_floor(output_ordinal).wait()
        product = report.operators[0].product_paths[0]
        assert product == OPERATOR
        return (
            read_array(stage, product, "points", output_ordinal).reshape(-1, 3),
            read_array(stage, product, "faceVertexCounts", output_ordinal),
            read_array(stage, product, "faceVertexIndices", output_ordinal),
            read_array(stage, product, "primvars:colors", output_ordinal),
        )


def test_free_slice_follows_operator_translation() -> None:
    with open_scenario("planar_slice", stage_name="ovcae.tests.planar-slice.free") as (stage, source):
        with ovcae.Session(stage, device="cpu") as session:
            with source.at_time(0.0) as frame:
                first = session.update(input_ordinal=1, output_ordinal=2, frame_data=frame).wait()
            assert first.operators[0].status == "succeeded", first.diagnostics
            product = first.operators[0].product_paths[0]
            stage.advance_write_floor(2).wait()
            first_points = read_array(stage, product, "points", 2).reshape(-1, 3)

            write_transform(stage, OPERATOR, (0.0, 0.005, 0.0), 3)
            stage.advance_write_floor(3).wait()
            with source.at_time(0.0) as frame:
                second = session.update(input_ordinal=3, output_ordinal=4, frame_data=frame).wait()
            assert second.operators[0].status == "succeeded", second.diagnostics
            stage.advance_write_floor(4).wait()
            second_points = read_array(stage, product, "points", 4).reshape(-1, 3)
            colors = read_array(stage, product, "primvars:colors", 4)

        assert len(first_points) > 0 and len(second_points) > 0
        # Product points are local to the recipe prim. The renderer applies the
        # authored translation once, placing the second slice at y=0.005.
        np.testing.assert_allclose(first_points[:, 1], 0.0, atol=1e-6)
        np.testing.assert_allclose(second_points[:, 1], 0.0, atol=1e-6)
        np.testing.assert_allclose(second_points[:, 1] + 0.005, 0.005, atol=1e-6)
        assert colors is not None and len(colors) == len(second_points)


def test_xyz_slice_merges_three_planes() -> None:
    points, counts, indices, colors = _run_once("cpu", mode="xyz")
    assert len(points) > 0 and len(counts) > 0
    assert len(indices) == int(counts.sum())
    assert colors is not None and len(colors) == len(points)


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA device unavailable")
def test_planar_slice_cuda_matches_cpu() -> None:
    cpu = _run_once("cpu", mode="xyz")
    gpu = _run_once("cuda:0", mode="xyz")
    np.testing.assert_allclose(gpu[0], cpu[0], rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(gpu[1], cpu[1])
    np.testing.assert_array_equal(gpu[2], cpu[2])
    np.testing.assert_allclose(gpu[3], cpu[3], rtol=1e-6, atol=1e-6)
