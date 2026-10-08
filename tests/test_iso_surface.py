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
from ._ovstage_helpers import read_array, write_scalar

wp = pytest.importorskip("warp")
pytest.importorskip("warp_simdata")

OPERATOR = "/World/Recipes/IsoSurface"


def _run_once(device: str):
    with open_scenario("iso_surface", stage_name=f"ovcae.tests.iso-surface.{device}") as (stage, source):
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
        counts = read_array(stage, product, "faceVertexCounts", 2)
        indices = read_array(stage, product, "faceVertexIndices", 2)
        colors = read_array(stage, product, "primvars:colors", 2)
        assert len(points) > 0
        assert colors is not None and len(colors) == len(points)
        return points, counts, indices


def test_iso_surface_from_qualified_dataset() -> None:
    _run_once("cpu")


def test_iso_surface_clears_empty_result() -> None:
    with (
        open_scenario("iso_surface", stage_name="ovcae.tests.iso-surface.empty") as (stage, source),
        ovcae.Session(stage, device="cpu") as session,
    ):
        with source.at_time(0.0) as frame:
            first = session.update(input_ordinal=1, output_ordinal=2, frame_data=frame).wait()
        assert first.operators[0].status == "succeeded", first.diagnostics
        product = first.operators[0].product_paths[0]
        stage.advance_write_floor(2).wait()
        assert read_array(stage, product, "points", 2) is not None

        write_scalar(stage, OPERATOR, "cae:viz:isoSurface:isoValue", np.float32(1.0e9), 3)
        stage.advance_write_floor(3).wait()
        with source.at_time(0.0) as frame:
            empty = session.update(input_ordinal=3, output_ordinal=4, frame_data=frame).wait()
        assert empty.operators[0].status == "succeeded", empty.diagnostics
        stage.advance_write_floor(4).wait()
        assert read_array(stage, product, "points", 4) is None


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA device unavailable")
def test_iso_surface_cuda_matches_cpu() -> None:
    cpu = _run_once("cpu")
    gpu = _run_once("cuda:0")
    np.testing.assert_allclose(gpu[0], cpu[0], rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(gpu[1], cpu[1])
    np.testing.assert_array_equal(gpu[2], cpu[2])
