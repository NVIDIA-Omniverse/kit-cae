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
from ._ovstage_helpers import read_array, write_schemas

wp = pytest.importorskip("warp")
pytest.importorskip("warp_simdata")

OPERATOR = "/World/Recipes/Points"


def _run_once(device: str):
    with (
        open_scenario("points", stage_name=f"ovcae.tests.points.{device}") as (stage, source),
        ovcae.Session(stage, device=device) as session,
    ):
        with source.at_time(0.0) as frame:
            report = session.update(input_ordinal=1, output_ordinal=2, frame_data=frame).wait()
        assert report.committable
        assert report.operators[0].status == "succeeded"
        product = report.operators[0].product_paths[0]
        assert product == OPERATOR
        stage.advance_write_floor(2).wait()
        return {
            "points": read_array(stage, product, "points", 2).reshape(-1, 3),
            "widths": read_array(stage, product, "widths", 2),
            "primvar_widths": read_array(stage, product, "primvars:widths", 2),
            "colors": read_array(stage, product, "primvars:colors", 2),
        }


def test_points_publish_file_backed_fields_and_max_count() -> None:
    values = _run_once("cpu")
    assert 0 < len(values["points"]) <= 8
    assert len(values["widths"]) == len(values["points"])
    np.testing.assert_array_equal(values["primvar_widths"], values["widths"])
    assert len(values["colors"]) == len(values["points"])


def test_points_replace_removed_optional_primvars() -> None:
    with (
        open_scenario("points", stage_name="ovcae.tests.points.replace") as (stage, source),
        ovcae.Session(stage, device="cpu") as session,
    ):
        with source.at_time(0.0) as frame:
            first = session.update(input_ordinal=1, output_ordinal=2, frame_data=frame).wait()
        product = first.operators[0].product_paths[0]
        stage.advance_write_floor(2).wait()
        assert read_array(stage, product, "primvars:colors", 2) is not None

        write_schemas(
            stage,
            OPERATOR,
            [
                "CaeVizOperatorAPI",
                "CaeVizPointsAPI",
                "CaeVizDatasetSelectionAPI:source",
            ],
            3,
        )
        stage.advance_write_floor(3).wait()
        with source.at_time(0.0) as frame:
            second = session.update(input_ordinal=3, output_ordinal=4, frame_data=frame).wait()
        assert second.committable
        stage.advance_write_floor(4).wait()
        assert read_array(stage, product, "primvars:colors", 4) is None
        assert read_array(stage, product, "widths", 4).tolist() == pytest.approx([0.02])


def test_points_cuda_matches_cpu() -> None:
    if not wp.is_cuda_available():
        pytest.skip("CUDA is unavailable")
    cpu = _run_once("cpu")
    gpu = _run_once("cuda:0")
    for name in cpu:
        np.testing.assert_allclose(gpu[name], cpu[name], rtol=1e-5, atol=1e-6)
