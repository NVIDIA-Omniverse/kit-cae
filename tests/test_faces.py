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

pytest.importorskip("cae_openusd_plugins")
wp = pytest.importorskip("warp")
pytest.importorskip("warp_simdata")

OPERATOR = "/World/Recipes/Faces"


def _run_faces(device: str):
    with open_scenario("faces", stage_name=f"ovcae.tests.faces.{device}") as (stage, source):
        with (
            ovcae.Session(stage, device=device) as session,
            source.at_time(0.0) as frame,
        ):
            report = session.update(input_ordinal=1, output_ordinal=2, frame_data=frame).wait()
        assert report.committable
        assert report.operators[0].status == "succeeded", report.diagnostics
        stage.advance_write_floor(2).wait()
        product = report.operators[0].product_paths[0]
        assert product == OPERATOR
        arrays = {
            "points": read_array(stage, product, "points", 2),
            "counts": read_array(stage, product, "faceVertexCounts", 2),
            "indices": read_array(stage, product, "faceVertexIndices", 2),
            "temperature": read_array(stage, product, "primvars:temperature", 2),
        }
        assert arrays["points"] is not None and len(arrays["points"].reshape(-1, 3)) > 0
        assert arrays["counts"] is not None and len(arrays["counts"]) > 0
        assert arrays["indices"] is not None
        assert int(arrays["counts"].sum()) == len(arrays["indices"])
        assert arrays["temperature"] is not None
        assert len(arrays["temperature"]) == len(arrays["points"].reshape(-1, 3))
        return arrays


def test_faces_from_payload_dataset() -> None:
    _run_faces("cpu")


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA device unavailable")
def test_faces_cuda_matches_cpu() -> None:
    cpu = _run_faces("cpu")
    gpu = _run_faces("cuda:0")
    for name in cpu:
        np.testing.assert_array_equal(gpu[name], cpu[name])


def test_mesh_replacement_removes_stale_primvars() -> None:
    with open_scenario("faces", stage_name="ovcae.tests.faces.replace") as (stage, source):
        with ovcae.Session(stage, device="cpu") as first_session, source.at_time(0.0) as frame:
            first = first_session.update(input_ordinal=1, output_ordinal=2, frame_data=frame).wait()
        product = first.operators[0].product_paths[0]
        stage.advance_write_floor(2).wait()
        assert read_array(stage, product, "primvars:temperature", 2) is not None

        write_schemas(
            stage,
            OPERATOR,
            ["CaeVizOperatorAPI", "CaeVizFacesAPI", "CaeVizDatasetSelectionAPI:source"],
            3,
        )
        stage.advance_write_floor(3).wait()
        # A replacement Session reconstructs exact ownership from the private
        # manifest and removes the predecessor's stale optional primvar.
        with ovcae.Session(stage, device="cpu") as second_session, source.at_time(0.0) as frame:
            second = second_session.update(input_ordinal=3, output_ordinal=4, frame_data=frame).wait()
        assert second.operators[0].status == "succeeded", second.diagnostics
        stage.advance_write_floor(4).wait()
        assert read_array(stage, product, "primvars:temperature", 4) is None


def test_applied_optional_field_selection_with_no_fieldnames_is_ignored() -> None:
    # An applied CaeVizFieldSelectionAPI:opacity with no authored fieldNames is
    # how the authoring layer leaves an optional role unset; it must not fail
    # the recipe the way a required-but-empty selection would.
    with open_scenario("faces_optional_unset", stage_name="ovcae.tests.faces.optional-unset") as (stage, source):
        with (
            ovcae.Session(stage, device="cpu") as session,
            source.at_time(0.0) as frame,
        ):
            report = session.update(input_ordinal=1, output_ordinal=2, frame_data=frame).wait()
        assert report.operators[0].status == "succeeded", report.diagnostics
        product = report.operators[0].product_paths[0]
        stage.advance_write_floor(2).wait()
        assert read_array(stage, product, "primvars:temperature", 2) is not None


def test_vector_field_selection_modes_from_file() -> None:
    with open_scenario("faces_vector", stage_name="ovcae.tests.faces.field-modes") as (stage, source):
        with (
            ovcae.Session(stage, device="cpu") as session,
            source.at_time(0.0) as frame,
        ):
            report = session.update(input_ordinal=1, output_ordinal=2, frame_data=frame).wait()
        assert report.operators[0].status == "succeeded", report.diagnostics
        product = report.operators[0].product_paths[0]
        stage.advance_write_floor(2).wait()
        velocity_x = read_array(stage, product, "primvars:velocity_x", 2)
        speed = read_array(stage, product, "primvars:speed", 2)
        assert velocity_x is not None and speed is not None
        assert len(velocity_x) == len(speed)
        assert np.all(np.isfinite(speed))
        assert np.all(speed + 1e-6 >= np.abs(velocity_x))
