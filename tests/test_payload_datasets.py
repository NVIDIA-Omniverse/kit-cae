# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Datasets composed from payloads resolve by prim path through the shared layer stack."""

from __future__ import annotations

import numpy as np
import pytest

import ovcae

from ._file_scenarios import SCENES, open_scenario
from ._ovstage_helpers import read_array

readers = pytest.importorskip("ovcae.readers")
pytest.importorskip("warp")
pytest.importorskip("warp_simdata")

HOLDER = "/World/Assets/StaticMixer"
RELATIVE = "Base/StaticMixer/StaticMixer_Default"
DATASET = f"{HOLDER}/{RELATIVE}"
DIRECT = "/World/Recipes/Faces"
QUALIFIED = "/World/Recipes/FacesByIdentifier"


def test_payload_datasets_resolve_by_path_and_relative_identifier() -> None:
    with readers.UsdDataSource.open(SCENES / "payload_faces.usda") as source:
        assert source.resolves(DATASET)
        assert source.resolves(HOLDER)
        with source.at_time(0.0) as frame:
            direct = frame._resolve_dataset(DATASET, identifier="", device="cpu", fields={})
            qualified = frame._resolve_dataset(HOLDER, identifier=RELATIVE, device="cpu", fields={})
            assert direct.get_num_elems() == qualified.get_num_elems() > 0
            describe = frame._provider.describe_component
            assert describe(DATASET, "", "geometry") == describe(HOLDER, RELATIVE, "geometry")
            with pytest.raises(ovcae.DatasetResolutionError, match="identifier is required"):
                frame._resolve_dataset(HOLDER, identifier="", device="cpu", fields={})
            with pytest.raises(ovcae.DatasetResolutionError, match="did not resolve"):
                frame._resolve_dataset(HOLDER, identifier="Base/Missing", device="cpu", fields={})


def test_faces_from_payload_datasets_through_ovstage() -> None:
    with open_scenario("payload_faces", stage_name="ovcae.tests.payload-faces") as (stage, source):
        with ovcae.Session(stage, device="cpu") as session, source.at_time(0.0) as frame:
            report = session.update(input_ordinal=1, output_ordinal=2, frame_data=frame).wait()
        assert report.committable, report.diagnostics
        statuses = {operator.operator_path: operator.status for operator in report.operators}
        assert statuses == {DIRECT: "succeeded", QUALIFIED: "succeeded"}, report.diagnostics
        stage.advance_write_floor(2).wait()
        direct = read_array(stage, DIRECT, "points", 2)
        qualified = read_array(stage, QUALIFIED, "points", 2)
        assert direct is not None and len(direct) > 0
        np.testing.assert_array_equal(direct, qualified)
        assert read_array(stage, DIRECT, "primvars:temperature", 2) is not None
