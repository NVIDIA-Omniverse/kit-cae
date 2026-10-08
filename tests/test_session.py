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

import ovstage
import pytest

import ovcae

from ._ovstage_helpers import delete_prim, read_scalar, write_scalar, write_schemas

OPERATOR = "/World/Recipes/Test"
PRODUCT_VALUE = "ovcae:test:productValue"


def test_stage_recipe_values_and_tombstone() -> None:
    stage = ovstage.Stage("ovcae.tests.session")
    try:
        write_schemas(stage, OPERATOR, ["CaeVizOperatorAPI", "OvcaeTestOperatorAPI"], 1)
        write_scalar(stage, OPERATOR, "cae:viz:operator:enabled", True, 1)
        write_scalar(stage, OPERATOR, "ovcae:test:value", 7.0, 1)
        stage.advance_write_floor(1).wait()

        with ovcae.MemoryDataSource() as source, ovcae.Session(stage) as session:
            with source.at_time(0.0) as frame:
                update = session.update(input_ordinal=1, output_ordinal=2, frame_data=frame)
            report = update.wait()
            assert report.committable
            assert report.operators[0].status == "succeeded"
            assert not report.diagnostics
            product = report.operators[0].product_paths[0]
            assert _write_floor(stage) == 1
            stage.advance_write_floor(2).wait()
            assert product == OPERATOR
            assert read_scalar(stage, product, PRODUCT_VALUE, 2) == 7.0

            write_scalar(stage, OPERATOR, "cae:viz:operator:enabled", True, 3)
            write_scalar(stage, OPERATOR, "ovcae:test:value", 9.0, 3)
            stage.advance_write_floor(3).wait()
            with source.at_time(1.0) as frame:
                report = session.update(input_ordinal=3, output_ordinal=4, frame_data=frame).wait()
            assert not report.diagnostics
            stage.advance_write_floor(4).wait()
            assert read_scalar(stage, product, PRODUCT_VALUE, 4) == 9.0

            write_scalar(stage, OPERATOR, "cae:viz:operator:enabled", False, 5)
            stage.advance_write_floor(5).wait()
            with source.at_time(2.0) as frame:
                report = session.update(input_ordinal=5, output_ordinal=6, frame_data=frame).wait()
            assert report.operators[0].status == "skipped"
            stage.advance_write_floor(6).wait()
            assert read_scalar(stage, product, PRODUCT_VALUE, 6) is None
            assert read_scalar(stage, OPERATOR, "ovcae:test:value", 6) == 9.0
            assert read_scalar(stage, OPERATOR, "cae:viz:operator:enabled", 6) is False

            delete_prim(stage, OPERATOR, 7)
            stage.advance_write_floor(7).wait()
            with source.at_time(3.0) as frame:
                report = session.update(input_ordinal=7, output_ordinal=8, frame_data=frame).wait()
            assert report.operators[0].status == "removed"
    finally:
        stage.destroy()


def test_stage_reservation_and_ordinal_validation() -> None:
    stage = ovstage.Stage("ovcae.tests.contract")
    try:
        stage.advance_write_floor(2).wait()
        source = ovcae.MemoryDataSource()
        session = ovcae.Session(stage)
        try:
            with pytest.raises(ovcae.StageSessionConflictError):
                ovcae.Session(stage)
            with source.at_time(0.0) as frame:
                with pytest.raises(ovcae.OrdinalError, match="not committed"):
                    session.update(input_ordinal=3, output_ordinal=4, frame_data=frame)
                with pytest.raises(ovcae.OrdinalError, match="above write floor"):
                    session.update(input_ordinal=2, output_ordinal=2, frame_data=frame)
        finally:
            session.close()
            source.close()

        replacement = ovcae.Session(stage)
        replacement.close()
    finally:
        stage.destroy()


def test_frame_must_be_active_at_submission() -> None:
    stage = ovstage.Stage("ovcae.tests.frame-lifetime")
    source = ovcae.MemoryDataSource()
    session = ovcae.Session(stage)
    try:
        with source.at_time(0.0) as frame:
            pass
        with pytest.raises(ovcae.FrameDataError, match="outside"):
            session.update(input_ordinal=0, output_ordinal=1, frame_data=frame)
    finally:
        session.close()
        source.close()
        stage.destroy()


def test_one_update_in_flight_and_close_waits() -> None:
    stage = ovstage.Stage("ovcae.tests.close")
    source = ovcae.MemoryDataSource()
    session = ovcae.Session(stage)
    try:
        with source.at_time(0.0) as frame:
            update = session.update(input_ordinal=0, output_ordinal=1, frame_data=frame)
            with pytest.raises(ovcae.UpdateInFlightError):
                session.update(input_ordinal=0, output_ordinal=2, frame_data=frame)
        assert update.done(timeout=30.0)
        assert update.done()
        session.close()
        assert update.wait().committable
        assert update.done()
        with pytest.raises(ovcae.SessionClosedError), source.at_time(0.0) as frame:
            session.update(input_ordinal=0, output_ordinal=1, frame_data=frame)
    finally:
        session.close()
        source.close()
        stage.destroy()


def _write_floor(stage) -> int:
    with stage.get_attribute_write_floor() as query:
        query.wait()
        return query.fetch()


def test_input_ordinal_is_a_minimum_publication() -> None:
    stage = ovstage.Stage("ovcae.tests.minimum-publication")
    try:
        write_schemas(stage, OPERATOR, ["CaeVizOperatorAPI", "OvcaeTestOperatorAPI"], 1)
        write_scalar(stage, OPERATOR, "cae:viz:operator:enabled", True, 1)
        write_scalar(stage, OPERATOR, "ovcae:test:value", 7.0, 1)
        stage.advance_write_floor(1).wait()
        write_scalar(stage, OPERATOR, "ovcae:test:value", 9.0, 2)
        stage.advance_write_floor(2).wait()

        with ovcae.MemoryDataSource() as source, ovcae.Session(stage) as session, source.at_time(0.0) as frame:
            report = session.update(input_ordinal=1, output_ordinal=3, frame_data=frame).wait()
            assert report.committable
            assert report.operators[0].status == "succeeded"
            assert _write_floor(stage) == 2
            stage.advance_write_floor(3).wait()
            assert read_scalar(stage, OPERATOR, PRODUCT_VALUE, 3) == 9.0
    finally:
        stage.destroy()
