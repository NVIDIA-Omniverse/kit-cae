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

import subprocess
import sys
import textwrap
from threading import Barrier
from typing import ClassVar

import numpy as np
import ovstage
import pytest
import warp as wp

import ovcae
from ovcae.operators import utils as op_utils

from ._ovstage_helpers import read_array, write_relationship, write_scalar, write_schemas, write_token

FAILING_RECIPE = "/World/Recipes/FailingFaces"
HEALTHY_RECIPE = "/World/Recipes/HealthyPoints"
POPULATED_RECIPE = "/Recipe"


def test_operator_failure_clears_only_its_product_and_keeps_update_committable() -> None:
    wp.init()
    stage = ovstage.Stage("ovcae.tests.operator-failure-isolation")
    source = ovcae.MemoryDataSource()
    registry = _two_operator_registry()
    try:
        _author_two_product_recipes(stage, width=0.1, ordinal=1)
        stage.advance_write_floor(1).wait()

        with ovcae.Session(stage, device="cpu", operators=registry) as session:
            with source.at_time(0.0) as frame:
                initial = session.update(input_ordinal=1, output_ordinal=2, frame_data=frame).wait()
            assert initial.committable
            assert [result.status for result in initial.operators] == ["succeeded", "succeeded"], initial.diagnostics
            stage.advance_write_floor(2).wait()
            assert read_array(stage, FAILING_RECIPE, "points", 2) is not None
            np.testing.assert_allclose(read_array(stage, HEALTHY_RECIPE, "widths", 2), [0.1])

            write_scalar(stage, FAILING_RECIPE, "cae:viz:faces:externalOnly", False, 3)
            write_scalar(stage, HEALTHY_RECIPE, "cae:viz:points:width", 0.3, 3)
            stage.advance_write_floor(3).wait()
            with source.at_time(1.0) as frame:
                isolated = session.update(input_ordinal=3, output_ordinal=4, frame_data=frame).wait()

        assert isolated.committable
        assert [(result.operator_path, result.status) for result in isolated.operators] == [
            (FAILING_RECIPE, "failed"),
            (HEALTHY_RECIPE, "succeeded"),
        ]
        assert [diagnostic.code for diagnostic in isolated.diagnostics] == ["OPERATOR_EXECUTION_FAILED"]
        stage.advance_write_floor(4).wait()
        assert read_array(stage, FAILING_RECIPE, "points", 4) is None
        np.testing.assert_allclose(read_array(stage, HEALTHY_RECIPE, "widths", 4), [0.3])
    finally:
        source.close()
        stage.destroy()


def test_dataset_resolution_failure_does_not_block_unrelated_product() -> None:
    wp.init()
    stage = ovstage.Stage("ovcae.tests.dataset-failure-isolation")
    source = ovcae.MemoryDataSource()
    registry = ovcae.operators.OperatorRegistry.with_builtins()
    registry.register(_HealthyPoints)
    try:
        write_schemas(
            stage,
            FAILING_RECIPE,
            ["CaeVizOperatorAPI", "CaeVizFacesAPI", "CaeVizDatasetSelectionAPI:source"],
            1,
        )
        write_token(stage, FAILING_RECIPE, "usd-prim-type", "Mesh", 1)
        write_scalar(stage, FAILING_RECIPE, "cae:viz:operator:enabled", True, 1)
        write_relationship(
            stage,
            FAILING_RECIPE,
            "cae:viz:dataset_selection:source:target",
            ["/World/Assets/Missing"],
            1,
        )
        _author_recipe(stage, HEALTHY_RECIPE, "CaeVizPointsAPI", "Points", 1)
        write_scalar(stage, HEALTHY_RECIPE, "cae:viz:points:width", 0.4, 1)
        stage.advance_write_floor(1).wait()

        with ovcae.Session(stage, device="cpu", operators=registry) as session, source.at_time(0.0) as frame:
            report = session.update(input_ordinal=1, output_ordinal=2, frame_data=frame).wait()

        assert report.committable
        assert [(result.operator_path, result.status) for result in report.operators] == [
            (FAILING_RECIPE, "failed"),
            (HEALTHY_RECIPE, "succeeded"),
        ]
        assert [diagnostic.code for diagnostic in report.diagnostics] == ["OPERATOR_EXECUTION_FAILED"]
        assert "was not found" in report.diagnostics[0].message
        stage.advance_write_floor(2).wait()
        assert read_array(stage, FAILING_RECIPE, "points", 2) is None
        np.testing.assert_allclose(read_array(stage, HEALTHY_RECIPE, "widths", 2), [0.4])
    finally:
        source.close()
        stage.destroy()


def test_publication_failure_is_reported_and_does_not_block_unrelated_product(monkeypatch) -> None:
    wp.init()
    stage = ovstage.Stage("ovcae.tests.publication-failure-isolation")
    source = ovcae.MemoryDataSource()
    registry = _two_operator_registry()
    try:
        _author_two_product_recipes(stage, width=0.1, ordinal=1)
        stage.advance_write_floor(1).wait()

        with ovcae.Session(stage, device="cpu", operators=registry) as session:
            with source.at_time(0.0) as frame:
                initial = session.update(input_ordinal=1, output_ordinal=2, frame_data=frame).wait()
            assert initial.committable
            assert [result.status for result in initial.operators] == ["succeeded", "succeeded"], initial.diagnostics
            stage.advance_write_floor(2).wait()

            write_scalar(stage, HEALTHY_RECIPE, "cae:viz:points:width", 0.6, 3)
            stage.advance_write_floor(3).wait()
            real_write_attributes = stage.write_attributes
            fail_next_write = True

            def write_attributes(query, writes, ordinal):
                nonlocal fail_next_write
                if fail_next_write:
                    fail_next_write = False
                    raise RuntimeError("injected OVStage publication failure")
                return real_write_attributes(query, writes, ordinal)

            monkeypatch.setattr(stage, "write_attributes", write_attributes)
            with source.at_time(1.0) as frame:
                report = session.update(input_ordinal=3, output_ordinal=4, frame_data=frame).wait()

        assert report.committable
        assert [(result.operator_path, result.status) for result in report.operators] == [
            (FAILING_RECIPE, "failed"),
            (HEALTHY_RECIPE, "succeeded"),
        ]
        assert [diagnostic.code for diagnostic in report.diagnostics] == ["PRODUCT_PUBLICATION_FAILED"]
        assert report.diagnostics[0].message == "injected OVStage publication failure"
        stage.advance_write_floor(4).wait()
        assert read_array(stage, FAILING_RECIPE, "points", 4) is None
        np.testing.assert_allclose(read_array(stage, HEALTHY_RECIPE, "widths", 4), [0.6])
    finally:
        source.close()
        stage.destroy()


def test_asynchronous_publication_failure_isolated_at_update_wait(monkeypatch) -> None:
    wp.init()
    stage = ovstage.Stage("ovcae.tests.async-publication-failure")
    source = ovcae.MemoryDataSource()
    registry = _two_operator_registry()
    try:
        _author_two_product_recipes(stage, width=0.7, ordinal=1)
        stage.advance_write_floor(1).wait()

        real_write_attributes = stage.write_attributes
        fail_next_write = True

        class FailedOperation:
            def wait(self):
                raise RuntimeError("injected asynchronous OVStage publication failure")

        def write_attributes(query, writes, ordinal):
            nonlocal fail_next_write
            if fail_next_write:
                fail_next_write = False
                return FailedOperation()
            return real_write_attributes(query, writes, ordinal)

        monkeypatch.setattr(stage, "write_attributes", write_attributes)
        with ovcae.Session(stage, device="cpu", operators=registry) as session, source.at_time(0.0) as frame:
            report = session.update(input_ordinal=1, output_ordinal=2, frame_data=frame).wait()

        assert report.committable
        assert [(result.operator_path, result.status) for result in report.operators] == [
            (FAILING_RECIPE, "failed"),
            (HEALTHY_RECIPE, "succeeded"),
        ]
        assert [diagnostic.code for diagnostic in report.diagnostics] == ["PRODUCT_PUBLICATION_FAILED"]
        assert report.diagnostics[0].message == "injected asynchronous OVStage publication failure"
        stage.advance_write_floor(2).wait()
        assert read_array(stage, FAILING_RECIPE, "points", 2) is None
        np.testing.assert_allclose(read_array(stage, HEALTHY_RECIPE, "widths", 2), [0.7])
    finally:
        source.close()
        stage.destroy()


def test_cleanup_failure_makes_the_output_ordinal_uncommittable(monkeypatch) -> None:
    stage = ovstage.Stage("ovcae.tests.cleanup-failure")
    source = ovcae.MemoryDataSource()
    registry = ovcae.operators.OperatorRegistry()
    registry.register(_SometimesFailingFaces)
    try:
        _author_recipe(stage, FAILING_RECIPE, "CaeVizFacesAPI", "Mesh", 1)
        write_scalar(stage, FAILING_RECIPE, "cae:viz:faces:externalOnly", True, 1)
        stage.advance_write_floor(1).wait()

        with ovcae.Session(stage, device="cpu", operators=registry) as session:
            with source.at_time(0.0) as frame:
                initial = session.update(input_ordinal=1, output_ordinal=2, frame_data=frame).wait()
            assert initial.committable
            stage.advance_write_floor(2).wait()

            write_scalar(stage, FAILING_RECIPE, "cae:viz:faces:externalOnly", False, 3)
            stage.advance_write_floor(3).wait()

            def delete_attributes(*_args, **_kwargs):
                raise RuntimeError("injected OVStage cleanup failure")

            monkeypatch.setattr(stage, "delete_attributes", delete_attributes)
            with source.at_time(1.0) as frame:
                report = session.update(input_ordinal=3, output_ordinal=4, frame_data=frame).wait()

        assert not report.committable
        assert [(result.operator_path, result.status) for result in report.operators] == [(FAILING_RECIPE, "failed")]
        assert [diagnostic.code for diagnostic in report.diagnostics] == [
            "PRODUCT_CLEANUP_FAILED",
            "OPERATOR_EXECUTION_FAILED",
        ]
        assert report.diagnostics[0].message == "injected OVStage cleanup failure"
        assert _write_floor(stage) == 3
    finally:
        source.close()
        stage.destroy()


def test_newer_population_ordinal_is_captured_by_the_next_update() -> None:
    wp.init()
    ovcae.register_schema_paths()
    stage = ovstage.Stage("ovcae.tests.population-update")
    source = ovcae.MemoryDataSource()
    registry = ovcae.operators.OperatorRegistry()
    registry.register(_HealthyPoints)
    scene = """#usda 1.0
(
    timeCodesPerSecond = 1
)

def Points "Recipe" (
    prepend apiSchemas = ["CaeVizOperatorAPI", "CaeVizPointsAPI"]
)
{
    bool cae:viz:operator:enabled = true
    float cae:viz:points:width.timeSamples = {
        0: 0.1,
        1: 0.6,
    }
}
"""
    try:
        ovstage.population.open_usd_from_string_with_desc(
            stage,
            scene,
            ordinal=1,
            time_code=0.0,
            descs=ovcae.ovstage_population_desc(domains=ovstage.PopulationDomain.ALL),
        )
        stage.advance_write_floor(1).wait()

        with ovcae.Session(stage, device="cpu", operators=registry) as session:
            with source.at_time(0.0) as frame:
                initial = session.update(input_ordinal=1, output_ordinal=2, frame_data=frame).wait()
            assert initial.committable
            assert [result.status for result in initial.operators] == ["succeeded"], initial.diagnostics
            stage.advance_write_floor(2).wait()
            np.testing.assert_allclose(read_array(stage, POPULATED_RECIPE, "widths", 2), [0.1])

            ovstage.population.update_from_usd_time(stage, ordinal=3, time_code=1.0)
            stage.advance_write_floor(3).wait()
            with source.at_time(1.0) as frame:
                updated = session.update(input_ordinal=3, output_ordinal=4, frame_data=frame).wait()

        assert updated.committable
        stage.advance_write_floor(4).wait()
        np.testing.assert_allclose(read_array(stage, POPULATED_RECIPE, "widths", 4), [0.6])
    finally:
        source.close()
        stage.destroy()


def test_sessions_on_distinct_stages_execute_concurrently() -> None:
    wp.init()
    stages = [ovstage.Stage(f"ovcae.tests.concurrent-stage-{index}") for index in range(2)]
    sources = [ovcae.MemoryDataSource() for _ in stages]
    registry = ovcae.operators.OperatorRegistry()
    registry.register(_ConcurrentPoints)
    _ConcurrentPoints.barrier = Barrier(2)
    try:
        for index, stage in enumerate(stages):
            _author_recipe(stage, HEALTHY_RECIPE, "CaeVizPointsAPI", "Points", 1)
            write_scalar(stage, HEALTHY_RECIPE, "cae:viz:points:width", 0.2 + index, 1)
            stage.advance_write_floor(1).wait()

        with (
            ovcae.Session(stages[0], device="cpu", operators=registry) as first_session,
            ovcae.Session(stages[1], device="cpu", operators=registry) as second_session,
            sources[0].at_time(0.0) as first_frame,
            sources[1].at_time(0.0) as second_frame,
        ):
            first_update = first_session.update(input_ordinal=1, output_ordinal=2, frame_data=first_frame)
            second_update = second_session.update(input_ordinal=1, output_ordinal=2, frame_data=second_frame)
            reports = [first_update.wait(), second_update.wait()]

        assert all(report.committable for report in reports)
        assert all([result.status for result in report.operators] == ["succeeded"] for report in reports)
        for index, stage in enumerate(stages):
            stage.advance_write_floor(2).wait()
            np.testing.assert_allclose(read_array(stage, HEALTHY_RECIPE, "widths", 2), [0.2 + index])
    finally:
        for source in sources:
            source.close()
        for stage in stages:
            stage.destroy()


@pytest.mark.parametrize(
    "script",
    [
        """
        import ovcae
        import cae_openusd_plugins
        cae_openusd_plugins.check_runtime(raise_on_error=True)
        cae_openusd_plugins.register_usd_plugins()
        from pxr import Usd
        usd_stage = Usd.Stage.CreateInMemory()
        import ovstage
        ovcae.register_schema_paths()
        execution_stage = ovstage.Stage("ovcae.tests.usd-first")
        execution_stage.destroy()
        del usd_stage
        """,
        """
        import ovstage
        import ovcae
        ovcae.register_schema_paths()
        execution_stage = ovstage.Stage("ovcae.tests.ovstage-first")
        import cae_openusd_plugins
        cae_openusd_plugins.check_runtime(raise_on_error=True)
        cae_openusd_plugins.register_usd_plugins()
        from pxr import Usd
        usd_stage = Usd.Stage.CreateInMemory()
        del usd_stage
        execution_stage.destroy()
        """,
    ],
    ids=["usd-core-first", "ovstage-first"],
)
def test_usd_core_and_ovstage_runtime_orders_close_in_fresh_process(script: str) -> None:
    subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _author_recipe(stage, path: str, operator_schema: str, prim_type: str, ordinal: int) -> None:
    write_schemas(stage, path, ["CaeVizOperatorAPI", operator_schema], ordinal)
    write_token(stage, path, "usd-prim-type", prim_type, ordinal)
    write_scalar(stage, path, "cae:viz:operator:enabled", True, ordinal)


def _two_operator_registry() -> ovcae.operators.OperatorRegistry:
    registry = ovcae.operators.OperatorRegistry()
    registry.register(_SometimesFailingFaces)
    registry.register(_HealthyPoints)
    return registry


def _author_two_product_recipes(stage, *, width: float, ordinal: int) -> None:
    _author_recipe(stage, FAILING_RECIPE, "CaeVizFacesAPI", "Mesh", ordinal)
    _author_recipe(stage, HEALTHY_RECIPE, "CaeVizPointsAPI", "Points", ordinal)
    write_scalar(stage, FAILING_RECIPE, "cae:viz:faces:externalOnly", True, ordinal)
    write_scalar(stage, HEALTHY_RECIPE, "cae:viz:points:width", width, ordinal)


@ovcae.operators.operator(required_schemas={"CaeVizFacesAPI"}, priority=200)
class _SometimesFailingFaces:
    def execute(self, context):
        if not context.prim.get_attribute("cae:viz:faces:externalOnly").get():
            raise RuntimeError("injected operator failure")
        return op_utils.mesh(
            points=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
            face_vertex_counts=np.asarray([3], dtype=np.int32),
            face_vertex_indices=np.asarray([0, 1, 2], dtype=np.int32),
        )


@ovcae.operators.operator(required_schemas={"CaeVizPointsAPI"}, priority=200)
class _HealthyPoints:
    def execute(self, context):
        width = context.prim.get_attribute("cae:viz:points:width").get()
        return op_utils.points(
            points=np.asarray([[float(width), 0.0, 0.0]], dtype=np.float32),
            widths=op_utils.primvar("widths", np.asarray([width], dtype=np.float32), "constant"),
        )


@ovcae.operators.operator(required_schemas={"CaeVizPointsAPI"}, priority=200)
class _ConcurrentPoints:
    barrier: ClassVar[Barrier] = Barrier(1)

    def execute(self, context):
        type(self).barrier.wait(timeout=5)
        width = context.prim.get_attribute("cae:viz:points:width").get()
        return op_utils.points(
            points=np.asarray([[float(width), 0.0, 0.0]], dtype=np.float32),
            widths=op_utils.primvar("widths", np.asarray([width], dtype=np.float32), "constant"),
        )


def _write_floor(stage) -> int:
    with stage.get_attribute_write_floor() as query:
        query.wait()
        return query.fetch()
