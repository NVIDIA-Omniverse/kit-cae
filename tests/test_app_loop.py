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

import time

import numpy as np
import ovstage
import pytest

import ovcae
from ovcae.app import UpdateLoop, element_layout, write_attribute

from ._ovstage_helpers import read_scalar, write_scalar, write_schemas, write_token

# usd-core answers schema questions from a registry it builds on the first
# schema read, so the bundle registers before any loop in this module asks.
ovcae.register_schema_paths()

RECIPE_PATH = "/World/Recipes/Test"
CAMERA = "/World/Camera"
VALUE = "ovcae:test:value"
PRODUCT_VALUE = "ovcae:test:productValue"


class RecordingSource(ovcae.MemoryDataSource):
    """A Memory Data Source that remembers every time it was asked for."""

    def __init__(self):
        super().__init__()
        self.times = []

    def at_time(self, seconds):
        self.times.append(seconds)
        return super().at_time(seconds)


@pytest.fixture
def scene():
    stage = ovstage.Stage("ovcae.tests.app-loop")
    try:
        write_schemas(stage, RECIPE_PATH, ["CaeVizOperatorAPI", "OvcaeTestOperatorAPI", "MaterialBindingAPI"], 1)
        write_scalar(stage, RECIPE_PATH, "cae:viz:operator:enabled", True, 1)
        write_scalar(stage, RECIPE_PATH, VALUE, 7.0, 1)
        write_token(stage, CAMERA, "usd-prim-type", "Camera", 1)
        write_scalar(stage, CAMERA, "focalLength", np.float32(50.0), 1)
        stage.advance_write_floor(1).wait()
        with RecordingSource() as source:
            yield stage, source
    finally:
        stage.destroy()


def _wait_for_commit(loop, timeout=30.0):
    started = time.perf_counter()
    while True:
        report = loop.pump()
        if report is not None:
            return report
        assert time.perf_counter() - started < timeout, "the update never committed"
        loop.await_evaluation(0.05)


def test_loop_owns_a_session_and_commits_the_first_update(scene):
    stage, source = scene
    with UpdateLoop(stage, source, ordinal=1, device="cpu") as loop:
        report = loop.update()
        assert report is not None and report.committable
        assert loop.ordinal == 2 and loop.last_report is report
        assert read_scalar(stage, RECIPE_PATH, PRODUCT_VALUE, 2) == 7.0
    with pytest.raises(ovcae.SessionClosedError):
        loop.session.update(input_ordinal=2, output_ordinal=3, frame_data=None)


def test_property_types_come_from_the_prim_schema_or_the_value(scene):
    stage, source = scene
    with UpdateLoop(stage, source, ordinal=1, device="cpu") as loop:
        assert loop.property_type(CAMERA, "focalLength") == "float"
        assert loop.property_type(CAMERA, "clippingRange") == "float2"
        assert loop.property_type(CAMERA, "omni:xform") == "matrix4d"
        assert loop.property_type(RECIPE_PATH, "visibility") == "token"
        assert loop.property_type(RECIPE_PATH, VALUE) == "double"  # the committed value's type
        assert loop.property_type(RECIPE_PATH, "nothing:here") is None
        with pytest.raises(TypeError, match="pass types="):
            loop.request({(RECIPE_PATH, "nothing:here"): object()})


def test_relationship_and_token_array_edits_take_their_committed_types(scene):
    stage, source = scene
    with UpdateLoop(stage, source, ordinal=1, device="cpu") as loop:
        loop.update()
        write_attribute(stage, CAMERA, "target", ["/World"], 3, type_name="rel")
        write_attribute(stage, CAMERA, "names", ["a", "b"], 3, type_name="token[]")
        stage.advance_write_floor(3).wait()
        loop.ordinal = 3
        assert loop.property_type(CAMERA, "target") == "rel"
        assert loop.property_type(CAMERA, "names") == "token[]"
        loop.write({(CAMERA, "target"): ["/World/Recipes"], (CAMERA, "names"): ("c",)})
        assert loop.value(CAMERA, "target") == ("/World/Recipes",)
        assert loop.value(CAMERA, "names", type_name="token[]") == ("c",)


def test_presentation_writes_advance_the_floor_without_an_update(scene):
    stage, source = scene
    with UpdateLoop(stage, source, ordinal=1, device="cpu") as loop:
        loop.update()
        asked = len(source.times)
        matrix = np.eye(4)
        matrix[3, :3] = (1.0, 2.0, 3.0)
        loop.write({(CAMERA, "omni:xform"): matrix, (CAMERA, "focalLength"): 35.0})
        assert loop.ordinal == 3 and not loop.evaluating and not loop.pending
        assert len(source.times) == asked
        assert np.array_equal(np.asarray(loop.value(CAMERA, "omni:xform")).reshape(4, 4), matrix)
        assert loop.value(CAMERA, "focalLength") == pytest.approx(35.0)
        loop.write({})
        assert loop.pump() is None and loop.ordinal == 3


def test_submit_refuses_while_an_update_is_in_flight_and_keeps_the_batch(scene):
    stage, source = scene
    with UpdateLoop(stage, source, ordinal=1, device="cpu") as loop:
        loop.update()
        loop.request({(RECIPE_PATH, VALUE): 2.0})
        loop.pump()
        assert loop.evaluating and loop.ordinal == 3
        loop.request({(RECIPE_PATH, VALUE): 5.0})
        with pytest.raises(ovcae.UpdateInFlightError, match="flush or pump"):
            loop.submit()
        # Nothing was written or dropped: the batch waits for the next pump after the commit.
        assert loop.pending == {(RECIPE_PATH, VALUE): 5.0} and loop.ordinal == 3
        loop.flush()
        assert loop.update() is not None
        assert read_scalar(stage, RECIPE_PATH, PRODUCT_VALUE, loop.ordinal) == 5.0


def test_write_refuses_while_an_update_is_in_flight(scene):
    stage, source = scene
    with UpdateLoop(stage, source, ordinal=1, device="cpu") as loop:
        loop.update()
        loop.request({(RECIPE_PATH, VALUE): 2.0})
        loop.pump()
        assert loop.evaluating
        with pytest.raises(ovcae.UpdateInFlightError, match="flush or pump"):
            loop.write({(CAMERA, "focalLength"): 24.0})
        loop.flush()
        loop.write({(CAMERA, "focalLength"): 24.0})
        assert loop.value(CAMERA, "focalLength") == pytest.approx(24.0)


def test_recipe_edits_run_an_update_asynchronously_through_pump(scene):
    stage, source = scene
    with UpdateLoop(stage, source, ordinal=1, device="cpu") as loop:
        loop.update()
        loop.request({(RECIPE_PATH, VALUE): 8.0})
        loop.request({(RECIPE_PATH, VALUE): 9.0})  # latest value per key wins
        assert loop.pending == {(RECIPE_PATH, VALUE): 9.0}
        assert loop.pump() is None  # submitted: written at 3, update 3 -> 4 in flight
        assert loop.evaluating and loop.ordinal == 3 and not loop.pending
        report = _wait_for_commit(loop)
        assert (report.input_ordinal, report.output_ordinal) == (3, 4)
        assert loop.ordinal == 4 and not loop.evaluating
        assert read_scalar(stage, RECIPE_PATH, PRODUCT_VALUE, 4) == 9.0
        # Edits requested while an update runs wait for the next pump after its commit.
        loop.request({(RECIPE_PATH, VALUE): 11.0})
        assert loop.pump() is None and loop.evaluating
        loop.request({(RECIPE_PATH, VALUE): 12.0})
        assert loop.pending
        report = _wait_for_commit(loop)
        assert read_scalar(stage, RECIPE_PATH, PRODUCT_VALUE, report.output_ordinal) == 11.0
        assert loop.pending == {(RECIPE_PATH, VALUE): 12.0}
        assert loop.update() is not None
        assert read_scalar(stage, RECIPE_PATH, PRODUCT_VALUE, loop.ordinal) == 12.0


def test_update_with_edits_writes_then_evaluates(scene):
    stage, source = scene
    with UpdateLoop(stage, source, ordinal=1, device="cpu") as loop:
        loop.update()
        report = loop.update({(RECIPE_PATH, VALUE): 3.0})
        assert (report.input_ordinal, report.output_ordinal) == (3, 4)
        assert read_scalar(stage, RECIPE_PATH, PRODUCT_VALUE, 4) == 3.0


def test_invalidate_and_time_code_drive_updates_without_edits(scene):
    stage, source = scene
    clock = {"now": 0.5}
    with UpdateLoop(stage, source, ordinal=1, device="cpu", time_code=lambda: clock["now"]) as loop:
        loop.update()
        assert source.times == [0.5]
        assert loop.submit() is None  # nothing pending, nothing dirty
        clock["now"] = 1.5
        loop.invalidate()
        report = loop.submit(wait=True)
        assert report is not None and source.times == [0.5, 1.5]
        assert (report.input_ordinal, report.output_ordinal) == (2, 3) and loop.ordinal == 3
        assert not loop.evaluating


def test_flush_commits_the_in_flight_update(scene):
    stage, source = scene
    with UpdateLoop(stage, source, ordinal=1, device="cpu") as loop:
        loop.update()
        loop.request({(RECIPE_PATH, VALUE): 1.0})
        loop.pump()
        assert loop.evaluating
        report = loop.flush()
        assert report is not None and not loop.evaluating and loop.flush() is None
        assert loop.ordinal == report.output_ordinal == 4


def test_uncommittable_update_raises_and_keeps_the_floor(scene, monkeypatch):
    stage, source = scene
    with UpdateLoop(stage, source, ordinal=1, device="cpu") as loop:
        loop.update()
        real_update = loop.session.update

        def failing_update(**kwargs):
            update = real_update(**kwargs)
            report = update.wait()
            broken = ovcae.UpdateReport(
                report.input_ordinal,
                report.output_ordinal,
                report.operators,
                (ovcae.Diagnostic("error", "TEST_FAILURE", "made uncommittable by the test"),),
                False,
            )
            monkeypatch.setattr(update, "wait", lambda: broken)
            return update

        monkeypatch.setattr(loop.session, "update", failing_update)
        loop.invalidate()
        with pytest.raises(ovcae.UpdateError) as excinfo:
            loop.submit(wait=True)
        assert excinfo.value.report is not None and not excinfo.value.report.committable
        assert loop.ordinal == 2 and not loop.evaluating


def test_write_attribute_covers_the_usd_value_types(scene):
    stage, _ = scene
    matrix = np.arange(16, dtype=np.float64).reshape(4, 4)
    write_attribute(stage, CAMERA, "omni:xform", matrix, 2, type_name="matrix4d")
    write_attribute(stage, "/World/Render", "resolution", (640, 480), 2, type_name="int2")
    write_attribute(stage, CAMERA, "clippingRange", (0.1, 100.0), 2, type_name="float2")
    write_attribute(stage, CAMERA, "visibility", "invisible", 2, type_name="token")
    write_attribute(stage, CAMERA, "count", 3, 2, type_name="int")
    write_attribute(stage, CAMERA, "flag", True, 2, type_name="bool")
    write_attribute(stage, CAMERA, "names", ["a", "b"], 2, type_name="token[]")
    write_attribute(stage, CAMERA, "target", ["/World/Recipes/Test"], 2, type_name="rel")
    write_attribute(stage, "/World/Seeds", "points", np.zeros((5, 3)), 2, type_name="point3f[]")
    stage.advance_write_floor(2).wait()
    from ovcae._stage_io import read_value

    assert np.array_equal(np.asarray(read_value(stage, CAMERA, "omni:xform", 2)).reshape(4, 4), matrix)
    assert tuple(read_value(stage, "/World/Render", "resolution", 2)) == (640, 480)
    assert read_value(stage, CAMERA, "clippingRange", 2) == pytest.approx((0.1, 100.0))
    assert read_value(stage, CAMERA, "visibility", 2) == "invisible"
    assert read_value(stage, CAMERA, "count", 2) == 3
    assert read_value(stage, CAMERA, "flag", 2) is np.True_ or read_value(stage, CAMERA, "flag", 2) is True
    assert read_value(stage, CAMERA, "names", 2, type_name="token[]") == ("a", "b")
    assert read_value(stage, CAMERA, "target", 2) == ("/World/Recipes/Test",)
    assert np.asarray(read_value(stage, "/World/Seeds", "points", 2, type_name="point3f[]")).shape == (5, 3)
    with pytest.raises(ValueError, match="one value"):
        write_attribute(stage, CAMERA, "clippingRange", (0.1, 100.0, 5.0), 3, type_name="float2")


@pytest.mark.parametrize(
    ("type_name", "layout"),
    [
        ("float", ("float32", 1, None)),
        ("int2", ("int32", 2, None)),
        ("point3f[]", ("float32", 3, "POINT")),
        ("matrix4d", ("float64", 16, "MATRIX")),
        ("color3f", ("float32", 3, "COLOR")),
        ("double", ("float64", 1, None)),
    ],
)
def test_element_layout(type_name, layout):
    assert element_layout(type_name) == layout


def test_committed_type_recovers_the_written_type(scene):
    stage, source = scene
    from ovcae.app import committed_type

    written = [
        ("omni:xform", "matrix4d", np.eye(4)),
        ("resolution", "int2", (640, 480)),
        ("focal", "float", 3.0),
        ("v", "double", 3.0),
        ("vis", "token", "x"),
        ("rel", "rel", ["/World"]),
        ("pts", "point3f[]", np.zeros((5, 3))),
        ("names", "token[]", ["a", "b"]),
        ("flag", "bool", True),
        ("cols", "color3f[]", np.zeros((5, 3))),
        ("cols4", "color4f[]", np.zeros((5, 4))),
        ("uv", "float2[]", np.zeros((5, 2))),
        ("ints", "int[]", [1, 2, 3]),
        ("d3", "double3", (1.0, 2.0, 3.0)),
    ]
    for name, type_name, value in written:
        write_attribute(stage, "/T", name, value, 2, type_name=type_name)
    stage.advance_write_floor(2).wait()
    for name, type_name, _ in written:
        assert committed_type(stage, "/T", name, 2) == type_name, name
    assert committed_type(stage, "/T", "absent", 2) is None
    # A committed vector array is rewritten with its own lane count; OVStage refuses a type change.
    with UpdateLoop(stage, source, ordinal=2, device="cpu") as loop:
        loop.write({("/T", "cols"): np.ones((5, 3)), ("/T", "uv"): np.ones((2, 2))})
        assert np.asarray(loop.value("/T", "cols")).size == 15 and committed_type(stage, "/T", "cols", 3) == "color3f[]"


def test_element_layout_rejects_tokens_and_unknown_types():
    for bad in ("token", "string", "asset", "frobnicate3"):
        with pytest.raises(ValueError):
            element_layout(bad)


@pytest.mark.usefixtures("_plugins")
def test_recipe_settings_and_seed_transforms_update_a_real_scene():
    from ._file_scenarios import open_scenario

    with (
        open_scenario("streamlines", stage_name="ovcae.tests.app-loop.streamlines") as (stage, source),
        UpdateLoop(stage, source, ordinal=1, device="cpu") as loop,
    ):
        recipe = "/World/Recipes/Streamlines"
        seeds = "/World/Controls/Seeds"
        assert loop.property_type(recipe, "cae:viz:streamlines:maxSteps") == "int"
        assert loop.property_type(recipe, "cae:viz:streamlines:direction") == "token"
        assert loop.property_type(recipe, "cae:viz:dataset_selection:seeds:target") == "rel"
        first = loop.update()
        assert first is not None and first.operators[0].status == "succeeded", first.diagnostics
        report = loop.update({(recipe, "cae:viz:streamlines:maxSteps"): 5})
        assert report is not None and report.operators[0].status == "succeeded", report.diagnostics
        assert loop.value(recipe, "cae:viz:streamlines:maxSteps") == 5
        moved = np.eye(4)
        moved[3, 2] = 0.25
        report = loop.update({(seeds, "omni:xform"): moved})  # the seed transform is a recipe input
        assert report is not None and report.operators[0].status == "succeeded", report.diagnostics
        assert loop.property_type(seeds, "omni:xform") == "matrix4d"


@pytest.fixture
def _plugins():
    pytest.importorskip("cae_openusd_plugins")
    pytest.importorskip("warp_simdata")


def test_property_types_are_read_once_per_key(scene, monkeypatch):
    from ovcae.app import _loop

    stage, source = scene
    calls = []
    real = _loop.committed_type

    def counting(*args, **kwargs):
        calls.append(args[1:3])
        return real(*args, **kwargs)

    monkeypatch.setattr(_loop, "committed_type", counting)
    with UpdateLoop(stage, source, ordinal=1, device="cpu") as loop:
        loop.update({(RECIPE_PATH, VALUE): 8.0})
        loop.update({(RECIPE_PATH, VALUE): 9.0})
        loop.write({(CAMERA, "focalLength"): 35.0})
        loop.write({(CAMERA, "focalLength"): 36.0})
    assert calls == [(RECIPE_PATH, VALUE), (CAMERA, "focalLength")]


def test_request_during_an_update_does_not_read_the_stage(scene, monkeypatch):
    """A read on a prim the update is publishing to fails that publication; unknown types wait for the submit."""
    stage, source = scene
    with UpdateLoop(stage, source, ordinal=1, device="cpu") as loop:
        loop.invalidate()
        loop.submit()
        assert loop.evaluating

        def no_reads(path, name):
            raise AssertionError(f"read {name} on {path} while an update was in flight")

        monkeypatch.setattr(loop, "property_type", no_reads)
        loop.request({(RECIPE_PATH, VALUE): 11.0})  # never seen: resolved by the submit that writes it
        loop.request({(RECIPE_PATH, "nothing:here"): object()})  # its TypeError moves to that submit too
        assert loop.pending == {
            (RECIPE_PATH, VALUE): 11.0,
            (RECIPE_PATH, "nothing:here"): loop.pending[RECIPE_PATH, "nothing:here"],
        }
        _wait_for_commit(loop)
        monkeypatch.undo()
        with pytest.raises(TypeError, match="pass types="):
            loop.submit()
        assert not loop.pending, "the batch that could not be typed is dropped whole"
        loop.update({(RECIPE_PATH, VALUE): 12.0})
        assert read_scalar(stage, RECIPE_PATH, PRODUCT_VALUE, loop.ordinal) == 12.0
