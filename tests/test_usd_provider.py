# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""USD description/loading coherence and committed whole-operator skipping."""

import numpy as np
import pytest

import ovcae

readers = pytest.importorskip("ovcae.readers")
from pxr import Sdf, Usd  # noqa: E402

from ovcae.readers._source import UsdFrameProvider  # noqa: E402


def make_source(path, *, scale=1, rate=1):
    """Author unequal schedules so accidental interpolation changes observable values."""
    stage = Usd.Stage.CreateInMemory()
    stage.SetTimeCodesPerSecond(rate)
    prim = stage.DefinePrim("/data", "OmniSciDataset")
    stage.SetDefaultPrim(prim)
    names = ["points", "cellTypes", "connectivityOffsets", "connectivityArray", "T", "unused"]
    prim.SetMetadata(
        "apiSchemas",
        Sdf.TokenListOp.CreateExplicit(
            [
                "OmniSciVtkUnstructuredGridAPI",
                "OmniSciFieldAPI:T",
                *[f"OmniSciArrayAPI:{name}" for name in names],
            ]
        ),
    )
    prim.CreateAttribute("omni:sci:field:T:name", Sdf.ValueTypeNames.String).Set("T")
    prim.CreateAttribute("omni:sci:field:T:association", Sdf.ValueTypeNames.Token).Set("node")
    points = prim.CreateAttribute("omni:sci:array:points:value", Sdf.ValueTypeNames.Float3Array)
    for time, factor in [(0, scale), (10, scale * 2)]:
        points.Set([(0, 0, 0), (factor, 0, 0), (0, factor, 0), (0, 0, factor)], time)
    for name, values in [("cellTypes", [10]), ("connectivityOffsets", [0, 4]), ("connectivityArray", [0, 1, 2, 3])]:
        prim.CreateAttribute(f"omni:sci:array:{name}:value", Sdf.ValueTypeNames.IntArray).Set(values, 0)
    temperature = prim.CreateAttribute("omni:sci:array:T:value", Sdf.ValueTypeNames.FloatArray)
    for time, value in [(0, 1), (5, 2), (10, 3)]:
        temperature.Set([value] * 4, time)
    unused = prim.CreateAttribute("omni:sci:array:unused:value", Sdf.ValueTypeNames.FloatArray)
    unused.Set([0], 0)
    unused.Set([7], 7)
    other = stage.DefinePrim("/other")
    other.CreateAttribute("unrelated", Sdf.ValueTypeNames.Float).Set(1, 1)
    stage.GetRootLayer().Export(str(path))
    return path


def test_shared_identity_held_loading_and_cached_discovery(tmp_path, monkeypatch):
    from ovcae.readers import _source

    path = make_source(tmp_path / "data.usda", rate=2)
    original = _source.dataset_time_samples
    discovered = []

    def discover(prim):
        discovered.append(prim.GetPath())
        return original(prim)

    monkeypatch.setattr(_source, "dataset_time_samples", discover)
    with readers.UsdDataSource.open(path) as source:
        identities = []
        for time, selected, scale, value in [(-1, 0, 1, 1), (2.5, 2.5, 1, 2), (3, 2.5, 1, 2), (20, 5, 2, 3)]:
            with source.at_time(time) as frame:
                provider = frame._provider
                descriptions = [
                    provider.describe_component("/data", "", c) for c in ("topology", "geometry", "field:T")
                ]
                assert descriptions[0] == descriptions[1] == descriptions[2]
                identity, sample = descriptions[0]
                assert sample.time == selected and sample.granularity == "dataset"
                identities.append(identity)
                dataset = frame._resolve_dataset("/data", identifier="", device="cpu", fields={"temperature": ["T"]})
                np.testing.assert_array_equal(dataset.handle.points.numpy()[1], [scale, 0, 0])
                np.testing.assert_array_equal(dataset.get_field("temperature").get_data().numpy(), [value] * 4)
        assert identities[1] == identities[2]
        assert identities[0] != identities[1] != identities[3]
        assert len(discovered) == 1
    with readers.UsdDataSource.open(path) as reopened, reopened.at_time(2.5) as frame:
        assert frame._provider.describe_component("/data", "", "geometry")[0] != identities[1]


def test_description_does_not_load_values(tmp_path, monkeypatch):
    path = make_source(tmp_path / "data.usda")
    with readers.UsdDataSource.open(path) as source, source.at_time(6) as frame:

        def unexpected_get(*args, **kwargs):
            pytest.fail("description read a scientific value")

        monkeypatch.setattr(Usd.Attribute, "Get", unexpected_get)
        assert frame._provider.describe_component("/data", "", "geometry")[1].time == 5


def test_usd_samples_drive_committed_faces_gate(tmp_path, monkeypatch):
    import ovstage

    from ovcae.operators._builtins.faces import Faces

    from ._ovstage_helpers import read_array
    from .test_change_detection import RECIPE, setup_recipe

    path = make_source(tmp_path / "data.usda")
    counts = dict(execute=0, resolve=0, writes=0)
    observed = []
    original_execute = Faces.execute

    def execute(self, context):
        counts["execute"] += 1
        observed.append(
            tuple(context.changes.input_changed("source", components={c}) for c in ("topology", "geometry"))
        )
        return original_execute(self, context)

    monkeypatch.setattr(Faces, "execute", execute)
    original_resolve = UsdFrameProvider.resolve_dataset

    def resolve(*args, **kwargs):
        counts["resolve"] += 1
        return original_resolve(*args, **kwargs)

    monkeypatch.setattr(UsdFrameProvider, "resolve_dataset", resolve)
    stage = ovstage.Stage("ovcae.tests.usd-change-detection")
    try:
        setup_recipe(stage)
        stage.advance_write_floor(1).wait()
        for method in ("write_attributes", "delete_attributes"):
            original = getattr(stage, method)

            def write(*args, _original=original, **kwargs):
                counts["writes"] += 1
                return _original(*args, **kwargs)

            monkeypatch.setattr(stage, method, write)
        with readers.UsdDataSource.open(path) as source, ovcae.Session(stage, device="cpu") as session:
            ordinal = 1
            for time, changed, value in [
                (0, True, 1),
                (1, False, 1),
                (5, True, 2),
                (6, False, 2),
                (7, True, 2),
                (10, True, 3),
            ]:
                before = counts.copy()
                with source.at_time(time) as frame:
                    report = session.update(input_ordinal=ordinal, output_ordinal=ordinal + 1, frame_data=frame).wait()
                ordinal += 1
                assert report.committable, report.diagnostics
                stage.advance_write_floor(ordinal).wait()
                assert report.operators[0].status == ("succeeded" if changed else "unchanged"), report.diagnostics
                if changed:
                    assert observed[-1] == (True, True)
                    assert counts["execute"] == before["execute"] + 1
                    assert counts["resolve"] == before["resolve"] + 1
                else:
                    assert counts == before
                np.testing.assert_array_equal(read_array(stage, RECIPE, "primvars:temperature", ordinal), [value] * 4)
            assert counts["execute"] == 4
    finally:
        stage.destroy()


@pytest.mark.parametrize("time", [float("nan"), float("inf"), -float("inf")])
def test_invalid_time_is_rejected(tmp_path, time):
    with (
        readers.UsdDataSource.open(make_source(tmp_path / "data.usda")) as source,
        pytest.raises(ovcae.DatasetResolutionError, match="finite"),
        source.at_time(time),
    ):
        pass


def test_held_provider_does_not_mutate_application_cached_stage(tmp_path):
    path = make_source(tmp_path / "data.usda")
    cache = Usd.StageCache()
    with Usd.StageCacheContext(cache):
        application_stage = Usd.Stage.Open(str(path))
        with readers.UsdDataSource.open(path) as source:
            assert source._stage != application_stage
            assert source._stage.GetInterpolationType() == Usd.InterpolationTypeHeld
            assert application_stage.GetInterpolationType() == Usd.InterpolationTypeLinear
