# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Generic discovery must cover schema patterns without reading scientific values."""

import pytest

pytest.importorskip("ovcae.readers")
from pxr import Sdf, Usd

from ovcae import DatasetResolutionError
from ovcae.data import held_time
from ovcae.readers._timeline import dataset_time_samples


def sampled(prim, name, times):
    attr = prim.CreateAttribute(name, Sdf.ValueTypeNames.Float)
    for time in times:
        attr.Set(time, time)
    return attr


def test_graph_union_includes_descendants_external_targets_and_cycles(monkeypatch):
    stage = Usd.Stage.CreateInMemory()
    root = stage.DefinePrim("/Zone")
    coordinates = stage.DefinePrim("/Coordinates")
    section = stage.DefinePrim("/Zone/Section")
    solution = stage.DefinePrim("/Solution")
    sampled(root, "origin", [0, 10])
    sampled(coordinates, "omni:sci:array:points:value", [1, 5])
    sampled(section, "omni:sci:array:connectivity:value", [0])
    pressure = sampled(solution, "omni:sci:array:pressure:value", [2, 5])
    sampled(stage.DefinePrim("/Unrelated"), "unused", [99])
    root.CreateRelationship("coordinates").SetTargets([coordinates.GetPath()])
    root.CreateRelationship("pressure").SetTargets([pressure.GetPath()])
    section.CreateRelationship("zone").SetTargets([root.GetPath()])
    # Property-to-relationship forwarding, including a cycle, is conservative.
    link = coordinates.CreateRelationship("forward")
    link.SetTargets([root.GetRelationship("coordinates").GetPath()])

    def unexpected_get(*args, **kwargs):
        pytest.fail("timeline discovery loaded an attribute value")

    monkeypatch.setattr(Usd.Attribute, "Get", unexpected_get)
    assert dataset_time_samples(root) == (0, 1, 2, 5, 10)


@pytest.mark.parametrize("target", ["/Missing", "/Data.missing"])
def test_missing_dependencies_are_not_static(target):
    stage = Usd.Stage.CreateInMemory()
    root = stage.DefinePrim("/Data")
    root.CreateRelationship("dependency").SetTargets([target])
    with pytest.raises(DatasetResolutionError, match="targets missing"):
        dataset_time_samples(root)


def test_default_single_sample_and_held_selection():
    stage = Usd.Stage.CreateInMemory()
    root = stage.DefinePrim("/Data")
    root.CreateAttribute("constant", Sdf.ValueTypeNames.Float).Set(42)
    assert dataset_time_samples(root) == ()
    assert held_time((), 5) is None
    sampled(root, "one", [0])
    assert dataset_time_samples(root) == (0,)
    assert [held_time((0, 5, 10), t) for t in (-1, 0, 4, 5, 9, 10, 20)] == [0, 0, 0, 5, 5, 10, 10]


def test_composed_offsets_and_instance_descendants():
    layer = Sdf.Layer.CreateAnonymous()
    source = Usd.Stage.Open(layer)
    root = source.DefinePrim("/Data")
    sampled(source.DefinePrim("/Data/Child"), "value", [0, 2])
    source.SetDefaultPrim(root)
    stage = Usd.Stage.CreateInMemory()
    instance = stage.DefinePrim("/Instance")
    instance.GetReferences().AddReference(layer.identifier, layerOffset=Sdf.LayerOffset(10, 2))
    instance.SetInstanceable(True)
    assert dataset_time_samples(instance) == (10, 14)


def test_unloaded_payload_does_not_look_static():
    layer = Sdf.Layer.CreateAnonymous()
    source = Usd.Stage.Open(layer)
    source.SetDefaultPrim(source.DefinePrim("/Data"))
    stage = Usd.Stage.CreateInMemory()
    root = stage.DefinePrim("/Data")
    root.GetPayloads().AddPayload(layer.identifier)
    stage.Unload(root.GetPath())
    with pytest.raises(DatasetResolutionError, match="unloaded payload"):
        dataset_time_samples(root)


def test_lazy_arrays_discovered_even_when_property_enumeration_omits_values(monkeypatch):
    stage = Usd.Stage.CreateInMemory()
    root = stage.DefinePrim("/Data")
    # Model the native reader boundary without requiring its plugin registration.
    monkeypatch.setattr(Usd.Prim, "GetAppliedSchemas", lambda prim: ["OmniSciArrayAPI:points"])
    sampled(root, "omni:sci:array:points:value", [0, 3])
    original = Usd.Prim.GetAttributes

    def attributes(prim):
        return [attr for attr in original(prim) if not attr.GetName().endswith(":value")]

    monkeypatch.setattr(Usd.Prim, "GetAttributes", attributes)
    assert dataset_time_samples(root) == (0, 3)


def test_value_clip_times_are_discovered_in_composed_stage_time(tmp_path):
    clip = Usd.Stage.CreateInMemory()
    sampled(clip.DefinePrim("/Data"), "value", [0, 2])
    path = tmp_path / "clip.usda"
    clip.GetRootLayer().Export(str(path))
    stage = Usd.Stage.CreateInMemory()
    root = stage.DefinePrim("/Data")
    root.CreateAttribute("value", Sdf.ValueTypeNames.Float)
    clips = Usd.ClipsAPI(root)
    clips.SetClipAssetPaths([Sdf.AssetPath(str(path))])
    clips.SetClipPrimPath("/Data")
    clips.SetClipActive([(10, 0)])
    clips.SetClipTimes([(10, 0), (14, 2)])
    assert dataset_time_samples(root) == (10, 14)
