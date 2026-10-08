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

from pathlib import Path

import pytest

import ovcae

from ._simdata import tetrahedron

SCENE = Path(__file__).parent / "data" / "scenes" / "faces.usda"
HOLDER = "/World/Assets/StaticMixer"


def _tetrahedron():
    return tetrahedron("cpu")


def test_memory_source_and_frame_lifecycle() -> None:
    source = ovcae.MemoryDataSource()
    with source.at_time(1.25) as frame:
        assert isinstance(frame, ovcae.FrameData)
        with pytest.raises(ovcae.ActiveFrameDataError):
            source.close()

    source.close()
    source.close()
    with pytest.raises(ovcae.DataSourceClosedError), source.at_time(2.0):
        pass


def test_frame_data_has_no_public_constructor() -> None:
    with pytest.raises(TypeError):
        ovcae.FrameData()  # type: ignore[call-arg]


def test_memory_source_claims_registered_keys() -> None:
    source = ovcae.MemoryDataSource({"/a": _tetrahedron(), ("/b", "zone"): _tetrahedron()})
    assert source.resolves("/a")
    assert source.resolves("/b")
    assert not source.resolves("/c")


def test_missing_file_raises_open_error(tmp_path: Path) -> None:
    readers = pytest.importorskip("ovcae.readers")
    with pytest.raises(ovcae.DataSourceOpenError, match="not a file"):
        readers.UsdDataSource.open(tmp_path / "future-file-source.usda")


def test_composite_routes_keys_and_owns_children() -> None:
    first = ovcae.MemoryDataSource({"/a": _tetrahedron()})
    second = ovcae.MemoryDataSource({"/b": _tetrahedron()})
    with ovcae.CompositeDataSource(first, second) as source:
        assert source.resolves("/a")
        assert source.resolves("/b")
        assert not source.resolves("/c")
        with source.at_time(0.0) as frame:
            dataset = frame._resolve_dataset("/b", identifier="", device="cpu", fields={})
            assert dataset.get_num_elems() == 1
            with pytest.raises(ovcae.DatasetResolutionError, match="no Data Source resolves"):
                frame._resolve_dataset("/c", identifier="", device="cpu", fields={})
            with pytest.raises(ovcae.ActiveFrameDataError):
                first.close()
    with pytest.raises(ovcae.DataSourceClosedError), first.at_time(0.0):
        pass


def test_composite_rejects_ambiguous_claims() -> None:
    source = ovcae.CompositeDataSource(
        ovcae.MemoryDataSource({"/a": _tetrahedron()}),
        ovcae.MemoryDataSource({"/a": _tetrahedron()}),
    )
    with source, source.at_time(0.0) as frame, pytest.raises(ovcae.DatasetResolutionError, match="claimed by 2"):
        frame._resolve_dataset("/a", identifier="", device="cpu", fields={})


def test_composite_updates_retain_child_leases() -> None:
    child = ovcae.MemoryDataSource({"/a": _tetrahedron()})
    source = ovcae.CompositeDataSource(child)
    with source.at_time(0.0) as frame:
        frame._retain_for_update()
    source.close()
    assert child.published()
    frame._release_from_update()
    assert not child.published()


def test_composite_requires_data_sources() -> None:
    with pytest.raises(ValueError):
        ovcae.CompositeDataSource()
    with pytest.raises(TypeError):
        ovcae.CompositeDataSource({})  # type: ignore[arg-type]


def test_holder_requires_identifier_for_multi_dataset_payload() -> None:
    readers = pytest.importorskip("ovcae.readers")
    with (
        readers.UsdDataSource.open(SCENE) as source,
        source.at_time(0.0) as frame,
        pytest.raises(ovcae.DatasetResolutionError, match="identifier is required"),
    ):
        frame._resolve_dataset(HOLDER, identifier="", device="cpu", fields={})


def test_holder_composes_datasets_and_is_not_transformable() -> None:
    readers = pytest.importorskip("ovcae.readers")
    from pxr import UsdGeom

    with readers.UsdDataSource.open(SCENE) as source:
        assert source.resolves(HOLDER)
        assert not source.resolves("/World/Missing")
        holder = source._stage.GetPrimAtPath(HOLDER)
        assert holder.HasAuthoredPayloads()
        assert holder.HasAPI("OmniSciFileFormatArgsCgnsAPI")
        assert not holder.IsA(UsdGeom.Xformable)
        assert source._stage.GetPrimAtPath(f"{HOLDER}/Base/StaticMixer/B1_P3").GetTypeName() == "OmniSciDataset"


def test_usd_source_accepts_application_stage() -> None:
    readers = pytest.importorskip("ovcae.readers")
    from pxr import Usd

    ovcae.register_schema_paths()
    stage = Usd.Stage.Open(str(SCENE))
    with readers.UsdDataSource(stage) as source:
        assert source._stage is stage
        with source.at_time(0.0) as frame:
            dataset = frame._resolve_dataset(HOLDER, identifier="/Base/StaticMixer/B1_P3", device="cpu", fields={})
            assert dataset.get_num_elems() > 0
    assert stage.GetInterpolationType() == Usd.InterpolationTypeLinear


def test_usd_source_retains_loaded_dataset_across_frames() -> None:
    readers = pytest.importorskip("ovcae.readers")
    identifier = "/Base/StaticMixer/B1_P3"
    fields = {"velocities": ("VelocityX", "VelocityY", "VelocityZ")}
    with readers.UsdDataSource.open(SCENE) as source:
        with source.at_time(0.0) as frame:
            first = frame._resolve_dataset(HOLDER, identifier=identifier, device="cpu", fields=fields)
            first.build_element_locator()
            first.add_field("scratch", first.get_field("velocities"))
        with source.at_time(0.0) as frame:
            second = frame._resolve_dataset(HOLDER, identifier=identifier, device="cpu", fields=fields)
            third = frame._resolve_dataset(HOLDER, identifier=identifier, device="cpu", fields={})
        # One load serves every frame at the same sample: the handle, the locator
        # built by an earlier operator, and the loaded field arrays are shared.
        assert second is not first and second.handle is first.handle
        assert second.has_accel_struct("element_locator")
        assert second.get_field("velocities") is first.get_field("velocities")
        # Field mappings stay private to each resolution.
        assert not second.has_field("scratch") and not third.has_field("velocities")
        assert len(source._loaded) == 1
        # Selecting another sample of the same prim replaces the retained load.
        ((identity, _base, _fields),) = source._loaded.values()
        replaced, replaced_fields = source.loaded((identity[0], identity[1], 42.0), "cpu", lambda: object())
        assert len(source._loaded) == 1 and replaced_fields == {} and replaced is not first
        assert source.loaded((identity[0], identity[1], 42.0), "cpu", lambda: object()) == (replaced, replaced_fields)
    assert source._loaded == {}


def test_memory_source_rejects_non_datasets_and_bad_keys() -> None:
    with pytest.raises(TypeError, match="Warp SimData datasets"):
        ovcae.MemoryDataSource({"/a": {"points": [[0, 0, 0]]}})
    with pytest.raises(TypeError, match="path string or a"):
        ovcae.MemoryDataSource({("/a", "zone", "extra"): _tetrahedron()})


def test_memory_source_attaches_selected_fields_under_role_names() -> None:
    import numpy as np

    dataset = tetrahedron(
        "cpu",
        Temperature=[1.0, 2.0, 3.0, 4.0],
        VX=[1.0, 0.0, 0.0, 0.0],
        VY=[0.0, 1.0, 0.0, 0.0],
        VZ=[0.0, 0.0, 1.0, 0.0],
    )
    with ovcae.MemoryDataSource({"/a": dataset}) as source, source.at_time(0.0) as frame:
        resolved = frame._resolve_dataset(
            "/a", identifier="", device="cpu", fields={"colors": ("Temperature",), "velocity": ("VX", "VY", "VZ")}
        )
        assert sorted(resolved.get_field_names()) == ["colors", "velocity"]
        assert resolved.get_field("colors").to_array().numpy().tolist() == [1.0, 2.0, 3.0, 4.0]
        velocity = resolved.get_field("velocity").to_array().numpy()
        assert velocity.shape == (4, 3) and np.allclose(velocity[2], [0.0, 0.0, 1.0])
        # The published dataset itself is untouched and keeps its own names.
        assert sorted(dataset.get_field_names()) == ["Temperature", "VX", "VY", "VZ"]
        with pytest.raises(ovcae.DatasetResolutionError, match="has: Temperature, VX, VY, VZ"):
            frame._resolve_dataset("/a", identifier="", device="cpu", fields={"colors": ("Pressure",)})


def test_memory_source_reports_device_mismatch() -> None:
    with (
        ovcae.MemoryDataSource({"/a": _tetrahedron()}) as source,
        source.at_time(0.0) as frame,
        pytest.raises(ovcae.DatasetResolutionError, match="lives on cpu but the recipe evaluates on cuda:0"),
    ):
        frame._resolve_dataset("/a", identifier="", device="cuda:0", fields={})


def test_memory_source_publish_changes_identity_and_frames_keep_their_version() -> None:
    first, second = _tetrahedron(), _tetrahedron()
    source = ovcae.MemoryDataSource({"/a": first})
    with source.at_time(0.0) as before:
        identity_before, sample = before._provider.describe_component("/a", "", "geometry")
        assert sample.time is None
        source.publish("/a", second)
        with source.at_time(0.0) as after:
            identity_after, _ = after._provider.describe_component("/a", "", "geometry")
            assert identity_after != identity_before
            assert after._resolve_dataset("/a", identifier="", device="cpu", fields={}).handle is second.handle
        # The earlier lease still serves the dataset it saw when it was taken.
        assert before._resolve_dataset("/a", identifier="", device="cpu", fields={}).handle is first.handle
    source.remove("/a")
    assert not source.resolves("/a")
    with pytest.raises(KeyError):
        source.remove("/a")
    source.close()


def test_memory_source_time_samples_hold_the_preceding_sample() -> None:
    early, late = _tetrahedron(), _tetrahedron()
    source = ovcae.MemoryDataSource()
    source.publish("/a", early, time=1.0)
    source.publish("/a", late, time=2.0)
    source.publish("/b", _tetrahedron())  # static, serves every time
    for requested, expected in ((0.0, early), (1.0, early), (1.5, early), (2.0, late), (9.0, late)):
        with source.at_time(requested) as frame:
            resolved = frame._resolve_dataset("/a", identifier="", device="cpu", fields={})
            assert resolved.handle is expected.handle, requested
            _identity, sample = frame._provider.describe_component("/a", "", "topology")
            assert sample.time == (1.0 if expected is early else 2.0)
            frame._resolve_dataset("/b", identifier="", device="cpu", fields={})
    source.publish("/a", _tetrahedron())  # a static publish replaces the whole series
    with source.at_time(0.0) as frame:
        assert frame._provider.describe_component("/a", "", "topology")[1].time is None
    source.close()
