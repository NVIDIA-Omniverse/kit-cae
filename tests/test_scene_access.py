# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Read-through values are lazy and owned; only scene handles expire."""

import numpy as np
import ovstage
import pytest

from ovcae.scene import ExpiredViewError, OutOfScopeError
from ovcae.scene._plan import plan_operator

from ._ovstage_helpers import (
    path_query,
    release_path_query,
    write_relationship,
    write_scalar,
    write_token,
    write_transform,
)


@pytest.fixture
def scene():
    stage = ovstage.Stage("ovcae.tests.scene-access")
    write_token(stage, "/Recipe", "usd-prim-type", "Mesh", 1)
    write_token(stage, "/Seeds", "usd-prim-type", "Points", 1)
    write_relationship(stage, "/Recipe", "cae:viz:dataset_selection:source:target", ["/Seeds"], 1)
    paths, plist, query = path_query(stage, "/Seeds")
    try:
        stage.write_attribute(query, "points", 1, [np.asarray([[1, 2, 3]], np.float32)], is_array=True).wait()
    finally:
        release_path_query(paths, plist, query)
    stage.advance_write_floor(1).wait()
    view = plan_operator(
        stage,
        operator_path="/Recipe",
        applied_api_schemas=frozenset({"CaeVizFacesAPI", "CaeVizDatasetSelectionAPI:source"}),
        input_ordinal=1,
    )
    try:
        yield stage, view
    finally:
        view._expire()
        stage.destroy()


def test_scene_arrays_are_owned_and_native_reads_are_released(scene):
    stage, view = scene
    attr = view.get_prim_at_path("/Seeds").get_attribute("points")
    values = attr.get()
    assert isinstance(values, np.ndarray)
    assert values.shape == (1, 3)
    values[0, 0] = 99
    np.testing.assert_array_equal(attr.get(), [[1, 2, 3]])
    # No native group remains pinned after get; writes and destruction are safe.
    paths, plist, query = path_query(stage, "/Seeds")
    try:
        stage.delete_attributes(query, ["points"], 2).wait()
    finally:
        release_path_query(paths, plist, query)
    stage.advance_write_floor(2).wait()
    view._expire()
    with pytest.raises(ExpiredViewError):
        attr.get()
    np.testing.assert_array_equal(values, [[99, 2, 3]])


def test_defaults_scope_and_saved_handles(scene):
    _stage, view = scene
    prim = view.get_prim_at_path("/Recipe")
    attr = prim.get_attribute("cae:viz:faces:externalOnly")
    relationship = prim.get_relationship("cae:viz:dataset_selection:source:target")
    assert attr.get() is True
    assert relationship.targets == ("/Seeds",)
    with pytest.raises(OutOfScopeError):
        prim.get_attribute("points")
    with pytest.raises(OutOfScopeError):
        view.get_prim_at_path("/Unrelated")
    view._expire()
    for access in (attr.get, lambda: relationship.targets, prim.attribute_names):
        with pytest.raises(ExpiredViewError):
            access()


def test_values_read_current_stage_without_a_snapshot(scene):
    stage, view = scene
    attr = view.get_prim_at_path("/Recipe").get_attribute("cae:viz:faces:externalOnly")
    assert attr.get() is True
    # Exercise the reader directly, outside an Update. Applications must keep
    # inputs stable during actual execution; the view does not snapshot them.
    write_scalar(stage, "/Recipe", "cae:viz:faces:externalOnly", False, 2)
    stage.advance_write_floor(2).wait()
    assert attr.get() is False


def test_dependency_planning_does_not_load_settings_transforms_or_points(scene, monkeypatch):
    stage, _view = scene
    from ovcae.scene import _plan, _views

    original = _views.read_value
    read_names = []

    def read(*args, **kwargs):
        read_names.append(args[2])
        assert args[2] not in {"points", "omni:xform", "cae:viz:faces:externalOnly"}
        return original(*args, **kwargs)

    monkeypatch.setattr(_views, "read_value", read)
    view = _plan.plan_operator(
        stage,
        operator_path="/Recipe",
        applied_api_schemas=frozenset({"CaeVizFacesAPI", "CaeVizDatasetSelectionAPI:source"}),
        input_ordinal=1,
    )
    try:
        assert "omni:xform" not in read_names
        assert "points" in dict(view.dependencies)["/Seeds"]
        assert "cae:viz:faces:externalOnly" in dict(view.dependencies)["/Recipe"]
    finally:
        view._expire()


def test_native_transform_is_read_only_when_requested(scene):
    from ovcae.operators.utils import get_local_transform

    stage, view = scene
    write_transform(stage, "/Seeds", (4.0, 5.0, 6.0), 2)
    stage.advance_write_floor(2).wait()
    np.testing.assert_array_equal(get_local_transform(view.get_prim_at_path("/Seeds"))[:3, 3], [4, 5, 6])


def test_populated_usd_composed_transform(tmp_path):
    from pxr import Gf, Usd, UsdGeom

    import ovcae
    from ovcae.operators.utils import get_local_transform

    path = tmp_path / "transform.usda"
    usd = Usd.Stage.CreateNew(str(path))
    prim = UsdGeom.Mesh.Define(usd, "/Recipe")
    xform = UsdGeom.Xformable(prim)
    xform.AddTranslateOp().Set(Gf.Vec3d(1, 2, 3))
    xform.AddRotateXYZOp().Set(Gf.Vec3f(20, 35, 70))
    xform.AddScaleOp().Set(Gf.Vec3f(2, 3, 4))
    expected = np.asarray(xform.GetLocalTransformation()).T
    usd.GetRootLayer().Save()
    ovcae.register_schema_paths()
    stage = ovstage.Stage("ovcae.tests.native-transform")
    view = None
    try:
        ovstage.population.open_usd_with_desc(
            stage,
            str(path),
            ordinal=1,
            time_code=0.0,
            descs=ovcae.ovstage_population_desc(domains=ovstage.PopulationDomain.RENDERING),
        )
        stage.advance_write_floor(1).wait()
        view = plan_operator(
            stage, operator_path="/Recipe", applied_api_schemas={"CaeVizPlanarSliceAPI"}, input_ordinal=1
        )
        np.testing.assert_allclose(get_local_transform(view.get_prim_at_path("/Recipe")), expected)
        assert "omni:xform" in dict(view.dependencies)["/Recipe"]
    finally:
        if view is not None:
            view._expire()
        stage.destroy()


def test_repeated_access_reuses_query_and_expiry_releases_it(scene, monkeypatch):
    from unittest.mock import Mock

    stage, _view = scene
    queries = []
    original = stage.query_from_path_list

    def query(plist):
        native = original(plist)
        result = Mock(wraps=native, handle=native.handle)
        queries.append(result)
        return result

    # Use a fresh view to observe query creation, independently of fixture planning.
    monkeypatch.setattr(stage, "query_from_path_list", query)
    from ovcae.scene._views import StageView

    fresh = StageView(stage, 1)
    try:
        assert fresh._read("/Recipe", "cae:viz:faces:externalOnly") is None
        assert fresh._read("/Recipe", "usd-prim-type") == "Mesh"
        assert len(queries) == 1
    finally:
        fresh._expire()
    queries[0].release.assert_called_once()


def test_failed_planning_releases_queries(scene, monkeypatch):
    from unittest.mock import Mock

    from ovcae.scene import _views

    stage, _view = scene
    queries = []
    original_query = stage.query_from_path_list
    original_read = _views.read_value

    def query(plist):
        native = original_query(plist)
        result = Mock(wraps=native, handle=native.handle)
        queries.append(result)
        return result

    def read(*args, **kwargs):
        if args[1] == "/Seeds":
            raise RuntimeError("injected planning failure")
        return original_read(*args, **kwargs)

    monkeypatch.setattr(stage, "query_from_path_list", query)
    monkeypatch.setattr(_views, "read_value", read)
    with pytest.raises(RuntimeError, match="planning failure"):
        plan_operator(
            stage,
            operator_path="/Recipe",
            applied_api_schemas={"CaeVizFacesAPI", "CaeVizDatasetSelectionAPI:source"},
            input_ordinal=1,
        )
    assert len(queries) == 2
    for item in queries:
        item.release.assert_called_once()


def test_token_decoding_uses_semantics_not_property_names(scene):
    from ovcae._stage_io import read_value

    stage, _view = scene
    write_token(stage, "/Recipe", "custom:choice", "selected", 2)
    write_scalar(stage, "/Recipe", "custom:mode", 7, 2)
    stage.advance_write_floor(2).wait()
    assert read_value(stage, "/Recipe", "custom:choice", 2) == "selected"
    assert read_value(stage, "/Recipe", "custom:mode", 2) == 7
