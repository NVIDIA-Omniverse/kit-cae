# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Native contracts needed for recipe invalidation, without reading payload values."""

import ovstage
import pytest

from ._ovstage_helpers import (
    delete_prim,
    path_query,
    release_path_query,
    write_relationship,
    write_scalar,
    write_schemas,
)


def changed(stage, path, attribute, start, end):
    """Collect per-key change membership, including zero-payload tombstones."""
    paths, plist, query = path_query(stage, path)
    try:
        with stage.read_attributes(
            query, [paths.intern_token(attribute)], ovstage.OrdinalRange.between(start, end)
        ) as read:
            read.wait()
            result = []
            for group in read.groups():
                try:
                    result.append(group.is_delete)
                finally:
                    stage.release_group(group)
            return result
    finally:
        release_path_query(paths, plist, query)


@pytest.fixture
def stage():
    value = ovstage.Stage("ovcae.tests.recipe-change-ranges")
    try:
        yield value
    finally:
        value.destroy()


def test_recipe_ranges_track_writes_not_value_equality_and_are_path_specific(stage):
    write_scalar(stage, "/Recipe", "setting", 1, 1)
    write_scalar(stage, "/Other", "setting", 1, 1)
    stage.advance_write_floor(1).wait()
    write_scalar(stage, "/Other", "setting", 2, 2)
    stage.advance_write_floor(2).wait()
    assert changed(stage, "/Recipe", "setting", 2, 2) == []
    write_scalar(stage, "/Recipe", "setting", 1, 3)
    stage.advance_write_floor(3).wait()
    assert changed(stage, "/Recipe", "setting", 3, 3) == [False]
    from ovcae._reuse import Recipe

    assert Recipe(2, (("/Recipe", ("setting",)),)).changed(stage, 3)
    assert changed(stage, "/Recipe", "absent", 3, 3) == []


@pytest.mark.parametrize("kind", ["attribute", "relationship", "schemas"])
def test_recipe_range_tracks_each_property_kind_and_deletion(stage, kind):
    if kind == "relationship":
        name = "target"
        write_relationship(stage, "/Recipe", name, ["/A"], 1)
        write_relationship(stage, "/Recipe", name, ["/B"], 2)
    elif kind == "schemas":
        name = "usd-schemas"
        write_schemas(stage, "/Recipe", ["CaeVizFacesAPI"], 1)
        write_schemas(stage, "/Recipe", ["CaeVizPointsAPI"], 2)
    else:
        name = "setting"
        write_scalar(stage, "/Recipe", name, 1, 1)
        write_scalar(stage, "/Recipe", name, 2, 2)
    stage.advance_write_floor(2).wait()
    assert changed(stage, "/Recipe", name, 2, 2) == [False]
    paths, plist, query = path_query(stage, "/Recipe")
    try:
        stage.delete_attributes(query, [name], 3).wait()
    finally:
        release_path_query(paths, plist, query)
    stage.advance_write_floor(3).wait()
    assert changed(stage, "/Recipe", name, 3, 3) == [True]


def test_deleted_target_is_visible_to_an_explicit_path_query(stage):
    write_scalar(stage, "/Target", "points", 1, 1)
    stage.advance_write_floor(1).wait()
    delete_prim(stage, "/Target", 2)
    stage.advance_write_floor(2).wait()
    assert changed(stage, "/Target", "points", 2, 2) == [True]
    write_scalar(stage, "/Target", "points", 2, 3)
    stage.advance_write_floor(3).wait()
    assert changed(stage, "/Target", "points", 3, 3) == [False]


def test_range_retention_frontier_and_current_end(stage):
    write_scalar(stage, "/Recipe", "setting", 1, 1)
    stage.advance_write_floor(1).wait()
    write_scalar(stage, "/Recipe", "setting", 2, 2)
    stage.advance_write_floor(2).wait()
    with stage.get_oldest_preserved_ordinal() as query:
        query.wait()
        frontier = query.fetch()
    assert frontier <= 2
    assert changed(stage, "/Recipe", "setting", 2, 2) == [False]
    # A fixed old interval is not a historical payload read.
    if frontier <= 1:
        with pytest.raises(ovstage.OvstageError):
            changed(stage, "/Recipe", "setting", 1, 1)


def test_recipe_checks_absent_properties_and_ignores_unrelated_writes(stage):
    from ovcae._reuse import Recipe

    write_scalar(stage, "/Recipe", "setting", 1, 1)
    stage.advance_write_floor(1).wait()
    recipe = Recipe(1, (("/Recipe", ("setting", "defaulted")),))
    write_scalar(stage, "/Other", "setting", 2, 2)
    stage.advance_write_floor(2).wait()
    assert not recipe.changed(stage, 2)
    write_scalar(stage, "/Recipe", "defaulted", 4, 3)
    stage.advance_write_floor(3).wait()
    assert recipe.changed(stage, 3)


def test_recipe_assumes_expired_prefix_unchanged_but_checks_retained_suffix(stage, monkeypatch):
    from contextlib import contextmanager
    from types import SimpleNamespace

    from ovcae._reuse import Recipe

    write_scalar(stage, "/Recipe", "setting", 1, 1)
    stage.advance_write_floor(1).wait()
    recipe = Recipe(1, (("/Recipe", ("setting",)),))
    stage.advance_write_floor(3).wait()

    @contextmanager
    def expired_frontier():
        # Exercise the reported frontier without depending on a native retention
        # depth or how aggressively this particular runtime coalesces markers.
        yield SimpleNamespace(wait=lambda: None, fetch=lambda: 3)

    monkeypatch.setattr(stage, "get_oldest_preserved_ordinal", expired_frontier)
    assert not recipe.changed(stage, 3)
    write_scalar(stage, "/Recipe", "setting", 2, 4)
    stage.advance_write_floor(4).wait()
    assert recipe.changed(stage, 4)


def test_plan_declares_absent_transforms_and_relationship_targets(stage):
    from ovcae.scene._plan import plan_operator

    from ._ovstage_helpers import write_token

    write_token(stage, "/Recipe", "usd-prim-type", "Mesh", 1)
    write_relationship(stage, "/Recipe", "cae:viz:dataset_selection:source:target", ["/Absent"], 1)
    stage.advance_write_floor(1).wait()
    view = plan_operator(
        stage,
        operator_path="/Recipe",
        applied_api_schemas=frozenset({"CaeVizPlanarSliceAPI", "CaeVizDatasetSelectionAPI:source"}),
        input_ordinal=1,
    )
    try:
        dependencies = dict(view.dependencies)
        assert {"omni:xform", "usd-schemas", "usd-prim-type"} <= set(dependencies["/Recipe"])
        assert set(dependencies["/Absent"]) == {"usd-schemas", "usd-prim-type"}
        assert view.get_prim_at_path("/Recipe").get_attribute("omni:xform").get() is None
        assert "points" not in dependencies["/Recipe"]  # generated product, not an input
    finally:
        view._expire()


def test_recipe_read_failure_in_retained_suffix_still_invalidates(stage, monkeypatch):
    from ovcae._reuse import Recipe

    def fail(*args, **kwargs):
        raise ovstage.OvstageError(1, "injected recipe read failure")

    monkeypatch.setattr(stage, "read_attributes", fail)
    recipe = Recipe(1, (("/Recipe", ("setting",)),))
    assert recipe.changed(stage, 10, oldest=8)


def test_batching_preserves_dependency_pairs(stage, monkeypatch):
    from ovcae._reuse import Recipe

    write_scalar(stage, "/Operator", "setting", 1, 1)
    for path in ("/SeedsA", "/SeedsB"):
        write_scalar(stage, path, "points", 1, 1)
    stage.advance_write_floor(1).wait()
    recipe = Recipe(1, (("/Operator", ("setting",)), ("/SeedsA", ("points",)), ("/SeedsB", ("points",))))
    # These are in the cross-product of names and paths, but not dependencies.
    write_scalar(stage, "/Operator", "points", 2, 2)
    write_scalar(stage, "/SeedsA", "setting", 2, 2)
    stage.advance_write_floor(2).wait()
    from unittest.mock import Mock

    read = Mock(wraps=stage.read_attributes)
    monkeypatch.setattr(stage, "read_attributes", read)
    assert not recipe.changed(stage, 2)
    assert read.call_count == 2  # three paths, two distinct property sets
    write_scalar(stage, "/SeedsB", "points", 3, 3)
    stage.advance_write_floor(3).wait()
    assert recipe.changed(stage, 3)
