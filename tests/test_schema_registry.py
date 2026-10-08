# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""usd-core's schema registry answers every schema question, bundled or custom."""

from __future__ import annotations

from pathlib import Path

import pytest

import ovcae.schemas
from ovcae.schemas import (
    OPERATORS,
    SchemaError,
    find_schema,
    is_usd_builtin_schema,
    plain_value,
    register_usd_plugin,
    schema_from_registry,
)
from ovcae.schemas import typed as T


@pytest.fixture(scope="module", autouse=True)
def registered():
    pytest.importorskip("pxr")
    register_usd_plugin()


def test_register_usd_plugin_reports_the_bundled_applied_schemas() -> None:
    identifiers = register_usd_plugin()
    assert "CaeVizFacesAPI" in identifiers and "CaeVizFieldSelectionAPI" in identifiers
    assert "CaeVizIndeXVolumeAPI" in identifiers
    assert identifiers == register_usd_plugin(ovcae.schemas.schema_resource_path())
    with pytest.raises(SchemaError, match=r"no plugInfo\.json"):
        register_usd_plugin(Path(__file__).parent)


def test_typed_module_exposes_one_class_per_bundled_schema() -> None:
    assert find_schema("CaeVizFacesAPI") is T.FacesAPI
    assert schema_from_registry("CaeVizFieldSelectionAPI") is T.FieldSelectionAPI
    assert T.FacesAPI.__module__ == "ovcae.schemas.typed" and T.FacesAPI.__name__ == "FacesAPI"
    assert T.FacesAPI.__can_only_apply_to__ == ("Mesh",)
    assert T.FacesAPI.__schema_doc__.startswith("API schema when applied to UsdGeomMesh")
    assert "FacesAPI" in T.__all__ and "IndeXVolumeAPI" in T.__all__ and "Attribute" in T.__all__
    assert "FacesAPI" in dir(T)
    assert find_schema("CaeVizNopeAPI") is None
    with pytest.raises(SchemaError, match="not a registered"):
        schema_from_registry("CaeVizNopeAPI")
    with pytest.raises(AttributeError):
        T.NopeAPI  # noqa: B018


def test_typed_classes_carry_schema_metadata() -> None:
    direction = T.StreamlinesAPI.direction
    assert direction.allowed_tokens == ("forward", "backward", "both")
    assert direction.display_name == "Direction" and direction.display_group == "Streamlines"
    assert direction.doc.startswith("Specifies the integration direction")
    assert T.StreamlinesAPI.initial_step_size.default == 0.2
    assert T.FieldMappingAPI.domain.default == (0.0, -1.0)
    assert T.FieldMappingAPI.domain.resolve("scales") == "cae:viz:field_mapping:scales:domain"
    assert T.RescaleRangeAPI.includes.targets == "properties"
    assert T.DatasetSelectionAPI.target.targets == "prims"
    assert T.DatasetSelectionAPI.identifier.default == "" and T.DatasetSelectionAPI.identifier.has_default
    assert list(T.DatasetTransformingAPI.properties()) == []
    assert T.ColormapTextureAPI.identifier.variability == "Uniform"


def test_registry_builds_classes_for_other_applied_schemas() -> None:
    klass = schema_from_registry("CollectionAPI")
    assert klass.__kind__ == "multipleApply" and klass.__namespace__ == "collection"
    assert "expansion_rule" in klass.properties()
    assert klass.expansion_rule.allowed_tokens == ("explicitOnly", "expandPrims", "expandPrimsAndProperties")
    assert klass.expansion_rule.resolve("lights") == "collection:lights:expansionRule"
    assert klass.expansion_rule.default == "expandPrims"


def test_builtin_usd_schemas_are_recognized() -> None:
    assert is_usd_builtin_schema("MaterialBindingAPI")
    assert is_usd_builtin_schema("CollectionAPI")
    assert not is_usd_builtin_schema("CaeVizFacesAPI")
    assert not is_usd_builtin_schema("CaeVizNopeAPI")


def test_plain_value_reduces_pxr_values() -> None:
    from pxr import Gf, Sdf, Vt

    assert plain_value(Gf.Vec2f(0.0, -1.0), "float2") == (0.0, -1.0)
    assert plain_value(Vt.TokenArray(["a", "b"]), "token[]") == ("a", "b")
    assert plain_value(Sdf.AssetPath("./x.png"), "asset") == "./x.png"
    assert plain_value(0.20000000298023224, "float") == 0.2
    assert plain_value(0.20000000298023224, "double") == 0.20000000298023224
    assert plain_value(True, "bool") is True and plain_value(None) is None


def test_scene_scope_comes_from_the_registry() -> None:
    from ovcae.scene._schemas import applied_properties

    properties, unknown = applied_properties(
        (
            "CaeVizFieldSelectionAPI:colors",
            "CaeVizFacesAPI",
            "MaterialBindingAPI",
            "CaeVizXAPI",
            "CaeVizFieldMappingAPI",
        )
    )
    assert unknown == ("CaeVizXAPI",)
    assert applied_properties(("KitOnlyThingAPI", "MaterialBindingAPI"))[1] == ()
    assert properties["cae:viz:faces:externalOnly"].has_default is True
    assert properties["cae:viz:field_selection:colors:fieldNames"].has_default is False
    assert properties["cae:viz:field_selection:colors:mode"].default == "unchanged"
    assert not any(name.startswith("material:") for name in properties)
    assert not any(name.startswith("cae:viz:field_mapping") for name in properties)


def test_operator_table_matches_builtin_declarations() -> None:
    pytest.importorskip("warp")
    from ovcae.operators import OperatorRegistry

    declared = {}
    for declaration in OperatorRegistry.with_builtins().snapshot()._declarations:
        schema = next(item for item in declaration.required_schemas if item.endswith("API") and ":" not in item)
        declared[schema] = (declaration.required_schemas, declaration.optional_schemas)
    table = {item.schema: (item.required, item.optional) for item in OPERATORS}
    assert table == declared


def test_operator_table_roles() -> None:
    streamlines = next(item for item in OPERATORS if item.schema == "CaeVizStreamlinesAPI")
    assert streamlines.prim_type == "BasisCurves"
    assert streamlines.dataset_roles == ("seeds", "source")
    assert streamlines.field_roles == ("velocities",)
    applied = ("CaeVizStreamlinesAPI", "CaeVizDatasetSelectionAPI:source", "CaeVizFieldSelectionAPI:velocities")
    assert streamlines.missing(applied) == ("CaeVizDatasetSelectionAPI:seeds",)
    assert not streamlines.matches(applied)
    assert streamlines.matches((*applied, "CaeVizDatasetSelectionAPI:seeds"))
