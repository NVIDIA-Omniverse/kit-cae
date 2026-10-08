# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import pytest

from ovcae.operators import InvalidRecipe, OperatorRegistrationError, operator
from ovcae.operators.utils import _field_selection


class _FakeAttribute:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value


class _FakePrim:
    """Minimal PrimView stand-in exposing only what _field_selection reads."""

    def __init__(self, attributes: dict):
        self._attributes = attributes

    def get_attribute(self, name: str):
        return _FakeAttribute(self._attributes.get(name))


def test_optional_field_selection_with_no_fieldnames_decodes_to_none():
    prim = _FakePrim({})
    assert _field_selection(prim, "opacity", optional=True) is None


def test_required_field_selection_with_no_fieldnames_still_raises():
    prim = _FakePrim({})
    with pytest.raises(InvalidRecipe, match="has no fieldNames"):
        _field_selection(prim, "opacity")


def test_component_declarations_reject_unsupported_roles_and_components():
    with pytest.raises(OperatorRegistrationError, match="not selected"):
        operator(required_schemas={"CaeVizFacesAPI"}, dataset_components={"source": {"geometry"}})
    with pytest.raises(OperatorRegistrationError, match="invalid dataset components"):
        operator(required_schemas={"CaeVizDatasetSelectionAPI:source"}, dataset_components={"source": {"fields"}})


def test_component_declarations_are_immutable():
    requirements = {"source": {"geometry"}}

    @operator(required_schemas={"CaeVizDatasetSelectionAPI:source"}, dataset_components=requirements)
    class Example:
        def execute(self, context):
            pass

    requirements["source"].add("topology")
    components = Example.__ovcae_operator_declaration__.dataset_components
    assert components["source"] == frozenset({"geometry"})
    with pytest.raises(TypeError):
        components["source"] = frozenset()
