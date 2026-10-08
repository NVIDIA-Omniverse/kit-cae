# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Recipe validation against usd-core's schema registry and the built-in operator contracts."""

from __future__ import annotations

from ovcae.schemas import OPERATORS, SchemaError, find_schema

from . import schemas as S

_OPERATORS = {info.schema: info for info in OPERATORS}


def validate_recipe(prim) -> tuple[str, ...]:
    """Return the problems that would stop ovcae from executing ``prim`` as a recipe.

    Checks that ``CaeVizOperatorAPI`` and exactly one built-in operator API are
    applied, that the prim type matches the operator, that every required role
    is applied and selects something, that applied ``CaeViz*`` tokens are
    well-formed and registered with usd-core, and that authored token
    attributes use allowed values. An empty tuple means the recipe is
    well-formed as far as the schemas can tell.
    """

    applied = tuple(prim.GetAppliedSchemas())
    problems = []
    if S.OperatorAPI.__schema__ not in applied:
        problems.append(f"{S.OperatorAPI.__schema__} is not applied")

    operators = [info for info in _OPERATORS.values() if info.schema in applied]
    if not operators:
        problems.append(f"no operator API applied; expected one of {', '.join(_OPERATORS)}")
    elif len(operators) > 1:
        problems.append(f"more than one operator API applied: {', '.join(info.schema for info in operators)}")
    type_name = str(prim.GetTypeName())
    for info in operators:
        if type_name != info.prim_type:
            problems.append(f"{info.schema} requires a {info.prim_type} prim; got {type_name or 'untyped'}")
        problems.extend(f"{info.schema} requires {selector}" for selector in info.missing(applied))
        for role in info.dataset_roles:
            selection = S.DatasetSelectionAPI.get(prim, role)
            if selection is not None and not selection.target:
                problems.append(f"dataset role {role!r} has no target")
        for role in info.field_roles:
            selection = S.FieldSelectionAPI.get(prim, role)
            if selection is not None and not selection.field_names:
                problems.append(f"field role {role!r} selects no fields")

    for token in applied:
        if not token.startswith("CaeViz"):
            continue
        try:
            klass, instance = parse_applied(token)
        except SchemaError as exc:
            problems.append(str(exc))
            continue
        for descriptor in klass.properties().values():
            if descriptor.kind != "attribute" or not descriptor.allowed_tokens:
                continue
            attribute = prim.GetAttribute(descriptor.resolve(instance))
            if attribute and attribute.HasAuthoredValue() and descriptor.problem(attribute.Get()):
                problems.append(
                    f"{attribute.GetName()} is {attribute.Get()!r}; allowed: {', '.join(descriptor.allowed_tokens)}"
                )
    return tuple(problems)


def parse_applied(token: str):
    """``CaeVizFieldSelectionAPI:colors`` -> (typed class, ``"colors"``); raises :class:`SchemaError` when malformed."""

    family, separator, instance = token.partition(":")
    klass = find_schema(family)
    if klass is None:
        raise SchemaError(f"unknown schema {family!r}")
    if klass.__kind__ == "multipleApply" and not instance:
        raise SchemaError(f"{family} is multiple-apply; {token!r} lacks an instance")
    if klass.__kind__ != "multipleApply" and separator:
        raise SchemaError(f"{family} is single-apply; {token!r} must not carry an instance")
    return klass, (instance or None)
