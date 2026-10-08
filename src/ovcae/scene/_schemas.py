# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Property definitions the input scope is built from, answered by usd-core's schema registry.

A recipe prim's applied API schemas decide which properties an operator may
read. usd-core knows every registered schema, bundled or custom, so the runtime
asks it rather than carrying a catalog of its own. Schemas usd-core itself
ships (``MaterialBindingAPI`` and friends) are presentation, not recipe
settings, and stay out of the scope so edits to them never re-execute a recipe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ovcae.schemas import find_schema, is_usd_builtin_schema


@dataclass(frozen=True)
class PropertyInfo:
    """One property an operator may read: its resolved name, kind, type, and schema fallback."""

    name: str
    kind: str
    type_name: str | None = None
    default: Any = None
    has_default: bool = False


def applied_properties(applied_api_schemas) -> tuple[dict[str, PropertyInfo], tuple[str, ...]]:
    """Properties contributed by ``applied_api_schemas``, plus the ``CaeViz*`` tokens usd-core does not know.

    Every schema usd-core knows and does not itself ship contributes its
    properties. Built-in usd-core schemas contribute nothing, and unknown
    tokens outside the ``CaeViz`` family are ignored rather than reported, so
    a Kit-only schema on a recipe prim does not warn on every update. A
    multiple-apply token without an instance name contributes nothing either;
    it names no properties.
    """

    properties: dict[str, PropertyInfo] = {}
    unknown = []
    for token in sorted(applied_api_schemas):
        family, separator, instance = token.partition(":")
        klass = find_schema(family)
        if klass is None:
            if family.startswith("CaeViz"):
                unknown.append(token)
            continue
        if is_usd_builtin_schema(family):
            continue
        if klass.__kind__ == "multipleApply" and not instance:
            continue
        for descriptor in klass.properties().values():
            name = descriptor.resolve(instance if separator else None)
            if descriptor.kind == "attribute":
                properties[name] = PropertyInfo(
                    name, "attribute", descriptor.type_name, descriptor.default, descriptor.has_default
                )
            else:
                properties[name] = PropertyInfo(name, "relationship")
    return properties, tuple(unknown)


__all__ = ["PropertyInfo", "applied_properties"]
