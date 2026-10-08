# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Typed schema classes for every bundled OmniCaeViz API schema, built from usd-core's registry.

``FacesAPI`` is ``CaeVizFacesAPI``, ``FieldSelectionAPI`` is
``CaeVizFieldSelectionAPI``, and so on for every applied API schema the bundle
declares; a class is built on first access and cached. Importing this module
registers the bundle with usd-core, so import it, like ``ovcae.authoring``,
before the process's first schema read. Each class wraps a prim through a
registered backend (``import ovcae.authoring`` registers ``Usd.Prim``) and
exposes schema properties as Python attributes::

    faces = FacesAPI.apply(prim)
    faces.external_only = False
    colors = FieldSelectionAPI.apply(prim, "colors")
    colors.field_names = ["Temperature"]
"""

from __future__ import annotations

from ._registry import register_usd_plugin, schema_from_registry
from ._typed import (
    AppliedSchema,
    Attribute,
    MultipleApplyAPI,
    PrimBackend,
    Relationship,
    SchemaError,
    SingleApplyAPI,
    backend_for,
    make_schema_class,
    register_backend,
)

_PREFIX = "CaeViz"
_SCHEMAS = {identifier.removeprefix(_PREFIX): identifier for identifier in register_usd_plugin()}

__all__ = [
    *sorted(_SCHEMAS),
    "AppliedSchema",
    "Attribute",
    "MultipleApplyAPI",
    "PrimBackend",
    "Relationship",
    "SchemaError",
    "SingleApplyAPI",
    "backend_for",
    "make_schema_class",
    "register_backend",
]


def __getattr__(name: str) -> type[AppliedSchema]:
    try:
        identifier = _SCHEMAS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    klass = schema_from_registry(identifier)
    globals()[name] = klass
    return klass


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
