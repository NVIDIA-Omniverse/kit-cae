# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Runtime-neutral access to the codeless OmniCaeViz schema bundle.

Two views of the same ``schema.usda``:

- the resource plug-in (``schema_resource_path``) for usd-core and OVStage
  registration; usd-core's schema registry is the single source of schema
  knowledge once the plug-in is registered (:func:`register_usd_plugin`);
- typed schema classes (``ovcae.schemas.typed``), built from usd-core's
  registry on first use, that read and author schema properties on any prim a
  backend adapts, without spelling property names. :func:`schema_from_registry`
  builds the same kind of class for any other applied API schema usd-core knows.

Importing this package never imports pxr; the functions that need it import
it on call, and ``ovcae.schemas.typed`` registers the bundle when imported.
"""

from __future__ import annotations

from pathlib import Path

from ._operators import OPERATORS, OperatorInfo
from ._registry import find_schema, is_usd_builtin_schema, plain_value, register_usd_plugin, schema_from_registry
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


def schema_resource_path() -> Path:
    """Return the directory containing the codeless OpenUSD resource plug-in."""

    path = Path(__file__).parent / "resources"
    required = (
        path / "schema.usda",
        path / "generatedSchema.usda",
        path / "plugInfo.json",
    )
    missing = [item.name for item in required if not item.is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete ovcae.schemas resource bundle at {path}: missing {', '.join(missing)}")
    return path


def schema_source_path() -> Path:
    """Return the canonical OmniCaeViz schema source."""

    return schema_resource_path() / "schema.usda"


__all__ = [
    "OPERATORS",
    "AppliedSchema",
    "Attribute",
    "MultipleApplyAPI",
    "OperatorInfo",
    "PrimBackend",
    "Relationship",
    "SchemaError",
    "SingleApplyAPI",
    "backend_for",
    "find_schema",
    "is_usd_builtin_schema",
    "make_schema_class",
    "plain_value",
    "register_backend",
    "register_usd_plugin",
    "schema_from_registry",
    "schema_resource_path",
    "schema_source_path",
]
