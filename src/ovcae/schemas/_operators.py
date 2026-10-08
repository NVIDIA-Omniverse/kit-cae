# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Hand-maintained table of the built-in operators' schema contracts.

``schema.usda`` documents operator roles only in prose, so this table records
them as data for validators and inspectors. ``tests/test_schema_registry.py``
asserts it matches the ``@operator`` declarations in ``ovcae.operators``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OperatorInfo:
    """One operator API: the prim type it applies to and the applied schemas it needs.

    Selectors use the ``@operator`` declaration syntax: ``Schema`` or
    ``Schema:instance``, with ``Schema:*`` matching any instance.
    """

    schema: str
    prim_type: str
    required: frozenset[str]
    optional: frozenset[str] = frozenset()

    @property
    def dataset_roles(self) -> tuple[str, ...]:
        return _roles(self.required, "CaeVizDatasetSelectionAPI")

    @property
    def field_roles(self) -> tuple[str, ...]:
        return _roles(self.required, "CaeVizFieldSelectionAPI")

    def matches(self, applied_schemas: frozenset[str] | tuple[str, ...]) -> bool:
        applied = frozenset(applied_schemas)
        return all(_selector_matches(selector, applied) for selector in self.required)

    def missing(self, applied_schemas: frozenset[str] | tuple[str, ...]) -> tuple[str, ...]:
        applied = frozenset(applied_schemas)
        return tuple(sorted(selector for selector in self.required if not _selector_matches(selector, applied)))


def _roles(selectors: frozenset[str], schema: str) -> tuple[str, ...]:
    prefix = f"{schema}:"
    roles = (item[len(prefix) :] for item in selectors if item.startswith(prefix))
    return tuple(sorted(role for role in roles if role != "*"))


def _selector_matches(selector: str, applied: frozenset[str]) -> bool:
    if selector.endswith(":*"):
        prefix = selector[:-1]
        return any(value.startswith(prefix) and len(value) > len(prefix) for value in applied)
    return selector in applied


_ANY_FIELDS = frozenset({"CaeVizFieldSelectionAPI:*"})

OPERATORS: tuple[OperatorInfo, ...] = (
    OperatorInfo(
        "CaeVizGlyphsAPI",
        "PointInstancer",
        frozenset({"CaeVizGlyphsAPI", "CaeVizDatasetSelectionAPI:source"}),
        _ANY_FIELDS | frozenset({"CaeVizFieldMappingAPI:scales"}),
    ),
    OperatorInfo(
        "CaeVizFacesAPI",
        "Mesh",
        frozenset({"CaeVizFacesAPI", "CaeVizDatasetSelectionAPI:source"}),
        _ANY_FIELDS,
    ),
    OperatorInfo(
        "CaeVizIsoSurfaceAPI",
        "Mesh",
        frozenset(
            {
                "CaeVizIsoSurfaceAPI",
                "CaeVizDatasetSelectionAPI:source",
                "CaeVizFieldSelectionAPI:contour",
                "CaeVizFieldSelectionAPI:colors",
            }
        ),
        _ANY_FIELDS,
    ),
    OperatorInfo(
        "CaeVizPlanarSliceAPI",
        "Mesh",
        frozenset({"CaeVizPlanarSliceAPI", "CaeVizDatasetSelectionAPI:source", "CaeVizFieldSelectionAPI:colors"}),
        _ANY_FIELDS,
    ),
    OperatorInfo(
        "CaeVizPointsAPI",
        "Points",
        frozenset({"CaeVizPointsAPI", "CaeVizDatasetSelectionAPI:source"}),
        _ANY_FIELDS,
    ),
    OperatorInfo(
        "CaeVizStreamlinesAPI",
        "BasisCurves",
        frozenset(
            {
                "CaeVizStreamlinesAPI",
                "CaeVizDatasetSelectionAPI:source",
                "CaeVizDatasetSelectionAPI:seeds",
                "CaeVizFieldSelectionAPI:velocities",
            }
        ),
        _ANY_FIELDS,
    ),
)
