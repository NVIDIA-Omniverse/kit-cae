# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Discover explicit recipe dependencies without loading execution-only values."""

from ._schemas import PropertyInfo, applied_properties
from ._views import PrimView, StageView

# Concrete UsdGeomPointBased schemas: every one carries a ``points`` attribute,
# so any of them can serve as a captured point-cloud input.
POINT_BASED_TYPES = frozenset(
    {"Points", "Mesh", "BasisCurves", "NurbsCurves", "HermiteCurves", "NurbsPatch", "TetMesh"}
)


def plan_operator(stage, *, operator_path, applied_api_schemas, input_ordinal, paths=None):
    """Build an allowed input scope from schemas and relationships.

    Only metadata needed to discover dependencies is read: prim type/schemas,
    and relationship targets. Recipe settings, transform values, and
    scene point arrays are loaded later, when requested through the view.
    Property definitions come from usd-core's schema registry, so a custom
    operator's schema is captured once its plug-in is registered.
    """
    view = StageView(stage, input_ordinal, paths=paths)
    try:
        definitions, view.unknown_schemas = applied_properties(applied_api_schemas)
        dependencies = {}

        def define(path, type_name, schemas, properties):
            dependencies[path] = tuple(sorted(set(properties) | {"usd-prim-type", "usd-schemas"}))
            view._prims[path] = PrimView(view, path, str(type_name or ""), schemas, properties)

        def read(path, name):
            return view._read(path, name)

        type_name = read(operator_path, "usd-prim-type")
        if "CaeVizPlanarSliceAPI" in applied_api_schemas:
            definitions.update(_transform_definitions())
        define(operator_path, type_name, sorted(applied_api_schemas), definitions)
        prim = view.get_prim_at_path(operator_path)
        targets = {target for name in prim.relationship_names() for target in prim.get_relationship(name).targets}
        for target in sorted(targets - {operator_path}):
            target_type = read(target, "usd-prim-type")
            target_schemas = read(target, "usd-schemas")
            properties = {}
            if target_type in POINT_BASED_TYPES:
                properties["points"] = PropertyInfo("points", "attribute", "point3f[]")
                properties.update(_transform_definitions())
            define(target, target_type, target_schemas or (), properties)
            if target_type is None and target_schemas is None:
                view._prims[target] = None
        view.dependencies = tuple(sorted(dependencies.items()))
        return view
    except BaseException:
        view._expire()
        raise


def _transform_definitions():
    """Track the native composed local transform, including when currently absent."""
    return {"omni:xform": PropertyInfo("omni:xform", "attribute", "matrix4d")}
