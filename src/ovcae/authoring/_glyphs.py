# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Glyph recipe and application-owned prototype authoring."""

from collections.abc import Sequence

from . import schemas as S
from ._common import Color, edit_context, resolve_prim, select_field, set_display_color, set_optional
from ._errors import AuthoringError
from ._materials import DEFAULT_COLORMAP, Range, setup_scalar_materials
from ._recipes import _begin_recipe
from ._usd import require_usd


def create_glyphs(
    stage,
    path,
    *,
    source,
    identifier: str = "",
    shape: str = "Sphere",
    prototype=None,
    colors: str | Sequence[str] | None = None,
    scales: str | Sequence[str] | None = None,
    orientations: str | Sequence[str] | None = None,
    orientations_mode: str | None = None,
    max_count: int | None = None,
    scale: float | None = None,
    use_cell_points: bool | None = None,
    enabled: bool = True,
    display_color: Color | None = (0.8, 0.8, 0.8),
    materials: bool = True,
    colormap: str = DEFAULT_COLORMAP,
    color_range: Range | None = None,
    edit_target=None,
):
    """Define a PointInstancer recipe with one Sphere, Cone, Arrow, or Custom prototype.

    ``prototype`` is a source prim/path required for ``shape="Custom"``; it is
    internally referenced under the recipe's prototype hierarchy. Arrows extend
    from the origin to +X; spheres have radius 0.5 and cones have height 1 along X.
    Use ``use_cell_points=False`` for point clouds. All instance arrays start empty.

    ``scales`` selects a scalar or three-component field, replacing the constant
    ``scale``. Orientations use xyzw quaternions in ``quaternion`` mode. The Kit
    compatibility token ``eulerAngles`` means direction vectors from +X, not angles.
    The application owns prototypes and must repopulate USD after hierarchy edits.
    """
    pxr = require_usd()
    if shape not in {"Sphere", "Cone", "Arrow", "Custom"}:
        raise AuthoringError(f"unsupported glyph shape {shape!r}")
    if shape != "Custom" and prototype is not None:
        raise AuthoringError('prototype requires shape="Custom"')
    template = resolve_prim(stage, prototype, what="glyph prototype") if shape == "Custom" else None
    destination = pxr.Sdf.Path(str(path))
    if template is not None and (
        destination.HasPrefix(template.GetPath()) or template.GetPath().HasPrefix(destination)
    ):
        raise AuthoringError("custom prototype must be outside the glyph recipe hierarchy")
    with edit_context(stage, edit_target):
        prim, recipe = _begin_recipe(
            stage, path, "PointInstancer", S.GlyphsAPI, source=source, identifier=identifier, enabled=enabled
        )
        instancer = pxr.UsdGeom.PointInstancer(prim)
        instancer.CreatePositionsAttr().Set(pxr.Vt.Vec3fArray())
        instancer.CreateProtoIndicesAttr().Set(pxr.Vt.IntArray())
        instancer.CreateScalesAttr().Set(pxr.Vt.Vec3fArray())
        instancer.CreateOrientationsAttr().Set(pxr.Vt.QuathArray())
        # Keep templates in a dedicated hierarchy. USD population follows the
        # prototypes relationship and expands composite shapes into renderer leaves.
        parent = stage.OverridePrim(prim.GetPath().AppendChild("Prototypes"))
        root_path = parent.GetPath().AppendChild("Xform")
        root = pxr.UsdGeom.Xform.Define(stage, root_path)
        if shape == "Custom":
            child = stage.DefinePrim(root.GetPath().AppendChild(template.GetName()))
            child.GetReferences().AddInternalReference(template.GetPath())
        elif shape == "Sphere":
            pxr.UsdGeom.Sphere.Define(stage, root.GetPath().AppendChild("Sphere")).CreateRadiusAttr().Set(0.5)
        else:
            cone = pxr.UsdGeom.Cone.Define(stage, root.GetPath().AppendChild("Cone"))
            cone.CreateAxisAttr().Set(pxr.UsdGeom.Tokens.x)
            cone.CreateHeightAttr().Set(0.5 if shape == "Arrow" else 1.0)
            cone.CreateRadiusAttr().Set(0.3 if shape == "Arrow" else 0.5)
            if shape == "Arrow":
                cone.AddTranslateOp().Set((0.75, 0.0, 0.0))
                cylinder = pxr.UsdGeom.Cylinder.Define(stage, root.GetPath().AppendChild("Cylinder"))
                cylinder.CreateAxisAttr().Set(pxr.UsdGeom.Tokens.x)
                cylinder.CreateHeightAttr().Set(0.5)
                cylinder.CreateRadiusAttr().Set(0.15)
                cylinder.AddTranslateOp().Set((0.25, 0.0, 0.0))
        instancer.CreatePrototypesRel().SetTargets([root.GetPath()])
        set_optional(
            recipe,
            max_count=max_count,
            scale=scale,
            use_cell_points=use_cell_points,
            orientations_mode=orientations_mode,
        )
        for role, fields in (("colors", colors), ("scales", scales), ("orientations", orientations)):
            if fields is not None:
                select_field(prim, role, fields)
        set_display_color(prim, display_color)
        if materials:
            setup_scalar_materials(prim, colormap=colormap, color_range=color_range)
    return prim
