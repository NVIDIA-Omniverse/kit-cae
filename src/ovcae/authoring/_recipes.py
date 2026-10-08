# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Recipe builders: one function per built-in ovcae operator.

Each builder defines the typed recipe prim ovcae expects, authors renderer-safe
placeholder geometry, applies the operator and role API schemas, and writes only
the values the caller supplied. Schema fallbacks cover everything else. Builders
return the prim and raise :class:`AuthoringError` instead of logging.

Builders also author the MDL material scaffolding Kit-CAE's create commands
produce (see :mod:`._materials`): ``colormap`` names a bundled colormap or an
asset path, and ``color_range`` optionally sets the shader ``domain`` so colors
show before a runtime rescale exists. Pass ``materials=False`` to skip all of it.
"""

from __future__ import annotations

from collections.abc import Sequence

from . import schemas as S
from ._common import (
    Color,
    define_typed_prim,
    edit_context,
    resolve_prim,
    select_dataset,
    select_field,
    set_display_color,
    set_optional,
)
from ._errors import AuthoringError
from ._materials import DEFAULT_COLORMAP, Range, setup_scalar_materials, setup_width_mapping
from ._usd import require_usd

Vec3 = tuple[float, float, float]

_STREAMLINE_PLACEHOLDER = [(0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.2, 0.0, 0.0), (0.3, 0.0, 0.0)]


def _begin_recipe(stage, path, type_name: str, operator_api, *, source, identifier: str, enabled: bool):
    """Define the recipe prim, apply the operator schemas, and select the source dataset.

    Returns the prim and the applied ``operator_api`` wrapper for the builder's settings.
    """

    # Authoring knows no dataset representation; whether the reader can resolve
    # this prim is decided at evaluation time. Only its existence is checked here.
    source_prim = resolve_prim(stage, source, what="source dataset")
    prim = define_typed_prim(stage, path, type_name)
    S.OperatorAPI.apply(prim).enabled = bool(enabled)
    recipe = operator_api.apply(prim)
    select_dataset(prim, "source", source_prim, identifier=identifier)
    return prim, recipe


def _empty_mesh(prim) -> None:
    pxr = require_usd()
    mesh = pxr.UsdGeom.Mesh(prim)
    mesh.CreatePointsAttr().Set(pxr.Vt.Vec3fArray())
    mesh.CreateFaceVertexCountsAttr().Set(pxr.Vt.IntArray())
    mesh.CreateFaceVertexIndicesAttr().Set(pxr.Vt.IntArray())
    mesh.CreateSubdivisionSchemeAttr().Set(pxr.UsdGeom.Tokens.none)


def create_faces(
    stage,
    path,
    *,
    source,
    identifier: str = "",
    colors: str | Sequence[str] | None = None,
    external_only: bool | None = None,
    enabled: bool = True,
    display_color: Color | None = (0.8, 0.8, 0.8),
    materials: bool = True,
    colormap: str = DEFAULT_COLORMAP,
    color_range: Range | None = None,
    edit_target=None,
):
    """Define a ``Mesh`` recipe with ``CaeVizFacesAPI`` extracting dataset faces.

    ``colors`` optionally selects a field into the ``colors`` role, published as a
    primvar of the same name.
    """

    with edit_context(stage, edit_target):
        prim, faces = _begin_recipe(
            stage, path, "Mesh", S.FacesAPI, source=source, identifier=identifier, enabled=enabled
        )
        _empty_mesh(prim)
        set_optional(faces, external_only=external_only)
        if colors is not None:
            select_field(prim, "colors", colors)
        set_display_color(prim, display_color)
        if materials:
            setup_scalar_materials(prim, colormap=colormap, color_range=color_range)
    return prim


def create_iso_surface(
    stage,
    path,
    *,
    source,
    contour: str,
    iso_value: float,
    identifier: str = "",
    colors: str | Sequence[str] | None = None,
    enabled: bool = True,
    display_color: Color | None = (0.8, 0.8, 0.8),
    materials: bool = True,
    colormap: str = DEFAULT_COLORMAP,
    color_range: Range | None = None,
    edit_target=None,
):
    """Define a ``Mesh`` recipe with ``CaeVizIsoSurfaceAPI`` contouring ``contour`` at ``iso_value``.

    The ``colors`` role is required by the operator; it defaults to the contour field.
    """

    with edit_context(stage, edit_target):
        prim, iso = _begin_recipe(
            stage, path, "Mesh", S.IsoSurfaceAPI, source=source, identifier=identifier, enabled=enabled
        )
        _empty_mesh(prim)
        iso.iso_value = float(iso_value)
        select_field(prim, "contour", contour)
        select_field(prim, "colors", contour if colors is None else colors)
        set_display_color(prim, display_color)
        if materials:
            setup_scalar_materials(prim, colormap=colormap, color_range=color_range)
    return prim


def create_planar_slice(
    stage,
    path,
    *,
    source,
    colors: str | Sequence[str],
    identifier: str = "",
    mode: str | None = None,
    origin: Vec3 | None = None,
    rotation_xyz: Vec3 | None = None,
    enabled: bool = True,
    display_color: Color | None = (0.8, 0.8, 0.8),
    materials: bool = True,
    colormap: str = DEFAULT_COLORMAP,
    color_range: Range | None = None,
    edit_target=None,
):
    """Define a ``Mesh`` recipe with ``CaeVizPlanarSliceAPI``.

    The prim's translation is a point on every plane and, in ``free`` mode, its
    rotated local +Y axis is the plane normal. The builder always authors translate,
    rotateXYZ, and scale ops so applications can manipulate the plane later.
    ``colors`` is required by the operator.
    """

    pxr = require_usd()
    with edit_context(stage, edit_target):
        prim, plane = _begin_recipe(
            stage, path, "Mesh", S.PlanarSliceAPI, source=source, identifier=identifier, enabled=enabled
        )
        _empty_mesh(prim)
        xformable = pxr.UsdGeom.Xformable(prim)
        xformable.ClearXformOpOrder()
        xformable.AddTranslateOp().Set(pxr.Gf.Vec3d(*(origin or (0.0, 0.0, 0.0))))
        xformable.AddRotateXYZOp().Set(pxr.Gf.Vec3f(*(rotation_xyz or (0.0, 0.0, 0.0))))
        xformable.AddScaleOp().Set(pxr.Gf.Vec3f(1.0, 1.0, 1.0))
        set_optional(plane, mode=mode)
        select_field(prim, "colors", colors)
        set_display_color(prim, display_color)
        if materials:
            # Kit-CAE's slice uses the unlit export and only rescales its domain.
            setup_scalar_materials(
                prim,
                sub_identifiers=("UnlitScalarColor",),
                bound="UnlitScalarColor",
                colormap=colormap,
                color_range=color_range,
                enable_includes_for=(),
            )
    return prim


def create_points(
    stage,
    path,
    *,
    source,
    identifier: str = "",
    colors: str | Sequence[str] | None = None,
    widths: str | Sequence[str] | None = None,
    max_count: int | None = None,
    width: float | None = None,
    use_cell_points: bool | None = None,
    enabled: bool = True,
    display_color: Color | None = (0.8, 0.8, 0.8),
    materials: bool = True,
    colormap: str = DEFAULT_COLORMAP,
    color_range: Range | None = None,
    edit_target=None,
):
    """Define a ``Points`` recipe with ``CaeVizPointsAPI`` publishing dataset nodes.

    ``widths`` optionally selects a field into the ``widths`` role; ``width`` is the
    constant fallback width.
    """

    pxr = require_usd()
    with edit_context(stage, edit_target):
        prim, recipe = _begin_recipe(
            stage, path, "Points", S.PointsAPI, source=source, identifier=identifier, enabled=enabled
        )
        points = pxr.UsdGeom.Points(prim)
        points.CreatePointsAttr().Set(pxr.Vt.Vec3fArray())
        points.CreateWidthsAttr().Set(pxr.Vt.FloatArray())
        points.SetWidthsInterpolation(pxr.UsdGeom.Tokens.constant)
        set_optional(recipe, max_count=max_count, width=width, use_cell_points=use_cell_points)
        if colors is not None:
            select_field(prim, "colors", colors)
        if widths is not None:
            select_field(prim, "widths", widths)
        set_display_color(prim, display_color)
        if materials:
            setup_scalar_materials(prim, colormap=colormap, color_range=color_range)
            setup_width_mapping(prim, width=width)
    return prim


def create_streamlines(
    stage,
    path,
    *,
    source,
    seeds,
    velocities: str | Sequence[str],
    identifier: str = "",
    colors: str | Sequence[str] | None = None,
    direction: str | None = None,
    max_steps: int | None = None,
    initial_step_size: float | None = None,
    min_step_size: float | None = None,
    max_step_size: float | None = None,
    tolerance: float | None = None,
    width: float | None = None,
    enabled: bool = True,
    display_color: Color | None = (0.8, 0.8, 0.8),
    materials: bool = True,
    colormap: str = DEFAULT_COLORMAP,
    color_range: Range | None = None,
    edit_target=None,
):
    """Define a ``BasisCurves`` recipe with ``CaeVizStreamlinesAPI``.

    ``seeds`` is a ``UsdGeomPointBased`` prim or path, such as ``Points`` from
    :func:`create_seed_points` or any ``Mesh`` whose points should seed the curves.
    ``velocities`` names the vector field, either one multi-component field or the
    component scalars in order. The role keeps the schema's ``unchanged`` mode so
    components are preserved.
    """

    pxr = require_usd()
    with edit_context(stage, edit_target):
        seeds_prim = resolve_prim(stage, seeds, what="seeds")
        if not seeds_prim.IsA(pxr.UsdGeom.PointBased):
            raise AuthoringError(
                f"seeds {seeds_prim.GetPath()} must be a UsdGeomPointBased prim; got {seeds_prim.GetTypeName()!r}"
            )
        prim, recipe = _begin_recipe(
            stage,
            path,
            "BasisCurves",
            S.StreamlinesAPI,
            source=source,
            identifier=identifier,
            enabled=enabled,
        )
        curves = pxr.UsdGeom.BasisCurves(prim)
        curves.CreatePointsAttr().Set(pxr.Vt.Vec3fArray([pxr.Gf.Vec3f(*p) for p in _STREAMLINE_PLACEHOLDER]))
        curves.CreateCurveVertexCountsAttr().Set([len(_STREAMLINE_PLACEHOLDER)])
        curves.CreateTypeAttr().Set(pxr.UsdGeom.Tokens.linear)
        curves.CreateWrapAttr().Set(pxr.UsdGeom.Tokens.nonperiodic)
        curves.CreateWidthsAttr().Set([0.01 if width is None else float(width)])
        curves.SetWidthsInterpolation(pxr.UsdGeom.Tokens.constant)
        select_dataset(prim, "seeds", seeds_prim)
        select_field(prim, "velocities", velocities)
        if colors is not None:
            select_field(prim, "colors", colors)
        set_optional(
            recipe,
            direction=direction,
            max_steps=max_steps,
            initial_step_size=initial_step_size,
            min_step_size=min_step_size,
            max_step_size=max_step_size,
            tolerance=tolerance,
            width=width,
        )
        set_display_color(prim, display_color)
        if materials:
            # Kit-CAE binds ScalarColor and also authors AnimatedStreaks, whose
            # enable_coloring is the one toggled when a colors field is present.
            setup_scalar_materials(
                prim,
                sub_identifiers=("ScalarColor", "AnimatedStreaks"),
                bound="ScalarColor",
                colormap=colormap,
                color_range=color_range,
                enable_includes_for=("AnimatedStreaks",),
            )
            setup_width_mapping(prim)
    return prim
