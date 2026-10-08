# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Non-operator control prims: seed points and reference geometry."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import NamedTuple

from ._common import Color, define_typed_prim, edit_context, set_display_color
from ._errors import AuthoringError
from ._usd import require_usd

Vec3 = tuple[float, float, float]


class Bounds(NamedTuple):
    """Axis-aligned bounds used to size controls; typically a dataset's extent."""

    minimum: Vec3
    maximum: Vec3

    @property
    def center(self) -> Vec3:
        return tuple((lo + hi) / 2.0 for lo, hi in zip(self.minimum, self.maximum, strict=True))

    @property
    def diagonal(self) -> float:
        return math.sqrt(sum((hi - lo) ** 2 for lo, hi in zip(self.minimum, self.maximum, strict=True)))

    @property
    def is_empty(self) -> bool:
        return any(hi < lo for lo, hi in zip(self.minimum, self.maximum, strict=True))


def _bounds(value) -> Bounds:
    if isinstance(value, Bounds):
        bounds = value
    else:
        try:
            minimum, maximum = value
            bounds = Bounds(tuple(float(v) for v in minimum), tuple(float(v) for v in maximum))
        except (TypeError, ValueError) as exc:
            raise AuthoringError("bounds must be a Bounds or a (minimum, maximum) pair of 3-vectors") from exc
    if len(bounds.minimum) != 3 or len(bounds.maximum) != 3 or bounds.is_empty:
        raise AuthoringError(f"bounds must be non-empty 3-D bounds; got {value!r}")
    return bounds


def create_seed_points(stage, path, points: Sequence[Vec3], *, width: float | None = None, edit_target=None):
    """Define an invisible ``Points`` prim that Streamlines can target as seeds.

    Seeds are ordinary scene geometry; the recipe transform applies once.
    """

    pxr = require_usd()
    coordinates = [pxr.Gf.Vec3f(*(float(v) for v in point)) for point in points]
    if not coordinates:
        raise AuthoringError("seed points must not be empty")
    with edit_context(stage, edit_target):
        prim = define_typed_prim(stage, path, "Points")
        geom = pxr.UsdGeom.Points(prim)
        geom.CreatePointsAttr().Set(pxr.Vt.Vec3fArray(coordinates))
        if width is not None:
            geom.CreateWidthsAttr().Set([float(width)])
            geom.SetWidthsInterpolation(pxr.UsdGeom.Tokens.constant)
        geom.CreateExtentAttr().Set(pxr.UsdGeom.PointBased.ComputeExtent(pxr.Vt.Vec3fArray(coordinates)))
        geom.CreateVisibilityAttr().Set(pxr.UsdGeom.Tokens.invisible)
    return prim


def sphere_points(count: int, *, center: Vec3 = (0.0, 0.0, 0.0), radius: float = 1.0) -> list[Vec3]:
    """Return ``count`` evenly distributed points on a sphere (Fibonacci lattice)."""

    if count < 1:
        raise AuthoringError("count must be at least one")
    if radius <= 0.0:
        raise AuthoringError("radius must be positive")
    golden = math.pi * (3.0 - math.sqrt(5.0))
    points = []
    for i in range(count):
        y = 1.0 - 2.0 * (i + 0.5) / count
        ring = math.sqrt(max(0.0, 1.0 - y * y))
        theta = golden * i
        points.append(
            (
                center[0] + radius * ring * math.cos(theta),
                center[1] + radius * y,
                center[2] + radius * ring * math.sin(theta),
            )
        )
    return points


def create_seed_sphere(
    stage,
    path,
    *,
    count: int = 64,
    bounds=None,
    center: Vec3 | None = None,
    radius: float | None = None,
    edit_target=None,
):
    """Define seed points on a sphere, sized from ``bounds`` like Kit-CAE's seed sphere.

    With ``bounds`` the sphere is centered on the bounds and its radius is five percent
    of the half diagonal unless ``center`` or ``radius`` override it. Without bounds
    both ``center`` and ``radius`` are required.
    """

    if bounds is not None:
        box = _bounds(bounds)
        center = box.center if center is None else center
        radius = 0.05 * box.diagonal / 2.0 if radius is None else radius
    if center is None or radius is None:
        raise AuthoringError("create_seed_sphere needs bounds or both center and radius")
    points = sphere_points(count, center=tuple(float(v) for v in center), radius=float(radius))
    return create_seed_points(stage, path, points, edit_target=edit_target)


def create_bounding_box(stage, path, bounds, *, color: Color | None = (1.0, 1.0, 1.0), edit_target=None):
    """Define a ``BasisCurves`` wireframe of ``bounds``; a plain reference, not an operator."""

    pxr = require_usd()
    box = _bounds(bounds)
    lo, hi = box.minimum, box.maximum
    corners = [
        (lo[0], lo[1], lo[2]),
        (hi[0], lo[1], lo[2]),
        (hi[0], hi[1], lo[2]),
        (lo[0], hi[1], lo[2]),
        (lo[0], lo[1], hi[2]),
        (hi[0], lo[1], hi[2]),
        (hi[0], hi[1], hi[2]),
        (lo[0], hi[1], hi[2]),
    ]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    points = pxr.Vt.Vec3fArray([pxr.Gf.Vec3f(*corners[index]) for edge in edges for index in edge])
    with edit_context(stage, edit_target):
        prim = define_typed_prim(stage, path, "BasisCurves")
        curves = pxr.UsdGeom.BasisCurves(prim)
        curves.CreatePointsAttr().Set(points)
        curves.CreateCurveVertexCountsAttr().Set([2] * len(edges))
        curves.CreateTypeAttr().Set(pxr.UsdGeom.Tokens.linear)
        curves.CreateWrapAttr().Set(pxr.UsdGeom.Tokens.nonperiodic)
        curves.CreateWidthsAttr().Set([box.diagonal * 0.005])
        curves.SetWidthsInterpolation(pxr.UsdGeom.Tokens.constant)
        curves.CreateExtentAttr().Set([pxr.Gf.Vec3f(*lo), pxr.Gf.Vec3f(*hi)])
        set_display_color(prim, color)
    return prim
