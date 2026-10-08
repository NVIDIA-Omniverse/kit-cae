# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Private source-neutral visualization products."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Primvar:
    name: str
    values: object
    interpolation: str


@dataclass(frozen=True)
class MeshProduct:
    points: object
    face_vertex_counts: object
    face_vertex_indices: object
    primvars: tuple[Primvar, ...] = ()


@dataclass(frozen=True)
class CurvesProduct:
    points: object
    curve_vertex_counts: object
    widths: Primvar
    primvars: tuple[Primvar, ...] = ()
    curve_type: str = "linear"
    basis: str | None = None
    wrap: str = "nonperiodic"


@dataclass(frozen=True)
class PointsProduct:
    points: object
    widths: Primvar
    primvars: tuple[Primvar, ...] = ()


@dataclass(frozen=True)
class InstancerProduct:
    """Instance arrays only; prototype hierarchy and relationships belong to the application."""

    positions: object
    proto_indices: object
    scales: object
    orientations: object
    primvars: tuple[Primvar, ...] = ()


@dataclass(frozen=True)
class EmptyProduct:
    reason: str
