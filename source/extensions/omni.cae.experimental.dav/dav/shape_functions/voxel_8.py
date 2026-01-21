# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
Voxel (8-node axis-aligned hexahedron) shape functions.

A voxel is a special case of hexahedron where edges are axis-aligned.
This allows for more efficient computation of parametric coordinates.

Node ordering follows VTK convention:
    4-------5
   /|      /|
  7-------6 |
  | |     | |
  | 0-----|-1
  |/      |/
  3-------2

Parametric coordinates: r, s, t ∈ [0, 1]
  Node 0: (0, 0, 0)
  Node 1: (1, 0, 0)
  Node 2: (1, 1, 0)
  Node 3: (0, 1, 0)
  Node 4: (0, 0, 1)
  Node 5: (1, 0, 1)
  Node 6: (1, 1, 1)
  Node 7: (0, 1, 1)
"""

import warp as wp

NUM_NODES = 8


@wp.func
def compute_shape_functions(pcoords: wp.vec3f) -> wp.vec(length=NUM_NODES, dtype=wp.float32):
    """
    Compute trilinear shape functions for voxel.

    Note: Shape functions are identical to hexahedron.

    Args:
        pcoords: Parametric coordinates (r, s, t) in [0, 1]³

    Returns:
        Shape function values at each of the 8 nodes
    """
    r = pcoords[0]
    s = pcoords[1]
    t = pcoords[2]

    rm = 1.0 - r
    sm = 1.0 - s
    tm = 1.0 - t

    weights = wp.vec(length=NUM_NODES, dtype=wp.float32)
    weights[0] = rm * sm * tm
    weights[1] = r * sm * tm
    weights[2] = r * s * tm
    weights[3] = rm * s * tm
    weights[4] = rm * sm * t
    weights[5] = r * sm * t
    weights[6] = r * s * t
    weights[7] = rm * s * t

    return weights


@wp.func
def compute_shape_derivatives(pcoords: wp.vec3f) -> wp.mat(shape=(NUM_NODES, 3), dtype=wp.float32):
    """
    Compute derivatives of shape functions with respect to parametric coordinates.

    Note: Derivatives are identical to hexahedron.

    Args:
        pcoords: Parametric coordinates (r, s, t) in [0, 1]³

    Returns:
        8x3 matrix where row i contains [dN_i/dr, dN_i/ds, dN_i/dt]
    """
    r = pcoords[0]
    s = pcoords[1]
    t = pcoords[2]

    rm = 1.0 - r
    sm = 1.0 - s
    tm = 1.0 - t

    derivs = wp.mat(shape=(NUM_NODES, 3), dtype=wp.float32)

    # Node 0: (0, 0, 0)
    derivs[0, 0] = -sm * tm
    derivs[0, 1] = -rm * tm
    derivs[0, 2] = -rm * sm

    # Node 1: (1, 0, 0)
    derivs[1, 0] = sm * tm
    derivs[1, 1] = -r * tm
    derivs[1, 2] = -r * sm

    # Node 2: (1, 1, 0)
    derivs[2, 0] = s * tm
    derivs[2, 1] = r * tm
    derivs[2, 2] = -r * s

    # Node 3: (0, 1, 0)
    derivs[3, 0] = -s * tm
    derivs[3, 1] = rm * tm
    derivs[3, 2] = -rm * s

    # Node 4: (0, 0, 1)
    derivs[4, 0] = -sm * t
    derivs[4, 1] = -rm * t
    derivs[4, 2] = rm * sm

    # Node 5: (1, 0, 1)
    derivs[5, 0] = sm * t
    derivs[5, 1] = -r * t
    derivs[5, 2] = r * sm

    # Node 6: (1, 1, 1)
    derivs[6, 0] = s * t
    derivs[6, 1] = r * t
    derivs[6, 2] = r * s

    # Node 7: (0, 1, 1)
    derivs[7, 0] = -s * t
    derivs[7, 1] = rm * t
    derivs[7, 2] = rm * s

    return derivs


@wp.func
def compute_parametric_coordinates(
    p0: wp.vec3f,
    p1: wp.vec3f,
    p2: wp.vec3f,
    p3: wp.vec3f,
    p4: wp.vec3f,
    p5: wp.vec3f,
    p6: wp.vec3f,
    p7: wp.vec3f,
    pos: wp.vec3f,
) -> wp.vec3f:
    """
    Compute parametric coordinates for a point in axis-aligned voxel.

    Since voxel is axis-aligned, we can compute parametric coordinates directly
    without iteration.

    Args:
        p0-p7: The 8 node positions in world coordinates
        pos: World position to find parametric coordinates for

    Returns:
        Parametric coordinates (r, s, t)
    """
    # For axis-aligned voxel, we can compute parametric coords directly
    # by linear interpolation along each axis

    # Get min and max corners
    min_pt = p0
    max_pt = p6

    # Handle degenerate cases
    dx = max_pt[0] - min_pt[0]
    dy = max_pt[1] - min_pt[1]
    dz = max_pt[2] - min_pt[2]

    r = 0.0 if wp.abs(dx) < 1.0e-20 else (pos[0] - min_pt[0]) / dx
    s = 0.0 if wp.abs(dy) < 1.0e-20 else (pos[1] - min_pt[1]) / dy
    t = 0.0 if wp.abs(dz) < 1.0e-20 else (pos[2] - min_pt[2]) / dz

    return wp.vec3f(r, s, t)


@wp.func
def is_inside(pcoords: wp.vec3f, tolerance: float = 1.0e-6) -> bool:
    """
    Check if parametric coordinates are inside the voxel.

    Args:
        pcoords: Parametric coordinates (r, s, t)
        tolerance: Tolerance for boundary checking

    Returns:
        True if inside, False otherwise
    """
    return (
        pcoords[0] >= -tolerance
        and pcoords[0] <= 1.0 + tolerance
        and pcoords[1] >= -tolerance
        and pcoords[1] <= 1.0 + tolerance
        and pcoords[2] >= -tolerance
        and pcoords[2] <= 1.0 + tolerance
    )
