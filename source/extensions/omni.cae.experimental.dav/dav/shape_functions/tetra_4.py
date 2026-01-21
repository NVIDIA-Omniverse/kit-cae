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
Tetrahedron (4-node) shape functions.

Node ordering follows VTK convention:
       3
      /|\\
     / | \\
    /  |  \\
   /   |   \\
  0----|----2
   \\   |   /
    \\  |  /
     \\ | /
      \\|/
       1

Parametric coordinates use barycentric coordinates: r, s, t
where the fourth coordinate is implicit: 1 - r - s - t

  Node 0: r=0, s=0, t=0 (implicit w=1)
  Node 1: r=1, s=0, t=0 (implicit w=0)
  Node 2: r=0, s=1, t=0 (implicit w=0)
  Node 3: r=0, s=0, t=1 (implicit w=0)
"""

import warp as wp

NUM_NODES = 4


@wp.func
def compute_shape_functions(pcoords: wp.vec3f) -> wp.vec(length=NUM_NODES, dtype=wp.float32):
    """
    Compute barycentric shape functions for tetrahedron.

    Args:
        pcoords: Parametric coordinates (r, s, t) - barycentric coordinates

    Returns:
        Shape function values at each of the 4 nodes
    """
    r = pcoords[0]
    s = pcoords[1]
    t = pcoords[2]

    weights = wp.vec(length=NUM_NODES, dtype=wp.float32)
    weights[0] = 1.0 - r - s - t  # Node 0
    weights[1] = r  # Node 1
    weights[2] = s  # Node 2
    weights[3] = t  # Node 3

    return weights


@wp.func
def compute_shape_derivatives(pcoords: wp.vec3f) -> wp.mat(shape=(NUM_NODES, 3), dtype=wp.float32):
    """
    Compute derivatives of shape functions with respect to parametric coordinates.

    For tetrahedron, the derivatives are constant (linear shape functions).

    Args:
        pcoords: Parametric coordinates (r, s, t) - not used for linear tet

    Returns:
        4x3 matrix where row i contains [dN_i/dr, dN_i/ds, dN_i/dt]
    """
    derivs = wp.mat(shape=(NUM_NODES, 3), dtype=wp.float32)

    # Node 0: N0 = 1 - r - s - t
    derivs[0, 0] = -1.0
    derivs[0, 1] = -1.0
    derivs[0, 2] = -1.0

    # Node 1: N1 = r
    derivs[1, 0] = 1.0
    derivs[1, 1] = 0.0
    derivs[1, 2] = 0.0

    # Node 2: N2 = s
    derivs[2, 0] = 0.0
    derivs[2, 1] = 1.0
    derivs[2, 2] = 0.0

    # Node 3: N3 = t
    derivs[3, 0] = 0.0
    derivs[3, 1] = 0.0
    derivs[3, 2] = 1.0

    return derivs


@wp.func
def compute_parametric_coordinates(p0: wp.vec3f, p1: wp.vec3f, p2: wp.vec3f, p3: wp.vec3f, pos: wp.vec3f) -> wp.vec3f:
    """
    Compute barycentric coordinates for a point in tetrahedron.

    This uses a direct analytical solution based on volume ratios.

    Args:
        p0, p1, p2, p3: The 4 node positions in world coordinates
        pos: World position to find parametric coordinates for

    Returns:
        Parametric coordinates (r, s, t) - barycentric coordinates
    """
    # Vectors from p0 to other vertices
    v1 = p1 - p0
    v2 = p2 - p0
    v3 = p3 - p0
    vp = pos - p0

    # Solve linear system using Cramer's rule
    # [v1 v2 v3] * [r s t]^T = vp

    # Compute determinant (6 * volume of tetrahedron)
    det = wp.dot(v1, wp.cross(v2, v3))

    if wp.abs(det) < 1.0e-20:
        # Degenerate tetrahedron
        return wp.vec3f(0.0, 0.0, 0.0)

    inv_det = 1.0 / det

    # Barycentric coordinates using volume ratios
    r = wp.dot(vp, wp.cross(v2, v3)) * inv_det
    s = wp.dot(v1, wp.cross(vp, v3)) * inv_det
    t = wp.dot(v1, wp.cross(v2, vp)) * inv_det

    return wp.vec3f(r, s, t)


@wp.func
def is_inside(pcoords: wp.vec3f, tolerance: float = 1.0e-6) -> bool:
    """
    Check if barycentric coordinates are inside the tetrahedron.

    Args:
        pcoords: Parametric coordinates (r, s, t)
        tolerance: Tolerance for boundary checking

    Returns:
        True if inside, False otherwise
    """
    r = pcoords[0]
    s = pcoords[1]
    t = pcoords[2]
    w = 1.0 - r - s - t

    return r >= -tolerance and s >= -tolerance and t >= -tolerance and w >= -tolerance
