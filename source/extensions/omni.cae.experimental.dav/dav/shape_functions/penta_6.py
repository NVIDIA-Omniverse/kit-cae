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
Pentahedron/Prism (6-node triangular prism) shape functions.

Node ordering follows VTK convention:
     5
    /|\\
   / | \\
  3-----4
  |  |  |
  |  2  |
  | / \\ |
  |/   \\|
  0-----1

Parametric coordinates: r, s, t where:
  - r, s are triangular coordinates (r, s >= 0, r + s <= 1) for the triangular base
  - t ∈ [0, 1] is the extrusion direction

  Node 0: (0, 0, 0)  - bottom triangle
  Node 1: (1, 0, 0)  - bottom triangle
  Node 2: (0, 1, 0)  - bottom triangle
  Node 3: (0, 0, 1)  - top triangle
  Node 4: (1, 0, 1)  - top triangle
  Node 5: (0, 1, 1)  - top triangle
"""

import warp as wp

NUM_NODES = 6


@wp.func
def compute_shape_functions(pcoords: wp.vec3f) -> wp.vec(length=NUM_NODES, dtype=wp.float32):
    """
    Compute shape functions for pentahedron (triangular prism).

    Args:
        pcoords: Parametric coordinates (r, s, t)

    Returns:
        Shape function values at each of the 6 nodes
    """
    r = pcoords[0]
    s = pcoords[1]
    t = pcoords[2]

    # Triangular coordinates for base
    w = 1.0 - r - s

    weights = wp.vec(length=NUM_NODES, dtype=wp.float32)

    # Bottom triangle (t = 0)
    weights[0] = w * (1.0 - t)
    weights[1] = r * (1.0 - t)
    weights[2] = s * (1.0 - t)

    # Top triangle (t = 1)
    weights[3] = w * t
    weights[4] = r * t
    weights[5] = s * t

    return weights


@wp.func
def compute_shape_derivatives(pcoords: wp.vec3f) -> wp.mat(shape=(NUM_NODES, 3), dtype=wp.float32):
    """
    Compute derivatives of shape functions with respect to parametric coordinates.

    Args:
        pcoords: Parametric coordinates (r, s, t)

    Returns:
        6x3 matrix where row i contains [dN_i/dr, dN_i/ds, dN_i/dt]
    """
    r = pcoords[0]
    s = pcoords[1]
    t = pcoords[2]

    tm = 1.0 - t

    derivs = wp.mat(shape=(NUM_NODES, 3), dtype=wp.float32)

    # Bottom triangle nodes
    # Node 0: (0, 0, 0) - w * (1-t)
    derivs[0, 0] = -tm
    derivs[0, 1] = -tm
    derivs[0, 2] = -(1.0 - r - s)

    # Node 1: (1, 0, 0) - r * (1-t)
    derivs[1, 0] = tm
    derivs[1, 1] = 0.0
    derivs[1, 2] = -r

    # Node 2: (0, 1, 0) - s * (1-t)
    derivs[2, 0] = 0.0
    derivs[2, 1] = tm
    derivs[2, 2] = -s

    # Top triangle nodes
    # Node 3: (0, 0, 1) - w * t
    derivs[3, 0] = -t
    derivs[3, 1] = -t
    derivs[3, 2] = 1.0 - r - s

    # Node 4: (1, 0, 1) - r * t
    derivs[4, 0] = t
    derivs[4, 1] = 0.0
    derivs[4, 2] = r

    # Node 5: (0, 1, 1) - s * t
    derivs[5, 0] = 0.0
    derivs[5, 1] = t
    derivs[5, 2] = s

    return derivs


@wp.func
def compute_parametric_coordinates(
    p0: wp.vec3f,
    p1: wp.vec3f,
    p2: wp.vec3f,
    p3: wp.vec3f,
    p4: wp.vec3f,
    p5: wp.vec3f,
    pos: wp.vec3f,
    tolerance: float = 1.0e-5,
    max_iterations: int = 10,
) -> wp.vec3f:
    """
    Compute parametric coordinates for a point in pentahedron using Newton-Raphson iteration.

    Args:
        p0-p5: The 6 node positions in world coordinates
        pos: World position to find parametric coordinates for
        tolerance: Convergence tolerance
        max_iterations: Maximum number of iterations

    Returns:
        Parametric coordinates (r, s, t)
    """
    # Initial guess: center of element
    pcoords = wp.vec3f(0.333, 0.333, 0.5)

    for _ in range(max_iterations):
        # Evaluate shape functions and derivatives
        weights = compute_shape_functions(pcoords)
        derivs = compute_shape_derivatives(pcoords)

        # Store points in a matrix (6 rows x 3 columns)
        # Each row represents a point's (x, y, z) coordinates
        points = wp.mat(shape=(NUM_NODES, 3), dtype=wp.float32)
        points[0, 0] = p0[0]
        points[0, 1] = p0[1]
        points[0, 2] = p0[2]
        points[1, 0] = p1[0]
        points[1, 1] = p1[1]
        points[1, 2] = p1[2]
        points[2, 0] = p2[0]
        points[2, 1] = p2[1]
        points[2, 2] = p2[2]
        points[3, 0] = p3[0]
        points[3, 1] = p3[1]
        points[3, 2] = p3[2]
        points[4, 0] = p4[0]
        points[4, 1] = p4[1]
        points[4, 2] = p4[2]
        points[5, 0] = p5[0]
        points[5, 1] = p5[1]
        points[5, 2] = p5[2]

        # Compute current position and Jacobian
        current_pos = wp.vec3f(0.0, 0.0, 0.0)
        jacobian = wp.mat(shape=(3, 3), dtype=wp.float32)

        for i in range(NUM_NODES):
            # Extract point i as a vec3f
            pt = wp.vec3f(points[i, 0], points[i, 1], points[i, 2])
            current_pos = current_pos + weights[i] * pt
            for j in range(3):
                for k in range(3):
                    jacobian[j, k] = jacobian[j, k] + derivs[i, k] * points[i, j]

        # Compute residual
        residual = pos - current_pos

        # Check convergence
        if wp.length(residual) < tolerance:
            break

        # Solve Jacobian * delta = residual for delta
        det = (
            jacobian[0, 0] * (jacobian[1, 1] * jacobian[2, 2] - jacobian[1, 2] * jacobian[2, 1])
            - jacobian[0, 1] * (jacobian[1, 0] * jacobian[2, 2] - jacobian[1, 2] * jacobian[2, 0])
            + jacobian[0, 2] * (jacobian[1, 0] * jacobian[2, 1] - jacobian[1, 1] * jacobian[2, 0])
        )

        if wp.abs(det) < 1.0e-20:
            break

        # Inverse of 3x3 matrix using Cramer's rule
        inv_det = 1.0 / det

        delta_r = inv_det * (
            residual[0] * (jacobian[1, 1] * jacobian[2, 2] - jacobian[1, 2] * jacobian[2, 1])
            - residual[1] * (jacobian[0, 1] * jacobian[2, 2] - jacobian[0, 2] * jacobian[2, 1])
            + residual[2] * (jacobian[0, 1] * jacobian[1, 2] - jacobian[0, 2] * jacobian[1, 1])
        )

        delta_s = inv_det * (
            residual[0] * (jacobian[1, 2] * jacobian[2, 0] - jacobian[1, 0] * jacobian[2, 2])
            - residual[1] * (jacobian[0, 2] * jacobian[2, 0] - jacobian[0, 0] * jacobian[2, 2])
            + residual[2] * (jacobian[0, 2] * jacobian[1, 0] - jacobian[0, 0] * jacobian[1, 2])
        )

        delta_t = inv_det * (
            residual[0] * (jacobian[1, 0] * jacobian[2, 1] - jacobian[1, 1] * jacobian[2, 0])
            - residual[1] * (jacobian[0, 0] * jacobian[2, 1] - jacobian[0, 1] * jacobian[2, 0])
            + residual[2] * (jacobian[0, 0] * jacobian[1, 1] - jacobian[0, 1] * jacobian[1, 0])
        )

        # Update parametric coordinates
        pcoords = wp.vec3f(pcoords[0] + delta_r, pcoords[1] + delta_s, pcoords[2] + delta_t)

    return pcoords


@wp.func
def is_inside(pcoords: wp.vec3f, tolerance: float = 1.0e-6) -> bool:
    """
    Check if parametric coordinates are inside the pentahedron.

    For a pentahedron, we need:
      - r >= 0, s >= 0, r + s <= 1 (inside triangular base)
      - t ∈ [0, 1] (inside extrusion)

    Args:
        pcoords: Parametric coordinates (r, s, t)
        tolerance: Tolerance for boundary checking

    Returns:
        True if inside, False otherwise
    """
    r = pcoords[0]
    s = pcoords[1]
    t = pcoords[2]

    return (
        r >= -tolerance and s >= -tolerance and (r + s) <= 1.0 + tolerance and t >= -tolerance and t <= 1.0 + tolerance
    )
