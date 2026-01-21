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
Pyramid (5-node) shape functions.

Node ordering follows VTK convention:
        4 (apex)
       /|\\
      / | \\
     /  |  \\
    /   |   \\
   /    |    \\
  0-----3-----
  |           |
  |           |
  1-----------2

Parametric coordinates: r, s, t where t ∈ [0, 1] is height
  Node 0: (0, 0, 0)  - base corner
  Node 1: (1, 0, 0)  - base corner
  Node 2: (1, 1, 0)  - base corner
  Node 3: (0, 1, 0)  - base corner
  Node 4: (0.5, 0.5, 1) - apex

The parametric coordinates at apex: r, s can be anything, t = 1
"""

import warp as wp

NUM_NODES = 5


@wp.func
def compute_shape_functions(pcoords: wp.vec3f) -> wp.vec(length=NUM_NODES, dtype=wp.float32):
    """
    Compute shape functions for pyramid.

    Args:
        pcoords: Parametric coordinates (r, s, t)

    Returns:
        Shape function values at each of the 5 nodes
    """
    r = pcoords[0]
    s = pcoords[1]
    t = pcoords[2]

    weights = wp.vec(length=NUM_NODES, dtype=wp.float32)

    # Handle singularity at apex (t = 1)
    if t >= 0.9999:
        weights[0] = 0.0
        weights[1] = 0.0
        weights[2] = 0.0
        weights[3] = 0.0
        weights[4] = 1.0
    else:
        # Standard pyramid shape functions
        # At base (t=0): standard bilinear
        # At apex (t=1): all weight at node 4
        tm = 1.0 - t

        # Base nodes: scaled by (1-t)
        weights[0] = (1.0 - r) * (1.0 - s) * tm
        weights[1] = r * (1.0 - s) * tm
        weights[2] = r * s * tm
        weights[3] = (1.0 - r) * s * tm

        # Apex node
        weights[4] = t

    return weights


@wp.func
def compute_shape_derivatives(pcoords: wp.vec3f) -> wp.mat(shape=(NUM_NODES, 3), dtype=wp.float32):
    """
    Compute derivatives of shape functions with respect to parametric coordinates.

    Args:
        pcoords: Parametric coordinates (r, s, t)

    Returns:
        5x3 matrix where row i contains [dN_i/dr, dN_i/ds, dN_i/dt]
    """
    r = pcoords[0]
    s = pcoords[1]
    t = pcoords[2]

    derivs = wp.mat(shape=(NUM_NODES, 3), dtype=wp.float32)

    # Handle singularity at apex
    if t >= 0.9999:
        # At apex, derivatives are zero except dt/dt for node 4
        for i in range(5):
            derivs[i, 0] = 0.0
            derivs[i, 1] = 0.0
            derivs[i, 2] = 0.0
        derivs[4, 2] = 1.0
    else:
        tm = 1.0 - t

        # Node 0: (0, 0, 0)
        derivs[0, 0] = -(1.0 - s) * tm
        derivs[0, 1] = -(1.0 - r) * tm
        derivs[0, 2] = -(1.0 - r) * (1.0 - s)

        # Node 1: (1, 0, 0)
        derivs[1, 0] = (1.0 - s) * tm
        derivs[1, 1] = -r * tm
        derivs[1, 2] = -r * (1.0 - s)

        # Node 2: (1, 1, 0)
        derivs[2, 0] = s * tm
        derivs[2, 1] = r * tm
        derivs[2, 2] = -r * s

        # Node 3: (0, 1, 0)
        derivs[3, 0] = -s * tm
        derivs[3, 1] = (1.0 - r) * tm
        derivs[3, 2] = -(1.0 - r) * s

        # Node 4: apex
        derivs[4, 0] = 0.0
        derivs[4, 1] = 0.0
        derivs[4, 2] = 1.0

    return derivs


@wp.func
def compute_parametric_coordinates(
    p0: wp.vec3f,
    p1: wp.vec3f,
    p2: wp.vec3f,
    p3: wp.vec3f,
    p4: wp.vec3f,
    pos: wp.vec3f,
    tolerance: float = 1.0e-5,
    max_iterations: int = 10,
) -> wp.vec3f:
    """
    Compute parametric coordinates for a point in pyramid using Newton-Raphson iteration.

    Args:
        p0-p4: The 5 node positions in world coordinates
        pos: World position to find parametric coordinates for
        tolerance: Convergence tolerance
        max_iterations: Maximum number of iterations

    Returns:
        Parametric coordinates (r, s, t)
    """
    # Initial guess: center of element
    pcoords = wp.vec3f(0.5, 0.5, 0.5)

    for _ in range(max_iterations):
        # Evaluate shape functions and derivatives
        weights = compute_shape_functions(pcoords)
        derivs = compute_shape_derivatives(pcoords)

        # Store points in a matrix (5 rows x 3 columns)
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
    Check if parametric coordinates are inside the pyramid.

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
