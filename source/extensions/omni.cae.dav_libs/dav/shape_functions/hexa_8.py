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
Hexahedron (8-node brick) shape functions.

Node ordering follows VTK convention:
    4-------7
   /|      /|
  5-------6 |
  | |     | |
  | 0-----|-3
  |/      |/
  1-------2

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

import dav

from .typing import ShapeAPI, UniformShapesLibraryAPI

NUM_NODES = 8


@dav.func
def compute_shape_functions(pcoords: wp.vec3f) -> wp.vec(length=dav.config.max_points_per_cell, dtype=wp.float32):
    """
    Compute trilinear shape functions for hexahedron.

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

    weights = wp.vec(length=dav.config.max_points_per_cell, dtype=wp.float32)
    assert NUM_NODES <= dav.config.max_points_per_cell, "NUM_NODES exceeds max_points_per_cell in config"

    weights[0] = rm * sm * tm
    weights[1] = r * sm * tm
    weights[2] = r * s * tm
    weights[3] = rm * s * tm
    weights[4] = rm * sm * t
    weights[5] = r * sm * t
    weights[6] = r * s * t
    weights[7] = rm * s * t

    return weights


@dav.func
def compute_shape_derivatives(pcoords: wp.vec3f) -> wp.mat(shape=(NUM_NODES, 3), dtype=wp.float32):
    """
    Compute derivatives of shape functions with respect to parametric coordinates.

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


@dav.func
def is_inside(pcoords: wp.vec3f, tolerance: float = 1.0e-6) -> bool:
    """
    Check if parametric coordinates are inside the hexahedron.

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


def get_shape(data_model: dav.DataModel, shapes_library: UniformShapesLibraryAPI) -> ShapeAPI:
    """
    Get the API class for Hexahedron shape functions.
    This is intended to be used internally by a data model and hence accepts arguments
    that are part of the data model protocol but not a complete data model itself.
    """

    @dav.func
    def compute_parametric_coordinates(
        dataset: data_model.DatasetHandle, cell: data_model.CellHandle, pos: wp.vec3f, cell_type: wp.int32, tolerance: float = 1.0e-5, max_iterations: int = 10
    ) -> wp.vec3f:
        """
        Compute parametric coordinates for a point in hexahedron using Newton-Raphson iteration.

        Args:
            dataset: The dataset containing the cell
            cell: The cell for which to compute parametric coordinates
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

            # Store points in a matrix (8 rows x 3 columns)
            # Each row represents a point's (x, y, z) coordinates
            points = wp.mat(shape=(NUM_NODES, 3), dtype=wp.float32)
            for i in range(wp.static(NUM_NODES)):
                pt = data_model.DatasetAPI.get_point(dataset, data_model.CellAPI.get_point_id(cell, i, dataset))
                # populate points using VTK ordering expected by shape functions by using
                # the point to the index given by ShapeFunctionsAPI.get_vtk_corner_node_index
                vtk_idx = shapes_library.get_vtk_corner_node_index(cell_type, i)
                points[vtk_idx, 0] = pt.x
                points[vtk_idx, 1] = pt.y
                points[vtk_idx, 2] = pt.z

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

    class HexahedronShape:
        """Implements the ShapeAPI protocol for Hexahedron shape functions."""

        @staticmethod
        @dav.func
        def is_point_in_cell(point: wp.vec3f, cell: data_model.CellHandle, dataset: data_model.DatasetHandle, cell_type: wp.int32) -> bool:
            pcoords = compute_parametric_coordinates(dataset, cell, point, cell_type)
            return is_inside(pcoords)

        @staticmethod
        @dav.func
        def get_weights(point: wp.vec3f, cell: data_model.CellHandle, dataset: data_model.DatasetHandle, cell_type: wp.int32) -> wp.vec(
            length=dav.config.max_points_per_cell, dtype=wp.float32
        ):
            pcoords = compute_parametric_coordinates(dataset, cell, point, cell_type)
            return compute_shape_functions(pcoords)

    return HexahedronShape
