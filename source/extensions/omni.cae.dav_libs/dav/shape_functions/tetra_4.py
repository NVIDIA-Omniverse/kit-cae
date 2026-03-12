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

import dav

from .typing import ShapeAPI, UniformShapesLibraryAPI

NUM_NODES = 4


@dav.func
def compute_shape_functions(pcoords: wp.vec3f) -> wp.vec(length=dav.config.max_points_per_cell, dtype=wp.float32):
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

    weights = wp.vec(length=dav.config.max_points_per_cell, dtype=wp.float32)
    assert NUM_NODES <= dav.config.max_points_per_cell, "NUM_NODES exceeds max_points_per_cell in config"

    weights[0] = 1.0 - r - s - t  # Node 0
    weights[1] = r  # Node 1
    weights[2] = s  # Node 2
    weights[3] = t  # Node 3

    return weights


@dav.func
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


@dav.func
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


def get_shape(data_model: dav.DataModel, shapes_library: UniformShapesLibraryAPI) -> ShapeAPI:
    """
    Get the API class for Tetrahedron shape functions.
    This is intended to be used internally by a data model and hence accepts arguments
    that are part of the data model protocol but not a complete data model itself.
    """

    @dav.func
    def compute_parametric_coordinates(dataset: data_model.DatasetHandle, cell: data_model.CellHandle, pos: wp.vec3f, cell_type: wp.int32) -> wp.vec3f:
        """
        Compute barycentric coordinates for a point in tetrahedron.

        This uses a direct analytical solution based on volume ratios.

        Args:
            dataset: The dataset containing the cell
            cell: The cell for which to compute parametric coordinates
            pos: World position to find parametric coordinates for

        Returns:
            Parametric coordinates (r, s, t) - barycentric coordinates
        """
        # Get the 4 node positions using dataset APIs
        vtk_idx = shapes_library.get_vtk_corner_node_index(cell_type, 0)
        p0 = data_model.DatasetAPI.get_point(dataset, data_model.CellAPI.get_point_id(cell, vtk_idx, dataset))

        vtk_idx = shapes_library.get_vtk_corner_node_index(cell_type, 1)
        p1 = data_model.DatasetAPI.get_point(dataset, data_model.CellAPI.get_point_id(cell, vtk_idx, dataset))

        vtk_idx = shapes_library.get_vtk_corner_node_index(cell_type, 2)
        p2 = data_model.DatasetAPI.get_point(dataset, data_model.CellAPI.get_point_id(cell, vtk_idx, dataset))

        vtk_idx = shapes_library.get_vtk_corner_node_index(cell_type, 3)
        p3 = data_model.DatasetAPI.get_point(dataset, data_model.CellAPI.get_point_id(cell, vtk_idx, dataset))

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

    class TetrahedronShape:
        """Implements the ShapeAPI protocol for Tetrahedron shape functions."""

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

    return TetrahedronShape
