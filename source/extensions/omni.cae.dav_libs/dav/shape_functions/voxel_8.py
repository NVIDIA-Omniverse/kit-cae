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

Node ordering follows VTK convention (x varies fastest, then y, then z):
    4-------5
   /|      /|
  6-------7 |
  | |     | |
  | 0-----|-1
  |/      |/
  2-------3

Parametric coordinates: r, s, t ∈ [0, 1]
  Node 0: (0, 0, 0)
  Node 1: (1, 0, 0)
  Node 2: (0, 1, 0)
  Node 3: (1, 1, 0)
  Node 4: (0, 0, 1)
  Node 5: (1, 0, 1)
  Node 6: (0, 1, 1)
  Node 7: (1, 1, 1)
"""

import warp as wp

import dav

from .typing import ShapeAPI, UniformShapesLibraryAPI

NUM_NODES = 8


@dav.func
def compute_shape_functions(pcoords: wp.vec3f) -> wp.vec(length=dav.config.max_points_per_cell, dtype=wp.float32):
    """
    Compute trilinear shape functions for voxel.

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

    weights[0] = rm * sm * tm  # (0, 0, 0)
    weights[1] = r * sm * tm  # (1, 0, 0)
    weights[2] = rm * s * tm  # (0, 1, 0)
    weights[3] = r * s * tm  # (1, 1, 0)
    weights[4] = rm * sm * t  # (0, 0, 1)
    weights[5] = r * sm * t  # (1, 0, 1)
    weights[6] = rm * s * t  # (0, 1, 1)
    weights[7] = r * s * t  # (1, 1, 1)

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

    # Node 2: (0, 1, 0)
    derivs[2, 0] = -s * tm
    derivs[2, 1] = rm * tm
    derivs[2, 2] = -rm * s

    # Node 3: (1, 1, 0)
    derivs[3, 0] = s * tm
    derivs[3, 1] = r * tm
    derivs[3, 2] = -r * s

    # Node 4: (0, 0, 1)
    derivs[4, 0] = -sm * t
    derivs[4, 1] = -rm * t
    derivs[4, 2] = rm * sm

    # Node 5: (1, 0, 1)
    derivs[5, 0] = sm * t
    derivs[5, 1] = -r * t
    derivs[5, 2] = r * sm

    # Node 6: (0, 1, 1)
    derivs[6, 0] = -s * t
    derivs[6, 1] = rm * t
    derivs[6, 2] = rm * s

    # Node 7: (1, 1, 1)
    derivs[7, 0] = s * t
    derivs[7, 1] = r * t
    derivs[7, 2] = r * s

    return derivs


@dav.func
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


def get_shape(data_model: dav.DataModel, shapes_library: UniformShapesLibraryAPI) -> ShapeAPI:
    """
    Get the API class for Voxel shape functions.
    This is intended to be used internally by a data model and hence accepts arguments
    that are part of the data model protocol but not a complete data model itself.
    """

    @dav.func
    def compute_parametric_coordinates(dataset: data_model.DatasetHandle, cell: data_model.CellHandle, pos: wp.vec3f, cell_type: wp.int32) -> wp.vec3f:
        """
        Compute parametric coordinates for a point in axis-aligned voxel.

        Since voxel is axis-aligned, we can compute parametric coordinates directly
        without iteration.

        Args:
            dataset: The dataset containing the cell
            cell: The cell for which to compute parametric coordinates
            pos: World position to find parametric coordinates for

        Returns:
            Parametric coordinates (r, s, t)
        """
        # Get the 8 node positions using dataset APIs
        points = wp.mat(shape=(NUM_NODES, 3), dtype=wp.float32)
        for i in range(wp.static(NUM_NODES)):
            pt = data_model.DatasetAPI.get_point(dataset, data_model.CellAPI.get_point_id(cell, i, dataset))
            # populate points using VTK ordering expected by shape functions by using
            # the point to the index given by ShapeFunctionsAPI.get_vtk_corner_node_index
            vtk_idx = shapes_library.get_vtk_corner_node_index(cell_type, i)
            points[vtk_idx, 0] = pt.x
            points[vtk_idx, 1] = pt.y
            points[vtk_idx, 2] = pt.z

        # For axis-aligned voxel, we can compute parametric coords directly
        # by linear interpolation along each axis
        # In VTK voxel ordering: p0 is at (0,0,0), p7 is at (1,1,1)

        # Get min and max corners
        min_pt = wp.vec3f(points[0, 0], points[0, 1], points[0, 2])  # Node 0: (0, 0, 0)
        max_pt = wp.vec3f(points[7, 0], points[7, 1], points[7, 2])  # Node 7: (1, 1, 1)

        # Handle degenerate cases
        dx = max_pt[0] - min_pt[0]
        dy = max_pt[1] - min_pt[1]
        dz = max_pt[2] - min_pt[2]

        r = 0.0 if wp.abs(dx) < 1.0e-20 else (pos[0] - min_pt[0]) / dx
        s = 0.0 if wp.abs(dy) < 1.0e-20 else (pos[1] - min_pt[1]) / dy
        t = 0.0 if wp.abs(dz) < 1.0e-20 else (pos[2] - min_pt[2]) / dz

        return wp.vec3f(r, s, t)

    class VoxelShape:
        """Implements the ShapeAPI protocol for Voxel shape functions."""

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

    return VoxelShape
