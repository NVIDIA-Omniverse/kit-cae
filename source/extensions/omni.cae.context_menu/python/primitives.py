# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Primitive geometry creation functions."""

__all__ = ["create_unit_sphere", "create_unit_box"]

from logging import getLogger

import numpy as np
import omni.kit.commands
from omni.usd import get_stage_next_free_path
from pxr import Sdf, Usd, UsdGeom, Vt

logger = getLogger(__name__)


def get_anchor_path(stage: Usd.Stage) -> Sdf.Path:
    """Get or create the anchor path for CAE objects.

    Args:
        stage: The USD stage

    Returns:
        The anchor path (typically /CAE under the default prim)
    """
    defaultPrim = stage.GetDefaultPrim()
    path = defaultPrim.GetPath().AppendChild("CAE") if defaultPrim else Sdf.Path("/CAE")
    if not stage.GetPrimAtPath(path):
        UsdGeom.Xform.Define(stage, path)
    return path


def create_unit_sphere(stage: Usd.Stage, path: Sdf.Path):
    """Create a unit sphere mesh with UV coordinates.

    Args:
        stage: The USD stage to create the sphere in
        path: The path where the sphere prim should be created

    Returns:
        The created UsdGeom.Mesh prim, or None if stage is invalid
    """
    if not stage:
        logger.error("missing stage")
        return None

    coords, faces, st, normals = _generate_unit_sphere_mesh_with_uv(16)
    glyph = UsdGeom.Mesh.Define(stage, path)
    glyph.CreateExtentAttr().Set([(-1, -1, -1), (1, 1, 1)])
    glyph.CreatePointsAttr().Set(Vt.Vec3fArray.FromNumpy(coords))
    glyph.CreateFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(faces))
    glyph.CreateFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(np.ones(faces.shape[0] // 3) * 3))
    glyph.CreateNormalsAttr().Set(normals)
    api = UsdGeom.PrimvarsAPI(glyph)
    api.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "vertex").Set(st)
    logger.info("Created unit sphere at '%s'", glyph.GetPath())
    return glyph


def _generate_unit_sphere_mesh_with_uv(resolution: float):
    """
    Generates a unit sphere mesh with texture coordinates (UV).

    Args:
    - resolution: int, the number of divisions along latitude and longitude.

    Returns:
    - vertices: np.ndarray, shape (n, 3), the coordinates of the points on the surface.
    - faces: np.ndarray, shape (m, 3), the indices of the vertices forming triangular faces.
    - uv: np.ndarray, shape (n, 2), the UV texture coordinates for each vertex.
    """
    # Create a grid in spherical coordinates
    theta = np.linspace(0, np.pi, resolution)  # latitude (0 to pi)
    phi = np.linspace(0, 2 * np.pi, resolution)  # longitude (0 to 2*pi)

    # Create a meshgrid for spherical coordinates
    theta, phi = np.meshgrid(theta, phi)

    # Convert spherical coordinates to Cartesian coordinates
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    # Stack the coordinates into a single array of vertices
    vertices = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T

    # Generate texture coordinates (UV mapping)
    u = phi / (2 * np.pi)  # Normalize phi to range 0 to 1
    v = theta / np.pi  # Normalize theta to range 0 to 1
    uv = np.vstack([u.ravel(), v.ravel()]).T

    # Create faces by connecting the vertices in the grid
    faces = []
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            # Vertices of each quad
            v1 = i * resolution + j
            v2 = v1 + 1
            v3 = v1 + resolution
            v4 = v3 + 1

            # Two triangles per quad
            faces.append([v1, v2, v4])
            faces.append([v1, v4, v3])

    faces = np.array(faces)
    normals = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
    return vertices, faces.flatten(), uv, normals


def create_unit_box(stage: Usd.Stage, path: Sdf.Path):
    """Create a unit box using basis curves.

    Args:
        stage: The USD stage to create the box in
        path: The path where the box prim should be created

    Returns:
        The created UsdGeom.BasisCurves prim, or None if stage is invalid
    """
    if not stage:
        logger.error("missing stage")
        return None

    basisCurves = UsdGeom.BasisCurves.Define(stage, path)
    basisCurves.CreateTypeAttr().Set(UsdGeom.Tokens.linear)
    basisCurves.SetWidthsInterpolation(UsdGeom.Tokens.constant)
    basisCurves.CreateWrapAttr().Set(UsdGeom.Tokens.nonperiodic)
    basisCurves.CreatePointsAttr().Set(
        [
            (0, 0, 0),
            (1, 0, 0),
            (1, 0, 0),
            (1, 1, 0),
            (1, 1, 0),
            (0, 1, 0),
            (0, 1, 0),
            (0, 0, 0),
            (0, 0, 1),
            (1, 0, 1),
            (1, 0, 1),
            (1, 1, 1),
            (1, 1, 1),
            (0, 1, 1),
            (0, 1, 1),
            (0, 0, 1),
            (0, 0, 0),
            (0, 0, 1),
            (1, 0, 0),
            (1, 0, 1),
            (1, 1, 0),
            (1, 1, 1),
            (0, 1, 0),
            (0, 1, 1),
        ]
    )
    basisCurves.CreateCurveVertexCountsAttr().Set([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
    basisCurves.CreateWidthsAttr().Set([(0.02)])
    basisCurves.CreateExtentAttr().Set([(0, 0, 0), (1, 1, 1)])
    logger.info("Created unit box at '%s'", basisCurves.GetPath())
    return basisCurves
