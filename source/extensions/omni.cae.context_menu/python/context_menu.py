# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

__all__ = ["get_algorithms_menu_dict", "get_flow_menu_dict"]


from functools import partial
from logging import getLogger
from pathlib import Path
from typing import Callable, Union

import numpy as np
import omni.kit.commands
from omni.cae.schema import cae
from omni.usd import get_stage_next_free_path
from pxr import Sdf, Usd, UsdGeom, UsdVol, Vt

logger = getLogger(__name__)


def get_hovered_prim(objects: dict, filter: Callable[[Usd.Prim], bool] = None) -> Union[Usd.Prim, None]:
    obj = objects.get("hovered_prim") if objects.get("use_hovered", False) else None
    return obj if obj is None or filter is None or filter(obj) else None


def get_selected_prims(objects: dict, filter: Callable[[Usd.Prim], bool] = None) -> list[Usd.Prim]:
    prim_list = [prim for prim in objects.get("prim_list", []) if prim.IsValid()]
    if filter is not None:
        # if filter is provided, all of the selected prims must satisfy the filter
        for prim in prim_list:
            if not filter(prim):
                logger.debug("Mismatched prim selected: %s", prim)
                return []
    return prim_list


def get_active_prims(objects: dict, filter: Callable[[Usd.Prim], bool] = None) -> list[Usd.Prim]:
    """
    We define active prims as [hovered prim + selected prims] if hovered prim is
    in the selected prims collection, otherwise just return the hovered prim.
    """
    selected_prims = get_selected_prims(objects, filter)
    hovered_prim = get_hovered_prim(objects, filter)
    if hovered_prim and hovered_prim in selected_prims:
        return selected_prims
    return [hovered_prim] if hovered_prim else []


def get_active_dataset_prim(objects: dict) -> Usd.Prim:
    active_prim = objects.get("hovered_prim") if objects.get("use_hovered", False) else None
    if active_prim and active_prim.IsA(cae.DataSet):
        return active_prim
    return None


def get_active_volume_prim(objects: dict) -> Usd.Prim:
    active_prim = objects.get("hovered_prim") if objects.get("use_hovered", False) else None
    if active_prim and active_prim.IsA(UsdVol.Volume):
        return active_prim
    return None


def get_anchor_path(stage: Usd.Stage) -> Sdf.Path:
    defaultPrim = stage.GetDefaultPrim()
    path = defaultPrim.GetPath().AppendChild("CAE") if defaultPrim else Sdf.Path("/CAE")
    if not stage.GetPrimAtPath(path):
        UsdGeom.Xform.Define(stage, path)
    return path


def create_unit_sphere(objects: dict):
    stage: Usd.Stage = objects.get("stage")
    if not stage:
        logger.error("missing stage")
        return

    path = get_stage_next_free_path(stage, get_anchor_path(stage).AppendChild("UnitSphere"), False)

    coords, faces, st, normals = _generate_unit_sphere_mesh_with_uv(16)
    glyph = UsdGeom.Mesh.Define(stage, path)
    glyph.CreateExtentAttr().Set([(-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)])
    glyph.CreatePointsAttr().Set(Vt.Vec3fArray.FromNumpy(coords))
    glyph.CreateFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(faces))
    glyph.CreateFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(np.ones(faces.shape[0] // 3) * 3))
    glyph.CreateNormalsAttr().Set(normals)
    api = UsdGeom.PrimvarsAPI(glyph)
    api.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "vertex").Set(st)
    logger.info("create '%s", glyph.GetPath())
    omni.kit.commands.execute("SelectPrimsCommand", new_selected_paths=[str(glyph.GetPath())], old_selected_paths=[])


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


def create_unit_box(objects: dict):
    stage: Usd.Stage = objects.get("stage")
    if not stage:
        logger.error("missing stage")
        return

    path = get_stage_next_free_path(stage, get_anchor_path(stage).AppendChild("UnitBox"), False)

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
    logger.info("create '%s", basisCurves.GetPath())
    omni.kit.commands.execute(
        "SelectPrimsCommand", new_selected_paths=[str(basisCurves.GetPath())], old_selected_paths=[]
    )


def create_with_single(schema: Usd.Typed, command: str, name: str, objects: dict):
    stage: Usd.Stage = objects.get("stage")
    dataset_prims = get_active_prims(objects, lambda prim: prim.IsA(schema))
    if not dataset_prims:
        logger.error("No selected %s prim. Cannot execute command '%s'", schema, command)
    else:
        # trigger command for each dataset
        paths_to_select = []
        for dataset_prim in dataset_prims:
            cname = f"{name}_{dataset_prim.GetName()}"
            prim_path = get_stage_next_free_path(stage, get_anchor_path(stage).AppendChild(cname), False)
            status = omni.kit.commands.execute(
                command, dataset_path=str(dataset_prim.GetPath()), prim_path=str(prim_path)
            )
            if status is not None and status[0]:
                paths_to_select.append(str(prim_path))
                logger.info("Created %s", status)
            else:
                logger.error("Unhandled command '%s'. Perhaps optional extensions are missing?", command)
        if paths_to_select:
            # select all created prims
            omni.kit.commands.execute("SelectPrimsCommand", new_selected_paths=paths_to_select, old_selected_paths=[])


def create_with_multiple(schema: Usd.Typed, command: str, name: str, objects: dict):
    stage: Usd.Stage = objects.get("stage")
    dataset_prims = get_active_prims(objects, lambda prim: prim.IsA(schema))
    if not dataset_prims:
        logger.error("No selected %s prim. Cannot execute command '%s'", schema, command)
    else:
        # trigger single command with all datasets
        cname = f"{name}_{dataset_prims[0].GetName()}" if len(dataset_prims) == 1 else name
        prim_path = get_stage_next_free_path(stage, get_anchor_path(stage).AppendChild(cname), False)
        status = omni.kit.commands.execute(
            command, dataset_paths=[str(prim.GetPath()) for prim in dataset_prims], prim_path=str(prim_path)
        )
        if status is not None and status[0]:
            omni.kit.commands.execute("SelectPrimsCommand", new_selected_paths=[str(prim_path)], old_selected_paths=[])
            logger.info("Created %s", status)
        else:
            logger.error("Unhandled command '%s'. Perhaps optional extensions are missing?", command)


def create_with_anchor(command: str, name: str, objects: dict):
    stage: Usd.Stage = objects.get("stage")
    prim_path = get_stage_next_free_path(stage, get_anchor_path(stage).AppendChild(name), False)
    status = omni.kit.commands.execute(command, prim_path=str(prim_path))
    if status is not None and status[0]:
        omni.kit.commands.execute("SelectPrimsCommand", new_selected_paths=[str(prim_path)], old_selected_paths=[])
        logger.info("Created %s", status)
    else:
        logger.error("Unhandled command '%s'. Perhaps optional extensions are missing?", command)


def get_icon_path(name) -> str:
    from carb.settings import get_settings

    style = get_settings().get_as_string("/persistent/app/window/uiStyle") or "NvidiaDark"
    current_path = Path(__file__).parent
    icon_path = current_path.parent.parent.parent.joinpath("icons") / style / f"{name}.svg"
    return str(icon_path)


def schema_isa(schema, objects: dict) -> bool:
    stage: Usd.Stage = objects.get("stage")
    return stage and len(get_active_prims(objects, lambda prim: prim.IsA(schema))) > 0


def schema_isa_str(schema: str, objects: dict) -> bool:
    stage: Usd.Stage = objects.get("stage")
    return stage and len(get_active_prims(objects, lambda prim: prim.GetTypeName() == schema)) > 0


def schema_hasa_str(schema: str, objects: dict) -> bool:
    stage: Usd.Stage = objects.get("stage")
    return stage and len(get_active_prims(objects, lambda prim: schema in prim.GetAppliedSchemas())) > 0


def supports_warp(objects) -> bool:
    from omni.kit.app import get_app

    manager = get_app().get_extension_manager()
    return manager.is_extension_enabled("omni.cae.algorithms.warp")


def get_algorithms_menu_dict():
    return {
        "name": {
            "CAE Algorithms": [
                {
                    "name": "Points",
                    "onclick_fn": partial(
                        create_with_single, cae.DataSet, "CreateCaeAlgorithmsExtractPoints", "Points"
                    ),
                    "show_fn": partial(schema_isa, cae.DataSet),
                },
                {
                    "name": "Glyphs",
                    "onclick_fn": partial(create_with_single, cae.DataSet, "CreateCaeAlgorithmsGlyphs", "Glyphs"),
                    "show_fn": partial(schema_isa, cae.DataSet),
                },
                {
                    "name": "External Faces",
                    "onclick_fn": partial(
                        create_with_single, cae.DataSet, "CreateCaeAlgorithmsExtractExternalFaces", "ExternalFaces"
                    ),
                    "show_fn": partial(schema_isa, cae.DataSet),
                },
                {
                    "name": "Slice (IndeX)",
                    "onclick_fn": partial(create_with_single, cae.DataSet, "CreateCaeIndeXSlice", "IndeXSlice"),
                    "show_fn": partial(schema_isa, cae.DataSet),
                },
                {
                    "name": "Slice (NanoVDB)",
                    "onclick_fn": partial(
                        create_with_single, cae.DataSet, "CreateCaeIndeXNanoVdbSlice", "IndeXNanoVdbSlice"
                    ),
                    "show_fn": partial(schema_isa, cae.DataSet),
                },
                {
                    "name": "Slice",
                    "onclick_fn": partial(
                        create_with_single, UsdVol.Volume, "CreateCaeIndeXVolumeSlice", "IndeXVolumeSlice"
                    ),
                    "show_fn": lambda objects: schema_isa(UsdVol.Volume, objects)
                    and (
                        schema_hasa_str("CaeIndeXVolumeAPI", objects)
                        or schema_hasa_str("CaeIndeXNanoVdbVolumeAPI", objects)
                    ),
                },
                {
                    "name": "Streamlines",
                    "onclick_fn": partial(create_with_single, cae.DataSet, "CreateCaeStreamlines", "Streamlines"),
                    "show_fn": partial(schema_isa, cae.DataSet),
                },
                {
                    "name": "Streamlines (NanoVDB)",
                    "onclick_fn": partial(
                        create_with_single, cae.DataSet, "CreateCaeNanoVdbStreamlines", "NanoVdbWarpStreamlines"
                    ),
                    "show_fn": partial(schema_isa, cae.DataSet),
                    "enabled_fn": supports_warp,
                },
                {
                    "name": "Volume (IndeX)",
                    "onclick_fn": partial(create_with_single, cae.DataSet, "CreateCaeIndeXVolume", "IndeXVolume"),
                    "show_fn": partial(schema_isa, cae.DataSet),
                },
                {
                    "name": "Volume (NanoVDB + IndeX)",
                    "onclick_fn": partial(
                        create_with_single, cae.DataSet, "CreateCaeNanoVdbIndeXVolume", "NanoVdbIndeXVolume"
                    ),
                    "show_fn": partial(schema_isa, cae.DataSet),
                },
                {},  # separator
                {
                    "name": "Unit Sphere",
                    "onclick_fn": create_unit_sphere,
                    "show_fn": lambda objects: objects.get("stage") is not None,
                },
                {
                    "name": "Unit Box",
                    "onclick_fn": create_unit_box,
                    "show_fn": lambda objects: objects.get("stage") is not None,
                },
                {
                    "name": "Bounding Box",
                    "onclick_fn": partial(
                        create_with_multiple, cae.DataSet, "CreateCaeAlgorithmsExtractBoundingBox", "BoundingBox"
                    ),
                    "show_fn": partial(schema_isa, cae.DataSet),
                },
            ]
        },
        "glyph": get_icon_path("menu"),
    }


def get_flow_menu_dict():
    return {
        "name": {
            "CAE Flow": [
                {
                    "name": "Environment",
                    "onclick_fn": partial(create_with_anchor, "CreateCaeFlowEnvironment", "FlowEnvironment"),
                    "show_fn": lambda objects: objects.get("stage") is not None,
                },
                {
                    "name": "DataSet Emitter",
                    "onclick_fn": partial(
                        create_with_single, cae.DataSet, "CreateCaeFlowDataSetEmitter", "DataSetEmitter"
                    ),
                    "show_fn": partial(schema_isa, cae.DataSet),
                },
                {
                    "name": "Volume Streamlines",
                    "onclick_fn": partial(create_with_anchor, "CreateCaeFlowSmoker", "VolumeStreamlines"),
                    "show_fn": lambda objects: objects.get("stage") is not None,
                },  # partial(schema_isa_str, "CaeFlowEnvironment")},
            ]
        },
        "glyph": get_icon_path("menu"),
    }
