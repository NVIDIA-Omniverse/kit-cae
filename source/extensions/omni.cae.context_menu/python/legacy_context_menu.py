# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

__all__ = [
    "get_algorithms_menu_dict",
    "get_flow_menu_dict",
]


import inspect
from functools import partial
from logging import getLogger
from pathlib import Path
from typing import Callable, Union

import omni.kit.commands
from omni.cae.schema import cae
from omni.kit.async_engine import run_coroutine
from omni.usd import get_stage_next_free_path
from pxr import Sdf, Usd, UsdGeom, UsdVol

logger = getLogger(__name__)


def get_hovered_prim(objects: dict, filter: Callable[[Usd.Prim], bool] = None) -> Union[Usd.Prim, None]:
    obj = objects.get("hovered_prim") if objects.get("use_hovered", False) else None
    return obj if obj is None or filter is None or filter(obj) else None


def get_selected_prims(objects: dict, filter: Callable[[Usd.Prim], bool] = None) -> list[Usd.Prim]:
    prim_list = [prim for prim in objects.get("prim_list", []) if prim]
    stage = objects.get("stage")
    prim_list = [prim if isinstance(prim, Usd.Prim) else stage.GetPrimAtPath(prim) for prim in prim_list]
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


def get_anchor_path(stage: Usd.Stage) -> Sdf.Path:
    """Get or create the anchor path for CAE objects."""
    defaultPrim = stage.GetDefaultPrim()
    path = defaultPrim.GetPath().AppendChild("CAE") if defaultPrim else Sdf.Path("/CAE")
    if not stage.GetPrimAtPath(path):
        UsdGeom.Xform.Define(stage, path)
    return path


def create_with_single(schema: Usd.Typed, command: str, name: str, objects: dict, **command_kwargs):
    run_coroutine(create_with_single_async(schema, command, name, objects, **command_kwargs))


async def create_with_single_async(
    schema: Usd.Typed, command: str, name: str, objects: dict, parent_path: Sdf.Path = None, **command_kwargs
):
    """Create prims using a command for each selected dataset.

    Args:
        schema: USD schema type to filter prims
        command: Command name to execute
        name: Base name for created prims
        objects: Dict containing stage and prim selection info
        parent_path: Optional parent path to use instead of anchor path
        **command_kwargs: Additional keyword arguments to pass to the command
    """
    stage: Usd.Stage = objects.get("stage")
    dataset_prims = get_active_prims(objects, lambda prim: prim.IsA(schema))
    if not dataset_prims:
        logger.error("No selected %s prim. Cannot execute command '%s'", schema, command)
    else:
        # trigger command for each dataset
        paths_to_select = []
        for dataset_prim in dataset_prims:
            cname = f"{name}_{dataset_prim.GetName()}"
            # Use provided parent_path or fallback to anchor path
            base_path = parent_path if parent_path is not None else get_anchor_path(stage)
            prim_path = get_stage_next_free_path(stage, base_path.AppendChild(cname), False)
            status, result = omni.kit.commands.execute(
                command, dataset_path=str(dataset_prim.GetPath()), prim_path=str(prim_path), **command_kwargs
            )
            if not status:
                logger.error("Failed to execute command '%s'. Perhaps optional extensions are missing?", command)
            else:
                if inspect.isawaitable(result):
                    await result
                paths_to_select.append(str(prim_path))
                logger.info("Created %s", result)
        if paths_to_select:
            # select all created prims
            omni.kit.commands.execute("SelectPrimsCommand", new_selected_paths=paths_to_select, old_selected_paths=[])


def create_with_multiple(schema: Usd.Typed, command: str, name: str, objects: dict, **command_kwargs):
    run_coroutine(create_with_multiple_async(schema, command, name, objects, **command_kwargs))


async def create_with_multiple_async(schema: Usd.Typed, command: str, name: str, objects: dict, **command_kwargs):
    stage: Usd.Stage = objects.get("stage")
    dataset_prims = get_active_prims(objects, lambda prim: prim.IsA(schema))
    if not dataset_prims:
        logger.error("No selected %s prim. Cannot execute command '%s'", schema, command)
    else:
        # trigger single command with all datasets
        cname = f"{name}_{dataset_prims[0].GetName()}" if len(dataset_prims) == 1 else name
        prim_path = get_stage_next_free_path(stage, get_anchor_path(stage).AppendChild(cname), False)
        status, result = omni.kit.commands.execute(
            command,
            dataset_paths=[str(prim.GetPath()) for prim in dataset_prims],
            prim_path=str(prim_path),
            **command_kwargs,
        )
        if not status:
            logger.error("Failed to execute command '%s'. Perhaps optional extensions are missing?", command)
        else:
            if inspect.isawaitable(result):
                await result
            omni.kit.commands.execute("SelectPrimsCommand", new_selected_paths=[str(prim_path)], old_selected_paths=[])
            logger.info("Created %s", result)


def create_with_anchor(command: str, name: str, objects: dict):
    run_coroutine(create_with_anchor_async(command, name, objects))


async def create_with_anchor_async(command: str, name: str, objects: dict):
    stage: Usd.Stage = objects.get("stage")
    prim_path = get_stage_next_free_path(stage, get_anchor_path(stage).AppendChild(name), False)
    status, result = omni.kit.commands.execute(command, prim_path=str(prim_path))
    if not status:
        logger.error("Failed to execute command '%s'. Perhaps optional extensions are missing?", command)
    else:
        if inspect.isawaitable(result):
            await result
        omni.kit.commands.execute("SelectPrimsCommand", new_selected_paths=[str(prim_path)], old_selected_paths=[])
        logger.info("Created %s", result)


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


def create_unit_sphere_wrapper(objects: dict):
    """Wrapper to handle path finding and selection for unit sphere creation."""
    from .primitives import create_unit_sphere

    stage = objects.get("stage")
    if not stage:
        logger.error("missing stage")
        return
    path = get_stage_next_free_path(stage, get_anchor_path(stage).AppendChild("UnitSphere"), False)
    prim = create_unit_sphere(stage, path)
    if prim:
        omni.kit.commands.execute("SelectPrimsCommand", new_selected_paths=[str(prim.GetPath())], old_selected_paths=[])


def create_unit_box_wrapper(objects: dict):
    """Wrapper to handle path finding and selection for unit box creation."""
    from .primitives import create_unit_box

    stage = objects.get("stage")
    if not stage:
        logger.error("missing stage")
        return
    path = get_stage_next_free_path(stage, get_anchor_path(stage).AppendChild("UnitBox"), False)
    prim = create_unit_box(stage, path)
    if prim:
        omni.kit.commands.execute("SelectPrimsCommand", new_selected_paths=[str(prim.GetPath())], old_selected_paths=[])


def get_algorithms_menu_dict():
    return {
        "name": {
            "CAE Algorithms (Legacy)": [
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
                    "onclick_fn": create_unit_sphere_wrapper,
                    "show_fn": lambda objects: objects.get("stage") is not None,
                },
                {
                    "name": "Unit Box",
                    "onclick_fn": create_unit_box_wrapper,
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
            "CAE Flow (Legacy)": [
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
