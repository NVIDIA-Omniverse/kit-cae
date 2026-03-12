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
    "get_sources_menu_dict",
    "get_operators_menu_dict",
    "get_flow_menu_dict",
    "get_add_menu_dict",
]


import inspect
from functools import partial, wraps
from logging import getLogger
from pathlib import Path
from typing import Callable, Optional, Union

import omni.kit.commands
from omni.cae.data.commands import execute_command
from omni.cae.schema import cae
from omni.cae.schema import viz as cae_viz
from omni.kit.async_engine import run_coroutine
from omni.usd import get_context, get_stage_next_free_path
from pxr import Sdf, Usd, UsdGeom, UsdVol

from .dialog import TypeSelectionDialog, ValidatedInputDialog

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


def can_apply_api_prim(api_schema: str, prim: Usd.Prim) -> bool:
    # check if API schema is single apply or multiple apply
    registry = Usd.SchemaRegistry()
    if registry.IsMultipleApplyAPISchema(api_schema):
        return prim.CanApplyAPI(api_schema, "cae_random_instance_name")
    return prim.CanApplyAPI(api_schema)


def can_apply_api(api_schema: str, objects: dict) -> bool:
    prims = get_selected_prims(objects)
    return all(can_apply_api_prim(api_schema, prim) for prim in prims)


def add_api(
    schema: Usd.Typed,
    api_schema: str,
    objects: dict = {},
    payload=None,
    suggestions_provider: Optional[Callable[[list[Usd.Prim]], list[str]]] = None,
):
    if payload is not None:
        objects = {}
        objects["stage"] = payload.get_stage()
        objects["prim_list"] = [path for path in payload]

    prims = get_selected_prims(objects, lambda prim: prim.IsA(schema))
    logger.info("add_api(%s, %s, %s)", schema, api_schema, prims)

    registry = Usd.SchemaRegistry()
    if registry.IsMultipleApplyAPISchema(api_schema):
        run_coroutine(add_api_async(prims, api_schema, suggestions_provider))
    else:
        for prim in prims:
            prim.ApplyAPI(api_schema)
            logger.info("Applied %s to prim %s", api_schema, prim.GetPath())


async def add_api_async(
    prims: list[Usd.Prim], api_schema: str, suggestions_provider: Optional[Callable[[list[Usd.Prim]], list[str]]] = None
):
    """Show a dialog to enter instance name and apply API to prims."""
    from .api_schema_dialog import APISchemaDialog

    if not prims:
        logger.error("No prims selected to apply API schema '%s'", api_schema)
        return

    # Show API schema dialog with provided suggestions provider
    dialog = APISchemaDialog(
        api_schema=api_schema,
        prims=prims,
        suggestions_provider=suggestions_provider,
        default_value="default",
    )
    instance_name = await dialog.exec()

    # Apply the API if a valid name was entered
    if instance_name is not None:
        for prim in prims:
            prim.ApplyAPI(api_schema, instance_name)
            logger.info("Applied %s:%s to prim %s", api_schema, instance_name, prim.GetPath())


def get_flow_layer_numbers(stage: Usd.Stage) -> list[int]:
    """Get the list of layer numbers for the flow environments."""
    from usdrt import Usd as UsdRT

    context = omni.usd.get_context()
    stage = context.get_stage() if context else None
    if stage is None:
        return []

    stage_rt = UsdRT.Stage.Attach(context.get_stage_id()) if context else None
    if stage_rt is None:
        return []

    paths = stage_rt.GetPrimsWithTypeName("FlowSimulate")
    if not paths:
        return []

    layer_numbers = []
    for path in paths:
        prim = stage.GetPrimAtPath(path.GetString())
        if prim and prim.HasAttribute("layer"):
            layer_numbers.append(prim.GetAttribute("layer").Get())
    return layer_numbers


def get_next_available_layer_number(stage: Usd.Stage) -> int:
    """Get the next available layer number for the flow environment."""
    layer_numbers = get_flow_layer_numbers(stage)
    if not layer_numbers:
        return 0

    return max(layer_numbers) + 1


def get_icon_path(name) -> str:
    from carb.settings import get_settings

    style = get_settings().get_as_string("/persistent/app/window/uiStyle") or "NvidiaDark"
    current_path = Path(__file__).parent
    icon_path = current_path.parent.parent.parent.joinpath("icons") / style / f"{name}.svg"
    return str(icon_path)


def schema_isa(schema, objects: dict) -> bool:
    stage: Usd.Stage = objects.get("stage")
    return stage and len(get_active_prims(objects, lambda prim: prim.IsA(schema))) > 0


def async_callback(async_fn: Callable):
    """Decorator to convert an async function into a sync callback that works with run_coroutine."""

    @wraps(async_fn)
    def wrapper(objects: dict, *args, **kwargs):
        run_coroutine(async_fn(objects, *args, **kwargs))

    return wrapper


def select_result(async_fn: Callable):
    @wraps(async_fn)
    async def wrapper(*args, **kwargs):
        path_or_paths = (
            await async_fn(*args, **kwargs) if inspect.iscoroutinefunction(async_fn) else async_fn(*args, **kwargs)
        )
        if isinstance(path_or_paths, str):
            path_or_paths = [path_or_paths]
        elif isinstance(path_or_paths, list):
            path_or_paths = [str(path) for path in path_or_paths]
        elif path_or_paths is None:
            return
        else:
            raise ValueError(f"Expected str or list of str, got {type(path_or_paths)}")
        await execute_command("SelectPrimsCommand", new_selected_paths=path_or_paths, old_selected_paths=[])

    return wrapper


class BoundingBox:

    @staticmethod
    def show(objects: dict) -> bool:
        return objects.get("stage") is not None

    @staticmethod
    def enabled(objects: dict) -> bool:
        active_prims = get_active_prims(objects, lambda prim: prim.IsA(cae.DataSet))
        return len(active_prims) > 0

    @staticmethod
    @async_callback
    @select_result
    async def onclick(objects: dict):
        stage = objects.get("stage")
        assert stage is not None, "missing stage"
        active_dataset_prims = get_active_prims(objects, lambda prim: prim.IsA(cae.DataSet))
        assert len(active_dataset_prims) > 0

        name = f"BoundingBox_{active_dataset_prims[0].GetName()}" if len(active_dataset_prims) == 1 else "BoundingBox"
        prim_path = get_stage_next_free_path(stage, get_anchor_path(stage).AppendChild(name), False)
        await execute_command(
            "CreateCaeVizBoundingBox",
            dataset_paths=[str(prim.GetPath()) for prim in active_dataset_prims],
            prim_path=prim_path,
        )
        return prim_path


class UnitSphere:
    @staticmethod
    def show(objects: dict) -> bool:
        return objects.get("stage") is not None

    @staticmethod
    @async_callback
    @select_result
    async def onclick(objects: dict):
        stage = objects.get("stage")
        assert stage is not None, "missing stage"

        # Prompt user for sphere resolution
        resolution = await UnitSphere.get_resolution()
        if resolution is None:
            return None

        # get active prims (optional)
        active_prims = get_active_prims(objects, lambda prim: prim.IsA(cae.DataSet) or prim.IsA(UsdGeom.Boundable))

        prim_path = get_stage_next_free_path(stage, get_anchor_path(stage).AppendChild("UnitSphere"), False)
        await execute_command(
            "CreateCaeVizMeshPrim",
            prim_type="UnitSphere",
            prim_path=prim_path,
            resolution=resolution,
            boundable_paths=[str(prim.GetPath()) for prim in active_prims],
        )
        return prim_path

    @staticmethod
    async def get_resolution() -> int:
        def validate_resolution(value: str) -> tuple[bool, str]:
            try:
                res = int(value)
                if res < 3:
                    return False, "Resolution must be at least 3"
                if res > 256:
                    return False, "Resolution must not exceed 256"
                return True, ""
            except ValueError:
                return False, "Resolution must be an integer"

        dialog = ValidatedInputDialog(
            title="Choose Sphere Resolution",
            label="Enter the number of divisions along latitude and longitude:",
            validator=validate_resolution,
            default_value="16",
            field_label="Resolution:",
            field_width=100,
        )
        result = await dialog.exec()
        return int(result) if result is not None else None


class VolumeSlice:
    @staticmethod
    def show(objects: dict) -> bool:
        return objects.get("stage") is not None

    @staticmethod
    def enabled(objects: dict) -> bool:
        """Enabled when the active prim is a single UsdVolVolume prim."""
        active_prims = get_active_prims(objects, lambda prim: prim.IsA(UsdVol.Volume))
        return len(active_prims) == 1

    @staticmethod
    @async_callback
    @select_result
    async def onclick(objects: dict):
        stage = objects.get("stage")
        assert stage is not None, "missing stage"
        active_prim = get_active_prims(objects, lambda prim: prim.IsA(UsdVol.Volume))[0]
        assert active_prim is not None, "missing active prim"

        shape = await VolumeSlice.get_shape()
        if shape is None:
            return None

        prim_path = get_stage_next_free_path(stage, get_anchor_path(stage).AppendChild("VolumeSlice"), False)
        await execute_command(
            "CreateCaeVizVolumeSlice", volume_path=str(active_prim.GetPath()), prim_path=prim_path, shape=shape
        )
        return prim_path

    @staticmethod
    async def get_shape() -> str:
        dialog = TypeSelectionDialog(
            title="Choose Slice Shape",
            label="Select the slice shape:",
            # disable non-planar shapes for now since omni.index.usd doesn't handle those.
            # options=["Plane", "Bi-Plane", "Sphere", "Custom"],
            options=["Plane", "Bi-Plane", "Tri-Plane"],
            default_index=0,
            field_label="Shape:",
            field_width=60,
        )
        shape = await dialog.exec()
        return shape


class OperatorsPoints:
    @staticmethod
    def show(objects: dict) -> bool:
        return objects.get("stage") is not None

    @staticmethod
    def enabled(objects: dict) -> bool:
        return schema_isa(cae.DataSet, objects)

    @staticmethod
    @async_callback
    @select_result
    async def onclick(objects: dict):
        stage = objects.get("stage")
        assert stage is not None, "missing stage"
        active_dataset_prims = get_active_prims(objects, lambda prim: prim.IsA(cae.DataSet))
        assert len(active_dataset_prims) > 0

        # Create points for each dataset individually
        created_paths = []
        for dataset_prim in active_dataset_prims:
            name = f"Points_{dataset_prim.GetName()}"
            prim_path = get_stage_next_free_path(stage, get_anchor_path(stage).AppendChild(name), False)
            await execute_command("CreateCaeVizPoints", dataset_path=str(dataset_prim.GetPath()), prim_path=prim_path)
            created_paths.append(prim_path)

        return created_paths


class OperatorsFaces:
    @staticmethod
    def show(objects: dict) -> bool:
        return objects.get("stage") is not None

    @staticmethod
    def enabled(objects: dict) -> bool:
        return schema_isa(cae.DataSet, objects)

    @staticmethod
    @async_callback
    @select_result
    async def onclick(objects: dict):
        stage = objects.get("stage")
        assert stage is not None, "missing stage"
        active_dataset_prims = get_active_prims(objects, lambda prim: prim.IsA(cae.DataSet))
        assert len(active_dataset_prims) > 0

        # Create faces for each dataset individually
        created_paths = []
        for dataset_prim in active_dataset_prims:
            name = f"Faces_{dataset_prim.GetName()}"
            prim_path = get_stage_next_free_path(stage, get_anchor_path(stage).AppendChild(name), False)
            await execute_command("CreateCaeVizFaces", dataset_path=str(dataset_prim.GetPath()), prim_path=prim_path)
            created_paths.append(prim_path)

        return created_paths


class OperatorsGlyphs:

    @staticmethod
    def show(objects: dict) -> bool:
        return objects.get("stage") is not None

    @staticmethod
    def enabled(objects: dict) -> bool:
        return schema_isa(cae.DataSet, objects)

    @staticmethod
    @async_callback
    @select_result
    async def onclick(objects: dict):
        stage = objects.get("stage")
        assert stage is not None, "missing stage"
        active_dataset_prims = get_active_prims(objects, lambda prim: prim.IsA(cae.DataSet))
        assert len(active_dataset_prims) > 0

        shape = await OperatorsGlyphs.get_shape()
        if shape is None:
            return None

        # Create glyphs for each dataset individually
        created_paths = []
        for dataset_prim in active_dataset_prims:
            name = f"Glyphs_{dataset_prim.GetName()}"
            prim_path = get_stage_next_free_path(stage, get_anchor_path(stage).AppendChild(name), False)
            await execute_command(
                "CreateCaeVizGlyphs", dataset_path=str(dataset_prim.GetPath()), prim_path=prim_path, shape=shape
            )
            created_paths.append(prim_path)

        return created_paths

    @staticmethod
    async def get_shape() -> str:
        dialog = TypeSelectionDialog(
            title="Choose Glyph Shape",
            label="Select the glyph shape:",
            options=["Sphere", "Cone", "Arrow", "Custom"],
            default_index=0,
            field_label="Shape:",
            field_width=60,
        )
        shape = await dialog.exec()
        return shape


class OperatorsStreamlines:
    @staticmethod
    def show(objects: dict) -> bool:
        return objects.get("stage") is not None

    @staticmethod
    def enabled(objects: dict) -> bool:
        return schema_isa(cae.DataSet, objects)

    @staticmethod
    @async_callback
    @select_result
    async def onclick(objects: dict):
        stage = objects.get("stage")
        assert stage is not None, "missing stage"
        active_dataset_prims = get_active_prims(objects, lambda prim: prim.IsA(cae.DataSet))
        assert len(active_dataset_prims) > 0

        streamlines_type = await OperatorsStreamlines.get_streamlines_type()
        if streamlines_type is None:
            return None
        created_paths = []
        for dataset_prim in active_dataset_prims:
            name = f"Streamlines_{dataset_prim.GetName()}"
            prim_path = get_stage_next_free_path(stage, get_anchor_path(stage).AppendChild(name), False)
            await execute_command(
                "CreateCaeVizStreamlines",
                dataset_path=str(dataset_prim.GetPath()),
                prim_path=prim_path,
                type=streamlines_type,
            )
            created_paths.append(prim_path)

        return created_paths

    @staticmethod
    async def get_streamlines_type() -> str:
        dialog = TypeSelectionDialog(
            title="Choose Streamlines Type",
            label="Select the streamlines type:",
            options=["standard", "nanovdb"],
            default_index=0,
            field_label="Type:",
            field_width=60,
        )
        streamlines_type = await dialog.exec()
        return streamlines_type


class OperatorsVolume:

    @staticmethod
    def show(objects: dict) -> bool:
        return objects.get("stage") is not None

    @staticmethod
    def enabled(objects: dict) -> bool:
        return schema_isa(cae.DataSet, objects)

    @staticmethod
    @async_callback
    @select_result
    async def onclick(objects: dict):
        stage = objects.get("stage")
        assert stage is not None, "missing stage"
        active_dataset_prims = get_active_prims(objects, lambda prim: prim.IsA(cae.DataSet))
        assert len(active_dataset_prims) > 0

        volume_type = await OperatorsVolume.get_volume_type()
        if volume_type is None:
            return None
        # Create volume for each dataset individually
        created_paths = []
        for dataset_prim in active_dataset_prims:
            name = f"Volume_{dataset_prim.GetName()}"
            prim_path = get_stage_next_free_path(stage, get_anchor_path(stage).AppendChild(name), False)
            await execute_command(
                "CreateCaeVizVolume", dataset_path=str(dataset_prim.GetPath()), prim_path=prim_path, type=volume_type
            )
            created_paths.append(prim_path)

        return created_paths

    @staticmethod
    async def get_volume_type() -> str:
        dialog = TypeSelectionDialog(
            title="Choose Volume Type",
            label="Select the volume type:",
            options=["irregular", "nanovdb"],
            default_index=0,
            field_label="Type:",
            field_width=60,
        )
        volume_type = await dialog.exec()
        return volume_type


def get_operators_menu_dict():
    return {
        "name": {
            "CAE Operators": [
                {
                    "name": "Points",
                    "onclick_fn": OperatorsPoints.onclick,
                    "show_fn": OperatorsPoints.show,
                    "enabled_fn": OperatorsPoints.enabled,
                },
                {
                    "name": "Faces",
                    "onclick_fn": OperatorsFaces.onclick,
                    "show_fn": OperatorsFaces.show,
                    "enabled_fn": OperatorsFaces.enabled,
                },
                {
                    "name": "Glyphs",
                    "onclick_fn": OperatorsGlyphs.onclick,
                    "show_fn": OperatorsGlyphs.show,
                    "enabled_fn": OperatorsGlyphs.enabled,
                },
                {
                    "name": "Streamlines",
                    "onclick_fn": OperatorsStreamlines.onclick,
                    "show_fn": OperatorsStreamlines.show,
                    "enabled_fn": OperatorsStreamlines.enabled,
                },
                {
                    "name": "Volume",
                    "onclick_fn": OperatorsVolume.onclick,
                    "show_fn": OperatorsVolume.show,
                    "enabled_fn": OperatorsVolume.enabled,
                },
            ]
        },
        "glyph": get_icon_path("menu"),
    }


def get_sources_menu_dict():
    return {
        "name": {
            "CAE Sources": [
                {
                    "name": "Bounding Box",
                    "onclick_fn": BoundingBox.onclick,
                    "enabled_fn": BoundingBox.enabled,
                    "show_fn": BoundingBox.show,
                },
                {
                    "name": "Unit Sphere",
                    "onclick_fn": UnitSphere.onclick,
                    "show_fn": UnitSphere.show,
                },
                {
                    "name": "Volume Slice",
                    "onclick_fn": VolumeSlice.onclick,
                    "show_fn": VolumeSlice.show,
                    "enabled_fn": VolumeSlice.enabled,
                },
            ]
        },
        "glyph": get_icon_path("menu"),
    }


class FlowEnvironment:

    @staticmethod
    def show(objects: dict) -> bool:
        return objects.get("stage") is not None

    @staticmethod
    @async_callback
    @select_result
    async def onclick(objects: dict):
        stage = objects.get("stage")
        assert stage is not None, "missing stage"

        layer_number = await FlowEnvironment.get_layer_number(stage)
        if layer_number is None:
            return None

        name = f"FlowSimulation_L{layer_number}"
        prim_path = get_stage_next_free_path(stage, get_anchor_path(stage).AppendChild(name), False)
        await execute_command("CreateCaeVizFlowEnvironment", prim_path=prim_path, layer_number=int(layer_number))

        return prim_path

    @staticmethod
    async def get_layer_number(stage: Usd.Stage) -> int:
        layer_numbers = get_flow_layer_numbers(stage)

        def validate_layer_number(layer_number: str) -> tuple[bool, str]:
            if not layer_number or not layer_number.strip():
                return False, "Layer number cannot be empty"
            try:
                layer_number = int(layer_number.strip())
                if layer_number < 0:
                    return False, "Layer number must be greater than or equal to 0"
                elif layer_number in layer_numbers:
                    return False, f"Layer number {layer_number} already exists"
                return True, ""
            except ValueError:
                return False, "Layer number must be an integer"

        dialog = ValidatedInputDialog(
            title="Choose Flow Layer",
            label=f"Enter the layer number for the flow environment.",
            validator=validate_layer_number,
            default_value=str(get_next_available_layer_number(stage)),
            field_label="Layer:",
            field_width=100,
        )
        layer_number = await dialog.exec()
        return int(layer_number) if layer_number is not None else None


class FlowFuelInjectorSphere:

    @staticmethod
    def show(objects: dict) -> bool:
        return objects.get("stage") is not None

    @staticmethod
    def enabled(objects: dict) -> bool:
        return FlowBoundary.enabled(objects)

    @staticmethod
    @async_callback
    @select_result
    async def onclick(objects: dict):
        stage = objects.get("stage")
        assert stage is not None, "missing stage"
        active_prims = get_active_prims(objects, lambda prim: prim.IsA(cae.DataSet) or prim.IsA(UsdGeom.Boundable))
        assert len(active_prims) > 0

        layer_number = await FlowFuelInjectorSphere.get_layer_selection(stage)
        if layer_number is None:
            return None
        simulation_prim = FlowFuelInjectorSphere.get_simulation_prim(stage, layer_number)

        # Create injector for all active prims
        created_paths = []
        name = f"FuelInjectorSphere_{active_prims[0].GetName()}" if len(active_prims) == 1 else "FuelInjectorSphere"
        parent_path = simulation_prim.GetPath() if simulation_prim else get_anchor_path(stage)
        prim_path = get_stage_next_free_path(stage, parent_path.AppendChild(name), False)
        await execute_command(
            "CreateCaeVizFlowSmokeInjector",
            prim_path=prim_path,
            layer_number=int(layer_number),
            mode="sphere",
            simulation_prim=simulation_prim,
            boundable_paths=[str(prim.GetPath()) for prim in active_prims],
        )
        created_paths.append(prim_path)

        return created_paths

    @staticmethod
    async def get_layer_selection(stage: Usd.Stage) -> int:
        layer_numbers = get_flow_layer_numbers(stage)
        if not layer_numbers:
            raise ValueError("No flow environment created. Please create a flow environment first.")

        if len(layer_numbers) == 1:
            return layer_numbers[0]

        dialog = TypeSelectionDialog(
            title="Choose Flow Layer",
            label="Select the flow layer:",
            options=[f"{layer_number}" for layer_number in layer_numbers],
            default_index=0,
            field_label="Layer:",
            field_width=50,
        )
        return await dialog.exec()

    @staticmethod
    def get_simulation_prim(stage: Usd.Stage, layer_number: int) -> Usd.Prim:
        flow_sim_prim_path = get_anchor_path(stage).AppendChild(f"FlowSimulation_L{layer_number}")
        flow_sim_prim = stage.GetPrimAtPath(flow_sim_prim_path)
        if flow_sim_prim and flow_sim_prim.IsValid():
            return flow_sim_prim
        logger.warning("FlowSimulation_L%s prim not found", layer_number)
        return None


class FlowDatasetInjector:

    @staticmethod
    def show(objects: dict) -> bool:
        return objects.get("stage") is not None

    @staticmethod
    def enabled(objects: dict) -> bool:
        stage = objects.get("stage")
        return schema_isa(cae.DataSet, objects) and stage and len(get_flow_layer_numbers(stage)) > 0

    @staticmethod
    @async_callback
    @select_result
    async def onclick(objects: dict):
        stage = objects.get("stage")
        assert stage is not None, "missing stage"
        active_dataset_prims = get_active_prims(objects, lambda prim: prim.IsA(cae.DataSet))
        assert len(active_dataset_prims) > 0

        layer_number = await FlowFuelInjectorSphere.get_layer_selection(stage)
        if layer_number is None:
            return None
        simulation_prim = FlowFuelInjectorSphere.get_simulation_prim(stage, layer_number)

        # Create injector for each dataset individually
        created_paths = []
        for dataset_prim in active_dataset_prims:
            name = f"DatasetInjector_{dataset_prim.GetName()}"
            parent_path = simulation_prim.GetPath() if simulation_prim else get_anchor_path(stage)
            prim_path = get_stage_next_free_path(stage, parent_path.AppendChild(name), False)
            await execute_command(
                "CreateCaeVizFlowDataSetEmitter",
                dataset_path=str(dataset_prim.GetPath()),
                prim_path=prim_path,
                layer_number=int(layer_number),
                simulation_prim=simulation_prim,
            )
            created_paths.append(prim_path)

        return created_paths


class FlowBoundary:

    @staticmethod
    def show(objects: dict) -> bool:
        return objects.get("stage") is not None

    @staticmethod
    def enabled(objects: dict) -> bool:
        stage = objects.get("stage")
        if not stage or len(get_flow_layer_numbers(stage)) == 0:
            return False

        if schema_isa(cae.DataSet, objects):
            return True

        if schema_isa(UsdGeom.Boundable, objects):
            return True

    @staticmethod
    @async_callback
    @select_result
    async def onclick(objects: dict):
        stage = objects.get("stage")
        assert stage is not None, "missing stage"
        active_prims = get_active_prims(objects, lambda prim: prim.IsA(cae.DataSet) or prim.IsA(UsdGeom.Boundable))
        assert len(active_prims) > 0

        layer_number = await FlowFuelInjectorSphere.get_layer_selection(stage)
        if layer_number is None:
            return None
        simulation_prim = FlowFuelInjectorSphere.get_simulation_prim(stage, layer_number)

        # Create boundary for all active prims
        created_paths = []
        name = f"BoundaryEmitter_{active_prims[0].GetName()}" if len(active_prims) == 1 else "BoundaryEmitter"
        parent_path = simulation_prim.GetPath() if simulation_prim else get_anchor_path(stage)
        prim_path = get_stage_next_free_path(stage, parent_path.AppendChild(name), False)
        await execute_command(
            "CreateCaeVizFlowBoundaryEmitter",
            boundable_paths=[str(prim.GetPath()) for prim in active_prims],
            prim_path=prim_path,
            layer_number=int(layer_number),
        )
        created_paths.append(prim_path)

        return created_paths


def get_flow_menu_dict():
    return {
        "name": {
            "CAE Flow": [
                {
                    "name": "Environment",
                    "onclick_fn": FlowEnvironment.onclick,
                    "show_fn": FlowEnvironment.show,
                },
                {},
                {
                    "name": "Fuel Injector (Sphere)",
                    "onclick_fn": FlowFuelInjectorSphere.onclick,
                    "show_fn": FlowFuelInjectorSphere.show,
                    "enabled_fn": FlowFuelInjectorSphere.enabled,
                },
                {
                    "name": "Dataset Injector",
                    "onclick_fn": FlowDatasetInjector.onclick,
                    "show_fn": FlowDatasetInjector.show,
                    "enabled_fn": FlowDatasetInjector.enabled,
                },
                {
                    "name": "Boundary",
                    "onclick_fn": FlowBoundary.onclick,
                    "show_fn": FlowBoundary.show,
                    "enabled_fn": FlowBoundary.enabled,
                },
            ]
        },
        "glyph": get_icon_path("menu"),
    }


def get_dataset_selection_suggestions(prims: list[Usd.Prim]) -> list[str]:
    """Generate suggestions for CaeVizDatasetSelectionAPI instance names.

    This provides contextual suggestions based on common dataset roles.
    """
    # Common dataset roles
    suggestions = ["source"]

    for prim in prims:
        if prim.HasAPI(cae_viz.StreamlinesAPI):
            suggestions.append("seeds")
    return suggestions


def get_field_selection_suggestions(prims: list[Usd.Prim]) -> list[str]:
    """Generate suggestions for CaeVizFieldSelectionAPI instance names.

    This provides contextual suggestions based on the applied API schemas.
    """
    # Common field roles
    suggestions = []

    for prim in prims:
        if prim.HasAPI(cae_viz.FacesAPI):
            suggestions.append("colors")
        if prim.HasAPI(cae_viz.StreamlinesAPI):
            suggestions.append("colors")
            suggestions.append("velocities")
            suggestions.append("widths")
        if prim.HasAPI(cae_viz.PointsAPI):
            suggestions.append("colors")
            suggestions.append("widths")
        if prim.HasAPI(cae_viz.GlyphsAPI):
            suggestions.append("colors")
            suggestions.append("orientations")
            suggestions.append("scales")
        if prim.HasAPI(cae_viz.IndeXVolumeAPI):
            suggestions.append("colors")
        if prim.IsA("FlowEmitterNanoVdb"):
            suggestions.append("velocities")
            suggestions.append("temperatures")

    return suggestions


def get_dependent_api_suggestions(
    prims: list[Usd.Prim], base_api_schema: str, extract_instance: bool = True
) -> list[str]:
    """Get instance names from a dependent API schema.

    For example, if applying CaeVizDatasetGaussianSplattingAPI, this will return
    all existing CaeVizDatasetSelectionAPI instance names.

    Args:
        prims: List of prims
        base_api_schema: The API schema to get instance names from
        extract_instance: If True, extract instance names; otherwise return full API names

    Returns:
        List of instance names from the base API schema
    """
    if not prims:
        return []

    # Get applied schemas from the first prim (assuming all prims have similar schemas)
    prim = prims[0]
    applied_schemas = prim.GetAppliedSchemas()

    instance_names = []
    for schema_str in applied_schemas:
        # Multi-apply schemas are in the format "APIName:instanceName"
        if ":" in schema_str:
            schema_name, instance_name = schema_str.split(":", 1)
            if schema_name == base_api_schema:
                instance_names.append(instance_name)

    return instance_names


def get_add_menu_dict():
    return {
        "name": {
            "CAE": [
                {
                    "name": "Operator Debugging",
                    "onclick_fn": partial(add_api, Usd.Typed, "CaeVizOperatorDebuggingAPI"),
                    "show_fn": partial(can_apply_api, "CaeVizOperatorDebuggingAPI"),
                },
                {
                    "name": "Operator Temporal",
                    "onclick_fn": partial(add_api, Usd.Typed, "CaeVizOperatorTemporalAPI"),
                    "show_fn": partial(can_apply_api, "CaeVizOperatorTemporalAPI"),
                },
                {
                    "name": "Dataset Selection",
                    "onclick_fn": partial(
                        add_api,
                        Usd.Typed,
                        "CaeVizDatasetSelectionAPI",
                        suggestions_provider=get_dataset_selection_suggestions,
                    ),
                    "show_fn": partial(can_apply_api, "CaeVizDatasetSelectionAPI"),
                },
                {
                    "name": "Dataset Gaussian Splatting",
                    "onclick_fn": partial(
                        add_api,
                        Usd.Typed,
                        "CaeVizDatasetGaussianSplattingAPI",
                        suggestions_provider=lambda prims: get_dependent_api_suggestions(
                            prims, "CaeVizDatasetSelectionAPI"
                        ),
                    ),
                    "show_fn": partial(can_apply_api, "CaeVizDatasetGaussianSplattingAPI"),
                },
                {
                    "name": "Dataset Voxelization",
                    "onclick_fn": partial(
                        add_api,
                        Usd.Typed,
                        "CaeVizDatasetVoxelizationAPI",
                        suggestions_provider=lambda prims: get_dependent_api_suggestions(
                            prims, "CaeVizDatasetSelectionAPI"
                        ),
                    ),
                    "show_fn": partial(can_apply_api, "CaeVizDatasetVoxelizationAPI"),
                },
                {
                    "name": "Dataset Temporal Traits",
                    "onclick_fn": partial(
                        add_api,
                        Usd.Typed,
                        "CaeVizDatasetTemporalTraitsAPI",
                        suggestions_provider=lambda prims: get_dependent_api_suggestions(
                            prims, "CaeVizDatasetSelectionAPI"
                        ),
                    ),
                    "show_fn": partial(can_apply_api, "CaeVizDatasetTemporalTraitsAPI"),
                },
                {
                    "name": "Field Selection",
                    "onclick_fn": partial(
                        add_api,
                        Usd.Typed,
                        "CaeVizFieldSelectionAPI",
                        suggestions_provider=get_field_selection_suggestions,
                    ),
                    "show_fn": partial(can_apply_api, "CaeVizFieldSelectionAPI"),
                },
                {
                    "name": "Field Mapping",
                    "onclick_fn": partial(
                        add_api,
                        Usd.Typed,
                        "CaeVizFieldMappingAPI",
                        suggestions_provider=lambda prims: get_dependent_api_suggestions(
                            prims, "CaeVizFieldSelectionAPI"
                        ),
                    ),
                    "show_fn": partial(can_apply_api, "CaeVizFieldMappingAPI"),
                },
                {
                    "name": "Field Thresholding",
                    "onclick_fn": partial(
                        add_api,
                        Usd.Typed,
                        "CaeVizFieldThresholdingAPI",
                        suggestions_provider=lambda prims: get_dependent_api_suggestions(
                            prims, "CaeVizFieldSelectionAPI"
                        ),
                    ),
                    "show_fn": partial(can_apply_api, "CaeVizFieldThresholdingAPI"),
                },
                # {
                #     "name": "Block List",
                #     "onclick_fn": partial(add_api, Usd.Typed, "CaeVizBlockListAPI"),
                #     "show_fn": partial(can_apply_api, "CaeVizBlockListAPI"),
                # },
            ]
        },
    }
