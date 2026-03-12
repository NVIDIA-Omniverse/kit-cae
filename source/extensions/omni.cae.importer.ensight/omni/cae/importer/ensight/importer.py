# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import asyncio
import os.path
from logging import getLogger

import omni.client.utils as clientutils
import omni.ui as ui
from omni.cae.data import progress
from omni.kit.tool.asset_importer import AbstractImporterDelegate
from omni.usd import get_context
from pxr import Sdf, Tf, Usd, UsdGeom, UsdUtils

logger = getLogger(__name__)


class Options:
    def __init__(self):
        self._time_scale = ui.SimpleFloatModel(1.0)
        self._time_offset = ui.SimpleFloatModel(0.0)

    def build_options(self, paths: list[str]):
        with ui.VStack(height=0, spacing=4):
            with ui.CollapsableFrame(
                "Time Options", build_header_fn=lambda _, title: ui.Label(title), height=0, collapsed=False
            ):
                with ui.VStack(height=0, spacing=4):
                    with ui.HStack(height=0, spacing=4):
                        ui.Label("Scale:", width=0)
                        ui.FloatField(name="TimeScale", model=self._time_scale, tooltip="Scale factor for time values.")
                    with ui.HStack(height=0, spacing=4):
                        ui.Label("Offset:", width=0)
                        ui.FloatField(
                            name="TimeOffset",
                            model=self._time_offset,
                            tooltip="Offset time values by this amount (applied after scaling)",
                        )

    @property
    def time_scale(self) -> float:
        return self._time_scale.get_value_as_float()

    @time_scale.setter
    def time_scale(self, value: float) -> None:
        self._time_scale.set_value(value)

    @property
    def time_offset(self) -> float:
        return self._time_offset.get_value_as_float()

    @time_offset.setter
    def time_offset(self, value: float) -> None:
        self._time_offset.set_value(value)


class EnSightGoldImporter(AbstractImporterDelegate):
    def __init__(self) -> None:
        super().__init__()
        self._options_map = {}

    @property
    def name(self):
        return "EnSight Gold Importer"

    @property
    def filter_regexes(self) -> list[str]:
        return [r".*\.case$"]

    @property
    def filter_descriptions(self) -> list[str]:
        return ["EnSight Gold files"]

    def show_destination_frame(self):
        return True

    def supports_usd_stage_cache(self):
        return True

    def get_options(self, path: str) -> Options:
        normalized_path = clientutils.normalize_url(path)
        if normalized_path not in self._options_map:
            self._options_map[normalized_path] = Options()
        return self._options_map[normalized_path]

    def build_options(self, paths: list[str]) -> None:
        for path in paths:
            self.get_options(path).build_options([path])

    async def convert_assets(self, paths: list[str], **kwargs):
        result = {}

        # we only support local assets for now.
        for path in filter(lambda uri: clientutils.is_local_url(uri), paths):
            normalized_path = clientutils.normalize_url(path)
            result[path] = await self._convert_asset(
                normalized_path, kwargs.get("import_as_reference"), kwargs.get("export_folder")
            )
        return result

    async def _convert_asset(self, path: str, import_as_reference: bool, export_folder: str):
        from omni.cae.delegate.ensight.ensight import TimeOffset, process_gold_case

        options = self.get_options(path)

        # Create TimeOffset object from options
        time_offset = TimeOffset()
        time_offset.offset = options.time_offset
        time_offset.scale = options.time_scale

        def populate_stage(stage: Usd.Stage):
            world = UsdGeom.Xform.Define(stage, "/World")
            stage.SetDefaultPrim(world.GetPrim())
            UsdGeom.SetStageUpAxis(stage, "Z")

            root = UsdGeom.Scope.Define(
                stage, world.GetPath().AppendChild(Tf.MakeValidIdentifier(os.path.basename(path)))
            )
            rootPath = root.GetPath()
            process_gold_case(stage, path, rootPath, time_offset)

        if import_as_reference:
            # when importing as reference, create a new stage file and then return that.
            output_dir = export_folder if export_folder else os.path.dirname(path)
            name, _ = os.path.splitext(os.path.basename(path))
            usd_path = os.path.join(output_dir, f"{name}.usda")
            # TODO: if file exists, warn!!!

            stage = Usd.Stage.CreateNew(usd_path)
            with progress.ProgressContext(f"Importing {os.path.basename(path)}") as context:
                context.notify(0.1)
                await asyncio.to_thread(populate_stage, stage)
                context.notify(0.9)
                stage.Save()
            return usd_path
        else:
            # when adding directly to stage, just create an in memory stage
            # and return its id
            stage = Usd.Stage.CreateInMemory()
            with progress.ProgressContext(f"Importing {os.path.basename(path)}") as context:
                context.notify(0.1)
                await asyncio.to_thread(populate_stage, stage)
                context.notify(0.9)
                stage_id = UsdUtils.StageCache.Get().Insert(stage)
            return stage_id.ToString()


async def import_to_stage(
    path: str,
    prim_path: str,
    *,
    time_scale: float = 1.0,
    time_offset: float = 0.0,
) -> Usd.Prim:
    """
    Import an EnSight Gold file into a USD stage at the given prim path.

    Args:
        path: The path to the EnSight Gold case file.
        prim_path: The path to the prim to import the EnSight file into.
        time_scale: Scale factor for time values.
        time_offset: Offset time values by this amount (applied after scaling).

    Returns:
        The imported prim.
    """
    importer = EnSightGoldImporter()
    if options := importer.get_options(path):
        options.time_scale = time_scale
        options.time_offset = time_offset
    else:
        raise RuntimeError(f"No options found for path: {path}")

    result = await importer.convert_assets([path], import_as_reference=False)
    stage_id = next(iter(result.values()))
    cache: Usd.StageCache = UsdUtils.StageCache.Get()
    cached_stage = cache.Find(Usd.StageCache.Id.FromString(stage_id))

    root_prim = cached_stage.GetDefaultPrim().GetChildren()[0]

    stage = get_context().get_stage()
    prim = stage.DefinePrim(prim_path)
    Sdf.CopySpec(
        cached_stage.GetRootLayer(),
        root_prim.GetPath(),
        stage.GetRootLayer(),
        prim.GetPath(),
    )
    return prim
