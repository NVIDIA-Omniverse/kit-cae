# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os.path
from logging import getLogger

import omni.client.utils as clientutils
import omni.ui as ui
from omni.cae.data import progress
from omni.cae.importer.npz.minimal_model import MinimalModal
from omni.client import get_local_file_async
from omni.kit.tool.asset_importer import AbstractImporterDelegate
from omni.usd import get_context
from pxr import Sdf, Tf, Usd, UsdUtils

logger = getLogger(__name__)


class Options:
    def __init__(self):
        self._time_scale = ui.SimpleFloatModel(1.0)
        self._time_offset = ui.SimpleFloatModel(0.0)
        self._time_source_model = MinimalModal(0, ["TimeStep", "TimeValue"])

    def build_options(self, paths: list[str]):
        with ui.VStack(height=0, spacing=4):
            with ui.CollapsableFrame(
                "Time Options", build_header_fn=lambda _, title: ui.Label(title), height=0, collapsed=False
            ):
                with ui.VStack(height=0, spacing=4):
                    with ui.HStack(height=0, spacing=4):
                        ui.Label("Source:", width=0)
                        combobox = ui.ComboBox(self._time_source_model)
                        combobox.set_tooltip("Choose the time source: Time step, Time values")
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

    @property
    def time_source(self) -> str:
        return self._time_source_model.current_text

    @time_source.setter
    def time_source(self, value: str) -> None:
        self._time_source_model.current_text = value


class CGNSAssetImporter(AbstractImporterDelegate):
    def __init__(self) -> None:
        super().__init__()
        self._options_map = {}

    @property
    def name(self) -> str:
        return "CAE CGNS Importer"

    @property
    def filter_regexes(self) -> list[str]:
        return [r".*\.cgns$"]

    @property
    def filter_descriptions(self) -> list[str]:
        return ["CGNS Files (*.cgns)"]

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
        for path in paths:
            normalized_path = clientutils.normalize_url(path)
            if converted_path := await self._convert_asset(
                normalized_path, kwargs.get("import_as_reference"), kwargs.get("export_folder")
            ):
                result[path] = converted_path
        return result

    async def _convert_asset(self, path: str, import_as_reference: bool, export_folder: str):
        # open the CGNS file, using CGNSFileFormat USD plugin.
        with progress.ProgressContext("Downloading .../%s" % os.path.basename(path)):
            _, local_path = await get_local_file_async(path)

        options = self.get_options(path)

        # we pass rootName as an argument to the layer so that it can be used
        # to create a valid identifier for the root prim. Otherwise, for remote files this ends up using
        # the name for the local copy in cache to create the root prim name, which is not correct.
        source_layer: Sdf.Layer = Sdf.Layer.FindOrOpen(
            str(local_path),
            {
                "rootName": Tf.MakeValidIdentifier(os.path.basename(path)),
                "assetPath": clientutils.make_file_url_if_possible(path),
                "timeScale": str(options.time_scale),
                "timeOffset": str(options.time_offset),
                "timeSource": str(options.time_source),
            },
        )

        if import_as_reference:
            # when importing as reference, create a new stage file and then return that.
            output_dir = export_folder if export_folder else os.path.dirname(path)
            name, _ = os.path.splitext(os.path.basename(path))
            usd_path = os.path.join(output_dir, f"{name}.usda")
            # TODO: if file exists, warn!!!

            stage = Usd.Stage.CreateNew(usd_path)
            stage.GetRootLayer().TransferContent(source_layer)
            stage.Save()
            return usd_path
        else:
            # when adding directly to stage, just create an in memory stage
            # and return its id
            stage = Usd.Stage.Open(source_layer)
            stage_id = UsdUtils.StageCache.Get().Insert(stage)
            return stage_id.ToString()


async def import_to_stage(
    path: str,
    prim_path: str,
    *,
    time_scale: float = 1.0,
    time_offset: float = 0.0,
    time_source: str = "TimeStep",
) -> Usd.Prim:
    """
    Import a CGNS file into a USD stage at the given prim path.

    Args:
        path: The path to the CGNS file.
        prim_path: The path to the prim to import the CGNS file into.
        time_scale: Scale factor for time values.
        time_offset: Offset time values by this amount (applied after scaling).
        time_source: The time source to use ("TimeStep" or "TimeValue").

    Returns:
        The imported prim.
    """
    importer = CGNSAssetImporter()
    if options := importer.get_options(path):
        options.time_scale = time_scale
        options.time_offset = time_offset
        options.time_source = time_source
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
