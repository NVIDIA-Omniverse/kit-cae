# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
#  its affiliates is strictly prohibited.

import asyncio
import os.path
from logging import getLogger

import omni.client.utils as clientutils
import omni.kit.tool.asset_importer as ai
import omni.ui as ui
from pxr import Usd, UsdUtils

logger = getLogger(__name__)

from omni.cae.data import progress


class VTKImporterOptions:
    import_as_timeseries = False


class VTKImporterOptionsBuilder:
    def __init__(self):
        self._options = VTKImporterOptions()
        self._clear()

    def _clear(self):
        self._built = False
        self._import_as_timeseries_checkbox = None
        self._import_as_timeseries_checkbox = None

    def build_pane(self, asset_paths: list[str]):
        self._options = self.get_import_options()
        self._built = True
        with ui.VStack(height=0, spacing=4):
            self._import_as_timeseries_checkbox = self._build_option_checkbox(
                "Import as Time Series", self._options.import_as_timeseries, "Treat all selected files as a time series"
            )

    def _build_option_checkbox(self, text, default_value, tooltip="", indent=0):
        with ui.HStack(height=0):
            if indent:
                ui.Spacer(width=indent, height=0)
            checkbox = ui.CheckBox(width=20)
            checkbox.model.set_value(default_value)
            label = ui.Label(text, alignment=ui.Alignment.LEFT, word_wrap=True)
            if tooltip:
                label.set_tooltip(tooltip)
            return checkbox

    def get_import_options(self):
        context = VTKImporterOptions()
        if self._built:
            context.import_as_timeseries = self._import_as_timeseries_checkbox.model.get_value_as_bool()
        return context


class VTKImporter(ai.AbstractImporterDelegate):
    def __init__(self):
        super().__init__()
        self._options_builder = VTKImporterOptionsBuilder()

    @property
    def name(self) -> str:
        return "CAE VTI Importer"

    @property
    def filter_regexes(self) -> list[str]:
        return [r".*\.vt[kiu]$"]

    @property
    def filter_descriptions(self) -> list[str]:
        return ["VTK Files (*.vtk, *.vti, *.vtu)"]

    def show_destination_frame(self):
        return True

    def supports_usd_stage_cache(self):
        return True

    def build_options(self, paths: list[str]) -> None:
        self._options_builder.build_pane(paths)

    async def convert_assets(self, paths: list[str], **kwargs):
        result = {}
        # we only support local assets for now.
        for path in filter(lambda uri: clientutils.is_local_url(uri), paths):
            normalized_path = clientutils.normalize_url(path)
            if converted_path := await self._convert_asset(
                normalized_path, kwargs.get("import_as_reference"), kwargs.get("export_folder")
            ):
                result[path] = converted_path
        return result

    async def _convert_asset(self, path: str, import_as_reference: bool, export_folder: str):
        from . import impl

        with progress.ProgressContext(f"Importing {os.path.basename(path)}"):
            if import_as_reference:
                # when importing as reference, create a new stage file and then return that.
                output_dir = export_folder if export_folder else os.path.dirname(path)
                name, _ = os.path.splitext(os.path.basename(path))
                usd_path = os.path.join(output_dir, f"{name}.usda")
                # TODO: if file exists, warn!!!

                stage = Usd.Stage.CreateNew(usd_path)
                await asyncio.to_thread(impl.populate_stage, path, stage)
                stage.Save()
                return usd_path
            else:
                # when adding directly to stage, just create an in memory stage
                # and return its id
                stage = Usd.Stage.CreateInMemory()
                await asyncio.to_thread(impl.populate_stage, path, stage)
                stage_id = UsdUtils.StageCache.Get().Insert(stage)
                return stage_id.ToString()
