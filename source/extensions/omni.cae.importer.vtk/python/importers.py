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
import omni.kit.tool.asset_importer as ai
import omni.ui as ui
from omni.cae.data import progress
from omni.client import get_local_file_async
from omni.usd import get_context
from pxr import Sdf, Usd, UsdUtils

logger = getLogger(__name__)


class Options:
    def __init__(self):
        # Placeholder for future options
        pass

    def build_options(self, paths: list[str]):
        with ui.VStack(height=0, spacing=4):
            # Placeholder for future options UI
            pass


class VTKImporter(ai.AbstractImporterDelegate):
    def __init__(self):
        super().__init__()
        self._options_map = {}

    @property
    def name(self) -> str:
        return "CAE VTK Importer"

    @property
    def filter_regexes(self) -> list[str]:
        return [r".*\.vt[kiusp]$"]

    @property
    def filter_descriptions(self) -> list[str]:
        return ["VTK Files (*.vtk, *.vti, *.vtu, *.vts, *.vtp)"]

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
        from . import impl

        with progress.ProgressContext(f"Downloading {os.path.basename(path)}"):
            _, local_path = await get_local_file_async(path)

        with progress.ProgressContext(f"Importing {os.path.basename(path)}"):
            if import_as_reference:
                # when importing as reference, create a new stage file and then return that.
                output_dir = export_folder if export_folder else os.path.dirname(path)
                name, _ = os.path.splitext(os.path.basename(path))
                usd_path = os.path.join(output_dir, f"{name}.usda")
                # TODO: if file exists, warn!!!

                stage = Usd.Stage.CreateNew(usd_path)
                await asyncio.to_thread(impl.populate_stage, path, local_path, stage)
                stage.Save()
                return usd_path
            else:
                # when adding directly to stage, just create an in memory stage
                # and return its id
                stage = Usd.Stage.CreateInMemory()
                await asyncio.to_thread(impl.populate_stage, path, local_path, stage)
                stage_id = UsdUtils.StageCache.Get().Insert(stage)
                return stage_id.ToString()


async def import_to_stage(path: str, prim_path: str) -> Usd.Prim:
    """
    Import a VTK file into a USD stage at the given prim path.

    Args:
        path: The path to the VTK file.
        prim_path: The path to the prim to import the VTK file into.

    Returns:
        The imported prim.
    """
    importer = VTKImporter()
    if options := importer.get_options(path):
        # Options can be configured here if needed in the future
        pass
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
