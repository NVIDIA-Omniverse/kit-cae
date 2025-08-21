# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
#  its affiliates is strictly prohibited.

__all__ = ["Extension"]

import os.path

# import omni.ui as ui
import omni.client.utils as clientutils
import omni.ext
import omni.kit.tool.asset_importer as ai
from omni.cae.data import progress
from omni.client import get_local_file_async
from pxr import Sdf, Tf, Usd, UsdUtils


class CGNSAssetImporter(ai.AbstractImporterDelegate):
    def __init__(self) -> None:
        super().__init__()

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

    def build_options(self, paths: list[str]) -> None:
        # with ui.VStack(height=0):
        #     ui.Label("test option")
        #     ui.Label("test option2")
        pass

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

        # we pass rootName as an argument to the layer so that it can be used
        # to create a valid identifier for the root prim. Otherwise, for remote files this ends up using
        # the name for the local copy in cache to create the root prim name, which is not correct.
        source_layer: Sdf.Layer = Sdf.Layer.FindOrOpen(
            str(local_path), {"rootName": Tf.MakeValidIdentifier(os.path.basename(path)), "assetPath": path}
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


class Extension(omni.ext.IExt):
    def on_startup(self, ext_id):
        self._importer = CGNSAssetImporter()
        ai.register_importer(self._importer)

    def on_shutdown(self):
        ai.remove_importer(self._importer)
