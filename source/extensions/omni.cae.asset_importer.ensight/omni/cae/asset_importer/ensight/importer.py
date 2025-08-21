# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
#  its affiliates is strictly prohibited.

import os.path
from logging import getLogger

import omni.client.utils as clientutils
from omni.kit.tool.asset_importer import AbstractImporterDelegate
from pxr import Tf, Usd, UsdGeom, UsdUtils

logger = getLogger(__name__)


class EnSightGoldImporter(AbstractImporterDelegate):

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

    def build_options(self, paths: list[str]) -> None:
        pass

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
        from omni.cae.ensight.ensight import process_gold_case

        def populate_stage(stage: Usd.Stage):
            world = UsdGeom.Xform.Define(stage, "/World")
            stage.SetDefaultPrim(world.GetPrim())
            UsdGeom.SetStageUpAxis(stage, "Z")

            root = UsdGeom.Scope.Define(
                stage, world.GetPath().AppendChild(Tf.MakeValidIdentifier(os.path.basename(path)))
            )
            rootPath = root.GetPath()
            process_gold_case(stage, path, rootPath)

        if import_as_reference:
            # when importing as reference, create a new stage file and then return that.
            output_dir = export_folder if export_folder else os.path.dirname(path)
            name, _ = os.path.splitext(os.path.basename(path))
            usd_path = os.path.join(output_dir, f"{name}.usda")
            # TODO: if file exists, warn!!!

            stage = Usd.Stage.CreateNew(usd_path)
            populate_stage(stage)
            stage.Save()
            return usd_path
        else:
            # when adding directly to stage, just create an in memory stage
            # and return its id
            stage = Usd.Stage.CreateInMemory()
            populate_stage(stage)
            stage_id = UsdUtils.StageCache.Get().Insert(stage)
            return stage_id.ToString()
