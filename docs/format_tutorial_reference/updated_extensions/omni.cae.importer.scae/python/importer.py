# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

import json
import os.path
import re
from logging import getLogger

import omni.client.utils as clientutils
from omni.cae.schema import cae
from omni.client import get_local_file_async
from omni.kit.tool.asset_importer import AbstractImporterDelegate
from omni.usd import get_context
from pxr import Sdf, Tf, Usd, UsdGeom, UsdUtils

logger = getLogger(__name__)


class ScaeAssetImporter(AbstractImporterDelegate):
    @property
    def name(self) -> str:
        return "Scae Importer"

    @property
    def filter_regexes(self) -> list[str]:
        return [r".*\.scae$"]

    @property
    def filter_descriptions(self) -> list[str]:
        return ["Scae Files (*.scae)"]

    def show_destination_frame(self):
        return True

    def supports_usd_stage_cache(self):
        return True

    def build_options(self, paths: list[str]) -> None:
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
        if import_as_reference:
            output_dir = export_folder if export_folder else os.path.dirname(path)
            name, _ = os.path.splitext(os.path.basename(path))
            usd_path = os.path.join(output_dir, f"{name}.usda")

            stage = Usd.Stage.CreateNew(usd_path)
            await self._populate_stage(stage, path)
            stage.Save()
            return usd_path
        stage = Usd.Stage.CreateInMemory()
        await self._populate_stage(stage, path)
        stage_id = UsdUtils.StageCache.Get().Insert(stage)
        return stage_id.ToString()

    @staticmethod
    def _normalize_name(name: str) -> str:
        return re.sub(r"[^a-z0-9]", "", name.lower())

    @classmethod
    def _is_coordinate_vector(cls, name: str) -> bool:
        normalized = cls._normalize_name(name)
        return normalized in {"coordinates", "coords", "points", "xyz", "gridcoordinates"}

    @classmethod
    def _coordinate_axis_index(cls, name: str) -> int | None:
        normalized = cls._normalize_name(name)
        if normalized in {"x", "coordx", "coordsx", "pointsx", "coordinatesx", "gridcoordinatesx"}:
            return 0
        if normalized in {"y", "coordy", "coordsy", "pointsy", "coordinatesy", "gridcoordinatesy"}:
            return 1
        if normalized in {"z", "coordz", "coordsz", "pointsz", "coordinatesz", "gridcoordinatesz"}:
            return 2
        return None

    async def _populate_stage(self, stage: Usd.Stage, path: str):
        world = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world.GetPrim())
        UsdGeom.SetStageUpAxis(stage, "Z")

        root = UsdGeom.Scope.Define(stage, world.GetPath().AppendChild(Tf.MakeValidIdentifier(os.path.basename(path))))
        root_path = root.GetPath()

        dataset = cae.DataSet.Define(stage, root_path.AppendChild("ScaeDataSet"))
        cae.PointCloudAPI.Apply(dataset.GetPrim())
        point_cloud_api = cae.PointCloudAPI(dataset)

        field_array_class = cae.ScaeFieldArray(stage.CreateClassPrim(root_path.AppendChild("ScaeFieldArrayClass")))
        field_array_class.CreateFileNamesAttr().Set([clientutils.make_file_url_if_possible(path)])
        field_array_class.CreateFieldAssociationAttr().Set(cae.Tokens.vertex)

        _, local_manifest = await get_local_file_async(path)
        with open(local_manifest, "r", encoding="utf-8") as stream:
            manifest = json.load(stream)
        arrays = manifest.get("arrays", {})
        if not isinstance(arrays, dict) or not arrays:
            raise RuntimeError(f"Scae manifest '{path}' does not contain a valid 'arrays' object")

        arrays_scope = UsdGeom.Scope.Define(stage, root_path.AppendChild("ScaeArrays"))
        coordinate_vector_target = None
        coordinate_component_targets = [None, None, None]

        for name in arrays.keys():
            field_array = cae.ScaeFieldArray.Define(
                stage, arrays_scope.GetPath().AppendChild(Tf.MakeValidIdentifier(name))
            )
            field_array.GetPrim().GetSpecializes().SetSpecializes([field_array_class.GetPath()])
            field_array.CreateArrayNameAttr().Set(name)
            if name in {"ElementConnectivity", "ElementStartOffset"}:
                field_array.CreateFieldAssociationAttr().Set(cae.Tokens.none)
            else:
                field_array.CreateFieldAssociationAttr().Set(cae.Tokens.vertex)

            if coordinate_vector_target is None and self._is_coordinate_vector(name):
                coordinate_vector_target = field_array.GetPath()
                continue

            axis_index = self._coordinate_axis_index(name)
            if axis_index is not None and coordinate_component_targets[axis_index] is None:
                coordinate_component_targets[axis_index] = field_array.GetPath()
                continue

            dataset.GetPrim().CreateRelationship(f"field:{Tf.MakeValidIdentifier(name)}").SetTargets(
                [field_array.GetPath()]
            )

        if coordinate_vector_target is not None:
            point_cloud_api.CreateCoordinatesRel().SetTargets([coordinate_vector_target])
            return

        if all(target is not None for target in coordinate_component_targets):
            point_cloud_api.CreateCoordinatesRel().SetTargets(coordinate_component_targets)
            return

        raise RuntimeError("Scae manifest must include coordinate array(s): 'Coordinates' or X/Y/Z components")


async def import_to_stage(path: str, prim_path: str) -> Usd.Prim:
    """
    Import a Scae file into a USD stage at the given prim path.

    Args:
        path: The path to the Scae manifest file.
        prim_path: The path to the prim to import the dataset into.

    Returns:
        The imported prim.
    """
    importer = ScaeAssetImporter()
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
