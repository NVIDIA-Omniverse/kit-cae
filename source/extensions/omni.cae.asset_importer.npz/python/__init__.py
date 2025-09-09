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
import re
from logging import getLogger

import numpy as np
import omni.client.utils as clientutils
import omni.ext
import omni.kit.tool.asset_importer as ai
import omni.ui as ui
from omni.cae.data import progress
from omni.cae.schema import cae, sids
from omni.client import get_local_file_async
from pxr import Tf, Usd, UsdGeom, UsdUtils

from .minimal_model import MinimalModal

logger = getLogger(__name__)


# For more examples on how to write such options dialogs
# refer to `omni.kit.tool.asset_importer/../builtin_options_builder.py`
class AssetContext:
    def __init__(self):
        self.mesh_schema_type = "SIDS Unstructured"
        self.allow_pickle = False


class ImporterOptionsBuilder:
    def __init__(self) -> None:
        self._context: AssetContext = AssetContext()
        self._mesh_schema_combo_box: ui.ComboBox = None
        self._allow_pickle_check_box: ui.CheckBox = None
        self._clear()

    def _clear(self):
        self._mesh_schema_combo_box = None

    def build_pane(self, asset_paths: list[str]):
        context = self.get_import_context()
        self._built = True
        with ui.VStack(height=0, spacing=4):
            models = ["SIDS Unstructured", "Point Cloud"]  # more can be added e.g. "SIDS Structured"
            self._mesh_schema_combo_box = self._build_option_combobox(
                default_index=models.index(context.mesh_schema_type),
                items=models,
                text="Mesh Schema",
                tooltip="Select schema to use for the CAEMesh.",
            )
            self._allow_pickle_check_box = self._build_option_checkbox(
                text="Allow Pickle",
                default_value=context.allow_pickle,
                tooltip="Allow 'pickle' for reading NPY files (only enable for trusted files)",
            )

    def get_import_context(self):
        context = self._context
        # if build_pane was called, ensure we update the context with the current values
        if self._mesh_schema_combo_box is not None:
            context.mesh_schema_type = self._mesh_schema_combo_box.model.current_text
        if self._allow_pickle_check_box is not None:
            context.allow_pickle = self._allow_pickle_check_box.model.get_value_as_bool()
        return context

    def _build_option_combobox(self, default_index, items, text, tooltip="", indent=0):
        model = MinimalModal(default_index, items)
        with ui.HStack(width=0, spacing=4):
            if indent:
                ui.Spacer(width=indent, height=0)
            # can't use model
            combobox = ui.ComboBox(model, width=200)
            label = ui.Label(text, alignment=ui.Alignment.LEFT, word_wrap=True)
            if tooltip:
                label.set_tooltip(tooltip)
            return combobox

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


class NPZAssetImporter(ai.AbstractImporterDelegate):
    def __init__(self) -> None:
        super().__init__()
        self._options_builder = ImporterOptionsBuilder()

    @property
    def name(self) -> str:
        return "CAE NumPy Importer"

    @property
    def filter_regexes(self) -> list[str]:
        return [r".*\.np[yz]$"]

    @property
    def filter_descriptions(self) -> list[str]:
        return ["NumPy Files (*.npz, *.npz)"]

    def show_destination_frame(self):
        return True

    def supports_usd_stage_cache(self):
        return True

    def build_options(self, paths: list[str]) -> None:
        self._options_builder.build_pane(paths)

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
        import_options = self._options_builder.get_import_context()

        def _make_valid(name: str):
            return Tf.MakeValidIdentifier(name)

        async def populate_stage(stage: Usd.Stage):
            world = UsdGeom.Xform.Define(stage, "/World")
            stage.SetDefaultPrim(world.GetPrim())
            UsdGeom.SetStageUpAxis(stage, "Z")

            root = UsdGeom.Scope.Define(stage, world.GetPath().AppendChild(_make_valid(os.path.basename(path))))
            rootPath = root.GetPath()
            caeFieldArrayClass = cae.NumPyFieldArray(
                stage.CreateClassPrim(rootPath.AppendChild("NumPyFieldArrayClass"))
            )
            caeFieldArrayClass.CreateFileNamesAttr().Set([clientutils.make_file_url_if_possible(path)])
            caeFieldArrayClass.CreateAllowPickleAttr().Set(import_options.allow_pickle)
            caeFieldArrayClass.CreateFieldAssociationAttr()

            if import_options.mesh_schema_type == "SIDS Unstructured":
                logger.info("Importing as 'SIDS Unstructured'")
                dataset = cae.DataSet.Define(stage, rootPath.AppendChild("NumPyDataSet"))
                sids.UnstructuredAPI.Apply(dataset.GetPrim())
                sidsAPI = sids.UnstructuredAPI(dataset.GetPrim())
                sidsAPI.CreateElementTypeAttr()
                sidsAPI.CreateElementConnectivityRel()
                sidsAPI.CreateElementStartOffsetRel()
                sidsAPI.CreateGridCoordinatesRel()
            elif import_options.mesh_schema_type == "Point Cloud":
                logger.info("Importing as 'Point Cloud'")
                dataset = cae.DataSet.Define(stage, rootPath.AppendChild("NumPyDataSet"))
                cae.PointCloudAPI.Apply(dataset.GetPrim())
                pcAPI = cae.PointCloudAPI(dataset)
                pcAPI.CreateCoordinatesRel()
                caeFieldArrayClass.CreateFieldAssociationAttr().Set(cae.Tokens.vertex)
            else:
                raise RuntimeError("Unsupported schema: %s", import_options.mesh_schema_type)

            scope = UsdGeom.Scope.Define(stage, rootPath.AppendChild("NumPyArrays"))
            logger.info("about to import %s, %s", path, import_options.allow_pickle)
            with progress.ProgressContext(f"Downloading dataset .../{os.path.basename(path)}"):
                _, local_file = await get_local_file_async(path)
            np_file = np.load(local_file, allow_pickle=import_options.allow_pickle)
            data = None
            logger.info("successfully opened")
            if hasattr(np_file, "files"):  # is a NpzFile
                logger.info("is an NPZFile")
                names = list(np_file.files)
                data = np_file
                # np_file.close()
            elif isinstance(np_file, np.ndarray) and np_file.dtype == object and isinstance(np_file.item(0), dict):
                logger.info("is ndarray with a dict")
                data = np_file.item(0)
                names = data.keys()
            else:
                raise RuntimeError("Unsupported file '%s'." % path)

            if import_options.mesh_schema_type == "SIDS Unstructured":
                # handle special arrays first.
                if "element_range" in names:
                    erange = data["element_range"]
                    if erange.ndim == 1 and erange.shape[0] == 2:
                        sidsAPI.CreateElementRangeStartAttr().Set(int(erange[0]))
                        sidsAPI.CreateElementRangeEndAttr().Set(int(erange[1]))
                    names.remove("element_range")
                if "element_type" in names:
                    etype = data["element_type"]
                    if etype.ndim == 1 and etype.shape[0] == 1:
                        token_map = {
                            2: sids.Tokens.NODE,
                            3: sids.Tokens.BAR_2,
                            5: sids.Tokens.TRI_3,
                            7: sids.Tokens.QUAD_4,
                            10: sids.Tokens.TETRA_4,
                            12: sids.Tokens.PYRA_5,
                            14: sids.Tokens.PENTA_6,
                            17: sids.Tokens.HEXA_8,
                            20: sids.Tokens.MIXED,
                        }
                        sidsAPI.CreateElementTypeAttr().Set(token_map.get(etype[0]))
                    else:
                        sidsAPI.CreateElementTypeAttr().Set(sids.Tokens.MIXED)
                    names.remove("element_type")

            coords = [None] * 3
            for name in names:
                fieldArray = cae.NumPyFieldArray.Define(stage, scope.GetPath().AppendChild(_make_valid(name)))
                fieldArray.GetPrim().GetSpecializes().SetSpecializes([caeFieldArrayClass.GetPath()])
                fieldArray.CreateArrayNameAttr().Set(name)

                if relName := self._detect_default_relationship(name):
                    if relName == "cae:sids:gridCoordinatesX":
                        coords[0] = fieldArray.GetPath()
                    elif relName == "cae:sids:gridCoordinatesY":
                        coords[1] = fieldArray.GetPath()
                    elif relName == "cae:sids:gridCoordinatesZ":
                        coords[2] = fieldArray.GetPath()
                    elif relName == "cae:sids:gridCoordinates":
                        coords[0] = fieldArray.GetPath()
                    else:
                        if dataset.GetPrim().HasRelationship(relName):
                            dataset.GetPrim().GetRelationship(relName).SetTargets({fieldArray.GetPath()})
                        else:
                            dataset.GetPrim().CreateRelationship(f"field:{_make_valid(name)}").SetTargets(
                                {fieldArray.GetPath()}
                            )
                else:
                    # add field array as a field to the mesh.
                    # dataset.CreateFieldRelationship(_make_valid(name), fieldArray.GetPath())
                    dataset.GetPrim().CreateRelationship(f"field:{_make_valid(name)}").SetTargets(
                        {fieldArray.GetPath()}
                    )
            coords = [x for x in filter(lambda i: i is not None, coords)]
            if coords:
                if import_options.mesh_schema_type == "SIDS Unstructured":
                    sidsAPI.GetGridCoordinatesRel().SetTargets(coords)
                elif import_options.mesh_schema_type == "Point Cloud":
                    pcAPI.GetCoordinatesRel().SetTargets(coords)

        if import_as_reference:
            # when importing as reference, create a new stage file and then return that.
            output_dir = export_folder if export_folder else os.path.dirname(path)
            name, _ = os.path.splitext(os.path.basename(path))
            usd_path = os.path.join(output_dir, f"{name}.usda")
            # TODO: if file exists, warn!!!

            stage = Usd.Stage.CreateNew(usd_path)
            await populate_stage(stage)
            stage.Save()
            return usd_path
        else:
            # when adding directly to stage, just create an in memory stage
            # and return its id
            stage = Usd.Stage.CreateInMemory()
            await populate_stage(stage)
            stage_id = UsdUtils.StageCache.Get().Insert(stage)
            return stage_id.ToString()

    def _detect_default_relationship(self, name):
        clean_name = re.sub(r"[^a-z0-9\s]", "", name.lower())
        items = {
            "cae:sids:elementConnectivity": [
                "elementconnectivity",
                "connectivity",
                "conn",
                "con",
                "elementconnectivity",
            ],
            "cae:sids:elementStartOffset": ["elementstartoffset", "startoffsets", "startoffset"],
            "cae:sids:gridCoordinates": ["gridcoordinates", "xyz", "coords", "points", "coordinates"],
            "cae:sids:gridCoordinatesX": ["gridcoordinatesx", "x", "coordsx", "coords0", "pointsx", "points0"],
            "cae:sids:gridCoordinatesY": ["gridcoordinatesy", "y", "coordsy", "coords1", "pointsy", "points1"],
            "cae:sids:gridCoordinatesZ": ["gridcoordinatesz", "z", "coordsz", "coords2", "pointsz", "points2"],
        }
        for key, choices in items.items():
            if clean_name.lower() in choices:
                return key
        return None


class Extension(omni.ext.IExt):
    def on_startup(self, ext_id):
        self._importer = NPZAssetImporter()
        ai.register_importer(self._importer)

    def on_shutdown(self):
        ai.remove_importer(self._importer)
