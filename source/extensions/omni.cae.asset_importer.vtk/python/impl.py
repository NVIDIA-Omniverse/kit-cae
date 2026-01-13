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

from omni.cae.schema import cae
from omni.cae.schema import vtk as cae_vtk
from omni.client import utils as clientutils
from pxr import Gf, Tf, Usd, UsdGeom
from vtkmodules.numpy_interface import dataset_adapter as dsa
from vtkmodules.vtkCommonDataModel import vtkDataObject
from vtkmodules.vtkCommonExecutionModel import vtkStreamingDemandDrivenPipeline as vtkSDDP
from vtkmodules.vtkIOLegacy import vtkDataSetReader
from vtkmodules.vtkIOXML import vtkXMLImageDataReader, vtkXMLUnstructuredGridReader

logger = getLogger(__name__)


# open the VTI file
def _make_valid(name: str):
    return Tf.MakeValidIdentifier(name)


def _add_arrays(dataset: cae.DataSet, stage, scope: UsdGeom.Scope, vtk_arrays: list[str], assoc, primClass):
    for vtk_name in vtk_arrays:
        name = _make_valid(vtk_name)
        fieldArray = cae_vtk.FieldArray.Define(stage, scope.GetPath().AppendChild(name))
        fieldArray.GetPrim().GetSpecializes().SetSpecializes([primClass.GetPath()])
        fieldArray.CreateFieldAssociationAttr().Set(assoc)
        fieldArray.CreateArrayNameAttr().Set(vtk_name)
        dataset.GetPrim().CreateRelationship(f"field:{name}").SetTargets({fieldArray.GetPath()})


def _add_special_array(name: str, dataset: cae.DataSet, stage, primClass, specialType):
    name = _make_valid(name)
    fieldArray = cae_vtk.FieldArray.Define(stage, dataset.GetPath().AppendChild(name))
    fieldArray.GetPrim().GetSpecializes().SetSpecializes([primClass.GetPath()])
    fieldArray.CreateFieldAssociationAttr().Set(cae.Tokens.none)
    fieldArray.CreateArrayNameAttr().Set(name)
    fieldArray.CreateSpecialAttr().Set(specialType)
    # dataset.GetPrim().CreateRelationship(f"field:{name}").SetTargets({fieldArray.GetPath()})
    return fieldArray


def populate_stage(path: str, stage: Usd.Stage):
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.SetStageUpAxis(stage, "Z")

    root = UsdGeom.Scope.Define(stage, world.GetPath().AppendChild(_make_valid(os.path.basename(path))))
    rootPath = root.GetPath()

    ext = os.path.splitext(path)[-1].lower()
    if ext == ".vti":
        reader = vtkXMLImageDataReader()
        reader.SetFileName(path)
        reader.UpdateInformation()
        pds = reader.GetPointDataArraySelection()
        cds = reader.GetCellDataArraySelection()
        point_arrays = [pds.GetArrayName(i) for i in range(pds.GetNumberOfArrays())]
        cell_arrays = [cds.GetArrayName(i) for i in range(cds.GetNumberOfArrays())]
    elif ext == ".vtu":
        reader = vtkXMLUnstructuredGridReader()
        reader.SetFileName(path)
        reader.UpdateInformation()
        pds = reader.GetPointDataArraySelection()
        cds = reader.GetCellDataArraySelection()
        point_arrays = [pds.GetArrayName(i) for i in range(pds.GetNumberOfArrays())]
        cell_arrays = [cds.GetArrayName(i) for i in range(cds.GetNumberOfArrays())]
    elif ext == ".vtk":
        reader = vtkDataSetReader()
        # for legacy VTK reader, there's no API to get arrays info without reading the dataset :/
        # so we just read it.
        reader.SetFileName(path)
        reader.ReadAllScalarsOn()
        reader.ReadAllVectorsOn()
        reader.ReadAllNormalsOn()
        reader.ReadAllTensorsOn()
        reader.Update()
        output = dsa.WrapDataObject(reader.GetOutputDataObject(0))
        point_arrays = output.PointData.keys()
        cell_arrays = output.CellData.keys()
    else:
        raise ValueError(f"Unknown file extension: {ext}")

    caeFieldArrayClass = cae_vtk.FieldArray(stage.CreateClassPrim(rootPath.AppendChild("VTKFieldArrayClass")))
    caeFieldArrayClass.CreateFileNamesAttr().Set([clientutils.make_file_url_if_possible(path)])

    reader.UpdateDataObject()
    output = reader.GetOutputDataObject(0)

    if output.IsA("vtkImageData"):
        outInfo = reader.GetOutputInformation(0)
        spacing = outInfo.Get(vtkDataObject.SPACING())
        wholeExtent = list(outInfo.Get(vtkSDDP.WHOLE_EXTENT()))

        if len(point_arrays) == 0 and len(cell_arrays) > 0:
            # if no point array are present, we present a dual grid instead to avoid cell-2-point conversion
            logger.info("Reading as cell-centered dual since no point-centered arrays are present.")
            wholeExtent[1] -= 1
            wholeExtent[3] -= 1
            wholeExtent[5] -= 1
            point_arrays = cell_arrays
            cell_arrays = []

        logger.info("Importing as 'DenseVolume'")
        dataset = cae.DataSet.Define(stage, rootPath.AppendChild("VTKImageData"))
        cae.DenseVolumeAPI.Apply(dataset.GetPrim())
        denseVolumeAPI = cae.DenseVolumeAPI(dataset)
        denseVolumeAPI.CreateSpacingAttr().Set(spacing)
        denseVolumeAPI.CreateMinExtentAttr().Set(Gf.Vec3i(wholeExtent[0], wholeExtent[2], wholeExtent[4]))
        denseVolumeAPI.CreateMaxExtentAttr().Set(Gf.Vec3i(wholeExtent[1], wholeExtent[3], wholeExtent[5]))
    elif output.IsA("vtkUnstructuredGrid"):
        logger.info("Importing as 'UnstructuredGrid'")
        dataset = cae.DataSet.Define(stage, rootPath.AppendChild("VTKUnstructuredGrid"))
        cae_vtk.UnstructuredGridAPI.Apply(dataset.GetPrim())
        cae.PointCloudAPI.Apply(dataset.GetPrim())
        unstructuredGridAPI = cae_vtk.UnstructuredGridAPI(dataset)
        pcAPI = cae.PointCloudAPI(dataset)

        # add special arrays first.
        faPrim = _add_special_array("Points", dataset, stage, caeFieldArrayClass, cae_vtk.Tokens.points)
        unstructuredGridAPI.CreatePointsRel().SetTargets({faPrim.GetPath()})
        pcAPI.CreateCoordinatesRel().SetTargets({faPrim.GetPath()})

        faPrim = _add_special_array(
            "ConnectivityOffsets", dataset, stage, caeFieldArrayClass, cae_vtk.Tokens.connectivity_offsets
        )
        unstructuredGridAPI.CreateConnectivityOffsetsRel().SetTargets({faPrim.GetPath()})

        faPrim = _add_special_array(
            "Connectivity", dataset, stage, caeFieldArrayClass, cae_vtk.Tokens.connectivity_array
        )
        unstructuredGridAPI.CreateConnectivityArrayRel().SetTargets({faPrim.GetPath()})

        faPrim = _add_special_array("CellTypes", dataset, stage, caeFieldArrayClass, cae_vtk.Tokens.cell_types)
        unstructuredGridAPI.CreateCellTypesRel().SetTargets({faPrim.GetPath()})
    else:
        raise ValueError("Unsupported dataset type.")

    if len(point_arrays) > 0:
        scope = UsdGeom.Scope.Define(stage, rootPath.AppendChild("PointData"))
        _add_arrays(dataset, stage, scope, point_arrays, cae.Tokens.vertex, caeFieldArrayClass)
    if len(cell_arrays) > 0:
        scope = UsdGeom.Scope.Define(stage, rootPath.AppendChild("CellData"))
        _add_arrays(dataset, stage, scope, cell_arrays, cae.Tokens.cell, caeFieldArrayClass)
