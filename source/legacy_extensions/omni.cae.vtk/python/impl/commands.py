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
from logging import getLogger

import numpy as np
from omni.cae.data import array_utils, cache, progress, usd_utils
from omni.cae.data.commands import (
    ComputeBounds,
    ConvertToMesh,
    ConvertToPointCloud,
    GenerateStreamlines,
    Mesh,
    PointCloud,
    Streamlines,
)
from omni.cae.schema import cae
from omni.cae.schema import vtk as cae_vtk
from pxr import Gf
from vtkmodules.numpy_interface import dataset_adapter as dsa
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkUnstructuredGrid

from ..commands import ConvertToVTKDataSet
from . import utils

logger = getLogger(__name__)


class OmniCaeDataSetConvertToMesh(ConvertToMesh):

    async def do(self) -> Mesh:
        logger.info("executing %s.do()", self.__class__.__name__)
        vtk_dataset = await ConvertToVTKDataSet.invoke(
            self.dataset, self.fields, forcePointData=False, timeCode=self.timeCode
        )
        hull = utils.generate_polydata_normals(utils.extract_surface(vtk_dataset), split=True)
        faceVertexIndices, faceVertexCounts = utils.get_polydata_polys(hull)

        mesh = Mesh()
        mesh.extents = utils.get_bbox(hull)
        mesh.points = hull.Points.astype(np.float32, copy=False)
        mesh.faceVertexIndices = faceVertexIndices
        mesh.faceVertexCounts = faceVertexCounts
        mesh.normals = hull.PointData["Normals"].astype(np.float32, copy=False)
        for field in self.fields:
            mesh.fields[field] = hull.PointData[field] if field in hull.PointData.keys() else hull.CellData[field]
        return mesh


class OmniCaeDataSetGenerateStreamlinesVTK(GenerateStreamlines):

    async def do(self) -> Streamlines:
        logger.info("executing %s.do()", self.__class__.__name__)
        vtk_dataset = await self.get_dataset()

        seeds = dsa.WrapDataObject(vtkUnstructuredGrid())
        seeds.Points = array_utils.as_numpy_array(self.seeds)
        streamlines: dsa.DataSet = utils.generate_streamlines(vtk_dataset, seeds, "velocity", self.dX, self.maxLength)

        if streamlines.Points is None:
            return None

        result = Streamlines()
        result.points = streamlines.Points
        result.curveVertexCounts = utils.get_polydata_lines(streamlines)
        result.fields["time"] = streamlines.PointData["IntegrationTime"]
        if "colors" in streamlines.PointData.keys():
            result.fields["scalar"] = streamlines.PointData["colors"]

        for field in self.extra_fields:
            result.fields[field] = streamlines.PointData[field]

        return result

    async def get_dataset(self) -> dsa.DataSet:
        fields = []
        fields += self.velocity_fields
        fields += [self.colorField] if self.colorField is not None else []
        fields += self.extra_fields

        fields = list(set(fields))  # unique

        cache_key = {
            "label": "OmniCaeDataSetGenerateStreamlinesVTK",
            "dataset": str(self.dataset.GetPath()),
            "fields": str(fields),
        }

        cache_state = {}

        if dataset := cache.get(str(cache_key), cache_state, timeCode=self.timeCode):
            return dataset

        vtk_dataset = await ConvertToVTKDataSet.invoke(
            self.dataset, fields, forcePointData=True, timeCode=self.timeCode
        )

        if len(self.velocity_fields) == 3:
            velocity = np.vstack([vtk_dataset.PointData[i] for i in self.velocity_fields]).transpose()
        else:
            velocity = vtk_dataset.PointData[self.velocity_fields[0]]

        dataset = dsa.WrapDataObject(vtk_dataset.NewInstance())
        dataset.CopyStructure(vtk_dataset.VTKObject)
        dataset.PointData.append(velocity, "velocity")
        if self.colorField is not None:
            dataset.PointData.append(vtk_dataset.PointData[self.colorField], "colors")

        for field in self.extra_fields:
            dataset.PointData.append(vtk_dataset.PointData[field], field)

        cache.put(str(cache_key), dataset, state=cache_state, sourcePrims=[self.dataset], timeCode=self.timeCode)
        return dataset


class CaeVtkUnstructuredGridConvertToPointCloud(ConvertToPointCloud):

    async def do(self) -> PointCloud:
        # Our strategy here is to
        # 1. Read the coordinates
        # 2. Read point arrays requested
        # 3. If any cell arrays are requested, the we don't have enough information here. So, we use `ConvertToVTKDataSet`
        #    operator to fetch the dataset while reading the cell arrays and forcing a cell-2-point conversion for the
        #    cell arrays.
        logger.info("executing %s.do()", self.__class__.__name__)
        dataset = self.dataset
        fields = self.fields
        assert dataset.HasAPI(cae_vtk.UnstructuredGridAPI), (
            "Dataset (%s) does not support cae_vtk.UnstructuredGridAPI!" % dataset
        )

        work_units: int = len(fields) + 1  # 1 for coords

        result = PointCloud()
        with progress.ProgressContext("Reading coordinates", scale=1 / work_units):
            result.points = await usd_utils.get_vecN_from_relationship(
                dataset, cae_vtk.Tokens.caeVtkPoints, 3, self.timeCode
            )

        point_arrays = {}
        cell_arrays = {}

        for fieldName in fields:
            fieldPrim = usd_utils.get_target_prim(dataset, f"field:{fieldName}")
            assoc = cae.FieldArray(fieldPrim).GetFieldAssociationAttr().Get(self.timeCode)
            if assoc == cae.Tokens.vertex:
                point_arrays[fieldName] = None
            elif assoc == cae.Tokens.cell:
                cell_arrays[fieldName] = None

        for idx, fieldName in enumerate(point_arrays.keys()):
            with progress.ProgressContext(
                "Reading field %s" % fieldName, shift=(1 + idx) / work_units, scale=1.0 / work_units
            ):
                fieldPrim = usd_utils.get_target_prim(dataset, f"field:{fieldName}")
                assoc = cae.FieldArray(fieldPrim).GetFieldAssociationAttr().Get(self.timeCode)
                assert assoc == cae.Tokens.vertex
                point_arrays[fieldName] = await usd_utils.get_array(fieldPrim, self.timeCode)

        if len(cell_arrays) > 0:
            # need to convert cell to point
            with progress.ProgressContext(
                "Reading field %s" % fieldName,
                shift=(1 + len(point_arrays)) / work_units,
                scale=len(cell_arrays) / work_units,
            ):
                vtk_dataset = await ConvertToVTKDataSet.invoke(
                    dataset, list(cell_arrays.keys()), forcePointData=True, timeCode=self.timeCode
                )
                for fieldName in cell_arrays.keys():
                    cell_arrays[fieldName] = vtk_dataset.PointData[fieldName]

        result.fields = {**point_arrays, **cell_arrays}
        return result


class CaeVtkUnstructuredGridConvertToVTKDataSet(ConvertToVTKDataSet):

    async def do(self):
        assert self.params.dataset.HasAPI(cae_vtk.UnstructuredGridAPI), (
            "Dataset (%s) does not support cae_vtk.UnstructuredGridAPI!" % self.params.dataset
        )

        dataset = dsa.WrapDataObject(vtkUnstructuredGrid())
        with progress.ProgressContext("Reading coordinates", scale=0.3):
            pts = await usd_utils.get_vecN_from_relationship(
                self.params.dataset, cae_vtk.Tokens.caeVtkPoints, 3, self.timeCode
            )
            pts_numpy = array_utils.as_numpy_array(pts)
            dataset.Points = pts_numpy
            dataset.__pts = pts_numpy

        with progress.ProgressContext("Reading connectivity", scale=0.3, shift=0.3):
            connectivity = await usd_utils.get_array_from_relationship(
                self.params.dataset, cae_vtk.Tokens.caeVtkConnectivityArray, self.timeCode
            )
            connectivity_numpy = array_utils.as_numpy_array(connectivity)

            offsets = await usd_utils.get_array_from_relationship(
                self.params.dataset, cae_vtk.Tokens.caeVtkConnectivityOffsets, self.timeCode
            )
            offsets_numpy = array_utils.as_numpy_array(offsets)

            types = await usd_utils.get_array_from_relationship(
                self.params.dataset, cae_vtk.Tokens.caeVtkCellTypes, self.timeCode
            )
            types_numpy = array_utils.as_numpy_array(types).astype(np.uint8, copy=False)
            # print(types_numpy, types_numpy.dtype)

            cells = vtkCellArray()
            cells.SetData(dsa.numpyTovtkDataArray(offsets_numpy), dsa.numpyTovtkDataArray(connectivity_numpy))
            dataset.VTKObject.SetCells(dsa.numpyTovtkDataArray(types_numpy), cells)

            dataset.__connectivity = connectivity_numpy
            dataset.__offsets = offsets_numpy
            dataset.__types = types_numpy

        dataset.__py_fields = []
        with progress.ProgressContext("Reading fields", shift=0.6, scale=0.3):
            for idx, fieldName in enumerate(self.params.fields):
                fieldPrim = usd_utils.get_target_prim(self.params.dataset, f"field:{fieldName}")
                assoc = cae.FieldArray(fieldPrim).GetFieldAssociationAttr().Get(self.timeCode)
                field = array_utils.as_numpy_array(await usd_utils.get_array(fieldPrim, self.timeCode))
                if assoc == cae.Tokens.vertex:
                    dataset.PointData.append(field, fieldName)
                    dataset.__py_fields.append(field)
                elif assoc == cae.Tokens.cell:
                    dataset.CellData.append(field, fieldName)
                    dataset.__py_fields.append(field)
                else:
                    raise usd_utils.QuietableException(
                        "Invalid field association (%s) detected for %s" % (assoc, fieldPrim)
                    )
        return dataset
