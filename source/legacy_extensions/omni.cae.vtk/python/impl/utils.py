# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from logging import getLogger

import numpy as np
from pxr import Gf
from vtkmodules.numpy_interface import dataset_adapter as dsa
from vtkmodules.vtkCommonDataModel import vtkDataObject
from vtkmodules.vtkFiltersCore import vtkCellDataToPointData, vtkCleanPolyData, vtkPolyDataNormals
from vtkmodules.vtkFiltersFlowPaths import vtkStreamTracer
from vtkmodules.vtkFiltersGeometry import vtkGeometryFilter
from vtkmodules.vtkIOLegacy import vtkDataSetWriter

logger = getLogger(__name__)


def extract_surface(dataset: dsa.DataSet):
    filter = vtkGeometryFilter()
    filter.SetInputData(dataset.VTKObject)
    filter.PassThroughCellIdsOff()
    filter.PassThroughPointIdsOff()
    filter.MergingOff()
    filter.DelegationOn()
    filter.FastModeOn()
    filter.Update()
    return dsa.WrapDataObject(filter.GetOutput())


def save_dataset(dataset: dsa.DataSet, fname: str):
    writer = vtkDataSetWriter()
    writer.SetInputData(dataset.VTKObject)
    writer.SetFileName(fname)
    writer.Write()


def cell_data_to_point_data(dataset: dsa.DataSet) -> dsa.DataSet:
    logger.info("converting cell data to point data")
    c2p = vtkCellDataToPointData()
    c2p.SetInputData(dataset.VTKObject)
    c2p.PassCellDataOff()
    c2p.ProcessAllArraysOn()
    c2p.Update()
    logger.info("done converting cell data to point data")
    return dsa.WrapDataObject(c2p.GetOutput())


def generate_polydata_normals(dataset: dsa.DataSet, split: bool):
    normals = vtkPolyDataNormals()
    normals.SetInputData(dataset.VTKObject)
    normals.SetSplitting(split)
    normals.Update()
    output = dsa.WrapDataObject(normals.GetOutput())
    assert "Normals" in output.PointData.keys()
    return output


def get_polydata_polys(dataset: dsa.DataSet):
    cells = dataset.VTKObject.GetPolys()
    offsets = dsa.vtkDataArrayToVTKArray(cells.GetOffsetsArray())
    counts = offsets[1:] - offsets[:-1]
    counts = counts.reshape([counts.shape[0], 1]).astype(np.int32, copy=False)
    conn = dsa.vtkDataArrayToVTKArray(cells.GetConnectivityArray())
    conn = conn.reshape([conn.shape[0], 1]).astype(np.int32, copy=False)
    return conn, counts


def get_polydata_lines(dataset: dsa.DataSet):
    offsets = dsa.vtkDataArrayToVTKArray(dataset.VTKObject.GetLines().GetOffsetsArray())
    lengths = offsets[1:] - offsets[:-1]
    # assert np.sum(lengths) != output.Points.shape[0]:
    return lengths.reshape([lengths.shape[0], 1]).astype(np.int32, copy=False)


def get_bbox(dataset: dsa.DataSet) -> Gf.Range3f:
    bds = [0.0] * 6
    dataset.GetBounds(bds)
    return Gf.Range3f((bds[0], bds[2], bds[4]), (bds[1], bds[3], bds[5]))


class StreamTracer:

    def __init__(self):
        self.tracer = vtkStreamTracer()
        self.tracer.SetIntegratorTypeToRungeKutta4()
        self.tracer.SetIntegrationStepUnit(self.tracer.LENGTH_UNIT)
        self.tracer.SetIntegrationDirectionToBoth()
        self.tracer.SetInterpolatorTypeToCellLocator()
        self.tracer.SetComputeVorticity(False)

        self.cleaner = vtkCleanPolyData()
        self.cleaner.ConvertLinesToPointsOff()
        self.cleaner.ConvertPolysToLinesOff()
        self.cleaner.ConvertStripsToPolysOff()
        self.cleaner.PointMergingOff()
        self.cleaner.SetInputConnection(self.tracer.GetOutputPort())

    def execute(self, dataset: dsa.DataSet, seeds: dsa.DataSet, velocity_name: str, dX: float, maxLength: int):
        self.tracer.SetSourceData(seeds.VTKObject)
        self.tracer.SetInputData(dataset.VTKObject)
        self.tracer.SetInputArrayToProcess(0, 0, 0, vtkDataObject.FIELD_ASSOCIATION_POINTS, velocity_name)
        self.tracer.SetInitialIntegrationStep(dX)
        self.tracer.SetMaximumNumberOfSteps(maxLength)
        self.tracer.SetMaximumPropagation(maxLength * dX * 10)
        self.cleaner.Update()
        return dsa.WrapDataObject(self.cleaner.GetOutput())


_stream_tracer = StreamTracer()


def generate_streamlines(dataset: dsa.DataSet, seeds: dsa.DataSet, velocity_name: str, dX: float, maxLength: int):
    global _stream_tracer
    # import gc
    # gc.collect()

    # save_dataset(clone, "/tmp/mesh.vtk")
    # save_dataset(seeds, "/tmp/seeds.vtk")
    return _stream_tracer.execute(dataset, seeds, velocity_name, dX, maxLength)
