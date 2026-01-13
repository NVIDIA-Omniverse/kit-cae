# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

__all__ = ["VTKDataDelegate"]

from logging import getLogger

import numpy as np
from omni.cae.data.delegates import DataDelegateBase
from omni.cae.schema import cae
from omni.cae.schema import vtk as cae_vtk
from omni.client import get_local_file
from pxr import Usd
from vtkmodules.numpy_interface import dataset_adapter as dsa
from vtkmodules.vtkIOLegacy import vtkDataSetReader
from vtkmodules.vtkIOXML import vtkXMLGenericDataObjectReader

logger = getLogger(__name__)


class VTKDataDelegate(DataDelegateBase):

    def __init__(self, ext_id):
        super().__init__(ext_id)
        self._reader_cache = {}

    def get_reader(self, fname: str):
        # this is bit of a HACK, but helps us avoid re-reading the same file
        # multiple times. Since VTK readers don't just read the specific array requested,
        # we end up lots of repeated reads
        if fname not in self._reader_cache:
            # get extension
            ext = fname.split(".")[-1].lower()
            if ext in ["vti", "vtu"]:
                reader = vtkXMLGenericDataObjectReader()
            elif ext == "vtk":
                reader = vtkDataSetReader()
            else:
                raise RuntimeError("Unsupported file %s", fname)
            reader.SetFileName(fname)
            self._reader_cache[fname] = reader
        return self._reader_cache[fname]

    def get_field_array(self, prim: Usd.Prim, time: Usd.TimeCode) -> np.ndarray:
        primT = cae_vtk.FieldArray(prim)
        arrayName = primT.GetArrayNameAttr().Get(time)
        fileNames = primT.GetFileNamesAttr().Get(time)
        assoc = primT.GetFieldAssociationAttr().Get(time)
        special = primT.GetSpecialAttr().Get(time)
        arrays = []
        for f in fileNames:
            fname = get_local_file(f.resolvedPath)[1]
            ext = fname.split(".")[-1].lower()
            if ext in ["vti", "vtu"]:
                reader = self.get_reader(fname)
                reader.UpdateInformation()

                if special != cae_vtk.Tokens.none:
                    # reader.GetPointDataArraySelection().DisableAllArrays()
                    # reader.GetCellDataArraySelection().DisableAllArrays()
                    # since we're reusing, never disable arrays to avoid modifying reader unnecessarily.
                    pass
                elif assoc == cae.Tokens.vertex and reader.GetPointDataArraySelection().ArrayExists(arrayName):
                    reader.GetPointDataArraySelection().EnableArray(arrayName)
                elif assoc == cae.Tokens.cell and reader.GetCellDataArraySelection().ArrayExists(arrayName):
                    reader.GetCellDataArraySelection().EnableArray(arrayName)
                elif assoc == cae.Tokens.vertex and reader.GetCellDataArraySelection().ArrayExists(arrayName):
                    # process as dual
                    reader.GetCellDataArraySelection().EnableArray(arrayName)
                    assoc = cae.Tokens.cell
            elif ext == "vtk":
                reader = self.get_reader(fname)
                # since legacy VTK reader doesn't really support array selection, we will just
                # read everything and then cache the results.
                reader.ReadAllScalarsOn()
                reader.ReadAllVectorsOn()
                reader.ReadAllNormalsOn()
                reader.ReadAllTensorsOn()
            else:
                raise ValueError("Unrecognized extension: %s" % ext)

            reader.Update()
            dataset = dsa.WrapDataObject(reader.GetOutput())
            if special == cae_vtk.Tokens.points:
                array = dataset.Points
            elif special == cae_vtk.Tokens.connectivity_offsets:
                array = dsa.vtkDataArrayToVTKArray(dataset.VTKObject.GetCells().GetOffsetsArray())
            elif special == cae_vtk.Tokens.connectivity_array:
                array = dsa.vtkDataArrayToVTKArray(dataset.VTKObject.GetCells().GetConnectivityArray())
            elif special == cae_vtk.Tokens.cell_types:
                array = dsa.vtkDataArrayToVTKArray(dataset.VTKObject.GetCellTypesArray())
            elif assoc == cae.Tokens.vertex:
                array = dataset.PointData[arrayName]
            elif assoc == cae.Tokens.cell:
                array = dataset.CellData[arrayName]
            else:
                raise RuntimeError(f"Failed to read {arrayName} from {f.resolvedPath}")

            # handle type conversion since IFieldArray does not support all types
            if np.issubdtype(array.dtype, np.integer) and array.itemsize < 4:
                array = array.astype(np.int32, copy=False)
            elif np.issubdtype(array.dtype, np.unsignedinteger) and array.itemsize < 4:
                array = array.astype(np.uint32, copy=False)
            elif np.issubdtype(array.dtype, np.floating) and array.itemsize < 4:
                array = array.astype(np.float32, copy=False)

            arrays.append(array)

        return np.concatenate(arrays) if arrays else None

    def can_provide(self, prim: Usd.Prim) -> bool:
        if prim and prim.IsValid() and prim.IsA(cae_vtk.FieldArray):
            primT = cae_vtk.FieldArray(prim)
            fileNames = primT.GetFileNamesAttr().Get(Usd.TimeCode.EarliestTime())
            # ensure all filenames have extension .vti
            return all(f.resolvedPath.split(".")[-1].lower() in ["vti", "vtk", "vtu"] for f in fileNames)
        return False
