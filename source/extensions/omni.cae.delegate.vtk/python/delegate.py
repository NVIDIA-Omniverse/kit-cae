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

import weakref
from logging import getLogger

import numpy as np
from omni.cae.data.delegates import DataDelegateBase
from omni.cae.schema import cae
from omni.cae.schema import vtk as cae_vtk
from omni.client import get_local_file
from omni.stageupdate import get_stage_update_interface
from pxr import Usd
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkIOLegacy import vtkDataSetReader
from vtkmodules.vtkIOXML import vtkXMLGenericDataObjectReader

logger = getLogger(__name__)


class ReaderCache:
    """
    A cache for VTK readers that allows us to share a reader across
    multiple get_field_array calls for the same file, which is important since VTK readers support limited
    ability to read parts of the data.

    If filename changed between calls for the same prim, the old cached reader is released
    unless its needed by some other prim. If the prim is deleted or resyned, the reader is released as well.

    If stage is reloaded, all cached readers are released since they will likely be invalid.
    """

    def __init__(self):
        # Maps prim_path -> (filename, reader); holds the strong ref to each reader.
        self._prim_to_reader: dict[str, tuple[str, object]] = {}
        # Weak lookup: filename -> reader. Entry disappears automatically when the last
        # prim holding a strong ref to that reader is removed from _prim_to_reader.
        self._file_to_reader: weakref.WeakValueDictionary = weakref.WeakValueDictionary()

    def _create_reader(self, fname: str):
        ext = fname.split(".")[-1].lower()
        if ext in ["vti", "vtu", "vts", "vtp"]:
            reader = vtkXMLGenericDataObjectReader()
            reader.SetFileName(fname)
        elif ext == "vtk":
            reader = vtkDataSetReader()
            reader.SetFileName(fname)
        else:
            raise RuntimeError("Unsupported file %s" % fname)
        return reader

    def get_reader(self, prim, filename: str):
        prim_path = str(prim.GetPath())

        existing = self._prim_to_reader.get(prim_path)
        if existing is not None and existing[0] == filename:
            return existing[1]

        # Prim is new or switched files — look up or create a reader for this file
        reader = self._file_to_reader.get(filename)
        if reader is None:
            reader = self._create_reader(filename)
            self._file_to_reader[filename] = reader

        # Store the strong ref; old entry (if any) is dropped here, releasing the
        # strong ref to the previous reader, which lets WeakValueDictionary clean it
        # up automatically if no other prim still holds it.
        self._prim_to_reader[prim_path] = (filename, reader)
        return reader

    def release_prim(self, path):
        """Release any cached reader associated with this prim path (call on prim delete/resync)."""
        self._prim_to_reader.pop(str(path), None)

    def clear(self):
        """Release all cached readers (call on stage reload)."""
        self._prim_to_reader.clear()


class VTKDataDelegate(DataDelegateBase):

    def __init__(self, ext_id):
        super().__init__(ext_id)
        self._reader_cache = ReaderCache()

        self._stage_subscription = get_stage_update_interface().create_stage_update_node(
            "cae.vtk.delegate",
            on_detach_fn=self._reader_cache.clear,
            on_prim_remove_fn=self._reader_cache.release_prim,
        )

    def __del__(self):
        if self._stage_subscription:
            del self._stage_subscription
            self._stage_subscription = None

    def get_field_array(self, prim: Usd.Prim, time: Usd.TimeCode) -> np.ndarray:
        primT = cae_vtk.FieldArray(prim)
        arrayName = primT.GetArrayNameAttr().Get(time)
        fileNames = primT.GetFileNamesAttr().Get(time)
        assoc = primT.GetFieldAssociationAttr().Get(time)
        special = primT.GetSpecialAttr().Get(time)
        logger.info("start reading %s (%s)", special, prim.GetPath())
        arrays = []
        for f in fileNames:
            fname = get_local_file(f.resolvedPath)[1]
            ext = fname.split(".")[-1].lower()
            if ext in ["vti", "vtu", "vts", "vtp"]:
                reader = self._reader_cache.get_reader(prim, fname)
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
                reader = self._reader_cache.get_reader(prim, fname)
                # since legacy VTK reader doesn't really support array selection, we will just
                # read everything and then cache the results.
                reader.ReadAllScalarsOn()
                reader.ReadAllVectorsOn()
                reader.ReadAllNormalsOn()
                reader.ReadAllTensorsOn()
            else:
                raise ValueError("Unrecognized extension: %s" % ext)

            reader.Update()
            vtk_dataset = reader.GetOutput()  # cache this since we use it a lot below
            array = None
            if special == cae_vtk.Tokens.points:
                array = vtk_to_numpy(vtk_dataset.GetPoints().GetData())
            elif special == cae_vtk.Tokens.connectivity_offsets:
                array = vtk_to_numpy(vtk_dataset.GetCells().GetOffsetsArray())
            elif special == cae_vtk.Tokens.connectivity_array:
                array = vtk_to_numpy(vtk_dataset.GetCells().GetConnectivityArray())
            elif special == cae_vtk.Tokens.cell_types:
                array = vtk_to_numpy(vtk_dataset.GetCellTypesArray())
            elif special == cae_vtk.Tokens.polyhedron_faces_offsets:
                if vtk_dataset.GetPolyhedronFaces():
                    array = vtk_to_numpy(vtk_dataset.GetPolyhedronFaces().GetOffsetsArray())
            elif special == cae_vtk.Tokens.polyhedron_faces_connectivity_array:
                if vtk_dataset.GetPolyhedronFaces():
                    array = vtk_to_numpy(vtk_dataset.GetPolyhedronFaces().GetConnectivityArray())
            elif special == cae_vtk.Tokens.polyhedron_face_locations_offsets:
                if vtk_dataset.GetPolyhedronFaceLocations():
                    array = vtk_to_numpy(vtk_dataset.GetPolyhedronFaceLocations().GetOffsetsArray())
            elif special == cae_vtk.Tokens.polyhedron_face_locations_connectivity_array:
                if vtk_dataset.GetPolyhedronFaceLocations():
                    array = vtk_to_numpy(vtk_dataset.GetPolyhedronFaceLocations().GetConnectivityArray())
            elif special == cae_vtk.Tokens.verts_connectivity_offsets:
                array = vtk_to_numpy(vtk_dataset.GetVerts().GetOffsetsArray())
            elif special == cae_vtk.Tokens.verts_connectivity_array:
                array = vtk_to_numpy(vtk_dataset.GetVerts().GetConnectivityArray())
            elif special == cae_vtk.Tokens.lines_connectivity_offsets:
                array = vtk_to_numpy(vtk_dataset.GetLines().GetOffsetsArray())
            elif special == cae_vtk.Tokens.lines_connectivity_array:
                array = vtk_to_numpy(vtk_dataset.GetLines().GetConnectivityArray())
            elif special == cae_vtk.Tokens.polys_connectivity_offsets:
                array = vtk_to_numpy(vtk_dataset.GetPolys().GetOffsetsArray())
            elif special == cae_vtk.Tokens.polys_connectivity_array:
                array = vtk_to_numpy(vtk_dataset.GetPolys().GetConnectivityArray())
            elif special == cae_vtk.Tokens.strips_connectivity_offsets:
                array = vtk_to_numpy(vtk_dataset.GetStrips().GetOffsetsArray())
            elif special == cae_vtk.Tokens.strips_connectivity_array:
                array = vtk_to_numpy(vtk_dataset.GetStrips().GetConnectivityArray())
            elif assoc == cae.Tokens.vertex:
                array = vtk_to_numpy(vtk_dataset.GetPointData().GetArray(arrayName))
            elif assoc == cae.Tokens.cell:
                array = vtk_to_numpy(vtk_dataset.GetCellData().GetArray(arrayName))
            else:
                raise RuntimeError(f"Failed to read {arrayName} from {f.resolvedPath}")

            if array is not None:
                # handle type conversion since IFieldArray does not support all types
                if np.issubdtype(array.dtype, np.integer) and array.itemsize < 4:
                    array = array.astype(np.int32, copy=False)
                elif np.issubdtype(array.dtype, np.unsignedinteger) and array.itemsize < 4:
                    array = array.astype(np.uint32, copy=False)
                elif np.issubdtype(array.dtype, np.floating) and array.itemsize < 4:
                    array = array.astype(np.float32, copy=False)

                arrays.append(array)
        if not arrays or len(arrays) == 0:
            return None
        elif len(arrays) > 1:
            logger.info("  concatenating ...")
            result = np.concatenate(arrays)
        else:
            result = arrays[0]
        logger.info("  done.")
        return result

    def can_provide(self, prim: Usd.Prim) -> bool:
        if prim and prim.IsValid() and prim.IsA(cae_vtk.FieldArray):
            primT = cae_vtk.FieldArray(prim)
            fileNames = primT.GetFileNamesAttr().Get(Usd.TimeCode.EarliestTime())
            # ensure all filenames have extension .vti
            return all(f.resolvedPath.split(".")[-1].lower() in ["vti", "vtk", "vtu", "vts", "vtp"] for f in fileNames)
        return False
