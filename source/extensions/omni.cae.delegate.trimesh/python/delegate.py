# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

__all__ = ["TrimeshDataDelegate"]

import weakref
from logging import getLogger

import numpy as np
import trimesh
from omni.cae.data.delegates import DataDelegateBase
from omni.client import get_local_file
from omni.stageupdate import get_stage_update_interface
from pxr import Usd

logger = getLogger(__name__)

# USD prim type name for CaeTrimeshFieldArray (defined in omniCaeTrimesh schema).
# String-based lookup is used until the compiled schema library is available.
_TRIMESH_PRIM_TYPE = "CaeTrimeshFieldArray"

# File extensions handled by this delegate.
TRIMESH_EXTENSIONS = frozenset(["stl", "obj", "ply", "off", "gltf", "glb", "3mf"])


class _MeshCache:
    """
    Cache trimesh mesh objects so that multiple get_field_array calls for the
    same file (e.g. vertices *and* faces from the same STL) only trigger one
    disk read.

    Strong references are kept per-prim; the file→mesh mapping uses weak
    references so meshes are released automatically when no prim needs them.
    If the prim is deleted/resynced, or the stage is reloaded, the associated
    strong reference is dropped.
    """

    def __init__(self):
        # Maps prim_path -> (filename, mesh); holds the strong ref to each mesh.
        self._prim_to_mesh: dict[str, tuple[str, trimesh.Trimesh]] = {}
        # Weak lookup: filename -> mesh.  Entry disappears when the last prim
        # that holds a strong ref to that mesh is removed from _prim_to_mesh.
        self._file_to_mesh: weakref.WeakValueDictionary = weakref.WeakValueDictionary()

    def get_mesh(self, prim: Usd.Prim, filename: str) -> trimesh.Trimesh:
        prim_path = str(prim.GetPath())

        existing = self._prim_to_mesh.get(prim_path)
        if existing is not None and existing[0] == filename:
            return existing[1]

        mesh = self._file_to_mesh.get(filename)
        if mesh is None:
            logger.info("loading trimesh from '%s'", filename)
            loaded = trimesh.load(filename, force="mesh")
            if not isinstance(loaded, trimesh.Trimesh):
                raise RuntimeError(
                    f"trimesh.load('{filename}') did not return a Trimesh object "
                    f"(got {type(loaded).__name__}). "
                    "Ensure the file contains a single surface mesh."
                )
            mesh = loaded
            self._file_to_mesh[filename] = mesh

        self._prim_to_mesh[prim_path] = (filename, mesh)
        return mesh

    def release_prim(self, path) -> None:
        """Drop any cached mesh associated with this prim path."""
        self._prim_to_mesh.pop(str(path), None)

    def clear(self) -> None:
        """Release all cached meshes (call on stage reload)."""
        self._prim_to_mesh.clear()


class TrimeshDataDelegate(DataDelegateBase):

    def __init__(self, ext_id: str):
        super().__init__(ext_id)
        self._mesh_cache = _MeshCache()

        self._stage_subscription = get_stage_update_interface().create_stage_update_node(
            "cae.trimesh.delegate",
            on_detach_fn=self._mesh_cache.clear,
            on_prim_remove_fn=self._mesh_cache.release_prim,
        )

    def __del__(self):
        if self._stage_subscription:
            del self._stage_subscription
            self._stage_subscription = None

    # ------------------------------------------------------------------
    # DataDelegateBase interface
    # ------------------------------------------------------------------

    def can_provide(self, prim: Usd.Prim) -> bool:
        if not (prim and prim.IsValid()):
            return False
        if prim.GetTypeName() != _TRIMESH_PRIM_TYPE:
            return False
        file_names_attr = prim.GetAttribute("fileNames")
        if not file_names_attr or not file_names_attr.IsValid():
            return False
        file_names = file_names_attr.Get(Usd.TimeCode.EarliestTime())
        if not file_names:
            return False
        return all(f.resolvedPath.rsplit(".", 1)[-1].lower() in TRIMESH_EXTENSIONS for f in file_names)

    def get_field_array(self, prim: Usd.Prim, time: Usd.TimeCode) -> np.ndarray:
        special = prim.GetAttribute("special").Get(time) or "none"
        array_name = prim.GetAttribute("arrayName").Get(time) or ""
        file_names = prim.GetAttribute("fileNames").Get(time) or []

        logger.info("trimesh delegate: special=%s prim=%s", special, prim.GetPath())

        arrays = []
        for f in file_names:
            fname = get_local_file(f.resolvedPath)[1]
            mesh = self._mesh_cache.get_mesh(prim, fname)
            array = self._extract(mesh, special, array_name, fname)
            if array is not None:
                arrays.append(_coerce_dtype(array))

        if not arrays:
            return None
        result = np.concatenate(arrays) if len(arrays) > 1 else arrays[0]
        logger.info("trimesh delegate: done, shape=%s dtype=%s", result.shape, result.dtype)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract(
        mesh: trimesh.Trimesh,
        special: str,
        array_name: str,
        fname: str,
    ) -> np.ndarray | None:
        if special == "vertices":
            return mesh.vertices

        if special == "faces":
            # Return a flat 1-D array of triangle vertex indices, matching the
            # VTK connectivity_array convention.
            return mesh.faces.flatten()

        if special == "face_offsets":
            # Offsets array: [0, 3, 6, ..., 3*M] for M triangles.
            n_faces = len(mesh.faces)
            return np.arange(0, (n_faces + 1) * 3, 3, dtype=np.int32)

        if special == "face_counts":
            # Every face is a triangle, so all counts are 3.
            return np.full(len(mesh.faces), 3, dtype=np.int32)

        if special == "vertex_normals":
            return mesh.vertex_normals

        if special == "face_normals":
            return mesh.face_normals

        if special == "vertex_colors":
            try:
                colors = mesh.visual.to_color().vertex_colors
                return np.asarray(colors, dtype=np.uint8)
            except Exception:
                logger.warning("vertex_colors not available for '%s'", fname)
                return None

        # special == "none": look up a named attribute
        if array_name:
            if array_name in mesh.vertex_attributes:
                return np.asarray(mesh.vertex_attributes[array_name])
            if array_name in mesh.face_attributes:
                return np.asarray(mesh.face_attributes[array_name])
            logger.warning(
                "array '%s' not found in vertex_attributes or face_attributes of '%s'",
                array_name,
                fname,
            )
            return None

        logger.warning(
            "trimesh delegate: special='none' but arrayName is empty for prim accessing '%s'",
            fname,
        )
        return None


def _coerce_dtype(array: np.ndarray) -> np.ndarray:
    """Normalise integer and float arrays to 32-bit.

    Smaller types are promoted and 64-bit types are narrowed so that
    IFieldArray always receives 32-bit elements."""
    if np.issubdtype(array.dtype, np.signedinteger) and array.itemsize != 4:
        return array.astype(np.int32, copy=False)
    if np.issubdtype(array.dtype, np.unsignedinteger) and array.itemsize != 4:
        return array.astype(np.uint32, copy=False)
    if np.issubdtype(array.dtype, np.floating) and array.itemsize != 4:
        return array.astype(np.float32, copy=False)
    return array
