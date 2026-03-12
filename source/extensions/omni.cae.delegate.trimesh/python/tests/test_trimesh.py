# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
import tempfile

import numpy as np
import omni.cae.delegate.trimesh
import omni.kit.test
import trimesh
from omni.cae.data import get_data_delegate_registry
from omni.cae.testing import new_stage
from pxr import Sdf, Usd


def _make_test_stl(path: str) -> trimesh.Trimesh:
    """Create a simple tetrahedron STL file and return the mesh."""
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [0.5, 0.5, 1.0],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 1, 3],
            [1, 2, 3],
            [0, 2, 3],
        ],
        dtype=np.int64,
    )
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.export(path)
    return mesh


def _define_trimesh_prim(stage: Usd.Stage, prim_path: str, stl_path: str, special: str) -> Usd.Prim:
    """Define a CaeTrimeshFieldArray prim with minimal required attributes."""
    prim = stage.DefinePrim(prim_path, "CaeTrimeshFieldArray")
    prim.CreateAttribute("fileNames", Sdf.ValueTypeNames.AssetArray).Set(Sdf.AssetPathArray([Sdf.AssetPath(stl_path)]))
    prim.CreateAttribute("special", Sdf.ValueTypeNames.Token).Set(special)
    prim.CreateAttribute("fieldAssociation", Sdf.ValueTypeNames.Token).Set("none")
    return prim


class TestTrimeshDelegate(omni.kit.test.AsyncTestCase):
    def setUp(self):
        self._tmp_dir = tempfile.mkdtemp()
        self._stl_path = os.path.join(self._tmp_dir, "test.stl")
        self._reference_mesh = _make_test_stl(self._stl_path)

    def tearDown(self):
        import shutil

        shutil.rmtree(self._tmp_dir, ignore_errors=True)

    async def test_vertices(self):
        async with new_stage() as stage:
            registry = get_data_delegate_registry()
            prim = _define_trimesh_prim(stage, "/World/Vertices", self._stl_path, "vertices")
            array = registry.get_field_array(prim)
            self.assertIsNotNone(array)
            self.assertEqual(array.ndim, 2)
            self.assertEqual(array.shape[1], 3)
            self.assertEqual(array.shape[0], len(self._reference_mesh.vertices))

    async def test_faces(self):
        async with new_stage() as stage:
            registry = get_data_delegate_registry()
            prim = _define_trimesh_prim(stage, "/World/Faces", self._stl_path, "faces")
            array = registry.get_field_array(prim)
            self.assertIsNotNone(array)
            # Flat triangle index array: 4 faces × 3 vertices each
            self.assertEqual(array.ndim, 1)
            self.assertEqual(array.shape[0], len(self._reference_mesh.faces) * 3)

    async def test_face_offsets(self):
        async with new_stage() as stage:
            registry = get_data_delegate_registry()
            prim = _define_trimesh_prim(stage, "/World/FaceOffsets", self._stl_path, "face_offsets")
            array = registry.get_field_array(prim)
            self.assertIsNotNone(array)
            n_faces = len(self._reference_mesh.faces)
            self.assertEqual(array.shape[0], n_faces + 1)
            self.assertEqual(array[0], 0)
            self.assertEqual(array[-1], n_faces * 3)

    async def test_face_counts(self):
        async with new_stage() as stage:
            registry = get_data_delegate_registry()
            prim = _define_trimesh_prim(stage, "/World/FaceCounts", self._stl_path, "face_counts")
            array = registry.get_field_array(prim)
            self.assertIsNotNone(array)
            n_faces = len(self._reference_mesh.faces)
            self.assertEqual(array.shape[0], n_faces)
            self.assertTrue(np.all(array == 3))

    async def test_vertex_normals(self):
        async with new_stage() as stage:
            registry = get_data_delegate_registry()
            prim = _define_trimesh_prim(stage, "/World/VNormals", self._stl_path, "vertex_normals")
            array = registry.get_field_array(prim)
            self.assertIsNotNone(array)
            self.assertEqual(array.shape[1], 3)
            self.assertEqual(array.shape[0], len(self._reference_mesh.vertices))

    async def test_face_normals(self):
        async with new_stage() as stage:
            registry = get_data_delegate_registry()
            prim = _define_trimesh_prim(stage, "/World/FNormals", self._stl_path, "face_normals")
            array = registry.get_field_array(prim)
            self.assertIsNotNone(array)
            self.assertEqual(array.shape[1], 3)
            self.assertEqual(array.shape[0], len(self._reference_mesh.faces))

    async def test_can_provide_true(self):
        async with new_stage() as stage:
            registry = get_data_delegate_registry()
            prim = _define_trimesh_prim(stage, "/World/Mesh", self._stl_path, "vertices")
            self.assertIsNotNone(registry.get_field_array(prim))

    async def test_wrong_type_not_provided(self):
        """Prims with unknown type should not be handled by this delegate."""
        async with new_stage() as stage:
            registry = get_data_delegate_registry()
            prim = stage.DefinePrim("/World/NotTrimesh", "CaeFieldArray")
            prim.CreateAttribute("fileNames", Sdf.ValueTypeNames.AssetArray).Set(
                Sdf.AssetPathArray([Sdf.AssetPath(self._stl_path)])
            )
            prim.CreateAttribute("fieldAssociation", Sdf.ValueTypeNames.Token).Set("none")
            # A bare CaeFieldArray prim should not be handled by the trimesh delegate
            from omni.cae.delegate.trimesh.delegate import TrimeshDataDelegate

            delegate = TrimeshDataDelegate.__new__(TrimeshDataDelegate)
            self.assertFalse(delegate.can_provide(prim))
