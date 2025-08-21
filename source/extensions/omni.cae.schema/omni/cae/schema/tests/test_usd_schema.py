# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
#  its affiliates is strictly prohibited.

import omni.kit.test
import omni.usd
from omni.cae.schema import cae, ensight, sids, vtk
from pxr import Sdf, Usd


class Test(omni.kit.test.AsyncTestCase):
    async def setUp(self):
        pass

    async def tearDown(self) -> None:
        pass

    async def test_schemas(self):
        stage = Usd.Stage.CreateInMemory()
        caeDataSet = cae.DataSet.Define(stage, "/DataSet")
        sids.UnstructuredAPI.Apply(caeDataSet.GetPrim())

        caeArray = cae.FieldArray.Define(stage, "/FieldArray0")
        caeArray.CreateFileNamesAttr(Sdf.AssetPathArray(["test.data"]))

        numpyArray = cae.NumPyFieldArray.Define(stage, "/FieldArray1")
        self.assertTrue(
            numpyArray.GetPrim().IsA(cae.FieldArray), "numpy.FieldArray must be a sub-type of cae.FieldArray"
        )

        cgnsFieldArray = cae.CgnsFieldArray.Define(stage, "/FieldArray2")
        self.assertTrue(
            cgnsFieldArray.GetPrim().IsA(cae.FieldArray), "cgns.FieldArray must be a sub-type of cae.FieldArray"
        )

        vtkFieldArray = vtk.FieldArray.Define(stage, "/FieldArray3")
        self.assertTrue(
            vtkFieldArray.GetPrim().IsA(cae.FieldArray), "vtk.FieldArray must be a sub-type of cae.FieldArray"
        )

        registry = Usd.SchemaRegistry()
        self.assertEqual(
            registry.GetSchemaTypeName(sids.UnstructuredAPI), "CaeSidsUnstructuredAPI", "Missing SidsUnstructuredAPI"
        )
        # stage.Save()
        # logging.warning("generated '/tmp/sample.usda'")
