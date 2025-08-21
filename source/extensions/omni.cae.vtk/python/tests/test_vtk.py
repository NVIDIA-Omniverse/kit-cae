# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
#  its affiliates is strictly prohibited.

import pathlib

import omni.kit.test
import omni.usd
from omni.cae.data import get_data_delegate_registry
from omni.cae.schema import vtk as cae_vtk
from pxr import Usd


class Test(omni.kit.test.AsyncTestCase):
    async def setUp(self):
        self._test_data = str(pathlib.Path(__file__).parent.joinpath("data"))
        self._registry = get_data_delegate_registry()

    async def tearDown(self) -> None:
        del self._registry

    def get_local_test_scene_path(self, relative_path: str) -> str:
        "compute the absolute path of the stage"
        return self._test_data + "/" + relative_path

    async def test_new_stage(self):
        stage = Usd.Stage.CreateInMemory()

        arrayVTK = cae_vtk.FieldArray.Define(stage, "/World/RTData")
        arrayVTK.CreateFileNamesAttr().Set([self.get_local_test_scene_path("wavelet.vti")])
        arrayVTK.CreateFieldAssociationAttr().Set("vertex")
        arrayVTK.CreateArrayNameAttr().Set("RTData")

        await omni.usd.get_context().attach_stage_async(stage)
        array = self._registry.get_field_array(arrayVTK.GetPrim())
        self.assertIsNotNone(array, "Failed to load array")
        self.assertEquals(array.shape[0], 9261, "Incorrect array size")
        self.assertEquals(array.ndim, 1, "Incorrect array dimensions")

        omni.usd.get_context().close_stage()
        del stage
