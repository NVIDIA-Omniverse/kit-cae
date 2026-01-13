# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import pathlib

import omni.cae.npz
import omni.kit.test
import omni.usd
from omni.cae.data import get_data_delegate_registry
from omni.cae.schema import cae
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

    async def test_npz_fields_usda(self):
        usd_context = omni.usd.get_context()
        await usd_context.open_stage_async(self.get_local_test_scene_path("npz_fields.usda"))
        stage: Usd.Stage = usd_context.get_stage()
        registry = get_data_delegate_registry()
        pressure = registry.get_field_array(stage.GetPrimAtPath("/World/Pressure"))
        self.assertTrue(pressure is not None)
        self.assertTrue(pressure.shape[0] > 1)
        missing = registry.get_field_array(stage.GetPrimAtPath("/World/Missing"))
        self.assertTrue(missing is None)
        omni.usd.get_context().close_stage()

    async def test_new_stage(self):
        stage = Usd.Stage.CreateInMemory()

        arrayNPZ = cae.NumPyFieldArray.Define(stage, "/World/Pressure")
        arrayNPZ.CreateFileNamesAttr().Set([self.get_local_test_scene_path("StaticMixer.npz")])
        arrayNPZ.CreateFieldAssociationAttr().Set("none")
        arrayNPZ.CreateArrayNameAttr().Set("Pressure")

        await omni.usd.get_context().attach_stage_async(stage)
        array = self._registry.get_field_array(arrayNPZ.GetPrim())
        self.assertIsNotNone(array)
        self.assertGreater(array.shape[0], 0)

        omni.usd.get_context().close_stage()
        del stage
