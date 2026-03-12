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

import omni.cae.delegate.npz
import omni.kit.test
import omni.usd
from omni.cae.data import get_data_delegate_registry
from omni.cae.schema import cae
from omni.cae.testing import get_test_data_path, get_test_stage_path, new_stage
from pxr import Usd


class Test(omni.kit.test.AsyncTestCase):
    async def test_npz_fields_usda(self):
        async with new_stage(get_test_stage_path("npz_fields.usda")) as stage:
            registry = get_data_delegate_registry()
            self.assertIsNotNone(registry)
            pressure = registry.get_field_array(stage.GetPrimAtPath("/World/Pressure"))
            self.assertIsNotNone(pressure)
            self.assertGreater(pressure.shape[0], 1)
            missing = registry.get_field_array(stage.GetPrimAtPath("/World/Missing"))
            self.assertIsNone(missing)

    async def test_new_stage(self):
        async with new_stage() as stage:
            registry = get_data_delegate_registry()
            self.assertIsNotNone(registry)
            arrayNPZ = cae.NumPyFieldArray.Define(stage, "/World/Pressure")
            arrayNPZ.CreateFileNamesAttr().Set([get_test_data_path("StaticMixer.npz")])
            arrayNPZ.CreateFieldAssociationAttr().Set("none")
            arrayNPZ.CreateArrayNameAttr().Set("Pressure")
            array = registry.get_field_array(arrayNPZ.GetPrim())
            self.assertIsNotNone(array)
            self.assertGreater(array.shape[0], 0)
