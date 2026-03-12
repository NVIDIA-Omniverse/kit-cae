# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import io

import omni.cae.delegate.nvdb
import omni.kit.test
import warp as wp
from omni.cae.data import array_utils, get_data_delegate_registry
from omni.cae.schema import nvdb as cae_nvdb
from omni.cae.testing import get_test_data_path, new_stage


class Test(omni.kit.test.AsyncTestCase):
    async def test_new_stage(self):
        async with new_stage() as stage:
            registry = get_data_delegate_registry()
            self.assertIsNotNone(registry)
            arrayNvdb = cae_nvdb.FieldArray.Define(stage, "/World/Pressure")
            arrayNvdb.CreateFileNamesAttr().Set([get_test_data_path("headsq.nvdb")])
            arrayNvdb.CreateFieldAssociationAttr().Set("vertex")
            array = registry.get_field_array(arrayNvdb.GetPrim())
            self.assertIsNotNone(array)
            self.assertGreater(array.shape[0], 0)

            # Validate that the returned buffer can be used to reconstruct a wp.Volume.
            # The field array holds the raw NanoVDB buffer as uint32 elements; view as
            # bytes and reload via wp.Volume.load_from_nvdb.
            raw_bytes = array_utils.as_warp_array(array)
            volume = wp.Volume(raw_bytes)
            self.assertIsNotNone(volume)
            self.assertEqual(volume.get_voxel_count(), 6291456)
