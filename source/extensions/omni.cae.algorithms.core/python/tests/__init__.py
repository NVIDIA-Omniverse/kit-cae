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
from pxr import Usd


class TestAlgorithms(omni.kit.test.AsyncTestCase):

    async def test_schema_hull_api(self):
        registry = Usd.SchemaRegistry()
        self.assertEqual(
            registry.GetAPISchemaTypeName("OmniCaeAlgorithmsStreamlinesAPI"), "CaeAlgorithmsStreamlinesAPI"
        )
        self.assertEqual(
            registry.GetTypeFromSchemaTypeName("CaeAlgorithmsStreamlinesAPI"), "OmniCaeAlgorithmsStreamlinesAPI"
        )

        # get definition with typename
        defn = registry.FindAppliedAPIPrimDefinition("CaeAlgorithmsStreamlinesAPI")
        self.assertIsNotNone(defn)
        self.assertIn("omni:cae:algorithms:streamlines:velocity", defn.GetPropertyNames())
        self.assertIn("omni:cae:algorithms:streamlines:dataset", defn.GetPropertyNames())

        # get defin with type str (will fail)
        defn = registry.FindAppliedAPIPrimDefinition("OmniCaeAlgorithmsStreamlinesAPI")
        self.assertIsNone(defn)
