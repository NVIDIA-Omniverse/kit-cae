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

import omni.kit.test


class Test(omni.kit.test.AsyncTestCase):
    async def setUp(self):
        self._test_data = str(pathlib.Path(__file__).parent.joinpath("data"))

    async def tearDown(self) -> None:
        pass

    def get_local_test_scene_path(self, relative_path: str) -> str:
        "compute the absolute path of the test data"
        return self._test_data + "/" + relative_path

    async def test_dav_import(self):
        """Test that DAV can be imported successfully"""
        try:
            import dav

            self.assertTrue(True, "DAV imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import DAV: {e}")
