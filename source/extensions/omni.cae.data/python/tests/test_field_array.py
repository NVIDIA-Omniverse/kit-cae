# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
#  its affiliates is strictly prohibited.

import logging

import numpy as np
import omni.kit.app
import omni.kit.test
import omni.usd
import warp as wp
from omni.cae.data import IFieldArray

logger = logging.getLogger(__name__)


class TestFieldArray(omni.kit.test.AsyncTestCase):

    def _test_np_array(self, narray: np.ndarray):
        farray = IFieldArray.from_numpy(narray)
        logger.info(f"array_interface: {farray.__array_interface__}")
        self.assertEqual(np.amin(narray), np.amin(farray))
        self.assertEqual(np.amax(narray), np.amax(farray))
        self.assertTrue(np.array_equal(narray, farray.numpy()))
        self.assertTrue(np.array_equal(narray, farray))

    def _test_wp_array(self, narray: wp.array):
        farray = IFieldArray.from_array(narray)
        # logger.info(f"cuda_array_interface: {farray.__cuda_array_interface__}")

        if narray.device.is_cpu:
            self.assertTrue(np.array_equal(narray, farray))
            self.assertTrue(np.array_equal(narray, farray.numpy()))
        else:
            inarray = narray.numpy()
            outarray = wp.array(farray).numpy()
            self.assertTrue(np.array_equal(inarray, outarray))

    def _test_array(self, narray: np.ndarray):
        self._test_np_array(narray)
        with wp.ScopedDevice("cpu"):
            self._test_wp_array(wp.array(narray))
        with wp.ScopedDevice("cuda"):
            self._test_wp_array(wp.array(narray))

    async def test_field_array_bindings(self):
        self._test_array(np.random.default_rng(99).uniform(low=0.0, high=100.0, size=[1024, 3]).astype(np.float32))
        self._test_array(np.random.default_rng(123).uniform(low=0.0, high=100.0, size=1024).astype(np.float64))
        self._test_array(np.random.default_rng(3).uniform(low=0.0, high=100.0, size=1024).astype(np.float64))
        self._test_array(np.random.default_rng(322).uniform(low=0.0, high=100.0, size=1024).astype(np.int32))
        self._test_array(np.random.default_rng(32).uniform(low=0.0, high=100.0, size=1024).astype(np.uint32))
        self._test_array(np.random.default_rng(36).uniform(low=0.0, high=100.0, size=1024).astype(np.int64))
        self._test_array(np.random.default_rng(36).uniform(low=0.0, high=100.0, size=1024).astype(np.uint64))
