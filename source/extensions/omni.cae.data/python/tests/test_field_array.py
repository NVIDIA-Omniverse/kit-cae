# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

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

    async def test_to_device_cpu_to_cpu(self):
        """Test copying array on CPU to CPU (should return same object)."""
        np_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        cpu_array = IFieldArray.from_numpy(np_array)

        # Same device should return original
        same_array = cpu_array.to_device(-1)
        self.assertIs(same_array, cpu_array, "Expected same object for same device")
        self.assertEqual(same_array.device_id, -1)

    async def test_to_device_cpu_to_gpu(self):
        """Test copying array from CPU to GPU."""
        np_array = np.random.default_rng(42).uniform(low=0.0, high=100.0, size=[128, 3]).astype(np.float32)
        cpu_array = IFieldArray.from_numpy(np_array)

        # Copy to GPU 0
        gpu_array = cpu_array.to_device(0)
        self.assertEqual(gpu_array.device_id, 0)
        self.assertEqual(gpu_array.shape, cpu_array.shape)
        self.assertEqual(gpu_array.dtype, cpu_array.dtype)

        # Verify data integrity
        gpu_data = wp.array(gpu_array).numpy()
        self.assertTrue(np.allclose(np_array, gpu_data))

    async def test_to_device_gpu_to_cpu(self):
        """Test copying array from GPU to CPU."""
        np_array = np.random.default_rng(42).uniform(low=0.0, high=100.0, size=256).astype(np.float64)

        # Create GPU array
        with wp.ScopedDevice("cuda:0"):
            wp_array = wp.array(np_array)
            gpu_array = IFieldArray.from_array(wp_array)
            self.assertEqual(gpu_array.device_id, 0)

            # Copy to CPU
            cpu_array = gpu_array.to_device(-1)
            self.assertEqual(cpu_array.device_id, -1)
            self.assertEqual(cpu_array.shape, gpu_array.shape)
            self.assertEqual(cpu_array.dtype, gpu_array.dtype)

            # Verify data integrity
            cpu_data = cpu_array.numpy()
            self.assertTrue(np.allclose(np_array, cpu_data))

    async def test_to_device_gpu_to_gpu_same(self):
        """Test copying array on GPU to same GPU (should return same object)."""
        np_array = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)

        with wp.ScopedDevice("cuda:0"):
            wp_array = wp.array(np_array)
            gpu_array = IFieldArray.from_array(wp_array)

            # Same device should return original
            same_array = gpu_array.to_device(0)
            self.assertIs(same_array, gpu_array, "Expected same object for same device")
            self.assertEqual(same_array.device_id, 0)

    async def test_to_device_round_trip(self):
        """Test round-trip CPU -> GPU -> CPU preserves data."""
        np_array = np.random.default_rng(123).uniform(low=-50.0, high=50.0, size=[64, 4]).astype(np.float32)

        # Start with CPU array
        cpu_array1 = IFieldArray.from_numpy(np_array)

        # Copy to GPU
        gpu_array = cpu_array1.to_device(0)
        self.assertEqual(gpu_array.device_id, 0)

        # Copy back to CPU
        cpu_array2 = gpu_array.to_device(-1)
        self.assertEqual(cpu_array2.device_id, -1)

        # Verify data integrity
        result = cpu_array2.numpy()
        self.assertTrue(np.allclose(np_array, result))

    async def test_to_device_method_chaining(self):
        """Test that to_device can be chained with other methods."""
        np_array = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)

        # Chain: create -> to GPU -> back to CPU -> to numpy
        result = IFieldArray.from_numpy(np_array).to_device(0).to_device(-1).numpy()

        self.assertTrue(np.array_equal(np_array, result))

    async def test_to_device_different_dtypes(self):
        """Test to_device with different data types."""
        dtypes = [np.float32, np.float64, np.int32, np.int64, np.uint32, np.uint64]

        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                np_array = np.arange(100, dtype=dtype).reshape(10, 10)
                cpu_array = IFieldArray.from_numpy(np_array)

                # Copy to GPU and back
                gpu_array = cpu_array.to_device(0)
                cpu_array2 = gpu_array.to_device(-1)

                # Verify
                result = cpu_array2.numpy()
                self.assertTrue(np.array_equal(np_array, result))
                self.assertEqual(result.dtype, dtype)

    async def test_reinterpret_same_size_types(self):
        """Test reinterpreting between types of the same size."""

        # float32 <-> int32 (both 4 bytes)
        np_array = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        float_array = IFieldArray.from_numpy(np_array)

        # Reinterpret as int32
        int_array = float_array.reinterpret(np.int32)
        self.assertEqual(int_array.device_id, float_array.device_id)
        self.assertEqual(tuple(int_array.shape), tuple(float_array.shape))
        self.assertEqual(int_array.dtype, np.int32)

        # Reinterpret back to float32
        float_array2 = int_array.reinterpret(np.float32)
        result = float_array2.numpy()
        self.assertTrue(np.allclose(np_array, result))

    async def test_reinterpret_size_change(self):
        """Test reinterpreting to different size types."""

        # int64 -> int32 (8 bytes -> 4 bytes, doubles count)
        np_array = np.array([1, 2, 3, 4], dtype=np.int64)
        int64_array = IFieldArray.from_numpy(np_array)

        # Reinterpret as int32 (should have 2x elements)
        int32_array = int64_array.reinterpret(np.int32)
        self.assertEqual(tuple(int32_array.shape), (8,))  # 4 int64 = 8 int32
        self.assertEqual(int32_array.dtype, np.int32)

    async def test_reinterpret_multidimensional(self):
        """Test reinterpreting multi-dimensional arrays."""

        # Create 2D array of float32
        np_array = np.arange(24, dtype=np.float32).reshape(6, 4)
        float_array = IFieldArray.from_numpy(np_array)

        # Reinterpret as int32 (same size, shape unchanged)
        int_array = float_array.reinterpret(np.int32)
        self.assertEqual(tuple(int_array.shape), (6, 4))

        # Reinterpret to int64 (larger type, last dim halves)
        int64_array = float_array.reinterpret(np.int64)
        self.assertEqual(tuple(int64_array.shape), (6, 2))  # 4 float32 = 2 int64

    async def test_reinterpret_zero_copy(self):
        """Verify that reinterpret is truly zero-copy."""

        # Create array
        np_array = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        float_array = IFieldArray.from_numpy(np_array)
        original_ptr = float_array.__array_interface__["data"][0]

        # Reinterpret
        int_array = float_array.reinterpret(np.int32)
        reinterpreted_ptr = int_array.__array_interface__["data"][0]

        # Should point to same memory
        self.assertEqual(original_ptr, reinterpreted_ptr, "Reinterpret should be zero-copy")

    async def test_reinterpret_invalid_size(self):
        """Test that reinterpret fails with incompatible sizes."""

        # Create array of 3 float32 values (12 bytes)
        np_array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        float_array = IFieldArray.from_numpy(np_array)

        # Try to reinterpret as int64 (8 bytes) - should fail
        # 12 bytes is not divisible by 8
        with self.assertRaises(Exception):
            float_array.reinterpret(np.int64)

    async def test_reinterpret_same_type(self):
        """Test that reinterpreting to same type returns original."""

        np_array = np.array([1, 2, 3, 4], dtype=np.int32)
        int_array = IFieldArray.from_numpy(np_array)

        # Reinterpret to same type should return original
        same_array = int_array.reinterpret(np.int32)
        self.assertIs(same_array, int_array)
