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
import omni.kit.test
from omni.cae.data import array_utils

logger = logging.getLogger(__name__)


class TestScalarStats(omni.kit.test.AsyncTestCase):
    """Tests for get_scalar_stats and compute_histogram with various dtypes."""

    def _validate_stats(self, stats, expected_min, expected_max, n, num_bins=32):
        """Common validation for get_scalar_stats results."""
        self.assertIn("counts", stats)
        self.assertIn("bin_edges", stats)
        self.assertIn("mean", stats)
        self.assertIn("min", stats)
        self.assertIn("max", stats)
        self.assertIn("median", stats)
        self.assertIn("q1", stats)
        self.assertIn("q2", stats)
        self.assertIn("q3", stats)
        self.assertIn("q4", stats)

        self.assertEqual(len(stats["counts"]), num_bins)
        self.assertEqual(len(stats["bin_edges"]), num_bins + 1)

        # The histogram uses half-open bins [min, max), so the single element
        # at exactly val_max may not be counted.  Allow off-by-one.
        total = sum(stats["counts"])
        self.assertTrue(n - 1 <= total <= n, f"Expected {n-1}..{n} counts, got {total}")

        # Min/max must match expected values.
        self.assertAlmostEqual(float(stats["min"]), float(expected_min), places=5)
        self.assertAlmostEqual(float(stats["max"]), float(expected_max), places=5)

    # ------------------------------------------------------------------
    # get_scalar_stats – float baselines
    # ------------------------------------------------------------------

    async def test_get_scalar_stats_float32(self):
        rng = np.random.default_rng(42)
        data = rng.uniform(0.0, 100.0, size=1024).astype(np.float32)
        stats = array_utils.get_scalar_stats(data, num_bins=32)
        self._validate_stats(stats, np.min(data), np.max(data), len(data))
        self.assertAlmostEqual(stats["mean"], float(np.mean(data)), delta=0.5)

    async def test_get_scalar_stats_float64(self):
        rng = np.random.default_rng(42)
        data = rng.uniform(-50.0, 50.0, size=512).astype(np.float64)
        stats = array_utils.get_scalar_stats(data, num_bins=16)
        self._validate_stats(stats, np.min(data), np.max(data), len(data), num_bins=16)
        self.assertAlmostEqual(stats["mean"], float(np.mean(data)), delta=0.5)

    # ------------------------------------------------------------------
    # get_scalar_stats – integer types (the core bug fix)
    # ------------------------------------------------------------------

    async def test_get_scalar_stats_int32(self):
        rng = np.random.default_rng(42)
        data = rng.integers(0, 1000, size=1024).astype(np.int32)
        stats = array_utils.get_scalar_stats(data, num_bins=32)
        self._validate_stats(stats, np.min(data), np.max(data), len(data))
        self.assertAlmostEqual(stats["mean"], float(np.mean(data)), delta=2.0)

    async def test_get_scalar_stats_int64(self):
        rng = np.random.default_rng(123)
        data = rng.integers(-500, 500, size=512).astype(np.int64)
        stats = array_utils.get_scalar_stats(data, num_bins=32)
        self._validate_stats(stats, np.min(data), np.max(data), len(data))
        self.assertAlmostEqual(stats["mean"], float(np.mean(data)), delta=2.0)

    # ------------------------------------------------------------------
    # get_scalar_stats – edge cases
    # ------------------------------------------------------------------

    async def test_get_scalar_stats_constant_int_array(self):
        """All-same-value array should not crash (bin_width == 0 path)."""
        data = np.full(256, 42, dtype=np.int32)
        stats = array_utils.get_scalar_stats(data, num_bins=32)
        self.assertEqual(stats["min"], 42)
        self.assertEqual(stats["max"], 42)
        self.assertAlmostEqual(stats["mean"], 42.0)

    async def test_get_scalar_stats_single_element_int(self):
        """Single-element array should not crash."""
        data = np.array([7], dtype=np.int32)
        stats = array_utils.get_scalar_stats(data, num_bins=32)
        self.assertEqual(stats["min"], 7)
        self.assertEqual(stats["max"], 7)
        self.assertAlmostEqual(stats["mean"], 7.0)

    # ------------------------------------------------------------------
    # compute_histogram – integer types
    # ------------------------------------------------------------------

    async def test_compute_histogram_int32(self):
        rng = np.random.default_rng(42)
        data = rng.integers(0, 100, size=1024).astype(np.int32)
        result = array_utils.compute_histogram(data, num_bins=10, range_min=0.0, range_max=100.0)
        self.assertEqual(len(result["counts"]), 10)
        self.assertEqual(len(result["bin_edges"]), 11)
        self.assertEqual(sum(result["counts"]), len(data))

    async def test_compute_histogram_int64(self):
        rng = np.random.default_rng(99)
        data = rng.integers(-1000, 1000, size=512).astype(np.int64)
        result = array_utils.compute_histogram(data, num_bins=20, range_min=-1000.0, range_max=1000.0)
        self.assertEqual(len(result["counts"]), 20)
        self.assertEqual(sum(result["counts"]), len(data))
