# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from logging import getLogger
from typing import Any

import numpy as np
import warp as wp

import dav
from dav.core import aot

logger = getLogger(__name__)


@dav.kernel
def _count_unique_values_kernel(array: wp.array(dtype=wp.int32), counts: wp.array(dtype=wp.int32)):
    idx = wp.tid()
    value = array[idx]
    if value >= 0 and value < counts.shape[0]:
        wp.atomic_add(counts, value, 1)


def get_unique_values(array: wp.array(dtype=wp.int32), max_value: int) -> wp.array(dtype=wp.int32):
    """
    Get unique values from a 1D Warp array with a known maximum value
    which is small enough to allow for a counting sort approach.

    Args:
        array: A 1D Warp array of any numeric type.
    """

    assert max_value < 10000, "Max value is too large for counting sort approach"

    device = array.device
    counts = wp.zeros(max_value + 1, dtype=wp.int32, device=device)
    wp.launch(_count_unique_values_kernel, dim=array.shape, inputs=[array, counts], device=device)
    counts_host = counts.numpy()
    unique_values = np.flatnonzero(counts_host)
    return wp.array(unique_values, dtype=wp.int32, device=device)


@dav.kernel
def _get_min_max_kernel(array: wp.array(dtype=Any), min_max: wp.array(dtype=Any)):
    idx = wp.tid()
    value = array[idx]
    wp.atomic_min(min_max, 0, value)
    wp.atomic_max(min_max, 1, value)


# Add overloads for AOT compilation for all requested scalar types
for T in aot.get_scalar_types():
    wp.overload(_get_min_max_kernel, {"array": wp.array(dtype=T), "min_max": wp.array(dtype=T)})


def get_scalar_min_max(array: wp.array) -> tuple[Any, Any]:
    """
    Get the minimum and maximum values from a 1D Warp array.

    Args:
        array: A 1D Warp array of any numeric type.
    Returns:
        A tuple containing the minimum and maximum values.
    """
    device = array.device
    assert array.shape[0] > 0, "Input array must not be empty"
    value = array[0:1].numpy().item()  # Get a single value to determine the type
    min_max = wp.array([value, value], dtype=array.dtype, device=device)
    wp.launch(_get_min_max_kernel, dim=array.shape, inputs=[array, min_max], device=device)
    min_max_host = min_max.numpy().tolist()
    return min_max_host[0], min_max_host[1]


if dav.config.compile_kernels_aot:
    from dav.core import aot

    logger.info("Compiling data model utility kernels ...")
    wp.compile_aot_module(__name__, device=aot.get_devices())
