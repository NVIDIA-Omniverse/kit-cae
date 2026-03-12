# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
r"""
Scoped timer utility.

This module provides a scoped timer utility that can be used to time compute operations.
Using this instead of directly using `warp.ScopedTimer` is recommend for code in DAV
to make it easy to respect `dav.config` settings.

Typical usage is to wrap a compute operation with a scoped timer.

Example:
    with scoped_timer("compute"):
        compute_operation()
"""

import warp as wp

from . import config


def scoped_timer(name: str, **kwargs):
    """
    Create a scoped timer.

    Args:
        name: The name of the timer.
        **kwargs: Additional keyword arguments to pass to the scoped timer.

    Returns:
        A scoped timer.
    """
    use_nvtx = kwargs.pop("use_nvtx", config.enable_nvtx)
    active = kwargs.pop("active", config.enable_timing)
    synchronize = kwargs.pop("synchronize", True)
    return wp.ScopedTimer(name, active=active, use_nvtx=use_nvtx, synchronize=synchronize, **kwargs)
