# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os

enable_timing = False
"""When enabled, all compute operations will be timed and the results will be printed to the console."""

enable_nvtx = False
"""When enabled and `enable_timing` is True, all compute operations will be profiled using NVTX."""

max_points_per_cell = 64
"""Maximum number of points per cell."""

compile_kernels_aot = os.environ.get("DAV_COMPILE_KERNELS_AOT", "0") not in ("0", "false", "False", "")
"""When enabled, kernels will be compiled AOT at module import time.
This can help reduce runtime overhead for the first call to these kernels, at the cost of increased import time.
Controlled by the ``DAV_COMPILE_KERNELS_AOT`` environment variable (set to ``1`` or ``true`` to enable)."""

aot_record_path = None
"""When set to a file path string, the AOT configuration is written (or overwritten) as JSON to
that path on every new kernel specialization encountered during JIT compilation.  Useful for
long-running applications where you want a continuously-updated snapshot of observed
specializations without needing a :class:`~dav.core.recorder.Recorder` context manager."""
