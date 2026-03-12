# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Kernel precompilation script run inside kit via --exec.

Arguments are received through environment variables set by the repo tool:
  DAV_COMPILE_KERNELS_AOT    - must be "1" (checked by dav.aot_compile internally)
  PRECOMPILE_KERNELS_JSON    - path to the AOT configuration JSON file
  PRECOMPILE_KERNELS_DEVICES - space-separated list of Warp device strings (e.g. "cuda cpu")
  WARP_CACHE_PATH            - (optional) kernel cache directory for Warp
"""

import logging
import os

import dav

logging.basicConfig(level=logging.INFO, format="[%(levelname)-5s] %(name)s — %(message)s")


import dav.aot_compile
import omni.kit.app
import warp as wp

json_path = os.environ.get("PRECOMPILE_KERNELS_JSON")
devices = os.environ.get("PRECOMPILE_KERNELS_DEVICES", "cuda").split()

dav.aot_compile.compile(config_path=json_path, devices=devices)

omni.kit.app.get_app().post_quit(0)
