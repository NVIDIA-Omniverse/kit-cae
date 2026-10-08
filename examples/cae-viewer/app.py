# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Open CAE USD scenes and manipulate seeds with viewport gizmos."""

from __future__ import annotations

import argparse
import ctypes
import logging
import os
from importlib.metadata import distribution
from pathlib import Path


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", nargs="?", type=Path, help="USD file to open; otherwise start with an empty view")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--fps", type=int, default=0, help="frame-rate limit; 0 means uncapped (default)")
    parser.add_argument("--device", default="cuda:0", help="Warp compute device")
    parser.add_argument("--headless", action="store_true", help="use OVUI's offscreen Vulkan backend")
    parser.add_argument("--frames", type=int, default=0, help="exit after N rendered frames (0 runs until closed)")
    parser.add_argument("--screenshot", type=Path, help="save the complete UI on exit; requires --frames")
    parser.add_argument(
        "--log-file", type=Path, help="write a timestamped DEBUG trace of edits, evaluation, and frames"
    )
    args = parser.parse_args(argv)
    if min(args.width, args.height) <= 0 or args.fps < 0 or args.frames < 0:
        parser.error("width and height must be positive; fps and frames must be nonnegative")
    if args.screenshot and not args.frames:
        parser.error("--screenshot requires a positive --frames count")
    return args


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.log_file:
        handler = logging.FileHandler(args.log_file, mode="w")
        handler.setFormatter(
            logging.Formatter("%(asctime)s.%(msecs)03d %(threadName)-8s %(name)s: %(message)s", "%H:%M:%S")
        )
        for name in ("cae_viewer", "ovcae"):
            logger = logging.getLogger(name)
            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG)
    if args.headless:
        os.environ["OMNIUI_HEADLESS"] = "1"
        os.environ["OMNIUI_BACKEND"] = "vulkan"

    # Load the wheel's CUDA runtime by absolute path before OVUI's native
    # extension. No system toolkit or process-wide library-path override is needed.
    runtime = distribution("nvidia-cuda-runtime-cu12").locate_file("nvidia/cuda_runtime/lib/libcudart.so.12")
    cuda_runtime = ctypes.CDLL(str(runtime), mode=ctypes.RTLD_GLOBAL)
    from cae_viewer.application import Viewer

    Viewer(args).run()
    del cuda_runtime


if __name__ == "__main__":
    main()
