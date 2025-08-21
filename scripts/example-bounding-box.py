# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
#  its affiliates is strictly prohibited.

import asyncio
import os.path

import omni.kit.commands
import omni.usd
from omni.kit.viewport.utility import get_active_viewport

# Usage:
# Copy paste this script into the Script Editor (Developer > Script Editor) or execute it on launch w/
# ./repo.sh launch -n omni.cae.kit -- --exec scripts/example-bounding-box.py


async def main():
    # 0. Open the CGNS file
    ctx = omni.usd.get_context()
    await ctx.open_stage_async(os.path.join(os.path.dirname(__file__), "../data/StaticMixer.cgns"))

    # 1. Generate the bounding box
    dataset_path: str = "/World/StaticMixer_cgns/Base/StaticMixer/GridCoordinates"
    viz_path = "/World/CAE/BoundingBox_GridCoordinates"
    omni.kit.commands.execute("CreateCaeAlgorithmsExtractBoundingBox", dataset_paths=[dataset_path], prim_path=viz_path)

    # 2. Generate the bounding box for 2 datasets
    dataset_paths: list[str] = [
        "/World/StaticMixer_cgns/Base/StaticMixer/in1",
        "/World/StaticMixer_cgns/Base/StaticMixer/in2",
    ]
    viz_path2 = "/World/CAE/BoundingBox"
    omni.kit.commands.execute("CreateCaeAlgorithmsExtractBoundingBox", dataset_paths=dataset_paths, prim_path=viz_path2)

    # 3. Select and frame the points
    ctx.get_selection().set_selected_prim_paths([viz_path], True)

    # let the stage update
    for i in range(10):
        await asyncio.sleep(0.1)
        await omni.kit.app.get_app().next_update_async()

    # https://docs.omniverse.nvidia.com/kit/docs/omni.usd/1.11.2+106/omni.usd.commands/omni.usd.commands.FramePrimsCommand.html
    viewport = get_active_viewport()
    camera_path = viewport.camera_path
    omni.kit.commands.execute("FramePrimsCommand", prim_to_move=camera_path, prims_to_frame=[viz_path], zoom=0.8)


asyncio.ensure_future(main())
