# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import asyncio

from omni.cae.data.commands import execute_command
from omni.cae.importer.cgns import import_to_stage
from omni.cae.testing import frame_prims, get_test_data_path, wait_for_update
from omni.usd import get_context

# Usage:
# Copy paste this script into the Script Editor (Developer > Script Editor) or execute it on launch w/
# ./repo.sh launch -n omni.cae.kit -- --exec scripts/example_bounding_box.py


async def main():
    # 0. Import the CGNS file
    ctx = get_context()

    # Import the CGNS file to the stage
    await import_to_stage(get_test_data_path("StaticMixer.cgns"), "/World/StaticMixer")

    # Base path for the imported dataset
    base_path = "/World/StaticMixer/Base/StaticMixer"

    # 1. Generate the bounding box for GridCoordinates
    dataset_path: str = f"{base_path}/GridCoordinates"
    viz_path = "/World/CAE/BoundingBox_GridCoordinates"
    await execute_command("CreateCaeVizBoundingBox", dataset_paths=[dataset_path], prim_path=viz_path)
    await wait_for_update()

    # 2. Generate the bounding box for 2 datasets (in1 and in2)
    dataset_paths: list[str] = [
        f"{base_path}/in1",
        f"{base_path}/in2",
    ]
    viz_path2 = "/World/CAE/BoundingBox_in1_in2"
    await execute_command("CreateCaeVizBoundingBox", dataset_paths=dataset_paths, prim_path=viz_path2)
    await wait_for_update()

    # 3. Select and frame the first bounding box
    ctx.get_selection().set_selected_prim_paths([viz_path], True)

    # Wait for the stage to update
    await wait_for_update()

    # Frame the camera on the bounding box
    await frame_prims([viz_path], zoom=0.8)


if __name__ == "__main__":
    asyncio.ensure_future(main())
