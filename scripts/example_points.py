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
from omni.cae.schema import viz as cae_viz
from omni.cae.testing import frame_prims, get_test_data_path, wait_for_update
from omni.usd import get_context
from pxr import Usd

# Usage:
# Copy paste this script into the Script Editor (Developer > Script Editor) or execute it on launch w/
# ./repo.sh launch -n omni.cae.kit -- --exec ./scripts/example_points.py


async def main():
    # 0. Import the CGNS file
    await import_to_stage(get_test_data_path("StaticMixer.cgns"), "/World/StaticMixer")

    ctx = get_context()
    stage: Usd.Stage = ctx.get_stage()

    # 1. Create bounding box
    dataset_path: str = "/World/StaticMixer/Base/StaticMixer/B1_P3"
    flow_solution_path: str = "/World/StaticMixer/Base/StaticMixer/Flow_Solution"
    bbox_path: str = "/World/CAE/BoundingBox_B1_P3"
    await execute_command("CreateCaeVizBoundingBox", dataset_paths=[dataset_path], prim_path=bbox_path)

    # 2. Create points
    points_path: str = "/World/CAE/Points_B1_P3"
    await execute_command("CreateCaeVizPoints", dataset_path=dataset_path, prim_path=points_path)
    points_prim: Usd.Prim = stage.GetPrimAtPath(points_path)

    points_api = cae_viz.PointsAPI(points_prim)
    points_api.CreateWidthAttr().Set(0.5)
    await wait_for_update()

    # Color by Temperature
    colors_fs_api = cae_viz.FieldSelectionAPI(points_prim, "colors")
    colors_fs_api.CreateTargetRel().SetTargets([f"{flow_solution_path}/Temperature"])
    await wait_for_update()

    # Width by Pressure
    field_mapping_api = cae_viz.FieldMappingAPI(points_prim, "widths")
    # Specifies the range for the widths field mapped to the range of the Velocity magnitude field
    field_mapping_api.CreateRangeAttr().Set((0.01, 0.2))
    widths_fs_api = cae_viz.FieldSelectionAPI(points_prim, "widths")
    widths_fs_api.CreateTargetRel().SetTargets(
        [
            f"{flow_solution_path}/VelocityX",
            f"{flow_solution_path}/VelocityY",
            f"{flow_solution_path}/VelocityZ",
        ]
    )
    # Mode can be used to compute magnitude of the vector field on the fly
    widths_fs_api.CreateModeAttr().Set(cae_viz.Tokens.vector_magnitude)
    await wait_for_update()

    # 3. Frame the bounding box
    await frame_prims([bbox_path], zoom=1.0)


if __name__ == "__main__":
    asyncio.ensure_future(main())
