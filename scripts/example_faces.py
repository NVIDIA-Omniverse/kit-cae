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

# Usage:
# Copy paste this script into the Script Editor (Developer > Script Editor) or execute it on launch w/
# ./repo.sh launch -n omni.cae.kit -- --exec scripts/example_faces.py


async def main():
    # 0. Import the CGNS file
    ctx = get_context()

    # Import the CGNS file to the stage
    await import_to_stage(get_test_data_path("StaticMixer.cgns"), "/World/StaticMixer")

    # Base path for the imported dataset
    base_path = "/World/StaticMixer/Base/StaticMixer"

    # 1. Generate the faces for StaticMixer_Default
    dataset_path: str = f"{base_path}/StaticMixer_Default"
    viz_path: str = "/World/CAE/ExternalFaces_StaticMixer_Default"
    await execute_command("CreateCaeVizFaces", dataset_path=dataset_path, prim_path=viz_path)
    await wait_for_update()

    # 2. Add color by Temperature field
    stage = ctx.get_stage()
    viz_prim = stage.GetPrimAtPath(viz_path)
    temp_field_path = f"{base_path}/Flow_Solution/Temperature"

    colors_fs_api = cae_viz.FieldSelectionAPI(viz_prim, "colors")
    colors_fs_api.CreateTargetRel().SetTargets([temp_field_path])
    await wait_for_update()

    # 3. Select and frame the mesh
    ctx.get_selection().set_selected_prim_paths([viz_path], True)

    # Wait for the stage to update
    await wait_for_update()

    # Frame the camera on the faces
    await frame_prims([viz_path], zoom=0.05)


if __name__ == "__main__":
    asyncio.ensure_future(main())
