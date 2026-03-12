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
from omni.cae.schema import cae
from omni.cae.schema import viz as cae_viz
from omni.cae.testing import frame_prims, get_test_data_path, wait_for_update
from omni.usd import get_context
from pxr import Usd

# Usage:
# Copy paste this script into the Script Editor (Developer > Script Editor) or execute it on launch w/
# ./repo.sh launch -n omni.cae.kit -- --exec scripts/example_streamlines.py


async def main():
    # 0. Import the CGNS file as SIDS Unstructured
    cgns_path = get_test_data_path("StaticMixer.cgns")
    await import_to_stage(cgns_path, "/World/StaticMixer")

    ctx = get_context()
    stage: Usd.Stage = ctx.get_stage()

    # 1. Create the streamlines and the seed sphere
    dataset_path: str = "/World/StaticMixer/Base/StaticMixer/B1_P3"
    flow_solution_path: str = "/World/StaticMixer/Base/StaticMixer/Flow_Solution"
    velocity_field_prefix: str = f"{flow_solution_path}/Velocity"
    temperature_field_path: str = f"{flow_solution_path}/Temperature"
    viz_path = "/World/CAE/Streamlines_B1_P3"
    sphere_path: str = "/World/CAE/Sphere"
    sphere_scale = [0.2] * 3
    sphere_translation = [-0.32, 1.69, 0.96]

    await execute_command("CreateCaeVizStreamlines", dataset_path=dataset_path, prim_path=viz_path, type="standard")
    await execute_command("CreateCaeVizMeshPrim", prim_type="UnitSphere", prim_path=sphere_path)
    await execute_command(
        "TransformPrimSRT", path=sphere_path, new_scale=sphere_scale, new_translation=sphere_translation
    )
    await wait_for_update()

    # 2. Get prims and set up streamlines
    viz_prim: Usd.Prim = stage.GetPrimAtPath(viz_path)
    sphere_prim: Usd.Prim = stage.GetPrimAtPath(sphere_path)

    # Set streamlines direction
    streamlines_api: cae_viz.StreamlinesAPI = cae_viz.StreamlinesAPI(viz_prim)
    streamlines_api.GetDirectionAttr().Set(cae_viz.Tokens.forward)

    # Set the seed target to the sphere prim
    ds_api: cae_viz.DatasetSelectionAPI = cae_viz.DatasetSelectionAPI(viz_prim, "seeds")
    ds_api.GetTargetRel().SetTargets({sphere_prim.GetPath()})

    # Set the velocity targets (V is a vector field with 3 components)
    vs_api = cae_viz.FieldSelectionAPI(viz_prim, "velocities")
    vs_api.GetTargetRel().SetTargets(
        [f"{velocity_field_prefix}X", f"{velocity_field_prefix}Y", f"{velocity_field_prefix}Z"]
    )

    # Set the color target
    colors_api = cae_viz.FieldSelectionAPI(viz_prim, "colors")
    colors_api.GetTargetRel().SetTargets([temperature_field_path])
    await wait_for_update()

    # Create a Bounding Box
    bbox_path = "/World/CAE/BoundingBox_B1_P3"
    await execute_command("CreateCaeVizBoundingBox", dataset_paths=[dataset_path], prim_path=bbox_path)
    await wait_for_update()

    # Frame the bounding box
    await frame_prims([bbox_path], zoom=1.0)

    # 3. Select the sphere
    ctx.get_selection().set_selected_prim_paths([sphere_path], True)


if __name__ == "__main__":
    asyncio.ensure_future(main())
