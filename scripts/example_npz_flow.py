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

import omni.kit.commands
import omni.timeline
from omni.cae.data.commands import execute_command
from omni.cae.importer.npz import import_to_stage
from omni.cae.schema import cae
from omni.cae.schema import viz as cae_viz
from omni.cae.testing import frame_prims, get_test_data_path, wait_for_update
from omni.usd import get_context

# Usage:
# Copy paste this script into the Script Editor (Developer > Script Editor) or execute it on launch w/
# ./repo.sh launch -n omni.cae.kit -- --exec scripts/example_npz_flow.py


async def main(skip_play: bool = False):
    # 0. Import the NPZ file as SIDS Unstructured
    npz_path = get_test_data_path("disk_out_ref.npz")
    await import_to_stage(npz_path, "/World/disk_out_ref_npz", schema_type="SIDS Unstructured")

    ctx = get_context()
    stage = ctx.get_stage()

    # 1. Fix field associations; this is needed for "SIDS Unstructured" schema type
    dataset_path = "/World/disk_out_ref_npz/NumPyDataSet"
    array_base_path = "/World/disk_out_ref_npz/NumPyArrays"
    array_paths = [f"{array_base_path}/{base}" for base in ["AsH3", "CH4", "GaMe3", "H2", "Pres", "Temp", "V"]]
    for array_path in array_paths:
        array_prim = stage.GetPrimAtPath(array_path)
        field_array_api = cae.FieldArray(array_prim)
        field_array_api.CreateFieldAssociationAttr().Set(cae.Tokens.vertex)

    # 2. Create a Bounding Box
    bbox_path = "/World/CAE/BoundingBox_NumPyDataSet"
    await execute_command("CreateCaeVizBoundingBox", dataset_paths=[dataset_path], prim_path=bbox_path)
    await wait_for_update()

    # 3. Frame the bounding box
    await frame_prims([bbox_path], zoom=1.0)

    # 4. Setup Flow Environment
    flow_environment_path = "/World/CAE/FlowSimulation_L0"
    await execute_command("CreateCaeVizFlowEnvironment", prim_path=flow_environment_path, layer_number=0)
    await wait_for_update()
    flow_environment_prim = stage.GetPrimAtPath(flow_environment_path)

    # 5. Create a Smoke Injector
    smoke_injector_path = "/World/CAE/SmokeInjector_NumPyDataSet"
    await execute_command(
        "CreateCaeVizFlowSmokeInjector",
        boundable_paths=[bbox_path],
        prim_path=smoke_injector_path,
        layer_number=0,
        mode="sphere",
        simulation_prim=flow_environment_prim,
    )
    await wait_for_update()

    # 6. Create a Boundary Emitter
    boundary_emitter_path = "/World/CAE/BoundaryEmitter_NumPyDataSet"
    await execute_command(
        "CreateCaeVizFlowBoundaryEmitter", boundable_paths=[bbox_path], prim_path=boundary_emitter_path, layer_number=0
    )
    await wait_for_update()

    # 6. Create a Data Set Emitter
    ds_emitter_path = "/World/CAE/DataSetEmitter_NumPyDataSet"
    await execute_command(
        "CreateCaeVizFlowDataSetEmitter",
        dataset_path=dataset_path,
        prim_path=ds_emitter_path,
        layer_number=0,
        simulation_prim=flow_environment_prim,
    )
    await wait_for_update()

    # 7. Set the velocity targets
    ds_emitter_prim = stage.GetPrimAtPath(ds_emitter_path)
    assert ds_emitter_prim.HasAPI(
        cae_viz.FieldSelectionAPI, "velocities"
    ), "Should have FieldSelectionAPI for velocities"
    velocities_fs_api = cae_viz.FieldSelectionAPI(ds_emitter_prim, "velocities")
    velocities_fs_api.CreateTargetRel().SetTargets([f"{array_base_path}/V"])
    await wait_for_update()

    # 8. Select the smoke injector
    ctx.get_selection().set_selected_prim_paths([smoke_injector_path], True)

    # 9. Play the animation
    if not skip_play:
        timeline = omni.timeline.get_timeline_interface()
        timeline.play()


if __name__ == "__main__":
    asyncio.ensure_future(main())
