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

import omni.timeline
from omni.cae.data.commands import execute_command
from omni.cae.importer.cgns import import_to_stage
from omni.cae.schema import viz as cae_viz
from omni.cae.testing import frame_prims, get_test_data_path, wait_for_update
from omni.usd import get_context
from pxr import Usd, UsdGeom, UsdVol

# Usage:
# Copy paste this script into the Script Editor (Developer > Script Editor) or execute it on launch w/
# ./repo.sh launch -n omni.cae.kit -- --exec ./scripts/example_temporal_interpolation.py


async def main(skip_play: bool = False):
    # 0. Import the CGNS file
    await import_to_stage(get_test_data_path("hex_timesteps.cgns"), "/World/hex_timesteps", time_scale=10.0)

    ctx = get_context()
    stage: Usd.Stage = ctx.get_stage()

    # 1. Generate the support volume for slicing
    dataset_path: str = "/World/hex_timesteps/Base/Zone/ElementsUniform"
    field_path: str = "/World/hex_timesteps/Base/Zone/SolutionVertex0001/PointSinusoid"
    vol_path: str = "/World/CAE/Volume_ElementsUniform"
    await execute_command("CreateCaeVizVolume", dataset_path=dataset_path, prim_path=vol_path, type="vdb")

    # 2. Set the field for the volume
    vol_prim: Usd.Prim = stage.GetPrimAtPath(vol_path)
    colors_fs_api = cae_viz.FieldSelectionAPI(vol_prim, "colors")
    colors_fs_api.CreateTargetRel().SetTargets([field_path])
    await wait_for_update()

    # 3. Enable temporal interpolation
    cae_viz.OperatorTemporalAPI.Apply(vol_prim)
    temporal_api = cae_viz.OperatorTemporalAPI(vol_prim)
    temporal_api.CreateEnableFieldInterpolationAttr().Set(True)
    await wait_for_update()

    # 4. Create the slice bi-plane
    slice_path: str = "/World/CAE/Slice_ElementsUniform"
    await execute_command("CreateCaeVizVolumeSlice", volume_path=vol_path, prim_path=slice_path, shape="Bi-Plane")

    # 5. Hide the volume
    UsdVol.Volume(vol_prim).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)

    # 6. Select the slice
    ctx.get_selection().set_selected_prim_paths([slice_path], True)

    # 7. Create a Bounding Box
    bbox_path = "/World/CAE/BoundingBox_ElementsUniform"
    await execute_command("CreateCaeVizBoundingBox", dataset_paths=[dataset_path], prim_path=bbox_path)
    await wait_for_update()

    # 7. Frame the bounding box
    await frame_prims([bbox_path], zoom=1.0)

    # 9. Select the slice
    ctx.get_selection().set_selected_prim_paths([slice_path], True)

    # 10. Play the animation
    if not skip_play:
        timeline = omni.timeline.get_timeline_interface()
        timeline.play()


if __name__ == "__main__":
    asyncio.ensure_future(main())
