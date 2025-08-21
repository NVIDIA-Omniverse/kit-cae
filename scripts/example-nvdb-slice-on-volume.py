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
from pxr import Gf, Usd, UsdGeom, UsdVol

# Usage:
# Copy paste this script into the Script Editor (Developer > Script Editor) or execute it on launch w/
# ./repo.sh launch -n omni.cae.kit -- --exec ./scripts/example-nvdb-slice-on-volume.py


async def main():
    # 0. Open the CGNS file
    ctx = omni.usd.get_context()
    ctx.open_stage(os.path.join(os.path.dirname(__file__), "../data/StaticMixer.cgns"))
    stage: Usd.Stage = ctx.get_stage()

    # 1. Generate the support volume for slicing
    dataset_path: str = "/World/StaticMixer_cgns/Base/StaticMixer/B1_P3"
    vol_path: str = "/World/CAE/IndeXVolume_B1_P3"
    omni.kit.commands.execute("CreateCaeNanoVdbIndeXVolume", dataset_path=dataset_path, prim_path=vol_path)

    vol_prim: Usd.Prim = stage.GetPrimAtPath(vol_path)
    UsdVol.Volume(vol_prim).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
    vol_prim.GetRelationship("omni:cae:index:nvdb:field").SetTargets(
        ["/World/StaticMixer_cgns/Base/StaticMixer/Flow_Solution/Eddy_Viscosity"]
    )
    vol_prim.GetAttribute("omni:cae:index:nvdb:maxResolution").Set(64)

    # 2 Create the slice on X axis
    slice_path: str = "/World/CAE/IndeXSlice_IndeXVolume_B1_P3"
    omni.kit.commands.execute("CreateCaeIndeXVolumeSlice", dataset_path=vol_path, prim_path=slice_path)

    # let the stage update
    for i in range(10):
        await asyncio.sleep(0.1)
        await omni.kit.app.get_app().next_update_async()

    # 3. Select and frame the points
    ctx.get_selection().set_selected_prim_paths([slice_path], True)

    # https://docs.omniverse.nvidia.com/kit/docs/omni.usd/1.11.2+106/omni.usd.commands/omni.usd.commands.FramePrimsCommand.html
    viewport = get_active_viewport()
    camera_path = viewport.camera_path
    omni.kit.commands.execute("FramePrimsCommand", prim_to_move=camera_path, prims_to_frame=[slice_path], zoom=0.5)


asyncio.ensure_future(main())
