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
import os

import omni.kit.commands
import omni.timeline
import omni.usd
from omni.cae.asset_importer.npz import NPZAssetImporter
from omni.kit.viewport.utility import get_active_viewport
from pxr import Gf

# Usage:
# Copy paste this script into the Script Editor (Developer > Script Editor) or execute it on launch w/
# ./repo.sh launch -n omni.cae.kit -- --exec scripts/example-npz-point-cloud.py


async def npz_to_usd(npz_path: str, mesh_schema_type):
    importer = NPZAssetImporter()
    importer._options_builder.get_import_context().mesh_schema_type = mesh_schema_type
    out = await importer.convert_assets([npz_path], import_as_reference=True)
    usd = next(iter(out.values()))
    return usd


async def main():
    # 0. Convert the NPZ file to USD and open it
    npz_path = os.path.join(os.path.dirname(__file__), "../data/disk_out_ref.npz")
    usd_path = await npz_to_usd(npz_path=npz_path, mesh_schema_type="Point Cloud")
    ctx = omni.usd.get_context()
    ctx.open_stage(usd_path)
    stage = ctx.get_stage()

    # 1. Fix field associations
    array_base_path = "/World/disk_out_ref_npz/NumPyArrays"
    array_paths = [f"{array_base_path}/{base}" for base in ["AsH3", "CH4", "GaMe3", "H2", "Pres", "Temp", "V"]]
    for array_path in array_paths:
        array_prim = stage.GetPrimAtPath(array_path)
        array_prim.GetAttribute("fieldAssociation").Set("vertex")

    # 2. Generate the volume data
    dataset_path = "/World/disk_out_ref_npz/NumPyDataSet"
    viz_path = "/World/CAE/NanoVdbIndeXVolume_NumPyDataSet"
    omni.kit.commands.execute("CreateCaeNanoVdbIndeXVolume", dataset_path=dataset_path, prim_path=viz_path)

    # 3. Set the velocity targets and setr the velocity scale
    viz_prim = stage.GetPrimAtPath(viz_path)
    viz_prim.GetRelationship("omni:cae:index:nvdb:field").SetTargets(["/World/disk_out_ref_npz/NumPyArrays/Temp"])

    # 4. Colormap and other params
    colormap_prim = stage.GetPrimAtPath(viz_path + "/Material/Colormap")
    colormap_prim.GetAttribute("domain").Set(Gf.Vec2f(0.0, 100.0))

    # 5. Zoom
    ctx.get_selection().set_selected_prim_paths([viz_path], True)
    # https://docs.omniverse.nvidia.com/kit/docs/omni.usd/1.11.2+106/omni.usd.commands/omni.usd.commands.FramePrimsCommand.html
    viewport = get_active_viewport()
    camera_path = viewport.camera_path
    omni.kit.commands.execute("FramePrimsCommand", prim_to_move=camera_path, prims_to_frame=[viz_path], zoom=0.05)


asyncio.ensure_future(main())
