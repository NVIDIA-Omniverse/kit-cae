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
import os

import omni.kit.commands
import omni.usd
from omni.cae.asset_importer.vtk import VTKImporter
from omni.kit.viewport.utility import get_active_viewport
from pxr import Sdf, Usd, UsdGeom

# Usage:
# Copy paste this script into the Script Editor (Developer > Script Editor) or execute it on launch w/
# ./repo.sh launch -n omni.cae_vtk.kit -- --exec scripts/example-headsq-vti.py


# 0. Import the vti file
async def vti_to_usd(vti_path: str):
    importer = VTKImporter()
    out = await importer.convert_assets([vti_path], import_as_reference=True)
    usd = next(iter(out.values()))
    return usd


async def main():
    # 0. Convert the NPZ file to USD and open it
    vti_path = os.path.join(os.path.dirname(__file__), "../data/headsq.vti")
    usd_path = await vti_to_usd(vti_path)
    ctx = omni.usd.get_context()
    ctx.open_stage(usd_path)
    stage: Usd.Stage = ctx.get_stage()

    # 2. Generate the volume data
    dataset_path = "/World/headsq_vti/VTKImageData"
    viz_path = "/World/CAE/NanoVdbIndeXVolume_VTKImageData"
    omni.kit.commands.execute("CreateCaeNanoVdbIndeXVolume", dataset_path=dataset_path, prim_path=viz_path)

    # 3. Specify the field
    viz_prim = stage.GetPrimAtPath(viz_path)
    viz_prim.GetRelationship("omni:cae:index:nvdb:field").SetTargets(["/World/headsq_vti/PointData/Scalars_"])

    # 4. Create Bounding Box
    bbox_path = "/World/CAE/BoundingBox_VTKDataSet"
    omni.kit.commands.execute(
        "CreateCaeAlgorithmsExtractBoundingBox", dataset_paths=[dataset_path], prim_path=bbox_path
    )

    # 5. Set width
    bbox_prim = stage.GetPrimAtPath(bbox_path)
    bbox_prim.CreateAttribute("omni:cae:algorithms:boundingBox:width", Sdf.ValueTypeNames.Float, custom=False).Set(1.0)

    # 6. Set scale
    xformApi = UsdGeom.XformCommonAPI(bbox_prim)
    xformApi.SetScale((0.5, 1.0, 1.0))

    # 6. Set ROI
    viz_prim.GetRelationship("omni:cae:index:nvdb:roi").SetTargets([bbox_path])

    # 7. Let the stage update
    for i in range(10):
        await asyncio.sleep(0.1)
        await omni.kit.app.get_app().next_update_async()

    # 8. Zoom
    ctx.get_selection().set_selected_prim_paths([viz_path], True)
    # https://docs.omniverse.nvidia.com/kit/docs/omni.usd/1.11.2+106/omni.usd.commands/omni.usd.commands.FramePrimsCommand.html
    viewport = get_active_viewport()
    camera_path = viewport.camera_path
    omni.kit.commands.execute("FramePrimsCommand", prim_to_move=camera_path, prims_to_frame=[bbox_path], zoom=0.9)


asyncio.ensure_future(main())
