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
from pxr import Usd, UsdGeom, UsdVol

# Usage:
# Copy paste this script into the Script Editor (Developer > Script Editor) or execute it on launch w/
# ./repo.sh launch -n omni.cae.kit -- --exec ./scripts/example_slice.py


async def main():
    # 0. Import the CGNS file
    await import_to_stage(get_test_data_path("StaticMixer.cgns"), "/World/StaticMixer")

    ctx = get_context()
    stage: Usd.Stage = ctx.get_stage()

    # 1. Generate the support volume for slicing
    dataset_path: str = "/World/StaticMixer/Base/StaticMixer/B1_P3"
    vol_path: str = "/World/CAE/IndeXVolume_B1_P3"
    await execute_command("CreateCaeVizVolume", dataset_path=dataset_path, prim_path=vol_path, type="irregular")

    # 2. Set the field for the volume
    vol_prim: Usd.Prim = stage.GetPrimAtPath(vol_path)
    colors_fs_api = cae_viz.FieldSelectionAPI(vol_prim, "colors")
    colors_fs_api.CreateTargetRel().SetTargets(["/World/StaticMixer/Base/StaticMixer/Flow_Solution/Eddy_Viscosity"])
    await wait_for_update()

    # 3. Create the slice on X axis
    slice_path: str = "/World/CAE/IndeXSlice_IndeXVolume_B1_P3"
    await execute_command("CreateCaeVizVolumeSlice", volume_path=vol_path, prim_path=slice_path, shape="Plane")
    slice_prim: Usd.Prim = stage.GetPrimAtPath(slice_path)

    # 4. Hide the volume
    UsdVol.Volume(vol_prim).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)

    # 5. Select the slice
    ctx.get_selection().set_selected_prim_paths([slice_path], True)

    # 6. Create a Bounding Box
    bbox_path = "/World/CAE/BoundingBox_IndeXVolume_B1_P3"
    await execute_command("CreateCaeVizBoundingBox", dataset_paths=[dataset_path], prim_path=bbox_path)
    await wait_for_update()

    # 7. Frame the bounding box
    await frame_prims([bbox_path], zoom=1.0)

    # 9. Select the slice
    ctx.get_selection().set_selected_prim_paths([slice_path], True)

    # 10. Frame the slice
    await frame_prims([slice_path], zoom=1.0)


if __name__ == "__main__":
    asyncio.ensure_future(main())
