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
from omni.cae.importer.npz import import_to_stage
from omni.cae.schema import viz as cae_viz
from omni.cae.testing import frame_prims, get_test_data_path, wait_for_update
from omni.usd import get_context
from pxr import Gf

# Usage:
# Copy paste this script into the Script Editor (Developer > Script Editor) or execute it on launch w/
# ./repo.sh launch -n omni.cae.kit -- --exec scripts/example_npz_point_cloud.py


async def main():
    # 0. Import the NPZ file as a point cloud
    npz_path = get_test_data_path("disk_out_ref.npz")
    await import_to_stage(npz_path, "/World/disk_out_ref_npz", schema_type="Point Cloud")

    ctx = get_context()
    stage = ctx.get_stage()

    # 1. Generate the volume data
    dataset_path = "/World/disk_out_ref_npz/NumPyDataSet"
    viz_path = "/World/CAE/Volume_NumPyDataSet"
    await execute_command("CreateCaeVizVolume", dataset_path=dataset_path, prim_path=viz_path, type="vdb")

    # 2. Set the field target for coloring
    viz_prim = stage.GetPrimAtPath(viz_path)

    assert viz_prim.HasAPI(cae_viz.FieldSelectionAPI, "colors"), "Should have FieldSelectionAPI for colors"
    colors_fs_api = cae_viz.FieldSelectionAPI(viz_prim, "colors")
    colors_fs_api.CreateTargetRel().SetTargets(["/World/disk_out_ref_npz/NumPyArrays/Temp"])

    # 3. Set the Gaussian Splatting parameters
    assert viz_prim.HasAPI(
        cae_viz.DatasetGaussianSplattingAPI, "source"
    ), "Should have DatasetGaussianSplattingAPI for source"
    gaussian_splatting_api = cae_viz.DatasetGaussianSplattingAPI(viz_prim, "source")
    gaussian_splatting_api.CreateRadiusFactorAttr().Set(2.5)
    await wait_for_update()

    # 4. Zoom
    ctx.get_selection().set_selected_prim_paths([viz_path], True)
    await frame_prims([viz_path], zoom=0.9)


if __name__ == "__main__":
    asyncio.ensure_future(main())
