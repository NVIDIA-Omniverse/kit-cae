# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os.path

import omni.kit.commands
import omni.usd
from omni.kit.viewport.utility import get_active_viewport
from pxr import Usd

# Usage:
# Copy paste this script into the Script Editor (Developer > Script Editor) or execute it on launch w/
# ./repo.sh launch -n omni.cae.kit -- --exec scripts/example-points.py

# 0. Open the CGNS file
ctx = omni.usd.get_context()
ctx.open_stage(os.path.join(os.path.dirname(__file__), "../data/StaticMixer.cgns"))

# 1. Generate the points
dataset_path: str = "/World/StaticMixer_cgns/Base/StaticMixer/GridCoordinates"
viz_path = "/World/CAE/Points_GridCoordinates"
omni.kit.commands.execute("CreateCaeAlgorithmsExtractPoints", dataset_path=dataset_path, prim_path=viz_path)

# 2. Select and frame the points
stage: Usd.Stage = ctx.get_stage()
viz_prim: Usd.Prim = stage.GetPrimAtPath(viz_path)

ctx.get_selection().set_selected_prim_paths([viz_path], True)
# https://docs.omniverse.nvidia.com/kit/docs/omni.usd/1.11.2+106/omni.usd.commands/omni.usd.commands.FramePrimsCommand.html
viewport = get_active_viewport()
camera_path = viewport.camera_path
omni.kit.commands.execute("FramePrimsCommand", prim_to_move=camera_path, prims_to_frame=[viz_path], zoom=0.05)

# 3. Set the point width
viz_prim.GetAttribute("omni:cae:algorithms:points:width").Set(0.25)

# 4. Set the point color target
viz_prim.GetRelationship("omni:cae:algorithms:points:colors").SetTargets(
    ["/World/StaticMixer_cgns/Base/StaticMixer/Flow_Solution/Temperature"]
)
