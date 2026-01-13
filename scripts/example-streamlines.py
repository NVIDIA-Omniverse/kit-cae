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
# ./repo.sh launch -n omni.cae_vtk.kit -- --exec ./scripts/example-streamlines.py

# 0. Open the CGNS file
ctx = omni.usd.get_context()
ctx.open_stage(os.path.join(os.path.dirname(__file__), "../data/StaticMixer.cgns"))

# 1. Generate the streamlines and the seed sphere
dataset_path: str = "/World/StaticMixer_cgns/Base/StaticMixer/B1_P3"
viz_path: str = "/World/CAE/Streamlines_B1_P3"
sphere_path: str = "/World/CAE/Sphere"
sphere_scale = 0.01
omni.kit.commands.execute("CreateCaeStreamlines", dataset_path=dataset_path, prim_path=viz_path)
omni.kit.commands.execute("CreateMeshPrim", prim_type="Sphere", prim_path=sphere_path)
omni.kit.commands.execute("TransformPrimSRT", path=sphere_path, new_scale=[sphere_scale, sphere_scale, sphere_scale])

# 2. Select and frame the points
stage: Usd.Stage = ctx.get_stage()
viz_prim: Usd.Prim = stage.GetPrimAtPath(viz_path)

ctx.get_selection().set_selected_prim_paths([sphere_path], True)
# https://docs.omniverse.nvidia.com/kit/docs/omni.usd/1.11.2+106/omni.usd.commands/omni.usd.commands.FramePrimsCommand.html
viewport = get_active_viewport()
camera_path = viewport.camera_path
omni.kit.commands.execute("FramePrimsCommand", prim_to_move=camera_path, prims_to_frame=[sphere_path], zoom=1.0)

# 3. Set the seed target to the sphere prim
viz_prim.GetRelationship("omni:cae:algorithms:streamlines:seeds").SetTargets([sphere_path])

# 4. Set the velocity targets
flow_solution_prim_path = "/World/StaticMixer_cgns/Base/StaticMixer/Flow_Solution"
velocity_targets = [
    f"{flow_solution_prim_path}/VelocityX",
    f"{flow_solution_prim_path}/VelocityY",
    f"{flow_solution_prim_path}/VelocityZ",
]
viz_prim.GetRelationship("omni:cae:algorithms:streamlines:velocity").SetTargets(velocity_targets)

# 5. Set the color target
viz_prim.GetRelationship("omni:cae:algorithms:streamlines:colors").SetTargets(
    ["/World/StaticMixer_cgns/Base/StaticMixer/Flow_Solution/Temperature"]
)
