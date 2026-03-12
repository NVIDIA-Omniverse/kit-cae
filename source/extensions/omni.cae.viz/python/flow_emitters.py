# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from logging import getLogger
from typing import Any

import dav
import numpy as np
import warp as wp
from omni.cae.data import array_utils, progress, usd_utils
from omni.cae.dav import index_utils as dav_index_utils
from omni.cae.schema import viz as cae_viz
from omni.usd import get_context
from pxr import Gf, Sdf, Usd, UsdShade, UsdVol, Vt

from . import utils as viz_utils
from .execution_context import ExecutionContext
from .operator import operator

logger = getLogger(__name__)


@operator()
class FlowNanoVDBEmitter:
    prim_type: str = "FlowEmitterNanoVdb"
    api_schemas: set[str] = {
        "CaeVizDatasetSelectionAPI:source",
        "CaeVizDatasetVoxelizationAPI:source",
    }

    optional_api_schemas: set[str] = {
        "CaeVizFieldSelectionAPI:velocities",
        "CaeVizFieldSelectionAPI:temperatures",
        "CaeVizFieldThresholdingAPI:velocities",
        "CaeVizFieldThresholdingAPI:temperatures",
    }

    async def get_source(self, prim: Usd.Prim, timeCode: Usd.TimeCode, device: str) -> dav.Dataset:
        return await viz_utils.get_input_dataset(prim, "source", timeCode=timeCode, device=device)

    async def exec(self, prim: Usd.Prim, device: str, context: ExecutionContext):
        source_dataset = await self.get_source(prim, context.timecode, device)
        if len(source_dataset.get_field_names()) == 0:
            raise usd_utils.QuietableException("No fields selected. At least one field is required.")

        field_map = {"velocities": "nanoVdbVelocities", "temperatures": "nanoVdbTemperatures"}
        couple_rate_map = {"velocities": "coupleRateVelocity", "temperatures": "coupleRateTemperature"}
        for field_name, vdb_attribute_name in field_map.items():
            if source_dataset.has_field(field_name):
                field = source_dataset.get_field(field_name)
                volume: wp.Volume = field.get_data()
                assert volume is not None, f"{field_name} field is required."
                assert isinstance(volume, wp.Volume), f"{field_name} field must be a wp.Volume."
                vdb_dataset = array_utils.get_nanovdb_as_field_array(volume).to_device(-1).numpy().view(dtype=np.uint32)
                with viz_utils.edit_context(prim):
                    prim.CreateAttribute(vdb_attribute_name, Sdf.ValueTypeNames.UIntArray, custom=True).Set(
                        Vt.UIntArray.FromNumpy(vdb_dataset)
                    )
                    prim.GetAttribute(couple_rate_map[field_name]).Set(120.0)
                    del couple_rate_map[field_name]

        # for fields that are not selected, set the couple rate to 0.0
        for couple_rate_name in couple_rate_map.values():
            attr = prim.GetAttribute(couple_rate_name)
            if not viz_utils.is_attr_locked(attr):
                attr.Set(0.0)

        # process auto-scaling for fields specified by CaeVizRescaleRangeAPI instances.
        viz_utils.process_rescale_range_apis(prim, source_dataset)

        # adjust velocity scale; we use heuristic where we assume we need to cover the max distance
        # of the volume in 1 seconds using the max velocity magnitude.
        if "velocities" in field_map:
            velocity_field = source_dataset.get_field("velocities")
            with progress.ProgressContext("Executing DAV [compute velocity range]"):
                velocity_max = velocity_field.get_range()[1]
            with progress.ProgressContext("Executing DAV [compute bounds]"):
                bds_min, bds_max = source_dataset.get_bounds()
            distance = np.linalg.norm(bds_max - bds_min)
            velocity_scale = float(distance / (velocity_max * 1.0))
            attr = prim.GetAttribute("velocityScale")
            if not viz_utils.is_attr_locked(attr):
                attr.Set(velocity_scale)

            voxel_size = min(velocity_field.get_data().get_voxel_size())
        else:
            voxel_size = None

        # Process CaeVizConfigureFlowEnvironmentAPI
        stage = prim.GetStage()
        if voxel_size is not None and prim.HasAPI(cae_viz.ConfigureFlowEnvironmentAPI, "source"):
            configure_flow_environment_api = cae_viz.ConfigureFlowEnvironmentAPI(prim, "source")
            rescale_mode = configure_flow_environment_api.GetRescaleModeAttr().Get()
            for target in configure_flow_environment_api.GetDensityCellSizeIncludesRel().GetForwardedTargets():
                attr = stage.GetAttributeAtPath(target)
                if attr and viz_utils.is_attr_locked(attr):
                    continue
                elif attr and attr.GetTypeName() == Sdf.ValueTypeNames.Float:
                    if rescale_mode == cae_viz.Tokens.exact:
                        attr.Set(voxel_size)
                    elif rescale_mode == cae_viz.Tokens.minimum:
                        attr.Set(min(attr.Get(), voxel_size))
                    elif rescale_mode == cae_viz.Tokens.maximum:
                        attr.Set(max(attr.Get(), voxel_size))
                else:
                    logger.warning(
                        f"Invalid attribute type for density cell size include {target}: {attr.GetTypeName()}"
                    )
