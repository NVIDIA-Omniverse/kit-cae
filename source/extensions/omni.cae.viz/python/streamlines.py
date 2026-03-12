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

import dav
import numpy as np
import omni.cae.dav as cae_dav
from dav.operators import streamlines as dav_streamlines
from omni.cae.data import progress, usd_utils
from omni.cae.schema import viz as cae_viz
from pxr import Usd
from usdrt import Sdf as SdfRT
from usdrt import UsdGeom as UsdGeomRT
from usdrt import Vt as VtRT

from . import utils as viz_utils
from .execution_context import ExecutionContext
from .operator import operator

logger = getLogger(__name__)


@operator()
class Streamlines:
    prim_type: str = "BasisCurves"
    api_schemas: set[str] = {
        "CaeVizStreamlinesAPI",
        "CaeVizDatasetSelectionAPI:source",
        "CaeVizDatasetSelectionAPI:seeds",
        "CaeVizFieldSelectionAPI:velocities",
    }

    optional_api_schemas: set[str] = {
        "CaeVizDatasetVoxelizationAPI:source",
        "CaeVizDatasetTransformingAPI:seeds",
        "CaeVizFieldSelectionAPI",
        "CaeVizFieldMappingAPI",
        "CaeVizFieldThresholdingAPI",
    }

    def deactivate(self, prim: Usd.Prim):
        prim_rt = UsdGeomRT.BasisCurves(usd_utils.get_prim_rt(prim))
        prim_rt.CreateVisibilityAttr().Set(UsdGeomRT.Tokens.invisible)

    async def get_source(self, prim: Usd.Prim, timeCode: Usd.TimeCode, device: str) -> dav.Dataset:
        return await viz_utils.get_input_dataset(
            prim, "source", timeCode=timeCode, device=device, required_fields=["velocities"]
        )

    async def exec(self, prim: Usd.Prim, device: str, context: ExecutionContext):
        # dav.config.enable_timing = True
        with dav.scoped_timer("prepare_input"):
            seeds_dataset = await viz_utils.get_selected_dataset(
                prim, "seeds", timeCode=context.timecode, device=device, needs_topology=False
            )
            source_dataset = await self.get_source(prim, context.timecode, device)

        # validate source dataset
        if not dav.utils.is_vector_dtype(source_dataset.get_field("velocities").dtype):
            raise ValueError(
                "Velocities field must be a vector field. Got {source_dataset.get_field('velocities').dtype}"
            )

        streamlines_api = cae_viz.StreamlinesAPI(prim)
        with progress.ProgressContext("Executing DAV [streamlines]"):
            result = dav_streamlines.compute(
                source_dataset,
                "velocities",
                seeds_dataset,
                initial_dt=streamlines_api.GetInitialStepSizeAttr().Get(),
                min_dt=streamlines_api.GetMinStepSizeAttr().Get(),
                max_dt=streamlines_api.GetMaxStepSizeAttr().Get(),
                max_steps=streamlines_api.GetMaxStepsAttr().Get(),
                tolerance=streamlines_api.GetToleranceAttr().Get(),
                direction=streamlines_api.GetDirectionAttr().Get(),
                # threshold=streamlines_api.GetThresholdAttr().Get(),
            )

        if result is None or result.get_num_points() == 0:
            raise usd_utils.QuietableException("No streamlines generated")

        with dav.scoped_timer("probe_fields"):
            result = cae_dav.probe_fields(source_dataset, result, exclude_fields={"velocities"})

        # Populate USD basis curves prim
        prim_rt = UsdGeomRT.BasisCurves(usd_utils.get_prim_rt(prim))
        viz_utils.set_array_attribute(prim_rt.GetPointsAttr(), result.handle.points)
        viz_utils.set_array_attribute(prim_rt.CreateCurveVertexCountsAttr(), result.handle.curve_vertex_counts)

        pvAPI = UsdGeomRT.PrimvarsAPI(prim_rt)
        rnd = np.random.default_rng(1986).random(result.handle.curve_vertex_counts.shape[0], dtype=np.float32)
        pvAPI.CreatePrimvar("rnd", SdfRT.ValueTypeNames.FloatArray, UsdGeomRT.Tokens.uniform).Set(
            VtRT.FloatArray(rnd.reshape(-1, 1))
        )

        # process any auto-scaling for fields specified by CaeVizRescaleRangeAPI instances.
        viz_utils.process_rescale_range_apis(prim, source_dataset)

        # handle primvars
        # exclude velocities field since it's used for computation only and not needed as primvar
        # "times" field is needed for animation, so even though there's no CaeVizFieldSelectionAPI
        # instance for it, we still need to include it in the primvars.
        viz_utils.process_field_selection_apis(prim, result, exclude_fields={"velocities"}, include_fields={"times"})

        # process widths
        viz_utils.process_widths(prim, result, fixed_width=streamlines_api.GetWidthAttr().Get())
        # dav.config.enable_timing = False
        prim_rt.CreateVisibilityAttr().Set(UsdGeomRT.Tokens.inherited)
