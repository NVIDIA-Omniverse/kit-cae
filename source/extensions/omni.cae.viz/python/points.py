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
import warp as wp
from dav.data_models.custom import point_cloud as dav_point_cloud
from dav.operators import point_cell_counts as dav_point_cell_counts
from dav.operators import point_field as dav_point_field
from dav.operators import point_splats as dav_point_splats
from omni.cae.data import array_utils, cache, progress, usd_utils
from omni.cae.schema import viz as cae_viz
from pxr import Usd
from usdrt import UsdGeom as UsdGeomRT
from usdrt import Vt as VtRT

from . import utils as viz_utils
from .execution_context import ExecutionContext
from .operator import operator

logger = getLogger(__name__)


def _subset(dataset: dav.Dataset, max_points: int, chosen_point_idxs: np.ndarray = None) -> dav.Dataset:
    # TODO: this can be optimized
    nb_points_in = dataset.get_num_points() if chosen_point_idxs is None else chosen_point_idxs.shape[0]
    if max_points > 0 and nb_points_in > max_points:
        logger.warning(f"Subsetting dataset to {max_points} points")
        stride = int(np.ceil(nb_points_in / max_points))
        if chosen_point_idxs is not None:
            mask = chosen_point_idxs[::stride]
        else:
            mask = np.s_[::stride]
    elif chosen_point_idxs is not None:
        mask = chosen_point_idxs
    else:
        # nothing to mask out; just return the dataset
        return dataset

    assert dataset.data_model == dav_point_cloud.DataModel
    pc_handle = dataset.handle
    pc_handle.points = wp.array(pc_handle.points.numpy()[mask], dtype=wp.vec3f, device="cpu")
    for field_name in dataset.get_field_names():
        field = dataset.get_field(field_name)
        if field.association == dav.AssociationType.CELL:
            logger.warning(f"Skipping cell-centered field: {field_name}")
            continue
        array = array_utils.as_warp_array(cae_dav.fetch_data(dataset, field_name)).numpy()
        array = array[mask]
        array = wp.array(array, dtype=wp.float32, device="cpu").to(dataset.device)
        dataset.add_field(field_name, dav.Field.from_array(array, dav.AssociationType.VERTEX), warn_if_exists=False)
    return dataset


def _compute_points(source_dataset: dav.Dataset, max_points: int, use_cell_points: bool) -> dav.Dataset:
    with progress.ProgressContext("Executing DAV [point_splats]"):
        points_dataset = dav_point_splats.compute(source_dataset, radius=0.0, sharpness=1.0)

    for field_name in source_dataset.get_field_names():
        # pass fields, converting cell-to-point data, if needed
        with progress.ProgressContext("Executing DAV [point_field]"):
            tmp_dataset = dav_point_field.compute(source_dataset, field_name, output_field_name="_point_field")
        points_dataset.add_field(field_name, tmp_dataset.get_field("_point_field"))

    if not use_cell_points:
        return _subset(points_dataset, max_points)

    with progress.ProgressContext("Executing DAV [point_cell_counts]"):
        point_cell_counts_ds = dav_point_cell_counts.compute(source_dataset, field_name="counts")
    countds_field = point_cell_counts_ds.get_field("counts")
    range = countds_field.get_range()
    if range[0] > 0:
        # all points are "chosen"
        return _subset(points_dataset, max_points)
    else:
        point_cell_counts_np = wp.array(cae_dav.fetch_data(point_cell_counts_ds, "counts"), copy=False).numpy()

        # get indices for points with non-zero cell counts
        chosen_point_idxs = np.where(point_cell_counts_np > 0)[0]
        return _subset(points_dataset, max_points, chosen_point_idxs)


@operator(supports_temporal=True, tick_on_time_change=True)
class Points:
    prim_type: str = "Points"
    api_schemas: set[str] = {
        "CaeVizPointsAPI",
        "CaeVizDatasetSelectionAPI:source",
        "CaeVizFieldSelectionAPI:colors",
        "CaeVizFieldSelectionAPI:widths",
    }

    optional_api_schemas: set[str] = {
        "CaeVizFieldSelectionAPI",
        "CaeVizFieldMappingAPI",
    }

    async def exec(self, prim: Usd.Prim, device: str, context: ExecutionContext):
        source_dataset = await viz_utils.get_input_dataset(
            prim, "source", timeCode=context.timecode, device=device, needs_topology=False
        )

        points_api = cae_viz.PointsAPI(prim)
        width = points_api.GetWidthAttr().Get()
        max_points = points_api.GetMaxCountAttr().Get()
        use_cell_points = points_api.GetUseCellPointsAttr().Get()

        if source_dataset.get_num_cells() == 0 and use_cell_points:
            logger.error(
                f"Dataset has no cells, but use_cell_points is True. Try setting {points_api.GetUseCellPointsAttr().GetPath()} to False."
            )

        points_dataset = _compute_points(source_dataset, max_points, use_cell_points)

        # Process rescale range APIs which depend on input dataset.
        viz_utils.process_rescale_range_apis(prim, source_dataset)

        cache_key = f"omni.cae.viz.points.Points:{prim.GetPath()}"
        if context.is_full_rebuild_needed():
            cache.remove(cache_key)
            cache.put(cache_key, points_dataset, timeCode=Usd.TimeCode.EarliestTime(), consumerPrims=[prim], force=True)
        cache.put(cache_key, points_dataset, timeCode=context.timecode, consumerPrims=[prim], force=True)
        await self.populate_points(prim, points_dataset, width, context)

    async def on_time_changed(self, prim: Usd.Prim, device: str, context: ExecutionContext):
        cache_key = f"omni.cae.viz.points.Points:{prim.GetPath()}"
        points_dataset = cache.get(cache_key, timeCode=context.timecode)
        if points_dataset is None:
            return
        if context.next_time_code and context.next_time_code != context.timecode:
            next_points_dataset = cache.get(cache_key, timeCode=context.next_time_code)
            if next_points_dataset is None:
                return
            factor = (context.raw_timecode.GetValue() - context.timecode.GetValue()) / (
                context.next_time_code.GetValue() - context.timecode.GetValue()
            )
            points_dataset = cae_dav.lerp_dataset(
                points_dataset,
                next_points_dataset,
                factor,
                fields=usd_utils.get_instances(prim, "CaeVizFieldSelectionAPI"),
            )

        points_api = cae_viz.PointsAPI(prim)
        width = points_api.GetWidthAttr().Get()
        await self.populate_points(prim, points_dataset, width, context)

    async def populate_points(
        self, prim: Usd.Prim, points_dataset: dav.Dataset, width: float, context: ExecutionContext
    ):
        prim_rt = UsdGeomRT.Points(usd_utils.get_prim_rt(prim))
        viz_utils.set_array_attribute(prim_rt.CreatePointsAttr(), points_dataset.handle.points)

        viz_utils.process_field_selection_apis(prim, points_dataset)
        viz_utils.process_widths(prim, points_dataset, fixed_width=width)


@operator(supports_temporal=True, tick_on_time_change=True)
class Glyphs:
    prim_type: str = "PointInstancer"
    api_schemas: set[str] = {
        "CaeVizGlyphsAPI",
        "CaeVizDatasetSelectionAPI:source",
    }

    optional_api_schemas: set[str] = {
        "CaeVizFieldSelectionAPI",
        "CaeVizFieldMappingAPI",
    }

    async def exec(self, prim: Usd.Prim, device: str, context: ExecutionContext):
        source_dataset = await viz_utils.get_input_dataset(
            prim, "source", timeCode=context.timecode, device=device, needs_topology=False
        )

        glyphs_api = cae_viz.GlyphsAPI(prim)
        max_points = glyphs_api.GetMaxCountAttr().Get(context.timecode)
        orientations_mode = glyphs_api.GetOrientationsModeAttr().Get(context.timecode)
        use_cell_points = glyphs_api.GetUseCellPointsAttr().Get(context.timecode)
        default_scale = glyphs_api.GetScaleAttr().Get(context.timecode)

        if source_dataset.get_num_cells() == 0 and use_cell_points:
            logger.error(
                f"Dataset has no cells, but use_cell_points is True. Try setting {glyphs_api.GetUseCellPointsAttr().GetPath()} to False."
            )

        points_dataset = _compute_points(source_dataset, max_points, use_cell_points)

        # Process rescale range APIs which depend on input dataset.
        viz_utils.process_rescale_range_apis(prim, source_dataset)

        cache_key = f"omni.cae.viz.points.Glyphs:{prim.GetPath()}"
        if context.is_full_rebuild_needed():
            cache.remove(cache_key)
            cache.put(cache_key, points_dataset, timeCode=Usd.TimeCode.EarliestTime(), consumerPrims=[prim], force=True)
        cache.put(cache_key, points_dataset, timeCode=context.timecode, consumerPrims=[prim], force=True)
        await self.populate_glyphs(prim, points_dataset, orientations_mode, default_scale, context)

    async def on_time_changed(self, prim: Usd.Prim, device: str, context: ExecutionContext):
        cache_key = f"omni.cae.viz.points.Glyphs:{prim.GetPath()}"
        points_dataset = cache.get(cache_key, timeCode=context.timecode)
        if points_dataset is None:
            return
        if context.next_time_code and context.next_time_code != context.timecode:
            next_points_dataset = cache.get(cache_key, timeCode=context.next_time_code)
            if next_points_dataset is None:
                return
            factor = (context.raw_timecode.GetValue() - context.timecode.GetValue()) / (
                context.next_time_code.GetValue() - context.timecode.GetValue()
            )
            points_dataset = cae_dav.lerp_dataset(
                points_dataset,
                next_points_dataset,
                factor,
                fields=usd_utils.get_instances(prim, "CaeVizFieldSelectionAPI"),
            )

        glyphs_api = cae_viz.GlyphsAPI(prim)
        orientations_mode = glyphs_api.GetOrientationsModeAttr().Get(context.timecode)
        default_scale = glyphs_api.GetScaleAttr().Get(context.timecode)
        await self.populate_glyphs(prim, points_dataset, orientations_mode, default_scale, context)

    async def populate_glyphs(
        self,
        prim: Usd.Prim,
        points_dataset: dav.Dataset,
        orientations_mode: str,
        default_scale: float,
        context: ExecutionContext,
    ):
        prim_rt = UsdGeomRT.PointInstancer(usd_utils.get_prim_rt(prim))
        viz_utils.set_array_attribute(prim_rt.CreatePositionsAttr(), points_dataset.handle.points)
        viz_utils.set_array_attribute(
            prim_rt.CreateProtoIndicesAttr(), np.zeros([points_dataset.get_num_points(), 1], dtype=np.int32)
        )

        if points_dataset.has_field("scales"):
            f_scales = cae_dav.fetch_data(points_dataset, "scales")

            if f_scales.ndim > 2 or (f_scales.ndim == 2 and f_scales.shape[1] not in [1, 3]):
                raise ValueError(f"Invalid scales shape {f_scales.shape}")

            # Apply field mapping if present
            f_scales = viz_utils.apply_field_mapping(prim, "scales", f_scales)

            if f_scales.ndim == 1 or f_scales.shape[1] == 1:
                # scales is a scalar field, convert it to a vec3f field since the "scales" primvar expects a vec3f array
                wp_array = wp.zeros(shape=f_scales.shape[0], dtype=wp.vec3f, device=array_utils.get_device(f_scales))
                wp.map(
                    lambda x: wp.vec3f(wp.float32(x)),
                    wp.array(f_scales, copy=False, device=wp_array.device),
                    out=wp_array,
                )
                f_scales = array_utils.IFieldArray.from_array(wp_array)

            viz_utils.set_array_attribute(prim_rt.CreateScalesAttr(), f_scales)
        else:
            np_scales = np.full([points_dataset.get_num_points(), 3], default_scale, dtype=np.float32)
            viz_utils.set_array_attribute(prim_rt.CreateScalesAttr(), np_scales)

        if points_dataset.has_field("orientations"):
            f_orientations = cae_dav.fetch_data(points_dataset, "orientations")
            if orientations_mode == "quaternion":
                if f_orientations.ndim != 2 or f_orientations.shape[1] != 4:
                    raise ValueError(f"Invalid orientations shape {f_orientations.shape}")
                quaternions = array_utils.as_warp_array(f_orientations).numpy()
                viz_utils.set_array_attribute(prim_rt.CreateOrientationsAttr(), quaternions)
            elif orientations_mode == "eulerAngles":
                if f_orientations.ndim != 2 or f_orientations.shape[1] != 3:
                    raise ValueError(f"Invalid orientations shape {f_orientations.shape}")
                # TODO: compute quaternions in warp
                quaternions = array_utils.compute_quaternions_from_directions(f_orientations)
                viz_utils.set_array_attribute(prim_rt.CreateOrientationsAttr(), quaternions)
            else:
                raise ValueError(f"Invalid orientations mode {orientations_mode}")
        else:
            prim_rt.CreateOrientationsAttr().Set([])

        viz_utils.process_field_selection_apis(prim, points_dataset, exclude_fields={"scales", "orientations"})
