# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
#  its affiliates is strictly prohibited.

__all__ = ["Streamlines"]

from logging import getLogger

import numpy as np
import warp as wp
from omni.cae.algorithms.core import Algorithm, set_shader_domain, set_shader_input
from omni.cae.data import IJKExtents, array_utils, progress, usd_utils
from omni.cae.data.commands import ComputeIJKExtents, ConvertToPointCloud, Voxelize
from omni.cae.schema import cae
from pxr import Gf, Sdf, Usd, UsdGeom
from usdrt import Sdf as SdfRt
from usdrt import UsdGeom as UsdGeomRt
from usdrt import Vt as VtRt

from .streamline_vdb import advect_vector_field

logger = getLogger(__name__)


class Streamlines(Algorithm):
    """Streamlines algorithm implementation using Warp."""

    _xform_ops = [
        "omni:fabric:localMatrix",
    ]

    def __init__(self, prim: Usd.Prim):
        super().__init__(prim, ["CaeAlgorithmsWarpStreamlinesAPI"])
        self._ns = "omni:cae:warp:streamlines"

        self._rel_tracker = usd_utils.ChangeTracker(self.stage)

        # xform ops
        for p in self._xform_ops:
            self._rel_tracker.TrackAttribute(p)

    def needs_update(self, timeCode: Usd.TimeCode) -> bool:
        if super().needs_update(timeCode):
            return True

        # if seeds were transformed, we need to reexecute.
        seeds_targets = self.prim.GetRelationship(f"{self._ns}:seeds").GetForwardedTargets()
        for t in seeds_targets:
            if self._rel_tracker.PrimChanged(t):
                return True

        return False

    @staticmethod
    def apply_xform(prim: Usd.Prim, coords: np.ndarray) -> np.ndarray:
        _xform_cache = UsdGeom.XformCache(Usd.TimeCode.EarliestTime())
        matrix: Gf.Matrix4d = _xform_cache.GetLocalTransformation(prim)[0]
        if matrix:
            coords_h = np.hstack([coords, np.ones((coords.shape[0], 1))])
            coords_h = coords_h @ matrix
            coords = coords_h[:, :3]  # / coords_h[:, 3]
        return coords

    async def execute(self, timeCode=None, force=True) -> None:
        await super().execute(timeCode, force)
        self._rel_tracker.ClearChanges()

    async def get_seeds(self, timeCode: Usd.TimeCode) -> wp.array:
        seeds_prim = usd_utils.get_target_prim(self.prim, f"{self._ns}:seeds")
        seeds_result = await ConvertToPointCloud.invoke(seeds_prim, [], timeCode)
        seeds = array_utils.as_numpy_array(seeds_result.points).astype(np.float32, copy=False)
        # apply xform
        xformed_seed_pts = Streamlines.apply_xform(seeds_prim, seeds)
        return wp.array(xformed_seed_pts, dtype=wp.vec3f, copy=False)

    async def get_dataset(self, timeCode: Usd.TimeCode) -> tuple[wp.Volume, float]:
        dataset_prim = usd_utils.get_target_prim(self.prim, f"{self._ns}:dataset")
        v_field_prims = usd_utils.get_target_prims(self.prim, f"{self._ns}:velocity")
        maxResolution = usd_utils.get_attribute(self.prim, f"{self._ns}:maxResolution")

        roi_prim: Usd.Prim = usd_utils.get_target_prim(self.prim, f"{self._ns}:roi", quiet=True)
        roi = usd_utils.get_bounds(roi_prim, timeCode, quiet=True)
        logger.info("Using ROI: %s", roi)

        with progress.ProgressContext("Computing ijkExtents", scale=0.1):
            ijk_extents: IJKExtents = await ComputeIJKExtents.invoke(
                dataset_prim, [maxResolution] * 3, roi=roi, timeCode=timeCode
            )
            voxel_size = ijk_extents.spacing[0]

        v_field_names = [usd_utils.get_field_name(dataset_prim, t) for t in v_field_prims]
        if len(v_field_names) != 1 and len(v_field_names) != 3:
            raise usd_utils.QuietableException("Invalid number of velocity fields specified!")

        v_assoc = cae.FieldArray(v_field_prims[0]).GetFieldAssociationAttr().Get()
        if v_assoc != cae.Tokens.vertex and v_assoc != cae.Tokens.cell:
            raise usd_utils.QuietableException("Invalid field association '%s' for velocity" % v_assoc)

        with progress.ProgressContext("Voxelizing", shift=0.1, scale=0.9):
            volume = await Voxelize.invoke(
                dataset_prim,
                v_field_names,
                bbox=ijk_extents.getRange(),
                voxel_size=voxel_size,
                device_ordinal=0,
                timeCode=timeCode,
            )
        assert volume is not None
        return volume, voxel_size

    async def execute_impl(self, timeCode: Usd.TimeCode) -> bool:
        logger.info("start executing streamlines (NanoVDB)")
        dX: float = usd_utils.get_attribute(self.prim, f"{self._ns}:dX")
        maxLength: int = usd_utils.get_attribute(self.prim, f"{self._ns}:maxLength")
        width: float = usd_utils.get_attribute(self.prim, f"{self._ns}:width")
        _ = usd_utils.get_target_prim(self.prim, f"{self._ns}:dataset")
        _ = usd_utils.get_target_prim(self.prim, f"{self._ns}:seeds")
        _ = usd_utils.get_target_prims(self.prim, f"{self._ns}:velocity")

        if dX <= 0.00001:
            raise RuntimeError(f"Invalid dX '{dX}'")
        if maxLength < 10:
            raise RuntimeError(f"Invalid maxLength '{maxLength}'")

        with progress.ProgressContext("Reading seeds", scale=0.1):
            seeds: wp.array = await self.get_seeds(timeCode)
        with progress.ProgressContext("Reading dataset", shift=0.1, scale=0.8):
            dataset, voxel_size = await self.get_dataset(timeCode)
        with progress.ProgressContext("Advecting", shift=0.9, scale=0.1):
            paths, scalars = advect_vector_field(
                initial_points=seeds, vdb=dataset, dt=dX * voxel_size, num_steps=maxLength
            )
        # print(seeds.shape, paths.shape, scalars.shape)

        curves = UsdGeomRt.BasisCurves(self.prim_rt)
        curves.CreatePointsAttr().Set(VtRt.Vec3fArray(paths.numpy()))
        # using np.int32 here fails on Windows builds, so using intc instead
        curves.CreateCurveVertexCountsAttr().Set(
            VtRt.IntArray(np.full(shape=(seeds.shape[0], 1), fill_value=maxLength, dtype=np.intc))
        )
        curves.CreateWidthsAttr().Set([(width)])

        primvarsApi = UsdGeomRt.PrimvarsAPI(self.prim_rt)
        if scalars:
            scalars_np = scalars.numpy()
            scalars_range = (np.amin(scalars_np), np.amax(scalars_np))
            primvarsApi.GetPrimvar("scalar").Set(VtRt.FloatArray(scalars_np.reshape(-1, 1)))

        else:
            primvarsApi.GetPrimvar("scalar").Set(
                VtRt.FloatArray(np.full(paths.shape[0], 0.0, dtype=np.float32).reshape(-1, 1))
            )

        if material := self.get_material("ScalarColor"):
            if scalars:
                await set_shader_domain(material, None, None, timeCode, domain=scalars_range)
                set_shader_input(material, "enable_coloring", Sdf.ValueTypeNames.Bool, True)
            else:
                set_shader_input(material, "enable_coloring", Sdf.ValueTypeNames.Bool, False)
        else:
            logger.warning("ScalarColor material not found")

        logger.info("done executing streamlines")
        return True
