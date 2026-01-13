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
from logging import getLogger

import numpy as np
import warp as wp
from omni.cae.data import Range3i
from omni.cae.schema import cae
from pxr import Gf, UsdGeom

# from .voxelization_sph import voxelize_sph
from . import array_utils, progress, settings, usd_utils, voxelization_dense_volume, voxelization_gaussian
from .command_types import (
    ComputeBounds,
    ComputeIJKExtents,
    ConvertToMesh,
    ConvertToPointCloud,
    Mesh,
    PointCloud,
    Voxelize,
)
from .types import IJKExtents

logger = getLogger(__name__)


class CaePointCloudConvertToPointCloud(ConvertToPointCloud):
    """Specialization for datasets that support cae.PointCloudAPI"""

    async def do(self):
        logger.info("executing %s.do()", self.__class__.__name__)
        timeCode = self.timeCode
        assert self.dataset.HasAPI(cae.PointCloudAPI)

        work_units: int = len(self.fields) + 3  # 3 for points

        result = PointCloud()
        with progress.ProgressContext("Reading coordinates", scale=3 / work_units):
            result.points = await usd_utils.get_vecN_from_relationship(
                self.dataset, cae.Tokens.caePointCloudCoordinates, 3, timeCode
            )
        result.fields = {}

        for idx, fieldName in enumerate(self.fields):
            with progress.ProgressContext(
                "Reading field %s" % fieldName, shift=(3 + idx) / work_units, scale=1.0 / work_units
            ):
                fieldPrim = usd_utils.get_target_prim(self.dataset, f"field:{fieldName}")
                assoc = cae.FieldArray(fieldPrim).GetFieldAssociationAttr().Get(timeCode)
                if assoc != cae.Tokens.vertex:
                    raise usd_utils.QuietableException(
                        "Invalid field association (%s) detected for %s" % (assoc, fieldPrim)
                    )
                result.fields[fieldName] = await usd_utils.get_array(fieldPrim, timeCode)
        return result


class CaePointCloudComputeBounds(ComputeBounds):

    async def do(self):
        logger.info("executing %s.do()", self.__class__.__name__)

        assert self.dataset.HasAPI(cae.PointCloudAPI)

        with progress.ProgressContext(scale=0.8):
            points = await usd_utils.get_vecN_from_relationship(
                self.dataset, cae.Tokens.caePointCloudCoordinates, 3, self.timeCode
            )

        with progress.ProgressContext(shift=0.8, scale=0.1):
            min = await asyncio.to_thread(np.min, points, axis=0)  # np.min(points, axis=0)

        with progress.ProgressContext(shift=0.9, scale=0.1):
            max = await asyncio.to_thread(np.max, points, axis=0)  # np.max(points, axis=0)

        return Gf.Range3d(Gf.Vec3d(min.tolist()), Gf.Vec3d(max.tolist()))


class CaeDenseVolumeComputeBounds(ComputeBounds):

    async def do(self):
        logger.info("executing %s.do()", self.__class__.__name__)

        assert self.dataset.HasAPI(cae.DenseVolumeAPI)
        denseVolumeAPI = cae.DenseVolumeAPI(self.dataset)
        ext_min = np.asarray(denseVolumeAPI.GetMinExtentAttr().Get(self.timeCode))
        ext_max = np.asarray(denseVolumeAPI.GetMaxExtentAttr().Get(self.timeCode))
        spacing = np.asarray(denseVolumeAPI.GetSpacingAttr().Get(self.timeCode))

        min = ext_min * spacing
        max = ext_max * spacing
        return Gf.Range3d(Gf.Vec3d(min.tolist()), Gf.Vec3d(max.tolist()))


class CaeDenseVolumeComputeIJKExtents(ComputeIJKExtents):

    async def do(self):
        logger.info("executing %s.do()", self.__class__.__name__)
        assert self.dataset.HasAPI(cae.DenseVolumeAPI)

        denseVolumeAPI = cae.DenseVolumeAPI(self.dataset)
        ext_min = denseVolumeAPI.GetMinExtentAttr().Get(self.timeCode)
        ext_max = denseVolumeAPI.GetMaxExtentAttr().Get(self.timeCode)
        spacing = denseVolumeAPI.GetSpacingAttr().Get(self.timeCode)
        extents = IJKExtents(ext_min, ext_max, spacing)

        if not self.roi.IsEmpty() and not extents.isEmpty():
            roi_ext = IJKExtents.fromBounds(self.roi, spacing)
            if roi_ext.intersect(extents):
                return roi_ext
            else:
                logger.error("ROI does not intersect with data extents. Ignoring ROI.")
        return extents


class OmniCaeDataSetComputeIJKExtents(ComputeIJKExtents):

    async def do(self):
        logger.info("executing %s.do()", self.__class__.__name__)
        assert self.dataset.IsA(cae.DataSet)

        bbox: Gf.Range3d = await ComputeBounds.invoke(self.dataset, self.timeCode)
        if not self.roi.IsEmpty():
            intersection = Gf.Range3d.GetIntersection(bbox, self.roi)
            if intersection.IsEmpty():
                logger.error("ROI does not intersect with data extents. Ignoring ROI.")
            else:
                bbox = intersection

        bbox_min: np.ndarray = np.asarray(bbox.min)
        bbox_max: np.ndarray = np.asarray(bbox.max)
        spacing = (bbox_max - bbox_min) / (np.asarray(self.maxDims) - 1)

        # we use uniform spacing for now
        uniform_spacing = np.amax(spacing)

        ext_min = np.around(bbox_min / uniform_spacing).astype(np.int32, copy=False)
        ext_max = np.around(bbox_max / uniform_spacing).astype(np.int32, copy=False)

        return IJKExtents(Gf.Vec3i(ext_min.tolist()), Gf.Vec3i(ext_max.tolist()), Gf.Vec3d([uniform_spacing] * 3))


class UsdGeomPointBasedConvertToPointCloud(ConvertToPointCloud):
    """Specialization for UsdGeom.Points"""

    def do(self):
        logger.info("executing %s.do()", self.__class__.__name__)
        assert self.dataset.IsA(UsdGeom.PointBased)
        assert not self.fields  # eventually, we may support primvars here

        primT = UsdGeom.PointBased(self.dataset)
        result = PointCloud()
        result.points = np.asarray(primT.GetPointsAttr().Get(self.timeCode))
        return result


class OmniCaeDataSetVoxelizeGaussianWarp(Voxelize):

    async def do(self):
        logger.info("executing %s.do()", self.__class__.__name__)

        with wp.ScopedDevice("cpu"), progress.ProgressContext(scale=0.6):
            result = await ConvertToPointCloud.invoke(self.dataset, self.fields, self.timeCode)

        if len(result.fields) == 3:
            field = array_utils.stack([result.fields[f] for f in self.fields])
        elif len(result.fields) == 1:
            field = result.fields[self.fields[0]]
        else:
            raise usd_utils.QuietableException("Unexpected number of fields %d" % len(result.fields))

        if self.bbox.isEmpty():
            bounds = np.array([np.amin(result.points, axis=0), np.amax(result.points, axis=0)])
            extents = np.around(bounds / self.voxelSize).astype(np.int32, copy=False)
            bbox = Range3i(extents[0], extents[1])
        else:
            bbox = self.bbox

        device = wp.get_cuda_device(self.deviceOrdinal) if self.deviceOrdinal >= 0 else wp.get_device("cpu")
        with wp.ScopedDevice(device), progress.ProgressContext("Voxelizing", shift=0.6, scale=0.4):
            vol: wp.Volume = voxelization_gaussian.voxelize(
                result.points,
                field,
                bbox,
                self.voxelSize,
                radius_factor=settings.get_warp_voxelization_radius_factor(),
                batch_size=settings.get_warp_voxelization_batch_size(),
            )
        return vol


class OmniCaeDataSetVoxelizeFlow(Voxelize):

    async def do(self):
        logger.info("executing %s.do()", self.__class__.__name__)
        with wp.ScopedDevice("cpu"), progress.ProgressContext(scale=0.6):
            result = await ConvertToPointCloud.invoke(self.dataset, self.fields, self.timeCode)

        from omni.flowusd import _flowusd

        iflow = _flowusd.acquire_flowusd_interface()
        iflow.init_persistent_voxelize_context()

        identity_xform = np.array(
            [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64
        )
        velocity = np.zeros((0, 3), dtype=np.float32)
        pressure = np.zeros(0, dtype=np.float32)
        if len(result.fields) == 1:
            pressure = array_utils.as_numpy_array(result.fields[self.fields[0]]).astype(np.float32, copy=False)
            assert pressure.shape[0] == result.points.shape[0]
            if True:  # FIXME: since iflow pressure doesn't work for me, we use velocity
                velocity = np.vstack([pressure, pressure, pressure]).transpose()
                pressure = np.zeros(0, dtype=np.float32)
                output_idx = 2
            else:
                output_idx = -2
        elif len(result.fields) == 3:
            arrays = [result.fields[fname] for fname in self.fields]
            velocity = array_utils.as_numpy_array(array_utils.stack(arrays))
            assert velocity.shape[0] == result.points.shape[0] and velocity.shape[1] == 3
            output_idx = 0

        with progress.ProgressContext("Voxelizing", shift=0.6, scale=0.4):
            output = iflow.voxelize_velocity_points_and_sync_v3(
                array_utils.as_numpy_array(result.points).astype(np.float32, copy=False).ravel(),  # points
                velocity.ravel(),  # velocities
                pressure.ravel(),  # pressure
                identity_xform,  # localToWorld
                identity_xform,  # vdbLocalToWorld
                (
                    self.bbox.numpy().astype(np.float32, copy=False).flatten() if not self.bbox.isEmpty() else []
                ),  # clipAABBs
                self.voxelSize,  # cellSize
                False,  # autoCellSize
                settings.get_flow_voxelization_max_blocks(),  # maxBlocks
                [],  # distances
                False,  # distanceIsColumnMajor
                [],  # distanceDim
                [],  # distanceRanges
                [],  # distanceAABBs
            )
        # print(output, len(output), output[output_idx])
        # iflow.save_nanovdb(output[output_idx], "/tmp/test.nvdb")

        device = wp.get_cuda_device(self.deviceOrdinal) if self.deviceOrdinal >= 0 else wp.get_device("cpu")
        array = wp.array(output[output_idx].view(np.byte), device=device)
        volume = wp.Volume(array)
        iflow.release_persistent_voxelize_context()
        return volume


@wp.kernel
def copy_dense_volume_to_nano_vdb_f(
    volume: wp.uint64, values: wp.array(dtype=wp.float32, ndim=3), ijk_origin: wp.vec3i
):
    i, j, k = wp.tid()
    wp.volume_store_f(volume, i + ijk_origin[0], j + ijk_origin[1], k + ijk_origin[2], values[i, j, k])


class CaeDenseVolumeVoxelize(Voxelize):

    async def do(self):
        denseVolumeAPI = cae.DenseVolumeAPI(self.dataset)

        array = array_utils.as_numpy_array(
            await usd_utils.get_array_from_relationship(self.dataset, f"field:{self.fields[0]}", self.timeCode)
        )
        min_ext = np.array(denseVolumeAPI.GetMinExtentAttr().Get(self.timeCode))
        max_ext = np.array(denseVolumeAPI.GetMaxExtentAttr().Get(self.timeCode))

        if self.bbox.isEmpty():
            min_roi_extent = min_ext
            max_roi_extent = max_ext
        else:
            min_roi_extent = np.asarray(self.bbox.min)
            max_roi_extent = np.asarray(self.bbox.max)

        device = wp.get_cuda_device(self.deviceOrdinal) if self.deviceOrdinal >= 0 else wp.get_device("cpu")
        with wp.ScopedDevice(device):
            vol: wp.Volume = voxelization_dense_volume.voxelize(
                array, (min_ext, max_ext), (min_roi_extent, max_roi_extent)
            )
        return vol


class CaeDenseVolumeVoxelizeFlow(CaeDenseVolumeVoxelize):
    pass


class CaeDenseVolumeVoxelizeGaussianWarp(CaeDenseVolumeVoxelize):
    pass


class CaeMeshConvertToMesh(ConvertToMesh):

    async def do(self):
        logger.info("executing %s.do()", self.__class__.__name__)
        assert self.dataset.HasAPI(cae.MeshAPI)

        nb_fields = len(self.fields)
        total_work_units = nb_fields + 1
        work_fraction = 1 / total_work_units

        with progress.ProgressContext("Reading mesh", scale=work_fraction):
            points = await usd_utils.get_vecN_from_relationship(
                self.dataset, cae.Tokens.caeMeshPoints, 3, self.timeCode
            )
            face_vertex_indices = await usd_utils.get_array_from_relationship(
                self.dataset, cae.Tokens.caeMeshFaceVertexIndices, self.timeCode
            )
            face_vertex_counts = await usd_utils.get_array_from_relationship(
                self.dataset, cae.Tokens.caeMeshFaceVertexCounts, self.timeCode
            )

            points = array_utils.as_numpy_array(points)
            face_vertex_indices = array_utils.as_numpy_array(face_vertex_indices)
            face_vertex_counts = array_utils.as_numpy_array(face_vertex_counts)

        mesh = Mesh()
        mesh.extents = Gf.Range3d(np.min(points, axis=0).tolist(), np.max(points, axis=0).tolist())
        mesh.points = points
        mesh.faceVertexIndices = face_vertex_indices
        mesh.faceVertexCounts = face_vertex_counts

        for idx, fname in enumerate(self.fields):
            with progress.ProgressContext(
                "Reading field %s" % fname, shift=(1 + idx) * work_fraction, scale=work_fraction
            ):
                mesh.fields[fname] = await usd_utils.get_array_from_relationship(
                    self.dataset, f"field:{fname}", self.timeCode
                )

        return mesh


class CaeMeshComputeBounds(ComputeBounds):

    async def do(self):
        logger.info("executing %s.do()", self.__class__.__name__)
        assert self.dataset.HasAPI(cae.MeshAPI)

        with progress.ProgressContext("Reading mesh", scale=0.8):
            points = await usd_utils.get_vecN_from_relationship(
                self.dataset, cae.Tokens.caeMeshPoints, 3, self.timeCode
            )
            points = array_utils.as_numpy_array(points)

        with progress.ProgressContext(shift=0.8, scale=0.1):
            min = await asyncio.to_thread(np.min, points, axis=0)  # np.min(points, axis=0)

        with progress.ProgressContext(shift=0.9, scale=0.1):
            max = await asyncio.to_thread(np.max, points, axis=0)  # np.max(points, axis=0)

        return Gf.Range3d(Gf.Vec3d(min.tolist()), Gf.Vec3d(max.tolist()))
