# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
#  its affiliates is strictly prohibited.

__all__ = [
    "ComputeBounds",
    "ComputeIJKExtents",
    "ConvertToMesh",
    "ConvertToPointCloud",
    "GenerateStreamlines",
    "Mesh",
    "PointCloud",
    "Streamlines",
    "Voxelize",
]

from logging import getLogger
from typing import Union

import warp as wp
from omni.cae.schema import cae
from omni.kit.commands import Command
from pxr import Gf, Usd, UsdGeom

from . import array_utils, cache, command_utils, settings, usd_utils
from .types import IJKExtents, Range3i
from .typing import FieldArrayLike

logger = getLogger(__name__)


class PointCloud:
    """A type to present a point cloud."""

    points: FieldArrayLike
    fields: dict[str, FieldArrayLike]

    def __init__(self):
        self.points = None
        self.fields = {}

    def numpy(self) -> "PointCloud":
        result = PointCloud()
        result.points = array_utils.as_numpy_array(self.points)
        result.fields = {k: array_utils.as_numpy_array(v) for k, v in self.fields.items()}
        return result

    def warp(self) -> "PointCloud":
        result = PointCloud()
        result.points = array_utils.as_warp_array(self.points)
        result.fields = {k: array_utils.as_warp_array(v) for k, v in self.fields.items()}
        return result


class ConvertToPointCloud(Command):
    """
    Base class for all CAE commands that can convert a CAE dataset to
    a point cloud representation. When a new schema type in introduced,
    one simply needs to register a command with name
    <SchemaTypeName>ConvertToPointCloud and register it with `omni.kit.commands`.
    """

    def __init__(self, dataset: Usd.Prim, fields: list[str], timeCode: Usd.TimeCode) -> None:
        self._dataset = dataset
        self._fields = fields
        self._timeCode = timeCode

    @property
    def dataset(self) -> Usd.Prim:
        return self._dataset

    @property
    def fields(self) -> list[str]:
        return self._fields

    @property
    def timeCode(self) -> Usd.TimeCode:
        return self._timeCode

    @classmethod
    async def invoke(cls, dataset: Usd.Prim, fields: list[str], timeCode: Usd.TimeCode) -> PointCloud:

        # validate args
        if not dataset.IsA(cae.DataSet) and not dataset.IsA(UsdGeom.PointBased):
            raise ValueError("%s must be a OmniCae.DataSet or UsdGeom.PointBased" % dataset)

        cache_key = {
            "label": "ConvertToPointCloud",
            "dataset": str(dataset.GetPath()),
        }

        cache_state = {}

        result = PointCloud()

        # validate that fields are field relationships on dataset.
        for f in fields:
            if not dataset.HasRelationship(f"field:{f}"):
                raise ValueError("'%s' if not a field name on %s" % (f, dataset))

            # check if field is cached
            key = {"field_name": f}
            key.update(cache_key)
            field = cache.get(str(key), cache_state, timeCode=timeCode)
            if field is not None:
                result.fields[f] = field

        coords_key = {"coordinates": True}
        coords_key.update(cache_key)
        result.points = cache.get(str(coords_key), cache_state, timeCode=timeCode)

        req_fields = list(filter(lambda f: f not in result.fields, fields))

        # check if we have all what we need from cache
        if result.points is not None and len(req_fields) == 0:
            # we have cached points and all requested arrays already, so just return them.
            return result

        result_new = await command_utils.execute(
            cls.__name__, dataset, dataset=dataset, fields=req_fields, timeCode=timeCode
        )
        assert len(result_new.fields) == len(req_fields), "Requested fields not found in dataset"

        # update cache with new fields
        for fname, f in result_new.fields.items():
            key = {"field_name": fname}
            key.update(cache_key)
            cache.put(str(key), f, state=cache_state, sourcePrims=[dataset], timeCode=timeCode)

        # update cache with new coordinates (even if we already have them)
        cache.put(str(coords_key), result_new.points, state=cache_state, sourcePrims=[dataset], timeCode=timeCode)

        # update result
        result.points = result_new.points
        result.fields.update(result_new.fields)
        assert len(result.fields) == len(fields)
        return result


class ComputeBounds(Command):
    """
    Base class for CAE commands to compute approximate bounding box for a dataset.
    When a new schema type in introduced, one simply needs to register a command with name
    <SchemaTypeName>ComputeBounds and register it with `omni.kit.commands`.
    """

    def __init__(self, dataset: Usd.Prim, timeCode: Usd.TimeCode) -> None:
        self._dataset: Usd.Prim = dataset
        self._timeCode: Usd.TimeCode = timeCode

    @property
    def dataset(self) -> Usd.Prim:
        return self._dataset

    @property
    def timeCode(self) -> Usd.TimeCode:
        return self._timeCode

    @classmethod
    async def invoke(cls, dataset: Usd.Prim, timeCode: Usd.TimeCode) -> Gf.Range3d:

        cache_key = {
            "label": "ComputeBounds",
            "dataset": str(dataset.GetPath()),
        }

        cache_state = {}

        cached_result = cache.get(str(cache_key), cache_state, timeCode=timeCode)
        if cached_result is not None:
            return cached_result

        result = await command_utils.execute(cls.__name__, dataset, dataset=dataset, timeCode=timeCode)
        cache.put(str(cache_key), result, state=cache_state, sourcePrims=[dataset], timeCode=timeCode)
        return result


class ComputeIJKExtents(Command):
    """
    Base class for CAE commands to compute structured extents for a prim.

    When a new schema type in introduced, one simply needs to register a command with name
    <SchemaTypeName>ComputeIJKExtents and register it with `omni.kit.commands`.
    """

    def __init__(self, dataset: Usd.Prim, max_dims: Gf.Vec3i, roi: Gf.Range3d, timeCode: Usd.TimeCode):
        self._dataset: Usd.Prim = dataset
        self._max_dims: Gf.Vec3i = Gf.Vec3i(max_dims)
        self._timeCode = timeCode
        self._roi: Gf.Range3d = roi

    @property
    def dataset(self) -> Usd.Prim:
        return self._dataset

    @property
    def maxDims(self) -> Gf.Vec3i:
        return self._max_dims

    @property
    def timeCode(self) -> Usd.TimeCode:
        return self._timeCode

    @property
    def roi(self) -> Gf.Range3d:
        return self._roi

    @classmethod
    async def invoke(
        cls,
        dataset: Usd.Prim,
        max_dims: Gf.Vec3i,
        roi: Gf.Range3d = None,
        timeCode: Usd.TimeCode = Usd.TimeCode.EarliestTime(),
    ) -> IJKExtents:
        if roi is None:
            roi = Gf.Range3d()
        max_dims = Gf.Vec3i(max_dims)
        return await command_utils.execute(
            cls.__name__, dataset, dataset=dataset, max_dims=max_dims, roi=roi, timeCode=timeCode
        )


class Mesh:
    """Type to represent the result of a ConvertToMesh command."""

    extents: Gf.Range3d
    points: FieldArrayLike
    faceVertexIndices: FieldArrayLike
    faceVertexCounts: FieldArrayLike
    normals: Union[None, FieldArrayLike]
    fields: dict[str, FieldArrayLike]

    def __init__(self):
        self.extents = Gf.Range3d()
        self.points = None
        self.faceVertexIndices = None
        self.faceVertexCounts = None
        self.normals = None
        self.fields = {}

    def numpy(self) -> "Mesh":
        result = Mesh()
        result.extents = self.extents
        result.points = array_utils.as_numpy_array(self.points)
        result.faceVertexIndices = array_utils.as_numpy_array(self.faceVertexIndices)
        result.faceVertexCounts = array_utils.as_numpy_array(self.faceVertexCounts)
        result.normals = array_utils.as_numpy_array(self.normals)
        result.fields = {k: array_utils.as_numpy_array(v) for k, v in self.fields.items()}
        return result

    def warp(self) -> "Mesh":
        result = Mesh()
        result.extents = self.extents
        result.points = array_utils.as_warp_array(self.points)
        result.faceVertexIndices = array_utils.as_warp_array(self.faceVertexIndices)
        result.faceVertexCounts = array_utils.as_warp_array(self.faceVertexCounts)
        result.normals = array_utils.as_warp_array(self.normals)
        result.fields = {k: array_utils.as_warp_array(v) for k, v in self.fields.items()}
        return result


class ConvertToMesh(Command):
    """
    Base class for CAE commands to convert a dataset to a surface Mesh.

    When a new schema type in introduced, one simply needs to register a command with name
    <SchemaTypeName>ConvertToMesh and register it with `omni.kit.commands`.
    """

    def __init__(self, dataset: Usd.Prim, fields: list[str], timeCode: Usd.TimeCode):
        self._dataset = dataset
        self._fields = fields
        self._timeCode = timeCode

    @property
    def dataset(self) -> Usd.Prim:
        return self._dataset

    @property
    def fields(self) -> list[str]:
        return self._fields

    @property
    def timeCode(self) -> Usd.TimeCode:
        return self._timeCode

    @classmethod
    async def invoke(cls, dataset: Usd.Prim, fields: list[str], timeCode: Usd.TimeCode) -> Mesh:

        # validate args
        if not dataset.IsA(cae.DataSet):
            raise ValueError("%s must be a OmniCae.DataSet or UsdGeom.PointBased" % dataset)

        # validate that fields are field relationships on dataset.
        for f in fields:
            if not dataset.HasRelationship(f"field:{f}"):
                raise ValueError("%s if not a field name on %s" % (f, dataset))

        cache_key = {
            "label": "ConvertToMesh",
            "dataset": str(dataset.GetPath()),
        }

        cache_state = {
            "fields": str(fields),
        }

        result = cache.get(str(cache_key), cache_state, timeCode=timeCode)
        if result is None:
            result = await command_utils.execute(
                cls.__name__, dataset, dataset=dataset, fields=fields, timeCode=timeCode
            )
            cache.put(str(cache_key), result, state=cache_state, sourcePrims=[dataset], timeCode=timeCode)
        return result


class Voxelize(Command):

    def __init__(
        self,
        dataset: Usd.Prim,
        fields: list[str],
        bbox: Range3i,
        voxel_size: float,
        device_ordinal: int,
        timeCode: Usd.TimeCode,
    ):
        self._dataset: Usd.Prim = dataset
        self._fields: list[str] = fields
        self._bbox: Range3i = bbox
        self._voxel_size: float = voxel_size
        self._device_ordinal: int = device_ordinal
        self._timeCode: Usd.TimeCode = timeCode

    @property
    def dataset(self) -> Usd.Prim:
        return self._dataset

    @property
    def fields(self) -> list[str]:
        return self._fields

    @property
    def bbox(self) -> Range3i:
        return self._bbox

    @property
    def voxelSize(self) -> float:
        return self._voxel_size

    @property
    def deviceOrdinal(self) -> int:
        return self._device_ordinal

    @property
    def timeCode(self) -> Usd.TimeCode:
        return self._timeCode

    @classmethod
    async def invoke(
        cls,
        dataset: Usd.Prim,
        fields: list[str],
        bbox: Union[None, Range3i],
        voxel_size: float,
        device_ordinal: int,
        timeCode: Usd.TimeCode,
    ) -> wp.Volume:

        # validate args
        if not dataset.IsA(cae.DataSet):
            raise ValueError("%s must be a OmniCae.DataSet or UsdGeom.PointBased" % dataset)

        if bbox is None:
            bbox = Range3i.empty()

        # validate that fields are field relationships on dataset.
        field_prims = []
        for f in fields:
            if not dataset.HasRelationship(f"field:{f}"):
                raise ValueError("%s if not a field name on %s" % (f, dataset))
            field_prims += usd_utils.get_target_prims(dataset, f"field:{f}")

        if len(fields) != 1 and len(fields) != 3:
            raise ValueError("Only one 1- or 3-component fields, or three 1-component fields are currently supported")

        cache_key = {"label": "Voxelize", "dataset": str(dataset.GetPath()), "fields": str(fields)}

        # not sure if we should add bbox and voxel_size to the cache_key or cache_state.
        # for now, we only cache 1 voxelization result for the (dataset, field).
        cache_state = {
            "bbox": str(bbox),
            "voxel_size": voxel_size,
            "device_ordinal": device_ordinal,
            "impl": settings.get_voxelization_impl(),
        }

        result = cache.get(str(cache_key), cache_state, timeCode=timeCode)
        if result is None:
            cmd_name = f"{cls.__name__}{settings.get_voxelization_impl()}"
            result = await command_utils.execute(
                cmd_name,
                dataset,
                dataset=dataset,
                fields=fields,
                bbox=bbox,
                voxel_size=voxel_size,
                device_ordinal=device_ordinal,
                timeCode=timeCode,
            )
            cache.put(str(cache_key), result, state=cache_state, sourcePrims=[dataset] + field_prims, timeCode=timeCode)

        return result


_generate_streamlines_impl = None


class Streamlines:
    """
    Type of the result of GenerateStreamlines.
    """

    # extents: Gf.Range3f
    points: FieldArrayLike
    curveVertexCounts: FieldArrayLike
    fields: dict[str, FieldArrayLike]

    def __init__(self):
        self.points = None
        self.curveVertexCounts = None
        self.fields = {}

    def numpy(self) -> "Streamlines":
        s = Streamlines()
        s.points = array_utils.as_numpy_array(self.points)
        s.curveVertexCounts = array_utils.as_numpy_array(self.curveVertexCounts)
        s.fields = {k: array_utils.as_numpy_array(v) for k, v in self.fields.items()}
        return s

    def warp(self) -> "Streamlines":
        s = Streamlines()
        s.points = array_utils.as_warp_array(self.points)
        s.curveVertexCounts = array_utils.as_warp_array(self.curveVertexCounts)
        s.fields = {k: array_utils.as_warp_array(v) for k, v in self.fields.items()}
        return s


class GenerateStreamlines(Command):

    def __init__(
        self,
        dataset: Usd.Prim,
        seeds: FieldArrayLike,
        velocity_fields: list[str],
        color_field: str,
        dX: float,
        maxLength: int,
        timeCode: Usd.TimeCode,
    ):
        self._dataset = dataset
        self._seeds = seeds
        self._velocity_fields = velocity_fields
        self._color_field = color_field
        self._dX = dX
        self._maxLength = maxLength
        self._timeCode: Usd.TimeCode = timeCode

    @property
    def dataset(self) -> Usd.Prim:
        return self._dataset

    @property
    def seeds(self) -> FieldArrayLike:
        return self._seeds

    @property
    def velocity_fields(self) -> list[str]:
        return self._velocity_fields

    @property
    def colorField(self) -> str:
        return self._color_field

    @property
    def dX(self) -> float:
        return self._dX

    @property
    def maxLength(self) -> int:
        return self._maxLength

    @property
    def timeCode(self) -> Usd.TimeCode:
        return self._timeCode

    @classmethod
    def override_impl(cls, name: str):
        global _generate_streamlines_impl
        _generate_streamlines_impl = name
        logger.info("override streamlines implementation with %s", name)

    @classmethod
    async def invoke(
        cls,
        dataset: Usd.Prim,
        seeds: FieldArrayLike,
        velocity_fields: list[str],
        color_field: str,
        dX: float,
        maxLength: int,
        timeCode: Usd.TimeCode,
    ) -> Union[None, Streamlines]:
        global _generate_streamlines_impl

        # validate args
        if not dataset.IsA(cae.DataSet):
            raise ValueError("%s must be a OmniCae.DataSet or UsdGeom.PointBased" % dataset)

        # validate that fields are field relationships on dataset.
        for f in velocity_fields:
            if not dataset.HasRelationship(f"field:{f}"):
                raise ValueError("%s if not a field name on %s" % (f, dataset))

        if len(velocity_fields) != 1 and len(velocity_fields) != 3:
            raise ValueError("Only one 1- or 3-component fields, or three 1-component fields are currently supported")

        if color_field is not None:
            if not dataset.HasRelationship(f"field:{color_field}"):
                raise ValueError("%s if not a field name on %s" % (color_field, dataset))

        impl = settings.get_streamline_impl() if _generate_streamlines_impl is None else _generate_streamlines_impl
        logger.info("Using streamlines implementation %s", impl)
        cmd_name = f"{cls.__name__}{impl}"
        return await command_utils.execute(
            cmd_name,
            dataset,
            dataset=dataset,
            seeds=seeds,
            velocity_fields=velocity_fields,
            color_field=color_field,
            dX=dX,
            maxLength=maxLength,
            timeCode=timeCode,
        )
