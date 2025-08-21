# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
#  its affiliates is strictly prohibited.

from logging import getLogger

import numpy as np
import warp as wp
from omni.cae.data import array_utils, cache, progress, usd_utils
from omni.cae.data.commands import ComputeBounds, ConvertToPointCloud, GenerateStreamlines, PointCloud, Streamlines
from omni.cae.schema import cae, sids
from pxr import Gf

from .. import sids_unstructured, types
from .streamline_elem_vol import advect_vector_field

logger = getLogger(__name__)


class CaeSidsUnstructuredConvertToPointCloud(ConvertToPointCloud):

    async def do(self):
        logger.info("executing %s.do()", self.__class__.__name__)

        timeCode = self.timeCode

        if not self.dataset.HasAPI(sids.UnstructuredAPI):
            raise usd_utils.QuietableException("Dataset (%s) does not support sids.UnstructuredAPI!" % self.dataset)

        cell_field_prims = {}
        point_field_prims = {}

        for fieldName in self.fields:
            fieldPrim = usd_utils.get_target_prim(self.dataset, f"field:{fieldName}")
            if not fieldPrim.IsA(cae.FieldArray):
                raise usd_utils.QuietableException("Invalid field prim " % fieldPrim)

            assc = usd_utils.get_attribute(fieldPrim, cae.Tokens.fieldAssociation, timeCode)
            if assc == cae.Tokens.cell:
                cell_field_prims[fieldName] = fieldPrim
            elif assc == cae.Tokens.vertex:
                point_field_prims[fieldName] = fieldPrim
            else:
                raise usd_utils.QuietableException("Invalid field association (%s) detected for %s" % (assc, fieldPrim))

        result = PointCloud()
        with wp.ScopedDevice("cpu"):
            eType = sids_unstructured.get_element_type(self.dataset)
            if eType == types.ElementType.NFACE_n:
                # For nface-n, we don't bother extracting chosen points since they generally cover all points
                result.points = await sids_unstructured.get_original_grid_coordinates(self.dataset, timeCode)
                ids = None
            else:
                ids = await sids_unstructured.get_grid_coordinates_ids(self.dataset, timeCode)
                result.points = await sids_unstructured.get_grid_coordinates(self.dataset, timeCode, ids=ids)

            # read point fields
            for fname, prim in point_field_prims.items():
                array = await usd_utils.get_array(prim, timeCode)
                if ids is not None:
                    result.fields[fname] = array_utils.lookup_index_1(array, ids)
                else:
                    result.fields[fname] = array

            # process cell fields
            if eType == types.ElementType.NGON_n and len(cell_field_prims) > 0:
                logger.error("Cell data extraction for NGON_n is not supported.")
                raise usd_utils.QuietableException("Cell data extraction for NGON_n is not supported.")

            cell_fields = await sids_unstructured.get_cell_field_as_point(
                self.dataset, cell_field_prims.values(), None, timeCode
            )
            for fname, array in zip(cell_field_prims.keys(), cell_fields):
                if ids is not None:
                    result.fields[fname] = array_utils.lookup_index_1(array, ids)
                else:
                    result.fields[fname] = array

        return result


class CaeSidsUnstructuredComputeBounds(ComputeBounds):

    async def do(self):
        logger.info("executing %s.do()", self.__class__.__name__)

        dataset = self.dataset

        if not dataset.HasAPI(sids.UnstructuredAPI):
            raise usd_utils.QuietableException("Dataset (%s) does not support sids.UnstructuredAPI!" % dataset)

        with wp.ScopedDevice("cpu"), progress.ProgressContext(scale=0.8):
            eType = sids_unstructured.get_element_type(dataset)
            if eType == types.ElementType.NFACE_n:
                logger.info("calling get_vecN_from_relationship for NFACE_n")
                coords = await usd_utils.get_vecN_from_relationship(
                    dataset, sids.Tokens.caeSidsGridCoordinates, 3, self.timeCode
                )
            else:
                logger.info("calling get_grid_coordinates")
                coords = await sids_unstructured.get_grid_coordinates(dataset, self.timeCode)

        with progress.ProgressContext("Computing Bounding Box", shift=0.8, scale=0.2):
            logger.info("now, computing bbox")
            bbox = Gf.Range3d(
                Gf.Vec3d(np.min(coords.numpy(), axis=0).tolist()), Gf.Vec3d(np.max(coords.numpy(), axis=0).tolist())
            )
            logger.info("bbox=%s", bbox)

        return bbox


# Dataset object
class CaeSidsUnstructuredDataset:
    def __init__(self, elementType, elementConnectivity, elementVertices, fields):
        self._elementType = elementType
        self._elementConnectivity = elementConnectivity
        self._elementVertices = elementVertices
        self._fields = fields

    @property
    def elementType(self):
        return self._elementType

    @property
    def elementConnectivity(self):
        return self._elementConnectivity

    @property
    def elementVertices(self):
        return self._elementVertices

    @property
    def fields(self):
        return self._fields


class CaeSidsUnstructuredGenerateStreamlinesWarp(GenerateStreamlines):

    async def do(self) -> Streamlines:
        logger.info("executing %s.do()", self.__class__.__name__)

        if not self.dataset.HasAPI(sids.UnstructuredAPI):
            raise usd_utils.QuietableException("Dataset (%s) does not support sids.UnstructuredAPI!" % self.dataset)

        dataset = await self.get_dataset()
        fieldX, fieldY, fieldZ = dataset.fields
        elemType, mesh_idx, mesh_vertices = dataset.elementType, dataset.elementConnectivity, dataset.elementVertices
        num_steps = self.maxLength
        streamlines_points, streamlines_scalars, streamlines_time = advect_vector_field(
            mesh_vertices, mesh_idx, elemType, (fieldX, fieldY, fieldZ), self.seeds, dt=self.dX, num_steps=num_steps
        )

        if streamlines_points is None:
            return None

        result = Streamlines()
        result.points = streamlines_points

        # note: paths include forward and backward steps, so number of steps is doubled (minus 1 for duplicated starting
        # point)
        result.curveVertexCounts = (2 * num_steps - 1) * np.ones(len(self.seeds))
        result.fields["scalar"] = streamlines_scalars
        result.fields["time"] = streamlines_time
        return result

    async def get_dataset(self) -> CaeSidsUnstructuredDataset:
        logger.info("Getting dataset for warp fem streamlines")
        fields = []
        fields += self.velocity_fields
        fields += [self.colorField] if self.colorField is not None else []

        cache_key = {
            "label": "CaeSidsUnstructuredGenerateStreamlinesWarp",
            "dataset": str(self.dataset.GetPath()),
            "fields": str(fields),
        }

        cache_state = {
            "timeCode": str(self.timeCode.GetValue()),
        }

        if dataset := cache.get(str(cache_key), cache_state):
            return dataset

        mesh_vertices = await sids_unstructured.get_original_grid_coordinates(self.dataset, self.timeCode)

        association = None
        field_arrays = []
        for fieldName in self.velocity_fields:
            fieldPrim = usd_utils.get_target_prim(self.dataset, f"field:{fieldName}")
            if not fieldPrim.IsA(cae.FieldArray):
                raise usd_utils.QuietableException("Invalid field prim " % fieldPrim)
            association = usd_utils.get_attribute(fieldPrim, cae.Tokens.fieldAssociation, self.timeCode)

            # Check data association type: Vertex or Cell
            if association is None or association == cae.Tokens.vertex:
                fieldData = await usd_utils.get_arrays_from_relationship(self.dataset, f"field:{fieldName}")
                field_arrays.append(wp.array(fieldData[0], dtype=wp.float32, copy=False))
            elif association == cae.Tokens.cell:
                with wp.ScopedDevice("cpu"):
                    vertex_field = await sids_unstructured.get_cell_field_as_point(
                        self.dataset, [fieldPrim], None, self.timeCode
                    )
                    field_arrays.append(vertex_field[0].to("cuda"))

        elements = await sids_unstructured.get_section(self.dataset, self.timeCode)
        dataset = CaeSidsUnstructuredDataset(
            elements.elementType,
            elements.elementConnectivity,
            mesh_vertices,
            (field_arrays[0], field_arrays[1], field_arrays[2]),
        )
        cache.put(str(cache_key), dataset, state=cache_state, sourcePrims=[self.dataset])
        return dataset
