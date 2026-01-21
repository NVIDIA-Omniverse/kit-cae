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
from dav.data_models.sids import nface_n as dav_sids_nface_n
from dav.data_models.sids import unstructured as dav_sids_unstructured
from dav.operators import probe as dav_probe
from dav.operators import streamlines as dav_streamlines
from dav.operators import voxelization as dav_voxelization
from omni.cae.data import Range3i, array_utils, cache, commands, progress, usd_utils
from omni.cae.data.commands import GenerateStreamlines, Streamlines, Voxelize
from omni.cae.schema import cae, sids
from omni.kit.commands import Command
from pxr import Usd

logger = getLogger(__name__)


class ConvertToDAVDataSet(Command):
    """
    Command to convert a USD Prim representing a CAE DataSet into a dav.DataSet.
    When introducing any new data model, one must provide a way to register the
    data model with DAV so that it can be processed by the algorithms provided by DAV.
    """

    def __init__(self, dataset: Usd.Prim, fields: list[str], device: str, timeCode: Usd.TimeCode) -> None:
        self._dataset = dataset
        self._fields = fields
        self._device = device
        self._timeCode = timeCode

    @property
    def dataset(self) -> Usd.Prim:
        """The CAE dataset prim to convert."""
        return self._dataset

    @property
    def fields(self) -> list[str]:
        """List of field names to include in the DAV dataset."""
        return self._fields

    @property
    def device(self) -> str:
        """Device to use for DAV processing (e.g., 'cpu', 'cuda', 'gpu')."""
        return self._device

    @property
    def timeCode(self) -> Usd.TimeCode:
        """Time code for data retrieval."""
        return self._timeCode

    @classmethod
    async def invoke(cls, dataset: Usd.Prim, fields: list[str], device: str, timeCode: Usd.TimeCode) -> dav.Dataset:
        """
        Convert a CAE dataset to a DAV DataSet.

        Args:
            dataset: The CAE dataset prim to convert
            fields: List of field names to include
            device: Device to use for DAV processing (e.g., 'cpu', 'cuda', 'gpu')
            timeCode: Time code for data retrieval

        Returns:
            A dav.DataSet object
        """

        # validate args
        if not dataset.IsA(cae.DataSet):
            raise ValueError("%s must be a OmniCae.DataSet" % dataset)

        # validate that fields are field relationships on dataset.
        field_prims = []
        for f in fields:
            if not dataset.HasRelationship(f"field:{f}"):
                raise ValueError("%s is not a field name on %s" % (f, dataset))
            field_prims += usd_utils.get_target_prims(dataset, f"field:{f}")

        cache_key = {
            "label": "ConvertToDAVDataSet",
            "dataset": str(dataset.GetPath()),
            "fields": str(fields),
            "device": device,
        }

        cache_state = {}
        dav_dataset = cache.get(str(cache_key), cache_state, timeCode=timeCode)
        if dav_dataset is None:
            dav_dataset = await commands.execute(
                cls.__name__, dataset, dataset=dataset, fields=fields, device=device, timeCode=timeCode
            )
            if dav_dataset:
                cache.put(
                    str(cache_key),
                    dav_dataset,
                    state=cache_state,
                    sourcePrims=[dataset] + field_prims,
                    timeCode=timeCode,
                )
        return dav_dataset

    async def do(self) -> dav.Dataset:
        """
        Execute the command to convert a CAE dataset to a DAV DataSet.

        This is a base implementation that should be overridden by subclasses
        for specific dataset types.

        Returns:
            A dav.DataSet object
        """
        raise NotImplementedError(
            f"Conversion to DAV DataSet not implemented for dataset type: {self.dataset.GetTypeName()}. "
            f"Please implement a subclass of ConvertToDAVDataSet for this dataset type."
        )


class CaeSidsUnstructuredConvertToDAVDataSet(ConvertToDAVDataSet):
    """
    Convert a SIDS unstructured dataset to a DAV DataSet.
    """

    async def do(self) -> dav.Dataset:
        """
        Execute the command to convert a SIDS unstructured dataset to a DAV DataSet.

        Returns:
            A dav.Dataset object with SIDS unstructured data model
        """
        logger.info("executing %s.do()", self.__class__.__name__)

        if not self.dataset.HasAPI(sids.UnstructuredAPI):
            raise usd_utils.QuietableException("Dataset (%s) does not support sids.UnstructuredAPI!" % self.dataset)

        device = self.device
        timeCode = self.timeCode

        # Get mesh vertices and elements
        from omni.cae.sids import sids_unstructured, types

        mesh_vertices = (await sids_unstructured.get_original_grid_coordinates(self.dataset, timeCode)).to(device)

        element_type = sids_unstructured.get_element_type(self.dataset)

        if element_type == types.ElementType.NGON_n:
            raise usd_utils.QuietableException("Conversion of NGON_n elements to DAV DataSet is not supported.")
        elif element_type == types.ElementType.NFACE_n:
            dataset = await self.nface_n_to_dav_dataset(self.dataset, mesh_vertices)
        else:
            dataset = await self.standard_to_dav_dataset(self.dataset, mesh_vertices)

        # Add fields
        for field_name in self.fields:
            field_prim = usd_utils.get_target_prim(self.dataset, f"field:{field_name}")
            field_array = await usd_utils.get_array(field_prim, timeCode)
            assoc = usd_utils.get_target_field_association(self.dataset, f"field:{field_name}")

            if assoc not in [cae.Tokens.vertex, cae.Tokens.cell]:
                logger.error("Unsupported field association (%s) for field %s", assoc, field_name)
                continue

            dav_assoc = dav.AssociationType.VERTEX if assoc == cae.Tokens.vertex else dav.AssociationType.CELL
            field_wp = wp.array(field_array, dtype=array_utils.to_warp_dtype(field_array), copy=False, device=device)
            dataset.add_field(field_name, dav.Field.from_array(field_wp, association=dav_assoc))

        return dataset

    async def standard_to_dav_dataset(self, prim, mesh_vertices) -> dav.DatasetLike:
        from omni.cae.sids import sids_unstructured

        device = self.device
        timeCode = self.timeCode

        elements = await sids_unstructured.get_section(prim, timeCode)

        # Build DAV DataSet_t
        dataset_t = dav_sids_unstructured.DatasetHandle()

        # dataset_t.hex_is_axis_aligned = True  # Add schema attr to make it possible to specify this
        dataset_t.element_type = elements.elementType
        dataset_t.element_range = elements.elementRange
        dataset_t.element_connectivity = elements.elementConnectivity.to(device)
        if elements.elementStartOffset is not None:
            dataset_t.element_start_offset = wp.array(
                elements.elementStartOffset.numpy().astype(np.int32), dtype=wp.int32, device=device
            )
        dataset_t.grid_coords = mesh_vertices

        # Create DAV dataset
        return dav.Dataset(dav_sids_unstructured.DataModel, dataset_t, device)

    async def nface_n_to_dav_dataset(self, nface_prim, mesh_vertices) -> dav.DatasetLike:
        from omni.cae.sids import sids_unstructured

        device = self.device
        timeCode = self.timeCode

        # process all ngon_n blocks
        ngon_prims = usd_utils.get_target_prims(nface_prim, sids.Tokens.caeSidsNgons)
        ngon_dav_datasets = []
        assert len(ngon_prims) > 0, "No ngon blocks found for nface_n dataset."
        for ngon_prim in ngon_prims:
            ngon_dav_dataset = await self.standard_to_dav_dataset(ngon_prim, mesh_vertices)
            ngon_dav_datasets.append(ngon_dav_dataset)

        ngon_dav_datasets = sorted(ngon_dav_datasets, key=lambda ds: ds.handle.element_range[0])

        nface_elements = await sids_unstructured.get_section(self.dataset, timeCode)

        # Build DAV DataSet_t
        dataset_t = dav_sids_nface_n.DatasetHandle()
        dataset_t.nface_n_block.element_type = nface_elements.elementType
        dataset_t.nface_n_block.element_range = nface_elements.elementRange
        dataset_t.nface_n_block.element_connectivity = nface_elements.elementConnectivity.to(device)
        dataset_t.nface_n_block.element_start_offset = wp.array(
            nface_elements.elementStartOffset.numpy().astype(np.int32), dtype=wp.int32, device=device
        )
        dataset_t.nface_n_block.grid_coords = mesh_vertices
        dataset_t.ngon_n_blocks = wp.array(
            [ngon_ds.handle for ngon_ds in ngon_dav_datasets], dtype=dav_sids_unstructured.DatasetHandle, device=device
        )
        dataset_t.ngon_n_element_range_starts = wp.array(
            [ngon_ds.handle.element_range[0] for ngon_ds in ngon_dav_datasets],
            dtype=wp.int32,
            device=device,
        )

        # Create DAV dataset
        return dav.Dataset(
            data_model=dav_sids_nface_n.DataModel,
            handle=dataset_t,
            device=device,
            ngon_dataset=tuple(ngon_dav_datasets),
        )


# Helper code for streamlines termination condition
@wp.struct
class TerminationConditionHandle:
    blocked_cells: wp.array(dtype=wp.int32)

    # hex faces encoded as: (-i,-j,-k,+i,+j,+k) -> (0,1,2,3,4,5)
    blocked_faces: wp.array(dtype=wp.int32)


@dav.cached
def get_tc_model(data_model):
    """Create a termination condition model for streamline computation.

    This function generates a termination condition model that checks if a streamline
    should terminate based on blocked cells. When a particle exits a blocked cell,
    the streamline terminates.

    Args:
        data_model: The data model used by the dataset

    Returns:
        TerminationConditionModel class with terminate method
    """

    class TerminationConditionModel:
        @staticmethod
        @wp.func
        def _get_face_direction_vector(face_id: int) -> wp.vec3f:
            """Get the normal vector for a given face ID."""
            # hex faces encoded as: (-i,-j,-k,+i,+j,+k) -> (0,1,2,3,4,5)
            if face_id == 0:
                return wp.vec3f(-1.0, 0.0, 0.0)
            elif face_id == 1:
                return wp.vec3f(0.0, -1.0, 0.0)
            elif face_id == 2:
                return wp.vec3f(0.0, 0.0, -1.0)
            elif face_id == 3:
                return wp.vec3f(1.0, 0.0, 0.0)
            elif face_id == 4:
                return wp.vec3f(0.0, 1.0, 0.0)
            elif face_id == 5:
                return wp.vec3f(0.0, 0.0, 1.0)

        @staticmethod
        @wp.func
        def terminate(
            tc_handle: TerminationConditionHandle,
            cursor: Any,
            next: Any,
            dt_used: float,
            v_used: wp.vec3f,
            ds: data_model.DatasetHandle,
        ) -> wp.bool:
            """Check if streamline should terminate at this step.

            Terminates if the current cell is in the blocked cells list.
            """
            cur_cell_id = data_model.DatasetAPI.get_cell_id_from_idx(ds, cursor.cell_idx)
            next_cell_id = data_model.DatasetAPI.get_cell_id_from_idx(ds, next.cell_idx)
            if cur_cell_id == next_cell_id:
                return False

            # If exiting a cell, check if we are exiting through a blocked face
            # For now, we terminate if exiting from any blocked cell
            end = tc_handle.blocked_cells.shape[0]
            idx = wp.lower_bound(tc_handle.blocked_cells, 0, end, cur_cell_id)
            if idx < end and tc_handle.blocked_cells[idx] == cur_cell_id:
                # determine which face we are exiting through;
                direction = wp.normalize(next.position - cursor.position)
                face_dir = TerminationConditionModel._get_face_direction_vector(tc_handle.blocked_faces[idx])
                dot = wp.dot(direction, face_dir)
                if dot > 0.0:
                    return True
            return False

    return TerminationConditionModel


class OmniCaeDataSetGenerateStreamlinesDAV(GenerateStreamlines):
    """
    Generate streamlines using DAV algorithms.
    """

    # FIXME:
    device: str = "cuda:0"

    async def do(self) -> Streamlines:
        logger.info("executing %s.do()", self.__class__.__name__)

        device = self.device
        dataset, velocity_field_name, color_field_name = await self.get_dataset(device)
        seeds = await self.get_seeds(device)
        tc_handle = await self.get_termination_condition_handle(device)

        result = Streamlines()
        result.fields = {}

        with dav.scoped_timer("dav.streamlines.compute", color="yellow"):
            vtk_result = dav_streamlines.compute(
                dataset,
                velocity_field_name,
                seeds,
                initial_dt=self.dX,
                min_dt=0.1 * self.dX,
                max_dt=0.9 * self.dX,
                max_steps=self.maxLength,
                tolerance=1e-1,
                tc_model=get_tc_model(dataset.data_model) if tc_handle is not None else None,
                tc_handle=tc_handle,
            )
        with dav.scoped_timer("dav.streamlines.extract", color="yellow"):
            result.points = vtk_result.handle.points.numpy()
            result.curveVertexCounts = np.diff(vtk_result.handle.cell_offsets.numpy())
            result.fields["time"] = vtk_result.fields["times"].get_data().numpy()
            if color_field_name is not None:
                # probe color field along streamlines
                probe_result = dav_probe.compute(dataset, color_field_name, vtk_result, "colors")
                result.fields["scalar"] = probe_result.fields["colors"].get_data().numpy()

        return result

    async def get_seeds(self, device) -> dav.Dataset:
        logger.info("Getting seeds for DAV streamlines")
        seed_points = wp.array(array_utils.as_numpy_array(self.seeds), dtype=wp.vec3f, device=device)

        # create dataset
        ds_handle = dav_sids_unstructured.DatasetHandle()
        ds_handle.grid_coords = seed_points
        ds_handle.element_type = dav_sids_unstructured.ET_NODE

        dataset = dav.Dataset(data_model=dav_sids_unstructured.DataModel, handle=ds_handle, device=device)
        return dataset

    async def get_dataset(self, device) -> tuple[dav.Dataset, str, str | None]:
        logger.info("Getting dataset for DAV streamlines")
        fields = []
        fields += self.velocity_fields
        fields += [self.colorField] if self.colorField is not None else []

        # Use ConvertToDAVDataSet to convert the dataset
        dav_dataset = await ConvertToDAVDataSet.invoke(
            dataset=self.dataset, fields=fields, device=device, timeCode=self.timeCode
        )

        if len(self.velocity_fields) == 1:
            velocity_field = dav_dataset.fields[self.velocity_fields[0]]
        elif len(self.velocity_fields) == 3:
            # create an SOA field for velocity
            velocity_field = dav.Field.from_arrays(
                [
                    dav_dataset.fields[self.velocity_fields[0]].get_data(),
                    dav_dataset.fields[self.velocity_fields[1]].get_data(),
                    dav_dataset.fields[self.velocity_fields[2]].get_data(),
                ],
                association=dav_dataset.fields[self.velocity_fields[0]].association,
            )
        else:
            raise usd_utils.QuietableException("Expected 1 or 3 velocity fields, got %d" % len(self.velocity_fields))

        dav_dataset.fields["cae:velocity"] = velocity_field
        if self.colorField is not None:
            if self.colorField not in dav_dataset.fields:
                raise usd_utils.QuietableException("Color field '%s' not found in DAV dataset fields" % self.colorField)

        return dav_dataset, "cae:velocity", self.colorField if self.colorField else None

    async def get_termination_condition_handle(self, device) -> TerminationConditionHandle:
        if not self.dataset.HasRelationship("field:block_list"):
            logger.info("No blocked cells/faces specified for termination condition.")
            return None

        if not usd_utils.get_target_prim(self.dataset, "field:block_list", quiet=True):
            logger.info("Block list relationship is empty; no termination condition will be applied.")
            return None

        cache_key = {
            "label": "GetTerminationConditionHandle",
            "dataset": str(self.dataset.GetPath()),
            "device": device,
        }

        if cached_dataset := cache.get(str(cache_key), timeCode=Usd.TimeCode.EarliestTime()):
            return cached_dataset

        block_list = array_utils.as_numpy_array(
            await usd_utils.get_array_from_relationship(self.dataset, "field:block_list")
        )
        block_list = block_list.reshape(-1, 2).astype(np.int32, copy=False)  # each entry is (cell_id, face_id)

        # sort by cell_id
        block_list = block_list[np.argsort(block_list[:, 0])]

        blocked_cells_wp = wp.array(block_list[:, 0] + 1, dtype=wp.int32, device=device)
        blocked_faces_wp = wp.array(block_list[:, 1], dtype=wp.int32, device=device)

        tc_handle = TerminationConditionHandle()
        tc_handle.blocked_cells = blocked_cells_wp
        tc_handle.blocked_faces = blocked_faces_wp

        cache.put(str(cache_key), tc_handle, timeCode=Usd.TimeCode.EarliestTime())
        return tc_handle


class OmniCaeDataSetVoxelizeDAV(Voxelize):

    async def get_dataset(self, device) -> dav.Dataset:
        logger.info("Getting dataset for DAV voxelization")
        # Use ConvertToDAVDataSet to convert the dataset
        dav_dataset = await ConvertToDAVDataSet.invoke(
            dataset=self.dataset, fields=self.fields, device=device, timeCode=self.timeCode
        )
        return dav_dataset

    async def do(self):
        logger.info("executing %s.do()", self.__class__.__name__)

        device = wp.get_cuda_device(self.deviceOrdinal) if self.deviceOrdinal >= 0 else wp.get_device("cpu")
        if device.is_cpu:
            raise usd_utils.QuietableException("DAV Voxelization currently only supports CUDA devices.")

        with progress.ProgressContext("Read Dataset", scale=0.6):
            dav_dataset = await self.get_dataset(device)

        if len(self.fields) == 1:
            field = dav_dataset.fields[self.fields[0]]
        elif len(self.fields) == 3:
            # create an SOA field for velocity
            field = dav.Field.from_arrays(
                [
                    dav_dataset.fields[self.fields[0]].get_data(),
                    dav_dataset.fields[self.fields[1]].get_data(),
                    dav_dataset.fields[self.fields[2]].get_data(),
                ],
                association=dav_dataset.fields[self.fields[0]].association,
            )
        else:
            raise usd_utils.QuietableException("Expected 1 or 3 velocity fields, got %d" % len(self.velocity_fields))

        dav_dataset.fields["cae:voxelize_field"] = field

        if self.bbox.isEmpty():
            min_bounds, max_bounds = dav_dataset.get_bounds()  # returns tuple of wp.vec3
            bounds = np.array([[min_bounds.x, min_bounds.y, min_bounds.z], [max_bounds.x, max_bounds.y, max_bounds.z]])
            extents = np.around(bounds / self.voxelSize).astype(np.int32, copy=False)
            bbox = Range3i(extents[0], extents[1])
        else:
            bbox = self.bbox

        dims = np.array(bbox.max) - np.array(bbox.min) + 1
        result = dav_voxelization.compute(
            dav_dataset,
            "cae:voxelize_field",
            origin=wp.vec3f(bbox.min[0], bbox.min[1], bbox.min[2]) * self.voxelSize,
            dims=dims,
            voxel_size=wp.vec3f(self.voxelSize, self.voxelSize, self.voxelSize),
            voxel_index_type="ij",
            use_nanovdb=True,
            output_field_name="cae:voxelized_field",
        )
        volume = result.fields["cae:voxelized_field"].get_data()
        return volume
