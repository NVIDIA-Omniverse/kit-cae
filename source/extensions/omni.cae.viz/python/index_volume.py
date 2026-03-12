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
from omni.cae.data import array_utils, cache, progress, usd_utils
from omni.cae.dav import index_utils as dav_index_utils
from omni.cae.index import bindings as index_bindings
from omni.cae.schema import viz as cae_viz
from omni.usd import get_context
from pxr import Sdf, Usd, UsdShade, UsdVol, Vt

from . import utils as viz_utils
from .execution_context import ExecutionContext
from .operator import operator

logger = getLogger(__name__)


@operator(supports_temporal=True, tick_on_time_change=True)
class IndeXVolume:
    """
    Operator for all volume rendering using NVIDIA IndeX. We create the same "pipeline" for both NanoVDB and Irregular volumes.
    The only difference is the importer and compute task used. That customization is done in the subclasses.
    """

    prim_type = "Volume"
    api_schemas = {
        "CaeVizIndeXVolumeAPI",
        "CaeVizDatasetSelectionAPI:source",
    }

    optional_api_schemas = {
        "CaeVizDatasetVoxelizationAPI:source",
        "CaeVizFieldThresholdingAPI",
        "CaeVizFieldSelectionAPI",
        "CaeVizRescaleRangeAPI",
        "CaeVizConfigureXACShaderAPI",
    }

    def get_nvindex_type(self, prim: Usd.Prim) -> str:
        if prim.HasAPI(cae_viz.DatasetVoxelizationAPI):
            return "vdb"
        else:
            return "irregular_volume"

    async def get_source(self, prim: Usd.Prim, timeCode: Usd.TimeCode, device: str) -> dav.Dataset:
        return await viz_utils.get_input_dataset(prim, "source", timeCode=timeCode, device=device)

    async def exec(self, prim: Usd.Prim, device: str, context: ExecutionContext):
        logger.info(
            f"IndexVolume.exec - reason: {context.reason.value}, timecode: {context.timecode}, raw: {context.raw_timecode}"
        )
        nv_volume_type = self.get_nvindex_type(prim)

        if nv_volume_type == "vdb" and device == "cpu":
            logger.warning(f"NanoVDBVolume operator must be executed on CUDA device, got {device}")
            device = "cuda"
        elif nv_volume_type == "irregular_volume" and device != "cpu":
            logger.warning(f"IrregularVolume operator must be executed on CPU device, got {device}")
            device = "cpu"

        source_dataset = await self.get_source(prim, context.timecode, device)
        if len(source_dataset.get_field_names()) == 0:
            raise usd_utils.QuietableException("No fields selected. At least one field is required.")

        cache_key = f"[viz:index_volume]::{prim.GetPath()}"
        if context.is_full_rebuild_needed():
            cache.remove(cache_key)  # necessary to remove all timecodes from cache
            # place earliest timecode in cache; this is used by the importer to initialize the volume
            cache.put(cache_key, source_dataset, timeCode=Usd.TimeCode.EarliestTime(), consumerPrims=[prim], force=True)

        # place current timecode in cache; this is used by the compute task to pass fields to IndeX
        cache.put(cache_key, source_dataset, timeCode=context.timecode, consumerPrims=[prim], force=True)

        with progress.ProgressContext("Executing DAV [compute bounds]"):
            bds = source_dataset.get_bounds()

        # block change to package all changes into a single pass.
        with Sdf.ChangeBlock():
            with viz_utils.edit_context(prim):
                # forcing it off at the beginning and toggling it at tend marks the UsdVolVolume
                # dirty causing IndeX to reexecute the compute task otherwise it can be lagging
                # unless some property on the value changed.
                prim.GetAttribute("nvindex:composite").Set(False)

                volume = UsdVol.Volume(prim)
                volume.CreateExtentAttr().Set([(bds[0][0], bds[0][1], bds[0][2]), (bds[1][0], bds[1][1], bds[1][2])])
                prim.GetAttribute("nvindex:type").Set(nv_volume_type)

                if importer := prim.GetChild("Importer"):
                    if context.is_full_rebuild_needed():
                        # importer is updated only on full rebuilds (and not temporal updates)
                        self.setup_importer(prim, importer, source_dataset, nv_volume_type, cache_key)
                else:
                    raise ValueError("Missing 'Importer' child prim for Volume")

                if loader := prim.GetChild("Material").GetChild("DataLoader"):
                    assert loader.IsA(UsdShade.Shader), f"DataLoader is not a shader: {loader.GetPath()}"
                    self.setup_data_loader(prim, UsdShade.Shader(loader), nv_volume_type, context, cache_key)
                else:
                    raise ValueError("Missing 'DataLoader' child prim for Volume")

            # Update ranges, if any, for the attributes specified in the CaeVizRescaleRangeAPI instances.
            viz_utils.process_rescale_range_apis(prim, source_dataset)

            if context.is_full_rebuild_needed():
                # no need to update this for temporal updates
                suggested_voxel_size = viz_utils.process_configure_xac_shader_apis(prim, source_dataset)

                # For NanoVDB volumes suggested_voxel_size is non-None. Now, we need to set voxel size for IndeX rendering
                # as the default (-1) doesn't since we use phyiscal bounds of the volume for Extents on UsdVolVolume unlike
                # what IndeX seems to prefer. So we need to adjust the sampling distance accordingly. However, we don't want
                # to always override user choice so we have a happy medium here by only updating the sampling distance if the
                # voxel size changes or if the user has explicitly blocked the update.
                if prim.GetCustomDataByKey("cae:viz:block_sampling_distance_update"):
                    suggested_voxel_size = None

                # update nvindex settings with the suggested voxel size;
                # to avoid replacing user settings too much we only do this if voxel-size changes.
                if (
                    suggested_voxel_size is not None
                    and prim.GetCustomDataByKey("cae:viz:last_voxel_size") != suggested_voxel_size
                ):

                    # these edits are intentionally done on the active authoring layer.
                    prim.SetCustomDataByKey("cae:viz:last_voxel_size", Vt.Float(suggested_voxel_size))
                    prim.SetCustomDataByKey("nvindex.renderSettings:samplingDistance", Vt.Float(suggested_voxel_size))

            with viz_utils.edit_context(prim):
                # last thing to do is to enable IndeX rendering for this volume.
                prim.GetAttribute("nvindex:composite").Set(True)

    def setup_importer(
        self,
        prim: Usd.Prim,
        importer: Usd.Prim,
        source_dataset: dav.Dataset,
        nv_volume_type: str,
        cache_key: str,
    ):
        enable_field_interpolation = (
            prim.HasAPI(cae_viz.OperatorTemporalAPI)
            and cae_viz.OperatorTemporalAPI(prim).GetEnableFieldInterpolationAttr().Get()
        )

        if nv_volume_type == "vdb":
            # Use the builtin empty init importer to initialize the NanoVDB volume.
            nb_attributes = len(source_dataset.get_field_names())
            importer.SetCustomDataByKey(
                "nvindex.importerSettings",
                {
                    "importer": Vt.Token("nv::index::plugin::openvdb_integration.NanoVDB_empty_init_importer"),
                    "nb_attributes": Vt.Int(nb_attributes if not enable_field_interpolation else nb_attributes * 2),
                },
            )
        elif nv_volume_type == "irregular_volume":
            # Use the PythonImporter for irregular volumes.
            importer.SetCustomDataByKey(
                "nvindex.importerSettings",
                {
                    "importer": "nv::omni::cae::index.PythonImporter",
                    "module_name": "omni.cae.viz.index_volume",
                    # Will end up being IndeXImporter_irregular_volume.
                    "class_name": "IndeXImporter_irregular_volume",
                    "params_prim_path": Vt.Token(str(prim.GetPath())),
                    "params_time_code": Vt.Token(str(Usd.TimeCode.EarliestTime())),
                    "params_enable_field_interpolation": Vt.Bool(enable_field_interpolation),
                    "params_cache_key": Vt.Token(cache_key),
                },
            )
        else:
            raise ValueError(f"Unknown NVINDEX volume type: {nv_volume_type}")

    def setup_data_loader(
        self, prim: Usd.Prim, loader: UsdShade.Shader, nv_volume_type: str, context: ExecutionContext, cache_key: str
    ):
        loader.CreateInput("module_name", Sdf.ValueTypeNames.String).Set("omni.cae.viz.index_volume")
        # Will end up being IndeXComputeTask_vdb or IndeXComputeTask_irregular_volume
        loader.CreateInput("class_name", Sdf.ValueTypeNames.String).Set(f"IndeXComputeTask_{nv_volume_type}")
        loader.CreateInput("enabled", Sdf.ValueTypeNames.Bool).Set(True)
        # For NanoVDB volumes we execute the compute task on the GPU, while for irregular volumes we execute on the CPU.
        loader.CreateInput("is_gpu_operation", Sdf.ValueTypeNames.Bool).Set(
            False if nv_volume_type == "irregular_volume" else True
        )
        loader.CreateInput("params_prim_path", Sdf.ValueTypeNames.Token).Set(str(prim.GetPath()))
        loader.CreateInput("params_time_code", Sdf.ValueTypeNames.String).Set(str(context.timecode))
        # note: next_time_code may be None
        loader.CreateInput("params_next_time_code", Sdf.ValueTypeNames.String).Set(str(context.next_time_code))
        loader.CreateInput("params_cache_key", Sdf.ValueTypeNames.Token).Set(cache_key)

    def deactivate(self, prim):
        with viz_utils.edit_context(prim):
            prim.GetAttribute("nvindex:composite").Set(False)

    async def on_time_changed(self, prim: Usd.Prim, device: str, context: ExecutionContext):
        """
        Called on every time change when tick_on_time_change=True.

        This is a lightweight hook for minimal updates on repeat timecodes.
        Just updates the DataLoader time parameters to trigger IndeX field data refresh.

        The context provides:
        - context.timecode: Snapped timecode (from get_bracketing_time_samples_for_prim)
        - context.raw_timecode: Original timeline timecode

        Check CaeVizOperatorTemporalAPI to determine if interpolation is needed:
        - When enableFieldInterpolation=True: Called for ALL raw timecodes
        - When enableFieldInterpolation=False: Called only for snapped timecodes
        """
        # Update DataLoader time parameters (triggers IndeX ComputeTask)
        with viz_utils.edit_context(prim):
            if loader := prim.GetChild("Material").GetChild("DataLoader"):
                s_loader = UsdShade.Shader(loader)
                s_loader.CreateInput("params_time_code", Sdf.ValueTypeNames.String).Set(str(context.timecode))
                # note: next_time_code may be None
                s_loader.CreateInput("params_next_time_code", Sdf.ValueTypeNames.String).Set(
                    str(context.next_time_code)
                )
            else:
                logger.warning(f"Missing DataLoader for {prim.GetPath()} in on_time_changed")

        # Update XAC shader temporal interpolation parameters.
        viz_utils.process_configure_xac_shader_apis_temporal(
            prim, context.timecode, context.next_time_code, context.raw_timecode
        )

        logger.info(
            f"Time tick for {prim.GetTypeName()} at {prim.GetPath()}: timecode={context.timecode}, raw={context.raw_timecode}"
        )


class IndeXBase:
    """A common base class for IndeX importer and compute task."""

    def __init__(self, params: dict):
        self.params = params.copy()

    @staticmethod
    def parse_time_code(tc: Any) -> Usd.TimeCode:
        if isinstance(tc, Usd.TimeCode):
            return tc
        elif tc is None or tc == str(Usd.TimeCode.EarliestTime()):
            return Usd.TimeCode.EarliestTime()
        elif tc == str(Usd.TimeCode.Default()):
            return Usd.TimeCode.Default()
        else:
            return Usd.TimeCode(float(tc))

    @property
    def prim(self):
        stage = get_context().get_stage()
        path = self.params.get("prim_path", "")
        return usd_utils.get_prim_at_path(stage, path)

    @property
    def cache_key(self) -> str:
        return self.params.get("cache_key", "")

    def get_time_code(self) -> Usd.TimeCode:
        return self.parse_time_code(self.params.get("time_code", Usd.TimeCode.EarliestTime()))

    def get_next_time_code(self) -> Usd.TimeCode:
        return self.parse_time_code(self.params.get("next_time_code", Usd.TimeCode.EarliestTime()))

    def has_source_dataset(self) -> bool:
        return self.get_source_dataset() is not None

    def get_source_dataset(self) -> dav.Dataset:
        return cache.get(self.cache_key, timeCode=self.get_time_code())

    def has_next_time_code(self) -> bool:
        next_tc = self.params.get("next_time_code", None)
        return next_tc is not None and str(next_tc) != str(None)

    def has_next_source_dataset(self) -> bool:
        next_tc = self.params.get("next_time_code", None)
        return next_tc is not None and str(next_tc) != str(None)

    def get_next_source_dataset(self) -> dav.Dataset:
        return cache.get(self.cache_key, timeCode=self.get_next_time_code())


class IndeXImporter_irregular_volume(IndeXBase):
    """An importer for irregular volumes."""

    def get_bounds(self) -> index_bindings.Bbox_float32:
        # Dummy bounds for example purposes
        bbox = index_bindings.Bbox_float32()

        volume = UsdVol.Volume(self.prim)
        exts = volume.GetExtentAttr().Get()
        bbox.min = np.array(exts[0])
        bbox.max = np.array(exts[1])
        return bbox

    def create_subset(self, bbox: index_bindings.Bbox_float32, factory: index_bindings.IData_subset_factory) -> Any:
        volume_prim = self.prim
        source_dataset = self.get_source_dataset()
        subset: index_bindings.IIrregular_volume_subset = factory.create_irregular_volume_subset()

        # count faces and face vertices
        nb_faces, nb_face_vertices = dav_index_utils.compute_face_summary(source_dataset)
        logger.info(f"nb_faces: {nb_faces:,}, nb_face_vertices: {nb_face_vertices:,}")

        params = index_bindings.Mesh_parameters()
        params.nb_vertices = source_dataset.get_num_points()
        params.nb_cells = source_dataset.get_num_cells()
        params.nb_faces = nb_faces
        params.nb_face_vtx_indices = nb_face_vertices
        params.nb_cell_face_indices = nb_faces  # since we won't be sharing faces

        logger.info(f"Mesh parameters: {params}")

        storage: index_bindings.Mesh_storage = subset.generate_mesh_storage(params)
        dav_index_utils.populate_mesh_storage(source_dataset, storage)

        instance_names = usd_utils.get_instances(volume_prim, "CaeVizFieldSelectionAPI")

        # Allocate attributes for current timestep
        dav_index_utils.allocate_attribute_storage(source_dataset, subset, instance_names, start_index=0)

        # If temporal field interpolation is enabled, allocate attributes for next timestep too
        if self.params.get("enable_field_interpolation", False):
            # Use the same source_dataset for allocation (structure is identical)
            num_fields = len(instance_names)
            dav_index_utils.allocate_attribute_storage(source_dataset, subset, instance_names, start_index=num_fields)

        return subset


class IndeXComputeTask_irregular_volume(IndeXBase):
    """A compute task for irregular volumes."""

    def launch_compute(self, dst_buffer: index_bindings.IDistributed_compute_destination_buffer):
        logger.info(f"Launching IndeXComputeTask_irregular_volume with params: {self.params}, {type(dst_buffer)}")
        if not self.has_source_dataset():
            logger.warning(f"No source dataset found for {self.prim.GetPath()}")
            return

        try:
            datasets = [self.get_source_dataset()]
            if self.has_next_time_code() and self.has_next_source_dataset():
                datasets.append(self.get_next_source_dataset())

            subset = dst_buffer.get_distributed_data_subset()
            instance_names = usd_utils.get_instances(self.prim, "CaeVizFieldSelectionAPI")

            for ds_idx, dataset in enumerate(datasets):
                start_index = ds_idx * len(instance_names)
                dav_index_utils.fill_attribute_storage(dataset, subset, instance_names, start_index)

        except Exception as e:
            logger.error(f"Failed to launch IndeXComputeTask_irregular_volume: {e}")


class IndeXComputeTask_vdb(IndeXBase):
    """A compute task for NanoVDB volumes."""

    def __init__(self, params: dict):
        super().__init__(params)
        logger.info(f"Created IndeXComputeTask_vdb[id={id(self)}] with params: {self.params}")

    def launch_compute(self, dst_buffer: index_bindings.IDistributed_compute_destination_buffer):
        logger.info(f"Launching IndeXComputeTask_vdb[id={id(self)}] with params: {self.params}, {type(dst_buffer)}")

        if not self.has_source_dataset():
            logger.warning(f"No source dataset found for {self.prim.GetPath()}")
            return

        try:
            datasets = [self.get_source_dataset()]
            if self.has_next_time_code() and self.has_next_source_dataset():
                datasets.append(self.get_next_source_dataset())

            cached_items = []
            for ds_idx, dataset in enumerate(datasets):
                subset = dst_buffer.get_distributed_data_subset()
                device_subset = subset.get_device_subset()
                device_id = device_subset.get_device_id()

                instance_names = usd_utils.get_instances(self.prim, "CaeVizFieldSelectionAPI")
                for idx, instance_name in enumerate(instance_names):
                    field_name = instance_name
                    if dataset.has_field(field_name):
                        field = dataset.get_field(field_name)
                        volume: wp.Volume = field.get_data()
                        # note: farray holds a reference to volume so we don't need to cache it
                        farray = array_utils.get_nanovdb_as_field_array(volume)

                        # ensure the field array is on the correct device
                        if farray.device_id != device_id:
                            logger.info(
                                f"Moving field array for '{field_name}' to device {device_id} (source: {farray.device_id})"
                            )
                            farray = farray.to_device(device_id)
                            cached_items.append(farray)
                        else:
                            cached_items.append(farray)

                        device_subset.adopt_field_array(ds_idx * len(instance_names) + idx, farray)
                    else:
                        logger.error(f"Field {field_name} not found in source dataset")

            # update the cached items until next invocation;
            # this is necessary since `adopt_field_array` does not copy the data or take ownership of the field array.
            # We set it up to automatically remove from cache when the volume prim is deleted. This is
            # very conservative but this avoid race condition when compute task is deleted but
            # the volume may still be need for rendering.
            volume_prim = self.prim
            cache.put_ex(
                f"[viz:index_compute_task_vdb]::{volume_prim.GetPath()}",
                cached_items,
                prims=[cache.PrimWatch(volume_prim, on="delete")],
                force=True,
                timeCode=Usd.TimeCode.EarliestTime(),
            )

        except Exception as e:
            logger.exception(f"Failed to launch IndeXComputeTask_vdb: {e}")
