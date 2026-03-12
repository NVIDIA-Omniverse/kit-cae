# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
__all__ = [
    "get_dataset",
    "get_field",
    "pass_fields",
    "probe_fields",
    "get_vector_magnitude_field",
    "get_selected_component_field",
    "fetch_data",
    "lerp_dataset",
]
from logging import getLogger
from typing import Any

import dav
import dav.operators.probe
import numpy as np
import warp as wp
from omni.cae.data import array_utils, progress, usd_utils
from omni.cae.schema import cae
from pxr import Usd

from .command_types import ConvertToDAVDataSet

logger = getLogger(__name__)


def _get_dav_association(assoc: str) -> dav.AssociationType:
    if assoc == cae.Tokens.vertex:
        return dav.AssociationType.VERTEX
    elif assoc == cae.Tokens.cell:
        return dav.AssociationType.CELL
    else:
        raise ValueError(f"Unsupported field association '{assoc}'")


def _get_num_components_from_array(array) -> int:
    if array.ndim == 1:
        return 1
    elif array.ndim == 2:
        return array.shape[1]
    else:
        raise ValueError(f"Array with ndim {array.ndim} is not supported for component selection.")


def fetch_data(dataset: dav.Dataset, field_name: str) -> array_utils.IFieldArray:
    """
    Fetches field data from a dataset as a IFieldArray.

    Parameters
    ----------
    dataset : dav.Dataset
        The dataset to fetch the field data from
    field_name : str
        The name of the field to fetch the data from

    Returns
    -------
    array_utils.IFieldArray
        The field data as an IFieldArray
    """
    if not dataset.has_field(field_name):
        raise ValueError(f"Field '{field_name}' not found in dataset.")

    field = dataset.get_field(field_name)
    wp_array = field.to_array()
    return array_utils.IFieldArray.from_array(wp_array)


async def get_dataset(
    dataset_prim: Usd.Prim, timeCode: Usd.TimeCode, device: str, needs_topology: bool, needs_geometry: bool
) -> dav.Dataset:
    """
    Internal utility to build a dataset from a dataset prim.

    Parameters
    ----------
    dataset_prim : Usd.Prim
        The prim from which to build the dataset
    timeCode : Usd.TimeCode
        The time code for which to fetch the dataset
    device : str
        The device on which to load the dataset
    needs_topology : bool
        Whether the dataset needs topology information
    needs_geometry : bool
        Whether the dataset needs geometry information

    Returns
    -------
    dav.Dataset
        The loaded dataset
    """
    # snap time-code to nearest time sample needed for dataset without fields.
    # this ensures that for datasets where only the fields are time-varying, we don't bother
    # geometry/topology fetching for the timesteps.
    timeCode = usd_utils.snap_time_code_to_prim(dataset_prim, timeCode, traverse_field_relationships=False)
    return await ConvertToDAVDataSet.invoke(dataset_prim, device, timeCode, needs_topology, needs_geometry)


async def get_field(field_prims: list[Usd.Prim], timeCode: Usd.TimeCode, device: str) -> dav.Field:
    """
    Utility to get a field from a list of field prims.
    Parameters
    ----------
    field_prims : list[Usd.Prim]
        The list of field prims to get the field from
    timeCode : Usd.TimeCode
        The time code for which to fetch the field
    device : str
        The device on which to load the field

    Returns
    -------
    dav.Field
        The loaded field
    """
    # this can never be since get_target_prims will throw usd_utils.QuietableException if arrays are missing.
    assert len(field_prims) > 0, f"No field prims found"

    if any(prim.HasAPI(cae.NanoVDBFieldArrayAPI) for prim in field_prims):
        if len(field_prims) > 1:
            raise ValueError(
                "CaeNanoVDBFieldArrayAPI only supports a single field prim; " f"{len(field_prims)} were provided."
            )
        prim = field_prims[0]
        nanovdb_api = cae.NanoVDBFieldArrayAPI(prim)

        origin_val = nanovdb_api.GetOriginAttr().Get(timeCode)
        origin = wp.vec3i(*origin_val) if origin_val is not None else wp.vec3i(0, 0, 0)

        # GetDimsAttr() will be available after schema regen+rebuild
        dims_val = prim.GetAttribute("cae:nanovdb_field_array:dims").Get(timeCode)
        if dims_val is None:
            raise ValueError("CaeNanoVDBFieldArrayAPI: 'dims' attribute is not set.")
        dims = wp.vec3i(*dims_val)

        association_token = cae.FieldArray(prim).GetFieldAssociationAttr().Get(timeCode)
        dav_association = _get_dav_association(association_token)

        array = (await usd_utils.get_arrays(field_prims, timeCode))[0]
        np_array = array.numpy().view(np.byte)  # NanoVDB buffer as uint32
        wp_raw = wp.array(np_array, device=device)  # Ensure the array is on the correct device
        volume = wp.Volume(wp_raw)
        return dav.Field.from_volume(volume, dims=dims, association=dav_association, origin=origin)

    associations = [cae.FieldArray(fa).GetFieldAssociationAttr().Get(timeCode) for fa in field_prims]

    if not all(assoc == associations[0] for assoc in associations):
        raise ValueError("Multiple different associations found; only one is supported currently.")

    dav_association = _get_dav_association(associations[0])
    arrays = await usd_utils.get_arrays(field_prims, timeCode)

    # To keep things a bit manageable, we only support selecting arrays where there's just 1 array
    # or multiple arrays with exactly one component each.
    if len(arrays) > 1 and any(_get_num_components_from_array(array) != 1 for array in arrays):
        raise ValueError(
            "FieldSelectionAPI with multiple target prims only supports arrays with a single component each."
        )

    # let's create warp arrays on the target device
    wp_arrays = [array_utils.as_warp_array(array).to(device) for array in arrays]

    # this will create an SoA field if there are multiple arrays, otherwise a AoS field is created.
    field = dav.Field.from_arrays(wp_arrays, dav_association)
    return field


async def get_vector_magnitude_field(field_prims: list[Usd.Prim], timeCode: Usd.TimeCode, device: str) -> dav.Field:
    """
    Utility to get a vector magnitude field from a list of field prims.
    Parameters
    ----------
    field_prims : list[Usd.Prim]
        The list of field prims to get the vector magnitude field from
    timeCode : Usd.TimeCode
        The time code for which to fetch the vector magnitude field
    device : str
        The device on which to load the vector magnitude field
    """
    base_field = await get_field(field_prims, timeCode, device)
    if not dav.utils.is_vector_dtype(base_field.dtype):
        # not a vector field, return the base field
        return base_field

    return dav.Field.from_field(base_field, magnitude=True)


async def get_selected_component_field(
    field_prims: list[Usd.Prim], timeCode: Usd.TimeCode, device: str, component_index: int
) -> dav.Field:
    """
    Utility to get a selected component field from a list of field prims.
    Parameters
    ----------
    field_prims : list[Usd.Prim]
        The list of field prims to get the selected component field from
    timeCode : Usd.TimeCode
        The time code for which to fetch the selected component field
    device : str
        The device on which to load the selected component field
    component_index : int
        The index of the component to fetch
    """
    base_field = await get_field(field_prims, timeCode, device)
    if not dav.utils.is_vector_dtype(base_field.dtype):
        # not a vector field, return the base field
        if component_index != 0:
            logger.warning(f"Component index {component_index} is not supported for scalar fields.")
        return base_field

    nb_components = dav.utils.get_vector_length(base_field.dtype)
    if component_index < 0 or component_index >= nb_components:
        raise ValueError(
            f"Component index {component_index} out of range for vector fields. Vector length is {nb_components}."
        )

    return dav.Field.from_field(base_field, component=component_index)


def probe_fields(
    source_dataset: dav.Dataset,
    target_dataset: dav.Dataset,
    *,
    fields: set[str] = set(),
    exclude_fields: set[str] = set(),
) -> dav.Dataset:
    """
    Utility to probe fields from a source dataset onto a target dataset based on matching associations and topology.

    Parameters
    ----------
    source_dataset : dav.Dataset
        The source dataset containing fields to probe from
    target_dataset : dav.Dataset
        The target dataset to probe fields onto
    fields : set[str], optional
        A set of field names to probe. If empty, all fields from the source dataset are probed, by default set()
    exclude_fields : set[str], optional
        A set of field names to exclude from probing. If empty, no fields are excluded, by default set()
    Returns
    -------
    dav.Dataset
        The target dataset with probed fields added
    """
    for field_name in source_dataset.get_field_names():
        if fields and field_name not in fields:
            logger.debug(f"Field '{field_name}' not in specified fields to probe; skipping.")
        elif field_name in exclude_fields:
            logger.debug(f"Field '{field_name}' in exclude fields; skipping.")
        elif target_dataset.has_field(field_name):
            logger.debug(f"Field '{field_name}' already exists in target dataset; skipping probe.")
        else:
            with progress.ProgressContext("Executing DAV [probe]"):
                target_dataset = dav.operators.probe.compute(
                    source_dataset, field_name, target_dataset, output_field_name=field_name
                )
    return target_dataset


@dav.cached
def get_extract_elements_kernel(field_model: dav.FieldModel, indices_model: dav.FieldModel, dtype: Any):
    """
    Get a kernel to extract elements from a field based on indices.
    """

    @wp.kernel(enable_backward=False, module="unique")
    def extract_elements_kernel(
        field: field_model.FieldHandle, indices: indices_model.FieldHandle, result: wp.array(dtype=dtype)
    ):
        tid = wp.tid()
        idx = indices_model.FieldAPI.get(indices, tid)
        result[tid] = field_model.FieldAPI.get(field, idx)

    return extract_elements_kernel


def extract_elements(field: dav.Field, indices: dav.Field) -> wp.array:
    """
    Extract elements from a field based on indices.
    """
    result = wp.zeros(indices.size, dtype=field.dtype, device=field.device)

    kernel = get_extract_elements_kernel(field.field_model, indices.field_model, field.dtype)
    wp.launch(kernel, dim=indices.size, inputs=[field.handle, indices.handle, result], device=field.device)
    return result


def pass_fields(
    source_dataset: dav.Dataset,
    target_dataset: dav.Dataset,
    *,
    fields: set[str] = set(),
    exclude_fields: set[str] = set(),
) -> dav.Dataset:
    """
    Utility to pass fields from a source dataset to a target dataset.
    """
    pt_idx = target_dataset.get_field("point_idx") if target_dataset.has_field("point_idx") else None
    cell_idx = target_dataset.get_field("cell_idx") if target_dataset.has_field("cell_idx") else None
    for field_name in source_dataset.get_field_names():
        if fields and field_name not in fields:
            logger.debug(f"Field '{field_name}' not in specified fields to pass; skipping.")
        elif field_name in exclude_fields:
            logger.debug(f"Field '{field_name}' in exclude fields; skipping.")
        elif target_dataset.has_field(field_name):
            logger.warning(f"Field '{field_name}' already exists in target dataset; skipping pass.")
        else:
            source_field = source_dataset.get_field(field_name)
            if source_field.association == dav.AssociationType.VERTEX:
                if pt_idx is None:
                    target_dataset.add_field(field_name, source_field)
                else:
                    # extract point indices from source field and use them to index the target field
                    wp_array = extract_elements(source_field, pt_idx)
                    new_field = dav.Field.from_array(wp_array, pt_idx.association)
                    target_dataset.add_field(field_name, new_field)
            elif source_field.association == dav.AssociationType.CELL:
                if cell_idx is None:
                    target_dataset.add_field(field_name, source_field)
                else:
                    # extract cell indices from source field and use them to index the target field
                    wp_array = extract_elements(source_field, cell_idx)
                    new_field = dav.Field.from_array(wp_array, cell_idx.association)
                    target_dataset.add_field(field_name, new_field)
            else:
                raise ValueError(f"Unsupported target association type {target_dataset.association}")
    return target_dataset


@wp.kernel(enable_backward=False, module="unique")
def _lerp_kernel(in1: wp.array(dtype=Any), in2: wp.array(dtype=Any), out: wp.array(dtype=Any), factor: float):
    tid = wp.tid()
    out[tid] = wp.lerp(in1[tid], in2[tid], factor)


@dav.cached
def _get_lerp_field_kernel(
    field1_model: dav.FieldModel, field2_model: dav.FieldModel, result_model: dav.FieldModel, dtype
):

    ScalarType = dav.utils.get_scalar_dtype(dtype)
    if dav.utils.is_integral_dtype(dtype):
        raise NotImplementedError("Interpolation of integral dtypes is not supported yet.")

    @wp.kernel(enable_backward=False, module="unique")
    def lerp_field_kernel(
        in1: field1_model.FieldHandle, in2: field2_model.FieldHandle, out: result_model.FieldHandle, factor: float
    ):
        tid = wp.tid()
        val1 = field1_model.FieldAPI.get(in1, tid)
        val2 = field2_model.FieldAPI.get(in2, tid)
        result = wp.lerp(val1, val2, ScalarType(factor))
        result_model.FieldAPI.set(out, tid, result)

    return lerp_field_kernel


def lerp_dataset(
    dataset1: dav.Dataset,
    dataset2: dav.Dataset,
    t: float,
    *,
    fields: set[str] = set(),
    exclude_fields: set[str] = set(),
) -> dav.Dataset:
    """
    Linearly interpolate between two datasets.

    This function performs linear interpolation (lerp) between two datasets with matching
    topology, interpolating both geometry (points) and field data. Supports both surface
    mesh and point cloud data models.

    Parameters
    ----------
    dataset1 : dav.Dataset
        The first dataset (at t=0.0) with either surface_mesh or point_cloud data model
    dataset2 : dav.Dataset
        The second dataset (at t=1.0) with the same data model as dataset1
    t : float
        Interpolation parameter, typically between 0.0 (returns dataset1) and 1.0 (returns dataset2)
    fields : set[str], optional
        A set of field names to interpolate. If empty, all common fields are interpolated, by default set()
    exclude_fields : set[str], optional
        A set of field names to exclude from interpolation, by default set()

    Returns
    -------
    dav.Dataset
        A new dataset with interpolated geometry and fields

    Raises
    ------
    ValueError
        If datasets don't have matching topology, data models, or devices

    Notes
    -----
    For surface meshes, topology (face connectivity) is preserved from dataset1.
    For point clouds, only point positions and fields are interpolated.
    """
    from dav.data_models.custom import point_cloud as dav_point_cloud
    from dav.data_models.custom import surface_mesh as dav_surface_mesh

    # Verify data models match
    if dataset1.data_model != dataset2.data_model:
        raise ValueError(
            f"Data model mismatch: dataset1 is '{dataset1.data_model}', dataset2 is '{dataset2.data_model}'"
        )

    # Verify supported data models
    if dataset1.data_model not in [dav_surface_mesh.DataModel, dav_point_cloud.DataModel]:
        raise ValueError(
            f"Unsupported data model '{dataset1.data_model}'. " f"Only 'surface_mesh' and 'point_cloud' are supported."
        )

    if dataset1.get_num_points() != dataset2.get_num_points():
        raise ValueError(
            f"Point count mismatch: dataset1 has {dataset1.get_num_points()} points, "
            f"dataset2 has {dataset2.get_num_points()} points"
        )

    # For surface meshes, verify cell count matches
    if dataset1.data_model == dav_surface_mesh.DataModel:
        if dataset1.get_num_cells() != dataset2.get_num_cells():
            raise ValueError(
                f"Cell count mismatch: dataset1 has {dataset1.get_num_cells()} cells, "
                f"dataset2 has {dataset2.get_num_cells()} cells"
            )

    if dataset1.device != dataset2.device:
        raise ValueError(
            f"Dataset devices mismatch: dataset1 is on {dataset1.device}, " f"dataset2 is on {dataset2.device}"
        )

    if t <= 0.0:
        return dataset1
    elif t >= 1.0:
        return dataset2

    # Interpolate points
    points1 = dataset1.handle.points
    points2 = dataset2.handle.points
    result_points = wp.zeros_like(points1, device=points1.device)
    wp.launch(
        _lerp_kernel,
        dim=points1.shape[0],
        inputs=[points1, points2, result_points, t],
        device=points1.device,
    )

    # Create result dataset based on data model
    if dataset1.data_model == dav_surface_mesh.DataModel:
        result_dataset = dav_surface_mesh.create_dataset(
            result_points,
            face_vertex_indices=dataset1.handle.face_vertex_indices,
            face_vertex_offsets=dataset1.handle.face_offsets,
        )
    else:  # point_cloud
        result_dataset = dav_point_cloud.create_dataset(result_points)

    # Get common fields to interpolate
    fields1 = set(dataset1.get_field_names())
    fields2 = set(dataset2.get_field_names())
    common_fields = fields1.intersection(fields2)

    # Filter fields based on input parameters
    if fields:
        fields_to_lerp = common_fields.intersection(fields)
    else:
        fields_to_lerp = common_fields

    fields_to_lerp = fields_to_lerp - exclude_fields

    # Interpolate fields
    for field_name in fields_to_lerp:
        field1 = dataset1.get_field(field_name)
        field2 = dataset2.get_field(field_name)

        # Verify field associations match
        if field1.association != field2.association:
            logger.warning(f"Field '{field_name}' has different associations; skipping interpolation.")
            continue

        # Verify field sizes match
        if field1.size != field2.size:
            logger.warning(f"Field '{field_name}' has different sizes; skipping interpolation.")
            continue

        if field1.dtype != field2.dtype:
            logger.warning(f"Field '{field_name}' has different dtypes; skipping interpolation.")
            continue

        result_field = dav.Field.from_array(
            wp.zeros(field1.size, dtype=field1.dtype, device=dataset1.device), field1.association
        )
        # Get field data
        lerp_field_kernel = _get_lerp_field_kernel(
            field1.field_model, field2.field_model, result_field.field_model, field1.dtype
        )
        wp.launch(
            lerp_field_kernel,
            dim=field1.size,
            inputs=[field1.handle, field2.handle, result_field.handle, t],
            device=dataset1.device,
        )

        result_dataset.add_field(field_name, result_field)

    return result_dataset
