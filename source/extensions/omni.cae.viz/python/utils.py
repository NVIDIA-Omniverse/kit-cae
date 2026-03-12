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
from dav.data_models.vtk import image_data as dav_data_models_vtk_image_data
from dav.operators import point_field as dav_point_field
from dav.operators import point_splats as dav_point_splats
from dav.operators import voxelization as dav_voxelization
from omni.cae.data import array_utils, cache, progress, usd_utils
from omni.cae.schema import cae
from omni.cae.schema import viz as cae_viz
from pxr import Gf, Sdf, Usd, UsdShade
from usdrt import Sdf as SdfRT
from usdrt import Usd as UsdRT
from usdrt import UsdGeom as UsdGeomRT
from usdrt import Vt as VtRT

logger = getLogger(__name__)


def edit_context(prim: Usd.Prim):
    """
    Returns the edit context for the given prim.
    For operators that update PXR USD prims (rather that UsdRt USD prims),
    the changes made by operators are typically not applied on the root layer but instead
    are applied on the session layer. This ensures that the changes don't clobber the root layer stage
    exports and also are not accidentally overridden by the user.

    Usage:

        with edit_context(prim):
            prim.GetAttribute("someAttr").Set("new_value")


    Caution:

        Never use `await` while an edit context is active. In general any long running operations should be
        avoided while an edit context is active. It's best to scope things such that the edit context is active
        only while updating properties on Prim and is deactivated as soon as the operation is complete.
    """
    stage = prim.GetStage()
    return Usd.EditContext(stage, stage.GetEditTargetForLocalLayer(stage.GetSessionLayer()))


def is_attr_locked(attr: Usd.Attribute) -> bool:
    """
    Returns True if the attribute is locked, False otherwise.
    """
    if not attr:
        raise ValueError("Attribute is None")
    if attr.HasCustomDataKey("omni:kit:locked") and attr.GetCustomDataByKey("omni:kit:locked"):
        return True
    return False


def process_rescale_range_apis(prim: Usd.Prim, dataset: dav.Dataset):
    """
    Processes the CaeVizRescaleRangeAPI schemas on the given prim and rescales the range of the attributes specified in the schemas.
    For every instance of CaeVizRescaleRangeAPI, the range of the attributes specified in the 'includes' relationship are rescaled
    to the range of the field specified in the instance name. The rescaling is done using the 'rescaleMode' attribute.

    Parameters
    ----------
    prim : Usd.Prim
        The prim to process
    dataset : dav.Dataset
        The dataset to obtain the field ranges from
    """
    stage = prim.GetStage()

    instance_names = set(usd_utils.get_instances(prim, "CaeVizRescaleRangeAPI"))
    for instance_name in instance_names:
        rescale_range_api = cae_viz.RescaleRangeAPI(prim, instance_name)
        rescale_mode = rescale_range_api.GetRescaleModeAttr().Get()

        if rescale_mode == cae_viz.Tokens.disable:
            continue

        include_targets = rescale_range_api.GetIncludesRel().GetForwardedTargets()
        min_include_targets = rescale_range_api.GetMinIncludesRel().GetForwardedTargets()
        max_include_targets = rescale_range_api.GetMaxIncludesRel().GetForwardedTargets()
        enable_include_targets = rescale_range_api.GetEnableIncludesRel().GetForwardedTargets()

        # filter all include targes to only consider attribute paths and then fetch those attributes from the stage
        attrs = [stage.GetAttributeAtPath(target) for target in include_targets if target.IsPrimPropertyPath()]

        # filter to only include attributes that can represent a float2
        attrs = [attr for attr in attrs if attr and attr.GetTypeName() == Sdf.ValueTypeNames.Float2]

        min_attrs = [stage.GetAttributeAtPath(target) for target in min_include_targets if target.IsPrimPropertyPath()]
        min_attrs = [attr for attr in min_attrs if attr and attr.GetTypeName() == Sdf.ValueTypeNames.Float]

        max_attrs = [stage.GetAttributeAtPath(target) for target in max_include_targets if target.IsPrimPropertyPath()]
        max_attrs = [attr for attr in max_attrs if attr and attr.GetTypeName() == Sdf.ValueTypeNames.Float]

        enable_attrs = [
            stage.GetAttributeAtPath(target) for target in enable_include_targets if target.IsPrimPropertyPath()
        ]
        enable_attrs = [attr for attr in enable_attrs if attr and attr.GetTypeName() == Sdf.ValueTypeNames.Bool]

        # filter to remove locked attributes i.e. attributes that have the "omni:kit:locked" custom data key set to True
        attrs = [attr for attr in attrs if not is_attr_locked(attr)]
        min_attrs = [attr for attr in min_attrs if not is_attr_locked(attr)]
        max_attrs = [attr for attr in max_attrs if not is_attr_locked(attr)]
        enable_attrs = [attr for attr in enable_attrs if not is_attr_locked(attr)]

        # if there are no attributes to process, skip
        if not attrs and not min_attrs and not max_attrs and not enable_attrs:
            continue

        # if the field is not present, set the enable attributes to False
        if not dataset.has_field(instance_name):
            for attr in enable_attrs:
                attr.Set(False)
            continue

        field = dataset.get_field(instance_name)
        range_min, range_max = field.get_range()  # for vectors, this returns the magnitude range
        for attr in attrs:
            if rescale_mode == cae_viz.Tokens.clamp:
                logger.info(f"Clamping range of attribute {attr.GetPath()} to ({range_min}, {range_max})")
                attr.Set((range_min, range_max))
            elif rescale_mode == cae_viz.Tokens.grow:
                cur_val = attr.Get()
                if cur_val is not None and cur_val[0] <= cur_val[1]:
                    v_min = min(cur_val[0], range_min)
                    v_max = max(cur_val[1], range_max)
                else:
                    v_min = range_min
                    v_max = range_max
                logger.info(f"Growing range of attribute {attr.GetPath()} to ({v_min}, {v_max})")
                attr.Set((v_min, v_max))

        for attr in min_attrs:
            if rescale_mode == cae_viz.Tokens.clamp:
                logger.info(f"Clamping range of attribute {attr.GetPath()} to {range_min}")
                attr.Set(range_min)
            elif rescale_mode == cae_viz.Tokens.grow:
                cur_val = attr.Get()
                if cur_val is not None and cur_val <= range_min:
                    v_min = range_min
                else:
                    v_min = cur_val
                logger.info(f"Growing range of attribute {attr.GetPath()} to {v_min}")
                attr.Set(v_min)

        for attr in max_attrs:
            if rescale_mode == cae_viz.Tokens.clamp:
                logger.info(f"Clamping range of attribute {attr.GetPath()} to {range_max}")
                attr.Set(range_max)
            elif rescale_mode == cae_viz.Tokens.grow:
                cur_val = attr.Get()
                if cur_val is not None and cur_val >= range_max:
                    v_max = range_max
                else:
                    v_max = cur_val
                logger.info(f"Growing range of attribute {attr.GetPath()} to {v_max}")
                attr.Set(v_max)

        for attr in enable_attrs:
            attr.Set(True)


def process_configure_xac_shader_apis(prim: Usd.Prim, dataset: dav.Dataset) -> float | None:
    """
    Processes the CaeVizConfigureXACShaderAPI schemas on the given prim and configures the XAC shader based on the field.

    For every instance of CaeVizConfigureXACShaderAPI, the voxel size and sample mode are configured based on the
    field type. The voxel size is configured based on the voxel size of the nvdb field specified in the
    instance name. The sample mode is configured based on the field type.

    Parameters
    ----------
    prim : Usd.Prim
        The prim to process
    dataset : dav.Dataset
        The dataset to obtain the field ranges from

    Returns
    -------
    float | None
        The suggested voxel size for IndeX rendering, None if not applicable

    Notes
    -----
    For NanoVDB volumes, the suggested voxel size is returned. For other volumes, None is returned.
    The value returned is the minimum voxel size of all the fields specified in the CaeVizConfigureXACShaderAPI instances.
    """
    stage = prim.GetStage()

    instance_names = set(usd_utils.get_instances(prim, "CaeVizConfigureXACShaderAPI"))
    index_voxel_size = None
    for instance_name in instance_names:
        if not dataset.has_field(instance_name):
            continue

        field = dataset.get_field(instance_name)
        if field.dtype not in [wp.float32, wp.vec3f]:
            logger.warning(
                "Automatic configuration of XAC shader is only supported for float32 and vec3f fields currently."
            )
            continue

        if hasattr(field.get_data(), "get_voxel_size"):
            voxel_size = field.get_data().get_voxel_size()
            voxel_size = np.array(voxel_size, dtype=np.float32).tolist()
            index_voxel_size = (
                min(index_voxel_size, min(voxel_size)) if index_voxel_size is not None else min(voxel_size)
            )
        else:
            voxel_size = [1.0, 1.0, 1.0]

        mode = 0 if field.dtype == wp.float32 else 1

        configure_xac_shader_api = cae_viz.ConfigureXACShaderAPI(prim, instance_name)

        for voxel_size_include in configure_xac_shader_api.GetVoxelSizeIncludesRel().GetForwardedTargets():
            attr = stage.GetAttributeAtPath(voxel_size_include)
            if attr and is_attr_locked(attr):
                continue
            elif attr and attr.GetTypeName() == Sdf.ValueTypeNames.Float3:
                attr.Set(tuple(voxel_size))
            else:
                logger.warning(
                    f"Invalid attribute type for voxel size include {voxel_size_include}: {attr.GetTypeName()}"
                )

        for sample_mode_include in configure_xac_shader_api.GetSampleModeIncludesRel().GetForwardedTargets():
            attr = stage.GetAttributeAtPath(sample_mode_include)
            if attr and is_attr_locked(attr):
                continue
            elif attr and attr.GetTypeName() == Sdf.ValueTypeNames.Int:
                attr.Set(int(mode))
            else:
                logger.warning(
                    f"Invalid attribute type for sample mode include {sample_mode_include}: {attr.GetTypeName()}"
                )

    return index_voxel_size


def process_configure_xac_shader_apis_temporal(
    prim: Usd.Prim, timecode: Usd.TimeCode, next_timecode: Usd.TimeCode, raw_timecode: Usd.TimeCode
):
    """
    Updates XAC shader temporal interpolation parameters.

    This function configures the shader parameters needed for field interpolation:
    - attrib_idx: int2 with (current_idx, next_idx) - set next to -1 to disable interpolation
    - time_codes: float3 with (current, next, raw) time code values

    Parameters
    ----------
    prim : Usd.Prim
        The volume prim to configure
    timecode : Usd.TimeCode
        Current snapped timecode (from get_bracketing_time_samples_for_prim)
    next_timecode : Usd.TimeCode or None
        Next bracketing timecode for interpolation (None if not available)
    raw_timecode : Usd.TimeCode
        Original timeline timecode before snapping
    """
    # Check if field interpolation is enabled
    enable_field_interpolation = (
        prim.HasAPI(cae_viz.OperatorTemporalAPI)
        and cae_viz.OperatorTemporalAPI(prim).GetEnableFieldInterpolationAttr().Get()
    )

    # Get field selection instances to determine attribute indices
    instance_names = usd_utils.get_instances(prim, "CaeVizFieldSelectionAPI")
    num_fields = len(instance_names)

    # Calculate time code values
    current_tc = float(timecode.GetValue())
    next_tc = float(next_timecode.GetValue()) if next_timecode else current_tc
    raw_tc = float(raw_timecode.GetValue())

    time_codes = Gf.Vec3f(current_tc, next_tc, raw_tc)

    stage = prim.GetStage()
    xac_instance_names = usd_utils.get_instances(prim, "CaeVizConfigureXACShaderAPI")
    for xac_instance_name in xac_instance_names:
        if xac_instance_name not in instance_names:
            logger.warning(
                f"CaeVizConfigureXACShaderAPI instance {xac_instance_name} not found in CaeVizFieldSelectionAPI instances"
            )
            continue

        # Calculate attribute indices
        current_attrib_idx = instance_names.index(xac_instance_name)

        # Set next to -1 to disable interpolation, or +num_fields to enable it
        next_attrib_idx = current_attrib_idx + num_fields if (enable_field_interpolation and next_timecode) else -1
        attrib_idx = Gf.Vec2i(current_attrib_idx, next_attrib_idx)

        xac_instance = cae_viz.ConfigureXACShaderAPI(prim, xac_instance_name)
        for attrib_idx_include in xac_instance.GetAttribIdxIncludesRel().GetForwardedTargets():
            attr = stage.GetAttributeAtPath(attrib_idx_include)
            if attr and is_attr_locked(attr):
                continue
            elif attr and attr.GetTypeName() == Sdf.ValueTypeNames.Int2:
                attr.Set(attrib_idx)

        for time_codes_include in xac_instance.GetTimeCodesIncludesRel().GetForwardedTargets():
            attr = stage.GetAttributeAtPath(time_codes_include)
            if attr and is_attr_locked(attr):
                continue
            elif attr and attr.GetTypeName() == Sdf.ValueTypeNames.Float3:
                attr.Set(time_codes)


def apply_field_mapping(prim: Usd.Prim, field_name: str, f_array):
    """
    Apply field mapping to an array if FieldMappingAPI is present.

    Parameters
    ----------
    prim : Usd.Prim
        The prim to check for FieldMappingAPI
    field_name : str
        The name of the field/instance
    f_array : array-like
        The array to remap

    Returns
    -------
    array-like
        The remapped array, or the original array if no mapping is present or if mapping is invalid
    """
    if prim.HasAPI(cae_viz.FieldMappingAPI, field_name):
        field_mapping_api = cae_viz.FieldMappingAPI(prim, field_name)
        fm_domain = field_mapping_api.GetDomainAttr().Get()
        fm_range = field_mapping_api.GetRangeAttr().Get()
        if fm_domain and fm_range and fm_domain[0] <= fm_domain[1]:
            return array_utils.remap_array(f_array, tuple(fm_domain), tuple(fm_range))
        elif fm_domain:
            logger.error(f"Invalid domain {fm_domain} for field {field_name}")
    return f_array


def process_field_selection_apis(
    prim: Usd.Prim, dataset: dav.Dataset, *, include_fields: set[str] = set(), exclude_fields: set[str] = set()
):
    """
    Processes the CaeVizFieldSelectionAPI schemas on the given prim and populates the primvars based on the fields.

    For every instance of CaeVizFieldSelectionAPI, the primvars are populated based on the fields specified in the
    instance name.

    Parameters
    ----------
    prim : Usd.Prim
        The prim to process
    dataset : dav.Dataset
        The dataset to populate the primvars into
    include_fields : set[str], optional
        A set of field names to include even if FieldSelectionAPI is not present for them.
    exclude_fields : set[str], optional
        A set of field names to exclude from processing
    """
    prim_rt = usd_utils.get_prim_rt(prim)
    pv_api = UsdGeomRT.PrimvarsAPI(prim_rt)

    instance_names = set(usd_utils.get_instances(prim, "CaeVizFieldSelectionAPI"))

    for field_name in dataset.get_field_names():
        if field_name in exclude_fields:
            continue

        if field_name not in instance_names and field_name not in include_fields:
            continue

        field = dataset.get_field(field_name)
        if field.association == dav.AssociationType.VERTEX:
            interpolation = UsdGeomRT.Tokens.vertex
        elif field.association == dav.AssociationType.CELL:
            interpolation = UsdGeomRT.Tokens.uniform
        else:
            raise ValueError(f"Unsupported association type {field.association} for primvar {field_name}")

        f_array = cae_dav.fetch_data(dataset, field_name)
        assert f_array is not None, f"Field {field_name} not found or cannot be passed as primvar"

        if f_array.ndim > 2:
            logger.error(f"Unsupported array shape {f_array.shape} for primvar {field_name}")
            continue

        # process field mapping API if present
        f_array = apply_field_mapping(prim, field_name, f_array)

        nb_comps = f_array.shape[1] if f_array.ndim > 1 else 1
        if nb_comps == 1:
            if f_array.dtype in [np.float32, np.float64]:
                pv_type = SdfRT.ValueTypeNames.FloatArray
            elif f_array.dtype in [np.int32, np.int64]:
                pv_type = SdfRT.ValueTypeNames.IntArray
            else:
                logger.error(f"Unsupported dtype {f_array.dtype} for primvar {field_name}")
                continue
        elif nb_comps == 3:
            if f_array.dtype in [np.float32, np.float64]:
                pv_type = SdfRT.ValueTypeNames.Float3Array
            elif f_array.dtype in [np.int32, np.int64]:
                pv_type = SdfRT.ValueTypeNames.Int3Array
            else:
                logger.error(f"Unsupported dtype {f_array.dtype} for primvar {field_name}")
                continue
        else:
            logger.warning(f"Unsupported number of components {nb_comps} for primvar {field_name}")
            continue

        logger.info(f"Creating primvar {field_name} with shape {f_array.shape} and dtype {f_array.dtype}")
        if not pv_api.GetPrimvar(field_name).GetAttr():
            pv = pv_api.CreatePrimvar(field_name, pv_type, interpolation)
        else:
            pv = pv_api.GetPrimvar(field_name)
            pv.SetInterpolation(interpolation)

        set_array_attribute(pv.GetAttr(), f_array)


def process_widths(prim: Usd.Prim, dataset: dav.Dataset, fixed_width: float):
    if prim.HasAPI(cae_viz.FieldSelectionAPI, "widths") and dataset.has_field("widths"):
        return  # widths are/will-be passed as primvar by process_field_selection_apis

    prim_rt = usd_utils.get_prim_rt(prim)
    pv_api = UsdGeomRT.PrimvarsAPI(prim_rt)

    # Once a primvar is created, USDRT doesn't like changing its interpolation type.
    # So, as soon as CaeFieldSelectionAPI:widths is specified, we always use vertex interpolation.
    # Otherwise, we always use constant interpolation regardless of whether constant or vertex-specific
    # widths are being used.
    if prim.HasAPI(cae_viz.FieldSelectionAPI, "widths"):
        pv = pv_api.CreatePrimvar("widths", SdfRT.ValueTypeNames.FloatArray, UsdGeomRT.Tokens.vertex)
        widths = wp.full((dataset.get_num_points(), 1), fixed_width, dtype=wp.float32, device=dataset.device)
        set_array_attribute(pv.GetAttr(), widths)
    else:
        pv = pv_api.CreatePrimvar("widths", SdfRT.ValueTypeNames.FloatArray, UsdGeomRT.Tokens.constant)
        widths = wp.array([fixed_width], dtype=wp.float32, device=dataset.device)
        set_array_attribute(pv.GetAttr(), widths)


async def get_selected_dataset(
    prim: Usd.Prim,
    instance_name: str,
    *,
    timeCode: Usd.TimeCode,
    device: str,
    needs_topology: bool = True,
    needs_geometry: bool = True,
) -> dav.Dataset:
    """
    Returns the selected dataset for the given instance name.

    Parameters
    ----------
    prim : Usd.Prim
        The prim to get the selected dataset from
    instance_name : str
        The instance name of the DatasetSelectionAPI
    timeCode : Usd.TimeCode
        The time code to get the dataset for
    device : str
        The device to get the dataset for
    needs_topology : bool, optional
        Whether the dataset needs topology information, by default True
    needs_geometry : bool, optional
        Whether the dataset needs geometry information, by default True

    Returns
    -------
    dav.Dataset
        The selected dataset
    """

    # TODO: we should manage caching here. A dataset cache with a consumer link to the "prim" and "instance_name"
    # can be setup. That way the cache is released if prim no longer needs it.

    # TODO: this should handle CaeVizDatasetTransformingAPI instances as well and return a transformed dataset.
    # DAV should add support for a data model for xformed datasets.

    if not prim.HasAPI(cae_viz.DatasetSelectionAPI, instance_name):
        raise ValueError(
            f"Prim {prim.GetPath()} does not have CaeVizDatasetSelectionAPI with instance name '{instance_name}'"
        )

    ds_api = cae_viz.DatasetSelectionAPI(prim, instance_name)
    dataset_prims = usd_utils.get_target_prims(ds_api.GetPrim(), ds_api.GetTargetRel().GetName())
    assert len(dataset_prims) > 0, f"No target prims found for DatasetSelectionAPI at {ds_api.GetPath()}"

    datasets = []
    for dataset_prim in dataset_prims:
        if not dataset_prim:
            continue
        dataset = await cae_dav.get_dataset(
            dataset_prim, timeCode, device=device, needs_topology=needs_topology, needs_geometry=needs_geometry
        )
        datasets.append(dataset.shallow_copy())
    try:
        return dav.DatasetCollection.from_datasets(datasets) if len(datasets) > 1 else datasets[0]
    except ValueError as e:
        raise ValueError(
            f"Failed to create DatasetCollection from datasets for DatasetSelectionAPI at {ds_api.GetPath()}: {e}"
        ) from e


async def add_selected_fields(
    dataset: dav.Dataset, prim: Usd.Prim, timeCode: Usd.TimeCode, required_instances: set[str] = set()
) -> dav.Dataset:
    """
    Utility to populate field selections in a dataset based on FieldSelectionAPI schemas on a prim.

    Parameters
    ----------
    dataset : dav.Dataset
        The dataset to populate field selections into
    prim : Usd.Prim
        The prim containing FieldSelectionAPI schemas
    timeCode : Usd.TimeCode
        The time code for which to fetch the fields
    required_instances : set[str], optional
        A set of instance names that are required; if fetching these fields fails, an exception is raised, by default set()

    Returns
    -------
    dav.Dataset
        The updated dataset with selected fields added
    """
    instance_names = usd_utils.get_instances(prim, "CaeVizFieldSelectionAPI")
    for n in required_instances:
        if n not in instance_names:
            raise usd_utils.QuietableException(f"Required field '{n}' not found on prim {prim.GetPath()}")

    for instance_name in instance_names:
        try:
            field = await get_selected_field(prim, instance_name, timeCode=timeCode, device=dataset.device)
            dataset.add_field(instance_name, field)
        except usd_utils.QuietableException as e:
            if instance_name in required_instances:
                raise  # required field
            else:
                # TODO: make this a once-only warning
                logger.info(f"Skipping optional field '{instance_name}' due to error: {e}")
    return dataset


async def get_selected_dataset_with_fields(
    prim: Usd.Prim,
    instance_name: str,
    *,
    timeCode: Usd.TimeCode,
    device: str,
    needs_topology: bool = True,
    needs_geometry: bool = True,
    required_fields: set[str] = set(),
) -> dav.Dataset:
    """
    Returns the selected dataset for the given instance name with the required fields added.
    Parameters
    ----------
    prim : Usd.Prim
        The prim to get the selected dataset from
    instance_name : str
        The instance name of the DatasetSelectionAPI
    timeCode : Usd.TimeCode
        The time code to get the dataset for
    device : str
        The device to get the dataset for
    needs_topology : bool, optional
        Whether the dataset needs topology information, by default True
    needs_geometry : bool, optional
        Whether the dataset needs geometry information, by default True
    required_fields : set[str], optional
        A set of field names that are required; if fetching these fields fails, an exception is raised, by default set()

    Returns
    -------
    dav.Dataset
        The selected dataset with the required fields added
    """
    dataset = await get_selected_dataset(
        prim,
        instance_name,
        timeCode=timeCode,
        device=device,
        needs_topology=needs_topology,
        needs_geometry=needs_geometry,
    )
    return await add_selected_fields(dataset, prim, timeCode, required_fields)


async def get_selected_field(prim: Usd.Prim, instance_name: str, *, timeCode: Usd.TimeCode, device: str) -> dav.Field:
    """
    Utility to get a field from a FieldSelectionAPI on a prim. This constructs dav.Field objects
    while implicitly handling magnitude and component selection as specified on the FieldSelectionAPI.

    Parameters
    ----------
    prim : Usd.Prim
        The prim containing the FieldSelectionAPI schema
    instance_name : str
        The instance name of the FieldSelectionAPI
    timeCode : Usd.TimeCode
        The time code for which to fetch the field
    device : str
        The device on which to load the field

    Returns
    -------
    dav.Field
        The loaded field
    """
    if not prim.HasAPI(cae_viz.FieldSelectionAPI, instance_name):
        raise ValueError(
            f"Prim {prim.GetPath()} does not have CaeVizFieldSelectionAPI with instance name '{instance_name}'"
        )

    f_selection_api = cae_viz.FieldSelectionAPI(prim, instance_name)
    field_prims = usd_utils.get_target_prims(f_selection_api.GetPrim(), f_selection_api.GetTargetRel().GetName())

    # this can never be since get_target_prims will throw usd_utils.QuietableException if arrays are missing.
    assert len(field_prims) > 0, f"No target prims found for FieldSelectionAPI at {f_selection_api.GetPath()}"

    # validate field associations
    associations = [cae.FieldArray(fa).GetFieldAssociationAttr().Get(timeCode) for fa in field_prims]
    if not all(assoc == associations[0] for assoc in associations):
        raise ValueError("Multiple different associations found; only one is supported currently.")

    if associations[0] not in [cae.Tokens.vertex, cae.Tokens.cell]:
        raise ValueError(
            f"Only vertex and cell associations are supported for FieldSelectionAPI:{instance_name} at {f_selection_api.GetPath()}. Got {associations[0]}."
        )

    selection_mode = f_selection_api.GetModeAttr().Get(timeCode)
    # snap time-code
    timeCode = usd_utils.snap_time_code_to_prims(field_prims, timeCode)

    cache_key = {
        "field_prims": [str(prim.GetPath()) for prim in field_prims],
    }
    cache_state = {
        "device": str(device),
        "selection_mode": str(selection_mode),
    }
    if cached_field := cache.get(str(cache_key), cache_state, timeCode=timeCode):
        return cached_field

    if selection_mode == cae_viz.Tokens.unchanged:
        field = await cae_dav.get_field(field_prims, timeCode=timeCode, device=device)
    elif selection_mode == cae_viz.Tokens.vector_magnitude:
        field = await cae_dav.get_vector_magnitude_field(field_prims, timeCode=timeCode, device=device)
    elif selection_mode == cae_viz.Tokens.selected_component:
        field = await cae_dav.get_selected_component_field(
            field_prims,
            timeCode=timeCode,
            device=device,
            component_index=f_selection_api.GetComponentIndexAttr().Get(timeCode),
        )
    else:
        raise ValueError(f"Unsupported FieldSelectionAPI mode '{selection_mode}'.")

    cache.put(str(cache_key), field, cache_state, timeCode=timeCode, sourcePrims=field_prims, consumerPrims=[prim])
    return field


class VoxelizationParameters:
    min_ijk: wp.vec3i
    max_ijk: wp.vec3i
    voxel_size: wp.vec3f

    def __init__(self, min_ijk: wp.vec3i, max_ijk: wp.vec3i, voxel_size: wp.vec3f):
        self.min_ijk = min_ijk
        self.max_ijk = max_ijk
        self.voxel_size = voxel_size


def get_voxelization_parameters(
    prim: Usd.Prim, instance_name: str, data_bounds: Gf.Range3d, timeCode: Usd.TimeCode
) -> VoxelizationParameters:
    """
    Returns the voxelization parameters for the given prim and data bounds.

    Parameters
    ----------
    prim : Usd.Prim
        The prim to get the voxelization parameters from
    instance_name : str
        The instance name of the DatasetVoxelizationAPI
    data_bounds : Gf.Range3d
        The bounds of the data to voxelize
    timeCode : Usd.TimeCode
        The time code to get the voxelization parameters for

    Returns
    -------
    VoxelizationParameters
        The voxelization parameters
    """
    if data_bounds.IsEmpty():
        raise ValueError("Data bounds are empty. Data bounds must be specified.")

    nvdb_api = cae_viz.DatasetVoxelizationAPI(prim, instance_name)
    # if ROI is specified, intersect the data bounds with the ROI bounds
    if roi_prim := usd_utils.get_target_prim(prim, nvdb_api.GetRoiRel().GetName(), quiet=True):
        if roi_bounds := usd_utils.get_bounds(roi_prim, timeCode, quiet=True):
            data_bounds.IntersectWith(roi_bounds)
            if data_bounds.IsEmpty():
                raise ValueError(
                    "Data bounds are empty after intersecting with ROI bounds. ROI may not intersect with data bounds."
                )

    data_bounds = np.array([data_bounds.GetMin(), data_bounds.GetMax()], dtype=np.float32)

    inflate_bounds = nvdb_api.GetInflateBoundsAttr().Get(timeCode)
    if inflate_bounds > 0:
        size = data_bounds[1] - data_bounds[0]
        inflation = size * inflate_bounds * 0.01
        data_bounds[0] -= inflation * 0.5
        data_bounds[1] += inflation * 0.5

    vox_size_mode = nvdb_api.GetVoxelSizeModeAttr().Get(timeCode)
    if vox_size_mode == cae_viz.Tokens.maxResolution:
        max_resolution = int(nvdb_api.GetMaxResolutionAttr().Get(timeCode))
        min_bounds = data_bounds[0]
        max_bounds = data_bounds[1]
        bounds_range = max_bounds - min_bounds

        # Start with initial voxel size based on max_resolution
        voxel_size = np.max(bounds_range) / max_resolution

        # Iteratively compute ijk bounds and adjust voxel_size if needed
        max_iterations = 10
        for iteration in range(max_iterations):
            # Compute ijk bounds using floor/ceil
            # Volume will extend from [voxel_size * ijk_min] to [voxel_size * ijk_max]
            ijk_min_float = min_bounds / voxel_size
            ijk_max_float = max_bounds / voxel_size

            ijk_min_arr = np.floor(ijk_min_float).astype(int)
            ijk_max_arr = np.ceil(ijk_max_float).astype(int)

            # Pad if bounds exactly align with voxel boundaries
            for i in range(3):
                if np.isclose(ijk_min_float[i], ijk_min_arr[i]):
                    ijk_min_arr[i] -= 1
                if np.isclose(ijk_max_float[i], ijk_max_arr[i]):
                    ijk_max_arr[i] += 1

            # Check dimensions (number of cells = ijk_max - ijk_min)
            dims = ijk_max_arr - ijk_min_arr
            max_dim = np.max(dims)

            if max_dim <= max_resolution:
                # Success! Verify that the volume encloses the bounds
                # Volume extends from [voxel_size * ijk_min] to [voxel_size * ijk_max]
                volume_min = voxel_size * ijk_min_arr
                volume_max = voxel_size * ijk_max_arr

                # Check enclosure with a small tolerance for floating point errors
                tolerance = 1e-6 * voxel_size
                if not np.all(volume_min <= min_bounds + tolerance):
                    raise RuntimeError(
                        f"Volume min {volume_min} does not enclose bounds min {min_bounds}\n"
                        f"  voxel_size={voxel_size}, ijk_min={ijk_min_arr}"
                    )
                if not np.all(volume_max >= max_bounds - tolerance):
                    raise RuntimeError(
                        f"Volume max {volume_max} does not enclose bounds max {max_bounds}\n"
                        f"  voxel_size={voxel_size}, ijk_max={ijk_max_arr}"
                    )

                break

            # Need to increase voxel_size to reduce dimensions
            scale_factor = max_dim / max_resolution
            voxel_size *= scale_factor * 1.01  # Add 1% margin to ensure convergence
        else:
            raise RuntimeError(f"Failed to compute valid voxelization parameters after {max_iterations} iterations")

        ijk_min = wp.vec3i(*ijk_min_arr.tolist())
        ijk_max = wp.vec3i(*ijk_max_arr.tolist())

        return VoxelizationParameters(ijk_min, ijk_max, wp.vec3f(voxel_size, voxel_size, voxel_size))

    elif vox_size_mode == cae_viz.Tokens.voxelSize:
        voxel_size = nvdb_api.GetVoxelSizeAttr().Get(timeCode)
        voxel_size = np.array(voxel_size, dtype=np.float32)
        ijk_min = wp.vec3i(np.floor(data_bounds[0] / voxel_size).astype(int))
        ijk_max = wp.vec3i(np.ceil(data_bounds[1] / voxel_size).astype(int))
        return VoxelizationParameters(ijk_min, ijk_max, wp.vec3f(voxel_size))
    else:
        raise ValueError(f"Unsupported NanoVDB voxel size mode: {vox_size_mode}")


async def get_input_dataset(
    prim: Usd.Prim,
    instance_name: str,
    *,
    timeCode: Usd.TimeCode,
    device: str,
    needs_topology: bool = True,
    needs_geometry: bool = True,
    needs_fields: bool = True,
    required_fields: set[str] = set(),
) -> dav.Dataset:
    """
    Returns the input dataset for the given instance name. Unlike get_selected_dataset, this will process
    any relevant api schemas on the prim for preparing the input dataset.

    This function automatically handles:
    - Loading the dataset specified by DatasetSelectionAPI
    - Applying DatasetGaussianSplattingAPI transformations (if present)
    - Applying DatasetVoxelizationAPI transformations (if present)
    - Applying DatasetTransformingAPI transformations (if present)
    - Loading all fields specified by FieldSelectionAPI instances

    Caching:
    --------
    Results are cached based on prim path and instance name. The cache is automatically invalidated when:
    - The prim or any of its properties change
    - Any of the dataset prims change
    - The cache mode setting changes

    Note: The cache invalidation is prim-level. Any change to the prim (including unrelated properties)
    will invalidate the cache. This ensures correctness but may result in more cache misses than a
    fine-grained state-based approach.

    Parameters
    ----------
    prim : Usd.Prim
        The prim to get the input dataset from
    instance_name : str
        The instance name of the DatasetSelectionAPI
    timeCode : Usd.TimeCode
        The time code to get the input dataset for
    device : str
        The device to get the input dataset for
    needs_topology : bool, optional
        Whether the caller needs access to dataset topology information, by default True
    needs_geometry : bool, optional
        Whether the caller needs access to dataset geometry information, by default True
    needs_fields : bool, optional
        Whether the caller needs access to dataset fields, by default True
    required_fields : set[str], optional
        A set of field names that are required; if fetching these fields fails, an exception is raised, by default set()

    Returns
    -------
    dav.Dataset
        The input dataset with all requested transformations and fields applied

    Raises
    ------
    ValueError
        If the prim does not have a DatasetSelectionAPI with the given instance name
    """
    cache_key = f"[viz:get_input_dataset]::{instance_name}:{prim.GetPath()}"
    ds = cache.get(cache_key, timeCode=timeCode)
    if ds:
        return ds

    if not prim.HasAPI(cae_viz.DatasetSelectionAPI, instance_name):
        raise ValueError(
            f"Prim {prim.GetPath()} does not have CaeVizDatasetSelectionAPI with instance name '{instance_name}'"
        )

    ds_api = cae_viz.DatasetSelectionAPI(prim, instance_name)
    dataset_prims = usd_utils.get_target_prims(ds_api.GetPrim(), ds_api.GetTargetRel().GetName())
    assert len(dataset_prims) > 0, f"No target prims found for DatasetSelectionAPI at {ds_api.GetPath()}"

    if prim.HasAPI(cae_viz.DatasetTransformingAPI, instance_name) or prim.HasAPI(
        cae_viz.DatasetGaussianSplattingAPI, instance_name
    ):
        needs_geometry = True

    if prim.HasAPI(cae_viz.DatasetVoxelizationAPI, instance_name) and not prim.HasAPI(
        cae_viz.DatasetGaussianSplattingAPI, instance_name
    ):
        # We need topology for voxelization to work, but not if gaussian splatting is also applied.
        needs_topology = True

    # if gaussian splatting is applied, needs_fields is true and we have any non-vertex centered fields, then we need topology
    if prim.HasAPI(cae_viz.DatasetGaussianSplattingAPI, instance_name) and (not needs_topology):
        # TODO: make this actually check for non-vertex centered fields
        needs_topology = True

    if needs_fields:
        dataset = await get_selected_dataset_with_fields(
            prim,
            instance_name,
            timeCode=timeCode,
            device=device,
            needs_topology=needs_topology,
            needs_geometry=needs_geometry,
            required_fields=required_fields,
        )
    else:
        dataset = await get_selected_dataset(
            prim,
            instance_name,
            timeCode=timeCode,
            device=device,
            needs_topology=needs_topology,
            needs_geometry=needs_geometry,
        )

    if prim.HasAPI(cae_viz.DatasetVoxelizationAPI, instance_name):
        dataset = await _process_dataset_voxelization(dataset, prim, instance_name, timeCode)
    if prim.HasAPI(cae_viz.DatasetTransformingAPI, instance_name):
        dataset = await _process_dataset_transforming(dataset, prim, instance_name, timeCode)

    # for now, if "prim" changes, just drop the cache. In future we may add support
    # for selectively invalidating the cache.
    cache.put(cache_key, dataset, timeCode=timeCode, sourcePrims=dataset_prims + [prim], consumerPrims=[prim])
    return dataset


async def _process_dataset_transforming(
    dataset: dav.Dataset, prim: Usd.Prim, instance_name: str, timeCode: Usd.TimeCode
) -> dav.Dataset:
    """
    Processes the dataset transforming API on the dataset.

    Parameters
    ----------
    dataset : dav.Dataset
        The dataset to process
    prim : Usd.Prim
        The prim to process
    instance_name : str
    """
    assert prim.HasAPI(
        cae_viz.DatasetTransformingAPI, instance_name
    ), f"Prim {prim.GetPath()} does not have CaeVizDatasetTransformingAPI with instance name '{instance_name}'"
    transforming_api = cae_viz.DatasetTransformingAPI(prim, instance_name)
    return dataset


async def _process_dataset_gaussian_splatting(
    dataset: dav.Dataset, prim: Usd.Prim, instance_name: str, timeCode: Usd.TimeCode, voxel_size: wp.vec3f
) -> dav.Dataset:
    """
    Processes the dataset gaussian splatting API on the dataset.

    Parameters
    ----------
    dataset : dav.Dataset
        The dataset to process
    prim : Usd.Prim
        The prim to process
    instance_name : str
    voxel_size : wp.vec3f
        The voxel size to use for the gaussian splatting
    """
    assert prim.HasAPI(
        cae_viz.DatasetGaussianSplattingAPI, instance_name
    ), f"Prim {prim.GetPath()} does not have CaeVizDatasetGaussianSplattingAPI with instance name '{instance_name}'"
    api = cae_viz.DatasetGaussianSplattingAPI(prim, instance_name)
    radius_factor = api.GetRadiusFactorAttr().Get(timeCode)
    radius = radius_factor * min(voxel_size)
    logger.info(f"Gaussian splatting radius: {radius:.3f}")
    sharpness = api.GetSharpnessAttr().Get(timeCode)
    with progress.ProgressContext("Executing DAV [point_splats]"):
        splatted_dataset = dav_point_splats.compute(dataset, radius, sharpness)
    # convert cell fields to point fields before splatting.
    for field_name in dataset.get_field_names():
        with progress.ProgressContext("Executing DAV [point_field]"):
            tmp_dataset = dav_point_field.compute(dataset, field_name, output_field_name="_point_field")
        splatted_dataset.add_field(field_name, tmp_dataset.get_field("_point_field"))
    return splatted_dataset


async def _process_dataset_voxelization(
    dataset: dav.Dataset, prim: Usd.Prim, instance_name: str, timeCode: Usd.TimeCode
) -> dav.Dataset:
    """
    Processes the dataset voxelization API on the dataset.

    Parameters
    ----------
    dataset : dav.Dataset
        The dataset to process
    prim : Usd.Prim
        The prim to process
    instance_name : str
    """
    # If no gaussian splatting is needed and every field is already a NanoVDB
    # volume, skip the entire voxelization pipeline.
    if not prim.HasAPI(cae_viz.DatasetGaussianSplattingAPI, instance_name):
        field_names = dataset.get_field_names()
        if field_names and all(isinstance(dataset.get_field(fn).get_data(), wp.Volume) for fn in field_names):
            logger.info(f"Skipping voxelization for dataset at {prim.GetPath()} since all fields are already volumes")
            return dataset

    with progress.ProgressContext("Executing DAV [compute bounds]"):
        data_bounds = dataset.get_bounds()
    data_bounds_r3d = Gf.Range3d(
        (data_bounds[0][0], data_bounds[0][1], data_bounds[0][2]),
        (data_bounds[1][0], data_bounds[1][1], data_bounds[1][2]),
    )
    vox_params = get_voxelization_parameters(prim, instance_name, data_bounds_r3d, timeCode)

    if prim.HasAPI(cae_viz.DatasetGaussianSplattingAPI, instance_name):
        # we need to voxelize the dataset after gaussian splatting
        dataset = await _process_dataset_gaussian_splatting(
            dataset, prim, instance_name, timeCode, vox_params.voxel_size
        )

    voxelized_dataset = None
    for field_name in dataset.get_field_names():
        # TODO process CaeFieldThresholdingAPI on the field to customize tiles for the genereated NanoVDB field.
        with progress.ProgressContext("Executing DAV [voxelization]"):
            vds = dav_voxelization.compute(
                dataset,
                field_name,
                min_ijk=vox_params.min_ijk,
                max_ijk=vox_params.max_ijk,
                voxel_size=vox_params.voxel_size,
                use_nanovdb=True,
                output_field_name=field_name,
            )
        if voxelized_dataset is None:
            voxelized_dataset = vds
        else:
            voxelized_dataset.add_field(field_name, vds.get_field(field_name))

    if voxelized_dataset is None:
        logger.info(f"No fields were voxelized for dataset at {prim.GetPath()}")

        # So we need to create an voxel grid without any fields.
        voxelized_dataset = dav_data_models_vtk_image_data.create_dataset(
            origin=wp.vec3f(0.0),
            spacing=wp.vec3f(*vox_params.voxel_size),
            extent_min=vox_params.min_ijk,
            extent_max=vox_params.max_ijk,
            device=dataset.device,
        )

    return voxelized_dataset


def set_array_attribute(attr: UsdRT.Attribute, array: array_utils.FieldArrayLike) -> None:
    """
    Sets the attribute value to the given array.
    """
    f_array = array_utils.IFieldArray.from_array(array)

    typename = attr.GetTypeName()
    assert typename.isArray, f"Attribute {attr.GetPath()} is not an array"
    scalar_type = typename.scalarType
    if scalar_type in [SdfRT.ValueTypeNames.Point3f, SdfRT.ValueTypeNames.Vector3f, SdfRT.ValueTypeNames.Float3]:
        _set_array_attribute(attr, array_utils.as_type(f_array, np.float32), VtRT.Vec3fArray)
    elif scalar_type in [SdfRT.ValueTypeNames.Point3d, SdfRT.ValueTypeNames.Vector3d, SdfRT.ValueTypeNames.Double3]:
        _set_array_attribute(attr, array_utils.as_type(f_array, np.float64), VtRT.Vec3dArray)
    elif scalar_type in [SdfRT.ValueTypeNames.Float2]:
        _set_array_attribute(attr, array_utils.as_type(f_array, np.float32), VtRT.Vec2fArray)
    elif scalar_type in [SdfRT.ValueTypeNames.Double2]:
        _set_array_attribute(attr, array_utils.as_type(f_array, np.float64), VtRT.Vec2dArray)
    elif scalar_type in [SdfRT.ValueTypeNames.Quatf, SdfRT.ValueTypeNames.Float4]:
        _set_array_attribute(attr, array_utils.as_type(f_array, np.float32), VtRT.Vec4fArray)
    elif scalar_type in [SdfRT.ValueTypeNames.Quatd, SdfRT.ValueTypeNames.Double4]:
        _set_array_attribute(attr, array_utils.as_type(f_array, np.float64), VtRT.Vec4dArray)
    elif scalar_type in [SdfRT.ValueTypeNames.Quath]:
        # Manually handle here, since we can't use _set_array_attribute because there's
        # no as_type() for half-floats.
        if f_array.device_id == -1:
            vt_array = VtRT.QuathArray(f_array.numpy())
        else:
            vt_array = VtRT.QuathArray(f_array)
        attr.Set(vt_array)
    elif scalar_type in [SdfRT.ValueTypeNames.Float]:
        _set_array_attribute(attr, array_utils.as_type(f_array, np.float32), VtRT.FloatArray)
    elif scalar_type in [SdfRT.ValueTypeNames.Double]:
        _set_array_attribute(attr, array_utils.as_type(f_array, np.float64), VtRT.DoubleArray)
    elif scalar_type == SdfRT.ValueTypeNames.Int:
        _set_array_attribute(attr, array_utils.as_type(f_array, np.int32), VtRT.IntArray)
    elif scalar_type == SdfRT.ValueTypeNames.Int64:
        _set_array_attribute(attr, array_utils.as_type(f_array, np.int64), VtRT.Int64Array)
    elif scalar_type == SdfRT.ValueTypeNames.UInt:
        _set_array_attribute(attr, array_utils.as_type(f_array, np.uint32), VtRT.UIntArray)
    elif scalar_type == SdfRT.ValueTypeNames.UInt64:
        _set_array_attribute(attr, array_utils.as_type(f_array, np.uint64), VtRT.UInt64Array)
    elif scalar_type == SdfRT.ValueTypeNames.Int2:
        _set_array_attribute(attr, array_utils.as_type(f_array, np.int32), VtRT.Vec2iArray)
    elif scalar_type == SdfRT.ValueTypeNames.Int3:
        _set_array_attribute(attr, array_utils.as_type(f_array, np.int32), VtRT.Vec3iArray)
    elif scalar_type == SdfRT.ValueTypeNames.Int4:
        _set_array_attribute(attr, array_utils.as_type(f_array, np.int32), VtRT.Vec4iArray)
    else:
        raise ValueError(f"Unsupported scalar type {scalar_type} for attribute {attr.GetPath()}")


def _set_array_attribute(attr: UsdRT.Attribute, f_array: array_utils.IFieldArray, attr_type) -> None:
    """
    Sets the attribute value to the given UsdRT.Array.
    """
    vt_array = array_utils.to_vtrt_array(f_array)
    attr.Set(vt_array)
    # attr.SyncDataToGpu()


def get_temporal_traits(prim: Usd.Prim, instance_name: str, attr_name: str) -> str:
    """
    Utility to get the temporal traits of a given attribute on a prim. This is used to determine whether
    we need to fetch topology/geometry information for a given prim when processing it.

    Parameters
    ----------
    prim : Usd.Prim
        The prim to get the temporal traits from
    instance_name : str
        The name of the API schema instance that contains the attribute
    attr_name : str
        The name of the attribute to get the temporal traits for
    Returns
    -------
    str
        The temporal traits of the attribute (e.g., "static", "varying", "undefined")
    """

    if not prim.HasAPI(cae_viz.DatasetTemporalTraitsAPI):
        return "undefined"

    temp_char_api = cae_viz.DatasetTemporalTraitsAPI(prim, instance_name)
    if attr_name == "topology":
        return temp_char_api.GetTopologyAttr().Get()
    elif attr_name == "geometry":
        return temp_char_api.GetGeometryAttr().Get()
    else:
        raise ValueError(
            f"Unsupported attribute name '{attr_name}' for temporal traits. Supported attributes are 'topology' and 'geometry'."
        )
