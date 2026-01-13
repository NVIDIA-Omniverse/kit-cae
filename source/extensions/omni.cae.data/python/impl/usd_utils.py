# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
  Module with utilities for making it easier to work with data delegate and USD prims.
  By design all methods raise exceptions instead of returning empty values or raising errors.
"""

__all__ = [
    "QuietableException",
    "quietable",
    "quietable_with_default",
    "assemble_vecN_arrays",
    "async_quietable",
    "async_quietable_with_default",
    "compute_and_set_range",
    "get_target_field_association",
    "get_target_field_associations",
    "get_target_paths",
    "get_target_path",
    "get_target_prim",
    "get_target_prims",
    "get_array",
    "get_arrays",
    "get_arrays_from_relationship",
    "get_array_from_relationship",
    "get_vecN_from_relationship",
    "get_vecN_from_relationships",
    "get_attribute",
    "get_prim_pxr",
    "get_field_name",
    "get_target_field_name",
    "get_target_field_names",
    "get_prim_at_path",
    "get_bounds",
    "get_bracketing_time_codes",
    "get_time_sample",
    "get_time_samples",
    "get_next_time_sample",
    "set_attribute",
    "ChangeTracker",
]

import asyncio
import bisect
import functools
import sys
import weakref
from logging import getLogger
from typing import Union

import numpy as np
from omni.cae.schema import cae
from pxr import Gf, Sdf, Tf, Usd, UsdGeom, UsdUtils
from usdrt import Rt
from usdrt import Usd as UsdRt

from .. import get_data_delegate_registry
from . import array_utils, cache, progress, range_utils
from .bindings import IFieldArray

logger = getLogger(__name__)


class QuietableException(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


def quietable(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        quiet = kwargs.pop("quiet", False)
        try:
            return func(*args, **kwargs)
        except QuietableException as e:
            if not quiet:
                raise
            logger.debug("Silenced exception %s", e)
        return None

    return wrapper


def quietable_with_default(val):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            quiet = kwargs.pop("quiet", False)
            try:
                return func(*args, **kwargs)
            except QuietableException as e:
                if not quiet:
                    raise
                logger.debug("Silenced exception %s", e)
            return val

        return wrapper

    return decorator


def async_quietable(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        quiet = kwargs.pop("quiet", False)
        try:
            return await func(*args, **kwargs)
        except QuietableException as e:
            if not quiet:
                raise
            logger.debug("Silenced exception %s", e)
        return None

    return wrapper


def async_quietable_with_default(val):
    async def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            quiet = kwargs.pop("quiet", False)
            try:
                return await func(*args, **kwargs)
            except QuietableException as e:
                if not quiet:
                    raise
                logger.debug("Silenced exception %s", e)
            return val

        return wrapper

    return decorator


@quietable
def get_target_paths(prim: Usd.Prim, relName: str) -> list[Sdf.Path]:
    if not prim:
        raise QuietableException("Invalid prim: %s" % prim)
    rel = prim.GetRelationship(relName)
    if not rel:
        raise QuietableException("Missing relationship '%s' on '%s'" % (relName, prim))
    targets = rel.GetForwardedTargets()
    if not targets:
        raise QuietableException("Missing targets on '%s" % rel)
    return targets


@quietable
def get_target_path(prim: Usd.Prim, relName: str) -> Sdf.Path:
    targets = get_target_paths(prim, relName)
    if len(targets) > 1:
        logger.warning("Only first target is being processed on %s.%s", prim, relName)
    return targets[0]


@quietable
def get_target_prim(prim: Usd.Prim, relName: str) -> Usd.Prim:
    path = get_target_path(prim, relName)
    if tprim := prim.GetStage().GetPrimAtPath(path):
        return tprim
    raise QuietableException(f"Missing target prim at path {path}")


@quietable_with_default([])
def get_target_prims(prim: Usd.Prim, relName: str) -> list[Usd.Prim]:
    paths = get_target_paths(prim, relName)
    stage = prim.GetStage()
    prims = []
    for p in paths:
        if tprim := stage.GetPrimAtPath(p):
            prims.append(tprim)
        else:
            raise QuietableException(f"Missing target prim at path {p}")
    if not prims:
        raise QuietableException(f"Missing valid targets prim at path {prim}->{relName}")
    return prims


@quietable
def get_target_field_association(prim: Usd.Prim, relName: str) -> str:
    field_prim = get_target_prim(prim, relName)
    if not field_prim.IsA(cae.FieldArray):
        raise QuietableException(f"Target prim {field_prim} is not a FieldArray")
    field_array = cae.FieldArray(field_prim)
    return field_array.GetFieldAssociationAttr().Get()


@quietable
def get_target_field_associations(prim: Usd.Prim, relNames: list[str]) -> list[str]:
    return [get_target_field_association(prim, relName) for relName in relNames]


@async_quietable
async def get_array(prim: Usd.Prim, timeCode: Usd.TimeCode = Usd.TimeCode.Default()) -> IFieldArray:
    """Given a FieldArray prim, fetch the array from the data delegate registry."""
    if not prim:
        raise QuietableException("Invalid prim %s" % prim)

    if not prim.IsA(cae.FieldArray):
        raise QuietableException("FieldArray prim expected at %s" % prim)

    registry = get_data_delegate_registry()
    label = f".../{prim.GetParent().GetName()}/{prim.GetName()}"
    with progress.ProgressContext(f"Fetching array for {label}"):
        array: IFieldArray = await registry.get_field_array_async(prim, timeCode)

    if array is None:
        raise QuietableException("Failed to fetch array for %s" % prim)
    logger.info("Got array %s (device_ordinal=%d)", prim, array.device_id)
    return array


@async_quietable
@progress.progress_context("Fetching arrays")
async def get_arrays(prims: list[Usd.Prim], timeCode: Usd.TimeCode = Usd.TimeCode.Default()) -> list[IFieldArray]:
    """Given a list of FieldArray prims, fetch the arrays from the data delegate registry."""
    arrays = []
    for idx, p in enumerate(prims):
        with progress.ProgressContext(
            f"Fetching array {idx} of {len(prims)}", shift=idx / len(prims), scale=1.0 / len(prims)
        ):
            arrays.append(await get_array(p, timeCode))
    return arrays


@async_quietable
@progress.progress_context("Fetching arrays from relationship")
async def get_arrays_from_relationship(
    prim: Usd.Prim, relName: str, timeCode=Usd.TimeCode.Default()
) -> list[IFieldArray]:
    """
    Given any prim, looks up relationship targets for  given relationship and fetches arrays from those.
    The relationship targets are expected to be FieldArray prims.
    """
    targets = get_target_prims(prim, relName)
    return await get_arrays(targets, timeCode)


@async_quietable
async def get_array_from_relationship(prim: Usd.Prim, relName: str, timeCode=Usd.TimeCode.Default()) -> IFieldArray:
    target = get_target_prim(prim, relName)
    return await get_array(target, timeCode)


@async_quietable
async def get_vecN_from_relationship(
    prim: Usd.Prim, relName: str, numComponents: int, timeCode=Usd.TimeCode.Default()
) -> IFieldArray:
    """
    A convenience method. Same as get_arrays_from_relationship except returns a multicomponent array for requested
    number of components if possible.
    """
    arrays = await get_arrays_from_relationship(prim, relName, timeCode)
    return await assemble_vecN_arrays(arrays, numComponents)


@async_quietable
async def assemble_vecN_arrays(arrays: list[IFieldArray], numComponents: int) -> IFieldArray:
    """
    Given a list of arrays, attempts to assemble them into a single multicomponent array.
    The following cases are supported:
    1. If there is a single array with shape (M, N) where N == numComponents, return that array.
    2. If there is a single array with shape (M,) and numComponents == 1, return that array.
    3. If there are numComponents arrays each with shape (M,), stack them to form an array with shape (M, N).
    """
    if len(arrays) == 1 and arrays[0].ndim == 2 and arrays[0].shape[1] == numComponents:
        return arrays[0]
    elif len(arrays) == 1 and arrays[0].ndim == 1 and numComponents == 1:
        return arrays[0]
    elif len(arrays) == numComponents:
        array = await asyncio.to_thread(array_utils.column_stack, arrays)
        if array.ndim == 2 and array.shape[1] == numComponents:
            return array
        else:
            raise QuietableException(
                f"Failed to assemble vecN array: expected shape (_, {numComponents}), got {array.shape}"
            )
    raise QuietableException(f"Failed to fetch {numComponents} components")


@async_quietable
async def get_vecN_from_relationships(
    prim: Usd.Prim, relNames: list[str], numComponents: int, timeCode=Usd.TimeCode.Default()
) -> IFieldArray:
    """
    A convenience method. Same as get_arrays_from_relationship except returns a multicomponent array for requested
    number of components if possible.
    """
    arrays = []
    for relName in relNames:
        arr = await get_arrays_from_relationship(prim, relName, timeCode)
        arrays += arr
    return await assemble_vecN_arrays(arrays, numComponents)


@quietable
def get_attribute(prim: Usd.Prim, attrName: str, timeCode=Usd.TimeCode.Default()) -> any:
    if not prim.HasAttribute(attrName):
        raise QuietableException(f"Missing attribute {prim}.{attrName}")
    attr = prim.GetAttribute(attrName)
    if not attr.HasAuthoredValue() and not attr.HasFallbackValue():
        raise QuietableException(f"Missing authored/default value for attribute {prim}.{attrName}")
    return attr.Get(timeCode.GetValue())


@quietable
def get_prim_pxr(prim: Union[Usd.Prim, UsdRt.Prim]) -> Usd.Prim:
    if prim and isinstance(prim, UsdRt.Prim):
        stage_id = prim.GetStage().GetStageId()
        cache = UsdUtils.StageCache.Get()
        stage: Usd.Stage = cache.Find(cache.Id.FromLongInt(stage_id))
        if not stage:
            raise QuietableException(f"Failed to locate PXR::UsdStage with id {stage_id}")
        return stage.GetPrimAtPath(str(prim.GetPath()))
    elif prim and isinstance(prim, Usd.Prim):
        return prim
    else:
        raise QuietableException(f"Invalid prim {prim}")


@quietable
def get_field_name(dataset_prim: Usd.Prim, field_prim: Usd.Prim) -> str:
    """Returns field relationship name (without the `field:` prefix)"""
    field_prim_path = field_prim.GetPath()

    for rel in dataset_prim.GetRelationships():
        if rel.GetNamespace().startswith("field") and field_prim_path in rel.GetTargets():
            return str(rel.GetName())[len("field:") :]
    raise QuietableException("%s is not a 'field:' on %s" % (field_prim, dataset_prim))


@quietable
def get_target_field_name(prim: Usd.Prim, relName: str, dataset_prim: Usd.Prim) -> str:
    """Returns field relationship name (without the `field:` prefix)"""
    field_prim = get_target_prim(prim, relName)
    return get_field_name(dataset_prim, field_prim)


@quietable_with_default([])
def get_target_field_names(prim: Usd.Prim, relName: str, dataset_prim: Usd.Prim) -> list[str]:
    """Returns field relationship names (without the `field:` prefix)"""
    field_prims = get_target_prims(prim, relName)
    return [get_field_name(dataset_prim, field_prim) for field_prim in field_prims]


@quietable
def get_prim_at_path(stage: Usd.Stage, path: str):
    prim = stage.GetPrimAtPath(path)
    if not prim:
        raise QuietableException(f"Prim not found at path {path}")
    return prim


@quietable_with_default(Gf.Range3d())
def get_bounds(prim: Usd.Prim, timeCode=Usd.TimeCode.Default()) -> Gf.Range3d:
    if not prim or not prim.IsA(UsdGeom.Boundable):
        raise QuietableException("Prim is not boundable")

    boundable = UsdGeom.Boundable(prim)
    bounds: Gf.BBox3d = boundable.ComputeLocalBound(timeCode, UsdGeom.Tokens.default_)
    return bounds.ComputeAlignedRange()


@async_quietable
async def compute_and_set_range(
    attr: Union[Usd.Attribute, UsdRt.Attribute],
    dataset: Usd.Prim,
    field_name_or_names: Union[str, list[str]],
    timeCode=Usd.TimeCode.EarliestTime(),
    precomputed_range: tuple[float, float] = None,
    force: bool = False,
):
    field_names = [field_name_or_names] if isinstance(field_name_or_names, str) else field_name_or_names

    if precomputed_range is None and range_utils.range_needs_update(attr, force):
        # need to get the field arrays from the dataset
        # we only want to fetch the arrays if we need to compute the range.
        field_arrays = []
        for f in field_names:
            field_arrays.extend(await get_arrays_from_relationship(dataset, f"field:{f}", timeCode))
    else:
        field_arrays = None

    return await range_utils.compute_and_set_range(attr, field_arrays, precomputed_range, force=force)


def get_bracketing_time_codes(prim: Usd.Prim, timeCode: Usd.TimeCode) -> tuple[Usd.TimeCode, Usd.TimeCode]:
    """
    This returns the two time samples that bracket the given time sample.

    To find the available time samples, this iterates over all attributes on the prim and its relationship
    targets. Only cae.FieldArray and cae.DataSet prims are considered.

    The returned value is defined as follows:

    1. if prim is not valid, return None
    2. if no time samples exist, return None
    3. if the timeCode is exactly equal to a time sample, return the same time sample as tuple
    4. if the timeCode is before the first time sample, return the first time sample as tuple
    5. if the timeCode is after the last time sample, return the last time sample
    6. if the timeCode is between two time samples, return the two time samples that bracket the given time
    """

    if not prim:
        return None

    tc_f = timeCode.GetValue()
    times_f: set[float] = set()
    stage: Usd.Stage = prim.GetStage()
    processed_prims: set[Usd.Prim] = set()

    def populate_time_samples(aprim: Usd.Prim) -> None:
        if not aprim.IsValid():
            return

        if aprim in processed_prims:
            return

        processed_prims.add(aprim)

        for attr in aprim.GetAuthoredAttributes():
            if bracket := attr.GetBracketingTimeSamples(tc_f):
                times_f.add(bracket[0])
                times_f.add(bracket[1])

        for rel in aprim.GetAuthoredRelationships():
            for tpath in rel.GetForwardedTargets():
                target = stage.GetPrimAtPath(tpath)
                if target.IsValid() and (target.IsA(cae.DataSet) or target.IsA(cae.FieldArray)):
                    populate_time_samples(target)

    populate_time_samples(prim)

    if len(times_f) == 0:
        # no timesamples exist, return None
        return None

    sorted_list = sorted(times_f)
    i = bisect.bisect_right(sorted_list, tc_f)

    if i == len(sorted_list):
        # out of range, return nearset time sample
        return (Usd.TimeCode(sorted_list[-1]), Usd.TimeCode(sorted_list[-1]))

    if sorted_list[i] == tc_f:
        # exact match, return the same time sample
        return (Usd.TimeCode(sorted_list[i]), Usd.TimeCode(sorted_list[i]))
    if i == 0:
        # before first time sample, return the first time sample
        return (Usd.TimeCode(sorted_list[0]), Usd.TimeCode(sorted_list[0]))
    if i > 0:
        # return the two time samples that bracket the given time
        return (Usd.TimeCode(sorted_list[i - 1]), Usd.TimeCode(sorted_list[i]))
    else:
        return (Usd.TimeCode(sorted_list[i]), Usd.TimeCode(sorted_list[i]))


def get_time_sample(prim: Usd.Prim, time: Usd.TimeCode) -> Usd.TimeCode:
    """
    This returns the time sample nearest to the given time, preferring the one that is less than or
    equal to the given time.

    To find the available time samples, this iterates over all attributes on the prim and its relationship
    targets. Only cae.FieldArray and cae.DataSet prims are considered.

    This function is useful when you want to find the authored time-sample for a cae.DataSet or cae.FieldArray
    that should be used to access the data for the given time. Since cae.DataSet and cae.FieldArray prims
    can only support discrete time samples, this function is useful to find the nearest time sample and
    avoid having to access data when the time is not available.
    """

    if not prim.IsValid():
        return Usd.TimeCode.EarliestTime()

    assert prim.IsA(cae.DataSet) or prim.IsA(cae.FieldArray), f"Prim {prim} is not a DataSet or FieldArray"

    times: set[float] = set()
    stage: Usd.Stage = prim.GetStage()

    for attr in prim.GetAuthoredAttributes():
        if bracket := attr.GetBracketingTimeSamples(time.GetValue()):
            times.add(bracket[0])

    for rel in prim.GetAuthoredRelationships():
        for tpath in rel.GetForwardedTargets():
            target = stage.GetPrimAtPath(tpath)
            if target.IsValid() and (target.IsA(cae.DataSet) or target.IsA(cae.FieldArray)):
                for attr in target.GetAuthoredAttributes():
                    if bracket := attr.GetBracketingTimeSamples(time.GetValue()):
                        times.add(bracket[0])

    if len(times) == 0:
        return Usd.TimeCode.EarliestTime()

    sorted_list = sorted(times)
    i = bisect.bisect_right(sorted_list, time.GetValue())
    return Usd.TimeCode(sorted_list[i] if i < len(sorted_list) else sorted_list[-1])


def get_next_time_sample(prim: Usd.Prim, tc: Usd.TimeCode) -> Usd.TimeCode:
    if not prim or not tc or tc.IsDefault() or tc.IsEarliestTime():
        return tc.EarliestTime()

    samples = get_time_samples(prim)
    if len(samples) < 2:
        return tc

    tc = get_time_sample(prim, tc)

    idx = samples.index(tc)
    if idx < len(samples) - 1:
        return samples[idx + 1]
    return tc


def get_time_samples(prim: Usd.Prim) -> list[Usd.TimeCode]:
    """
    This returns all the time samples available on the given prim and its relationship targets.
    Only cae.FieldArray and cae.DataSet prims are considered.
    """

    if not prim.IsValid():
        return []

    times: set[float] = set()
    stage: Usd.Stage = prim.GetStage()

    for attr in prim.GetAuthoredAttributes():
        for time in attr.GetTimeSamples():
            times.add(time)

    for rel in prim.GetAuthoredRelationships():
        for tpath in rel.GetForwardedTargets():
            target = stage.GetPrimAtPath(tpath)
            if target.IsValid() and (target.IsA(cae.DataSet) or target.IsA(cae.FieldArray)):
                for attr in target.GetAuthoredAttributes():
                    for time in attr.GetTimeSamples():
                        times.add(time)

    return [Usd.TimeCode(x) for x in sorted(list(times))]  # sorted list of time samples


@quietable
def set_attribute(
    attr: Usd.Attribute, array: IFieldArray, timeCode: Usd.TimeCode = Usd.TimeCode.Default(), xform=None
) -> bool:
    """
    Set the attribute value to the given array if it has changed since the last time it was set.
    This is handy to avoid adding new time samples when the value hasn't changed.
    """
    if not attr:
        raise QuietableException("Invalid attribute")

    # this is not a foolproof way to check if the array has changed, but it's good enough for now.
    # we can add a more sophisticated check if needed.
    checksum = array_utils.checksum(array)
    if attr.GetCustomDataByKey("omni.cae.kit:last_checksum") != checksum:
        attr.Set(xform(array) if xform else array, timeCode)
        attr.SetCustomDataByKey("omni.cae.kit:last_checksum", checksum)
        logger.debug("[set_attribute]: Set %s at %s [checksum=%s]", attr, timeCode, checksum)
        return True
    return False


class ChangeTracker:
    """
    Helper class that tracks changes to a USD stage. This is simply a wrapper around Rt.ChangeTracker
    that exposes limited API that accepts pxr.Usd prims instead of usdrt.Usd prims.

    This helps us keep algorithm implementations void of any UsdRt dependencies.
    """

    _rt_stage_weakref_map = {}

    def __init__(self, stage: Usd.Stage) -> None:
        cache = UsdUtils.StageCache.Get()
        id = cache.GetId(stage)

        stage_ref = ChangeTracker._rt_stage_weakref_map.get(id)
        stage = stage_ref() if stage_ref is not None else None
        if stage is None:
            stage = UsdRt.Stage.Attach(id.ToLongInt())
            ChangeTracker._rt_stage_weakref_map[id] = weakref.ref(stage)

        self._rt_stage = stage
        self._tracker = Rt.ChangeTracker(self._rt_stage)

    def PrimOrTargetsChanged(self, prim_or_path) -> None:
        self._paths = set()
        return self._PrimOrTargetsChangedInternal(prim_or_path)

    def _PrimOrTargetsChangedInternal(self, prim_or_path) -> None:
        if hasattr(prim_or_path, "GetPath"):
            path = str(prim_or_path.GetPath())
        else:
            path = str(prim_or_path)

        if path in self._paths:
            return False

        self._paths.add(path)

        # recursively check if any of relationship targets have changed
        prim = self._rt_stage.GetPrimAtPath(path)
        if not prim:
            return False

        if self._tracker.PrimChanged(prim):
            logger.info("Prim changed %s", prim)
            logger.info("  changed paths %s", self._tracker.GetChangedAttributes(prim))
            return True

        for rel in prim.GetRelationships():
            for target in rel.GetForwardedTargets():
                if self._PrimOrTargetsChangedInternal(target):
                    return True

        return False

    def PrimChanged(self, prim_or_path) -> None:
        if hasattr(prim_or_path, "GetPath"):
            path = str(prim_or_path.GetPath())
        else:
            path = str(prim_or_path)

        return self._tracker.PrimChanged(path)

    def AttributeChanged(self, attr_or_path) -> None:
        if hasattr(attr_or_path, "GetPath"):
            path = str(attr_or_path.GetPath())
        else:
            path = str(attr_or_path)

        return self._tracker.AttributeChanged(path)

    def ClearChanges(self) -> None:
        return self._tracker.ClearChanges()

    def TrackAttribute(self, attrName: str) -> None:
        return self._tracker.TrackAttribute(str(attrName))

    def TrackSchemaProperties(self, schema_name: str):
        logger.debug(f"{id(self)}: tracking schema properties for {schema_name}")
        registry = Usd.SchemaRegistry()
        defn = registry.FindAppliedAPIPrimDefinition(schema_name) or registry.FindConcretePrimDefinition(schema_name)

        if not defn:
            logger.error(f"Schema {schema_name} not found. Properties will not be tracked.")
            return

        for pname in defn.GetPropertyNames():
            logger.debug(f"{id(self)}: tracking {pname}")
            self.TrackAttribute(pname)

    def TrackCaeFieldArrayProperties(self) -> None:
        registry = Usd.SchemaRegistry()
        baseT: Tf.Type = registry.GetConcreteTypeFromSchemaTypeName("CaeFieldArray")
        assert not baseT.isUnknown

        for t in baseT.GetAllDerivedTypes():
            self.TrackSchemaProperties(registry.GetConcreteSchemaTypeName(t))
