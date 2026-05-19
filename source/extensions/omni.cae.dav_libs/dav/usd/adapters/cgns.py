# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Adapters for OmniCgns schemas."""

import asyncio
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
import warp as wp
from pxr import Usd

import dav
from dav.data_models.sids import nface_n as dav_sids_nface_n
from dav.data_models.sids import sids_shapes
from dav.data_models.sids import unstructured as dav_sids_unstructured

from ..exceptions import USDAdapterError
from ..utils import get_sci_array, get_sci_field, get_sci_field_infos, get_target_prim, get_target_prims, has_sci_array
from .base import PrimAdapter


@wp.kernel
def _map_nface_to_ngon_cell_field_indices_kernel(
    connectivity: wp.array(dtype=wp.int32), offsets: wp.array(dtype=wp.int32), indices: wp.array(dtype=wp.int32), ngon_start: wp.int32, ngon_count: wp.int32, field_offset: wp.int32
):
    cell_idx = wp.tid()
    start = offsets[cell_idx]
    end = offsets[cell_idx + 1]
    cell_field_idx = field_offset + cell_idx

    for offset in range(start, end):
        face_id = connectivity[offset]
        if face_id < 0:
            face_id = -face_id

        face_idx = face_id - ngon_start
        if face_idx >= 0 and face_idx < ngon_count:
            wp.atomic_min(indices, face_idx, cell_field_idx)


@wp.kernel
def _validate_ngon_cell_field_indices_kernel(indices: wp.array(dtype=wp.int32), missing_sentinel: wp.int32, missing_info: wp.array(dtype=wp.int32)):
    idx = wp.tid()
    if indices[idx] == missing_sentinel:
        wp.atomic_add(missing_info, 0, 1)
        wp.atomic_min(missing_info, 1, idx)


async def _zone_grid_coordinates(zone_prim: Usd.Prim, *, device: str, time_code: Usd.TimeCode):
    from pxr import OmniCgns

    tokens = OmniCgns.Tokens
    return await get_sci_array(zone_prim, [tokens.gridCoordinatesX, tokens.gridCoordinatesY, tokens.gridCoordinatesZ], time_code, device=device, target_dtype=wp.vec3f)


async def _standard_section_to_dataset(prim: Usd.Prim, grid_coordinates: wp.array, *, device: str, time_code: Usd.TimeCode):
    from pxr import OmniCgns

    elem_api = OmniCgns.UnstructuredElementsAPI(prim)
    element_type = sids_shapes.get_element_type_from_string(elem_api.GetElementTypeAttr().Get().lower())
    element_range = elem_api.GetElementRangeAttr().Get()
    e_start, e_end = int(element_range[0]), int(element_range[1])

    int32_max = np.iinfo(np.int32).max
    if e_start > int32_max or e_end > int32_max:
        raise USDAdapterError(f"Element range on prim {prim.GetPath()} exceeds the maximum value supported by DAV int32 indexing")

    tokens = OmniCgns.Tokens
    connectivity = await get_sci_array(prim, tokens.elementConnectivity, time_code, device=device, target_dtype=wp.int32)

    e_start_offsets = None
    if has_sci_array(prim, tokens.elementStartOffset):
        e_start_offsets = await get_sci_array(prim, tokens.elementStartOffset, time_code, device=device, target_dtype=wp.int32)

    return dav_sids_unstructured.create_dataset(
        grid_coords=grid_coordinates, element_type=element_type, element_range=wp.vec2i(e_start, e_end), element_connectivity=connectivity, element_start_offset=e_start_offsets
    )


async def _nface_n_section_to_dataset(nface_prim: Usd.Prim, zone_prim: Usd.Prim, grid_coordinates: wp.array, *, device: str, time_code: Usd.TimeCode):
    from pxr import OmniCgns

    zone_api = OmniCgns.ZoneAPI(zone_prim)
    stage = nface_prim.GetStage()
    ngon_dav_handles = []
    for section_path in zone_api.GetSectionsRel().GetTargets():
        section_prim = stage.GetPrimAtPath(section_path)
        if not section_prim or not section_prim.HasAPI(OmniCgns.UnstructuredElementsAPI):
            continue

        sec_type = sids_shapes.get_element_type_from_string(OmniCgns.UnstructuredElementsAPI(section_prim).GetElementTypeAttr().Get().lower())
        if sec_type == sids_shapes.ET_NGON_n:
            ngon_dav_handles.append((await _standard_section_to_dataset(section_prim, grid_coordinates, device=device, time_code=time_code)).handle)

    if not ngon_dav_handles:
        raise USDAdapterError(f"No NGON_n sections were found for NFACE_n prim {nface_prim.GetPath()}")

    nface_dataset = await _standard_section_to_dataset(nface_prim, grid_coordinates, device=device, time_code=time_code)
    return dav_sids_nface_n.create_dataset(nface_dataset.handle, ngon_dav_handles)


class _SectionDescriptor(NamedTuple):
    prim: Usd.Prim
    element_type: int
    start: int
    end: int

    @property
    def count(self) -> int:
        return self.end - self.start + 1


class _NfaceSectionInfo(NamedTuple):
    prim: Usd.Prim
    count: int
    field_offset: int


def _get_section_metadata(section_prim: Usd.Prim) -> tuple[int, int, int]:
    """Return ``(element_type, range_start, range_end)`` for a CGNS section prim in one API construction."""
    from pxr import OmniCgns

    elem_api = OmniCgns.UnstructuredElementsAPI(section_prim)
    element_type = sids_shapes.get_element_type_from_string(elem_api.GetElementTypeAttr().Get().lower())
    element_range = elem_api.GetElementRangeAttr().Get()
    return element_type, int(element_range[0]), int(element_range[1])


def _collect_zone_section_descriptors(zone_prim: Usd.Prim) -> list[_SectionDescriptor]:
    """Walk a zone's sections once, returning descriptors sorted by element range start."""
    from pxr import OmniCgns

    zone_api = OmniCgns.ZoneAPI(zone_prim)
    stage = zone_prim.GetStage()
    descriptors = []
    for section_path in zone_api.GetSectionsRel().GetTargets():
        section_prim = stage.GetPrimAtPath(section_path)
        if not section_prim or not section_prim.HasAPI(OmniCgns.UnstructuredElementsAPI):
            continue
        element_type, start, end = _get_section_metadata(section_prim)
        descriptors.append(_SectionDescriptor(prim=section_prim, element_type=element_type, start=start, end=end))
    descriptors.sort(key=lambda d: d.start)
    return descriptors


def _build_nface_layouts(descriptors: list[_SectionDescriptor]) -> tuple[list[_NfaceSectionInfo], int, list[_NfaceSectionInfo], int]:
    """Walk descriptors once and return both candidate field-offset layouts.

    The FlowSolution cell field may be laid out either over every non-NGON section
    (``..._all_cells``) or over NFACE_n sections only (``..._nface_only``). The caller
    picks the layout whose total cell count matches ``field.size``.

    Returns ``(infos_all_cells, total_all_cells, infos_nface_only, total_nface_only)``.
    """
    infos_all_cells: list[_NfaceSectionInfo] = []
    infos_nface_only: list[_NfaceSectionInfo] = []
    total_all_cells = 0
    total_nface_only = 0
    for desc in descriptors:
        if desc.element_type == sids_shapes.ET_NGON_n:
            continue
        if desc.element_type == sids_shapes.ET_NFACE_n:
            infos_all_cells.append(_NfaceSectionInfo(prim=desc.prim, count=desc.count, field_offset=total_all_cells))
            infos_nface_only.append(_NfaceSectionInfo(prim=desc.prim, count=desc.count, field_offset=total_nface_only))
            total_nface_only += desc.count
        total_all_cells += desc.count
    return infos_all_cells, total_all_cells, infos_nface_only, total_nface_only


async def _fetch_nface_arrays(nface_prim: Usd.Prim, time_code: Usd.TimeCode, *, device: str) -> tuple[wp.array, wp.array]:
    """Fetch ``(elementConnectivity, elementStartOffset)`` for an NFACE_n section concurrently."""
    from pxr import OmniCgns

    tokens = OmniCgns.Tokens
    connectivity, offsets = await asyncio.gather(
        get_sci_array(nface_prim, tokens.elementConnectivity, time_code, device=device, target_dtype=wp.int32),
        get_sci_array(nface_prim, tokens.elementStartOffset, time_code, device=device, target_dtype=wp.int32),
    )
    if connectivity.ndim != 1:
        raise USDAdapterError(f"NFACE_n ElementConnectivity on {nface_prim.GetPath()} must be one-dimensional")
    if offsets.ndim != 1:
        raise USDAdapterError(f"NFACE_n ElementStartOffset on {nface_prim.GetPath()} must be one-dimensional")
    return connectivity, offsets


def compute_ngon_cell_field_indices_from_arrays(
    nface_arrays: list[tuple[wp.array, wp.array, int, int]], *, ngon_start: int, ngon_count: int, field_size: int, device: str
) -> tuple[wp.array, int, int]:
    """Map NGON faces to the smallest referencing cell's field index.

    ``nface_arrays`` is a list of ``(connectivity, offsets, nface_count, field_offset)`` tuples,
    one per NFACE_n section. Returns ``(indices, missing_count, first_missing_face_id)``;
    when ``missing_count`` is zero, ``first_missing_face_id`` should be ignored.
    """
    missing_sentinel = int(field_size)
    indices = wp.full(shape=ngon_count, value=missing_sentinel, dtype=wp.int32, device=device)

    for connectivity, offsets, nface_count, field_offset in nface_arrays:
        wp.launch(_map_nface_to_ngon_cell_field_indices_kernel, dim=nface_count, inputs=[connectivity, offsets, indices, ngon_start, ngon_count, field_offset], device=device)

    missing_info = wp.array([0, ngon_count], dtype=wp.int32, device=device)
    wp.launch(_validate_ngon_cell_field_indices_kernel, dim=ngon_count, inputs=[indices, missing_sentinel, missing_info], device=device)

    missing_count, first_missing_idx = missing_info.numpy().tolist()
    return indices, int(missing_count), ngon_start + int(first_missing_idx)


async def _compute_ngon_cell_field_indices(
    ngon_prim: Usd.Prim, nface_infos: list[_NfaceSectionInfo], ngon_start: int, ngon_end: int, *, field_size: int, device: str, time_code: Usd.TimeCode
) -> wp.array:
    ngon_count = ngon_end - ngon_start + 1
    int32_max = np.iinfo(np.int32).max
    if ngon_count > int32_max or field_size > int32_max:
        raise USDAdapterError(f"NGON_n cell field subset for {ngon_prim.GetPath()} exceeds the supported int32 index range")

    fetched = await asyncio.gather(*(_fetch_nface_arrays(info.prim, time_code, device=device) for info in nface_infos))
    nface_arrays = [(connectivity, offsets, info.count, info.field_offset) for info, (connectivity, offsets) in zip(nface_infos, fetched, strict=True)]

    indices, missing_count, first_missing_id = compute_ngon_cell_field_indices_from_arrays(
        nface_arrays, ngon_start=ngon_start, ngon_count=ngon_count, field_size=field_size, device=device
    )
    if missing_count:
        raise USDAdapterError(f"Could not find NFACE_n cell data indices for {missing_count} NGON_n faces on {ngon_prim.GetPath()}; first missing face id is {first_missing_id}")
    return indices


async def _remap_ngon_cell_field(ngon_prim: Usd.Prim, zone_prim: Usd.Prim, field: dav.Field, *, device: str, time_code: Usd.TimeCode) -> dav.Field:
    """Project a zone-level cell field onto an NGON_n section's faces.

    A FlowSolution stores one value per zone cell; an NGON_n section exposes per-face
    geometry. To render a cell field on those faces, each face is mapped to the field
    index of the (smallest-numbered) NFACE_n cell that references it. Fields that are
    already laid out per-face (size matches the NGON face count) and non-cell-associated
    fields are returned unchanged.
    """
    if field.association != dav.AssociationType.CELL:
        return field
    ngon_element_type, ngon_start, ngon_end = _get_section_metadata(ngon_prim)
    if ngon_element_type != sids_shapes.ET_NGON_n:
        return field
    ngon_count = ngon_end - ngon_start + 1

    descriptors = _collect_zone_section_descriptors(zone_prim)
    infos_all, total_all, infos_nface, total_nface = _build_nface_layouts(descriptors)
    if field.size == total_all:
        nface_infos, total_field_cells = infos_all, total_all
    elif field.size == total_nface:
        nface_infos, total_field_cells = infos_nface, total_nface
    else:
        if not infos_nface or field.size == ngon_count:
            return field
        raise USDAdapterError(
            f"Cell field size {field.size} on NGON_n dataset {ngon_prim.GetPath()} does not match the referenced NFACE_n cell count "
            f"({total_nface} NFACE-only / {total_all} all-cell) or NGON_n face count {ngon_count}"
        )

    if not nface_infos:
        return field

    indices = await _compute_ngon_cell_field_indices(ngon_prim, nface_infos, ngon_start, ngon_end, field_size=total_field_cells, device=device, time_code=time_code)
    return field.subset(indices)


def _get_zone_prim(section_prim: Usd.Prim) -> Usd.Prim:
    from pxr import OmniCgns

    elem_api = OmniCgns.UnstructuredElementsAPI(section_prim)
    zone_targets = elem_api.GetZoneRel().GetTargets()
    if not zone_targets:
        raise USDAdapterError(f"No parent zone relationship found on prim {section_prim.GetPath()}")
    return section_prim.GetStage().GetPrimAtPath(zone_targets[0])


def _get_flow_solution_prims(ds_prim: Usd.Prim) -> list[Usd.Prim]:
    from pxr import OmniCgns

    if ds_prim.HasAPI(OmniCgns.ZoneAPI):
        zone_api = OmniCgns.ZoneAPI(ds_prim)
        return get_target_prims(ds_prim, zone_api.GetFlowSolutionsRel().GetName(), quiet=True)

    if ds_prim.HasAPI(OmniCgns.UnstructuredElementsAPI):
        elem_api = OmniCgns.UnstructuredElementsAPI(ds_prim)
        zone_prim = get_target_prim(ds_prim, elem_api.GetZoneRel().GetName(), quiet=True)
        if zone_prim:
            zone_api = OmniCgns.ZoneAPI(zone_prim)
            return get_target_prims(zone_prim, zone_api.GetFlowSolutionsRel().GetName(), quiet=True)

    return []


async def _get_flow_solution_field(zone_prim: Usd.Prim, field_names: str | Sequence[str], *, device: str, time_code: Usd.TimeCode):
    requested_names = [field_names] if isinstance(field_names, str) else [name for name in field_names if name]
    if not requested_names:
        raise USDAdapterError("field_names cannot be empty")

    flow_solution_prims = _get_flow_solution_prims(zone_prim)
    if not flow_solution_prims:
        raise USDAdapterError(f"No FlowSolution prims found for zone {zone_prim.GetPath()}")

    from pxr import OmniSci

    arrays = []
    association = None
    for field_name in requested_names:
        field = None
        for flow_solution_prim in flow_solution_prims:
            if flow_solution_prim.HasAPI(OmniSci.FieldAPI, field_name):
                field = await get_sci_field(flow_solution_prim, field_name, device=device, time_code=time_code)
                break
        if field is None:
            raise USDAdapterError(f"Field '{field_name}' was not found in any FlowSolution prim for zone {zone_prim.GetPath()}")

        if association is None:
            association = field.association
        elif association != field.association:
            raise USDAdapterError(f"Field components {requested_names} do not share the same association on zone {zone_prim.GetPath()}")
        data = field.get_data()
        arrays.extend(data if isinstance(data, list) else [data])

    assert association is not None
    return dav.Field.from_arrays(arrays, association)


class CgnsAdapter(PrimAdapter):
    """Convert OmniCgns prims into DAV datasets and fields."""

    def can_handle(self, prim: Usd.Prim) -> bool:
        from pxr import OmniCgns

        return prim.IsValid() and (prim.HasAPI(OmniCgns.ZoneAPI) or prim.HasAPI(OmniCgns.UnstructuredElementsAPI))

    async def to_dataset(self, prim: Usd.Prim, *, device: str, time_code: Usd.TimeCode):
        from pxr import OmniCgns

        if prim.HasAPI(OmniCgns.ZoneAPI):
            return await self._zone_to_dataset(prim, device=device, time_code=time_code)

        if prim.HasAPI(OmniCgns.UnstructuredElementsAPI):
            return await self._section_to_dataset(prim, device=device, time_code=time_code)

        raise USDAdapterError(f"Unsupported CGNS prim {prim.GetPath()}")

    async def list_fields(self, prim: Usd.Prim, *, time_code: Usd.TimeCode):
        from pxr import OmniCgns

        del time_code
        zone_prim = prim if prim.HasAPI(OmniCgns.ZoneAPI) else _get_zone_prim(prim)

        field_infos = []
        for flow_solution_prim in _get_flow_solution_prims(zone_prim):
            field_infos.extend(get_sci_field_infos(flow_solution_prim))

        deduped = {}
        for field_info in field_infos:
            deduped[field_info.name] = field_info
        return list(deduped.values())

    async def get_field(self, prim: Usd.Prim, *, field_names: str | Sequence[str], device: str, time_code: Usd.TimeCode):
        from pxr import OmniCgns

        zone_prim = prim if prim.HasAPI(OmniCgns.ZoneAPI) else _get_zone_prim(prim)
        field = await _get_flow_solution_field(zone_prim, field_names, device=device, time_code=time_code)
        if prim.HasAPI(OmniCgns.UnstructuredElementsAPI):
            field = await _remap_ngon_cell_field(prim, zone_prim, field, device=device, time_code=time_code)
        return field

    async def _section_to_dataset(self, prim: Usd.Prim, *, device: str, time_code: Usd.TimeCode):
        from pxr import OmniCgns

        zone_prim = _get_zone_prim(prim)
        grid_coordinates = await _zone_grid_coordinates(zone_prim, device=device, time_code=time_code)

        element_type = sids_shapes.get_element_type_from_string(OmniCgns.UnstructuredElementsAPI(prim).GetElementTypeAttr().Get().lower())
        if element_type == sids_shapes.ET_MIXED:
            raise USDAdapterError(f"Mixed element types are not yet supported for prim {prim.GetPath()}")
        if element_type == sids_shapes.ET_NFACE_n:
            return await _nface_n_section_to_dataset(prim, zone_prim, grid_coordinates, device=device, time_code=time_code)
        return await _standard_section_to_dataset(prim, grid_coordinates, device=device, time_code=time_code)

    async def _zone_to_dataset(self, zone_prim: Usd.Prim, *, device: str, time_code: Usd.TimeCode):
        from pxr import OmniCgns

        grid_coordinates = await _zone_grid_coordinates(zone_prim, device=device, time_code=time_code)

        section_datasets = []
        stage = zone_prim.GetStage()
        zone_api = OmniCgns.ZoneAPI(zone_prim)
        for section_path in zone_api.GetSectionsRel().GetTargets():
            section_prim = stage.GetPrimAtPath(section_path)
            if not section_prim or not section_prim.HasAPI(OmniCgns.UnstructuredElementsAPI):
                continue

            elem_type = sids_shapes.get_element_type_from_string(OmniCgns.UnstructuredElementsAPI(section_prim).GetElementTypeAttr().Get().lower())
            if elem_type == sids_shapes.ET_NGON_n:
                continue
            if elem_type == sids_shapes.ET_MIXED:
                raise USDAdapterError(f"Mixed element types are not yet supported for prim {section_prim.GetPath()}")
            if elem_type == sids_shapes.ET_NFACE_n:
                dataset = await _nface_n_section_to_dataset(section_prim, zone_prim, grid_coordinates, device=device, time_code=time_code)
            else:
                dataset = await _standard_section_to_dataset(section_prim, grid_coordinates, device=device, time_code=time_code)
            section_datasets.append(dataset)

        if not section_datasets:
            raise USDAdapterError(f"No convertible sections were found for zone {zone_prim.GetPath()}")
        if len(section_datasets) == 1:
            return section_datasets[0]
        return dav.DatasetCollection.from_datasets(section_datasets)
