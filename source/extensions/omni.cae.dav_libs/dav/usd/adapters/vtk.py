# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Adapters for OmniSci VTK schemas."""

from collections.abc import Sequence

import warp as wp
from pxr import Usd

from dav.data_models.vtk import image_data, polydata, structured_grid, unstructured_grid

from ..exceptions import USDAdapterError
from ..utils import get_sci_array, get_sci_field, get_sci_field_infos, has_sci_array
from .base import PrimAdapter


def _as_vec3i(value, *, prim: Usd.Prim, attr_name: str) -> wp.vec3i:
    if value is None or len(value) < 3:
        raise USDAdapterError(f"Attribute '{attr_name}' is missing or malformed on prim {prim.GetPath()}")
    return wp.vec3i(int(value[0]), int(value[1]), int(value[2]))


def _as_vec3f(value, *, prim: Usd.Prim, attr_name: str) -> wp.vec3f:
    if value is None or len(value) < 3:
        raise USDAdapterError(f"Attribute '{attr_name}' is missing or malformed on prim {prim.GetPath()}")
    return wp.vec3f(float(value[0]), float(value[1]), float(value[2]))


def _has_non_empty_array(value) -> bool:
    if value is None:
        return False
    try:
        return len(value) > 0
    except TypeError:
        return True


async def _get_points(prim: Usd.Prim, *, instance: str, device: str, time_code: Usd.TimeCode):
    return await get_sci_array(prim, instance, time_code, device=device, target_dtype=wp.vec3f)


async def _get_optional_int32_array(prim: Usd.Prim, *, instance: str, device: str, time_code: Usd.TimeCode):
    if not has_sci_array(prim, instance):
        return None
    return await get_sci_array(prim, instance, time_code, device=device, target_dtype=wp.int32)


class VtkAdapter(PrimAdapter):
    """Convert OmniSci VTK dataset prims into DAV VTK datasets."""

    def can_handle(self, prim: Usd.Prim) -> bool:
        from pxr import OmniSciVtk

        return prim.IsValid() and any(
            prim.HasAPI(api)
            for api in (OmniSciVtk.UnstructuredGridAPI, OmniSciVtk.StructuredGridAPI, OmniSciVtk.ImageDataAPI, OmniSciVtk.RectilinearGridAPI, OmniSciVtk.PolyDataAPI)
        )

    async def to_dataset(self, prim: Usd.Prim, *, device: str, time_code: Usd.TimeCode):
        from pxr import OmniSciVtk

        if prim.HasAPI(OmniSciVtk.UnstructuredGridAPI):
            return await self._unstructured_to_dataset(prim, device=device, time_code=time_code)
        if prim.HasAPI(OmniSciVtk.StructuredGridAPI):
            return await self._structured_to_dataset(prim, device=device, time_code=time_code)
        if prim.HasAPI(OmniSciVtk.ImageDataAPI):
            return await self._image_data_to_dataset(prim, device=device, time_code=time_code)
        if prim.HasAPI(OmniSciVtk.RectilinearGridAPI):
            # TODO: Add rectilinear-grid support once DAV grows a matching data model.
            raise USDAdapterError(f"VTK rectilinear grid prim {prim.GetPath()} is not yet supported by DAV")
        if prim.HasAPI(OmniSciVtk.PolyDataAPI):
            return await self._polydata_to_dataset(prim, device=device, time_code=time_code)
        raise USDAdapterError(f"Unsupported VTK prim {prim.GetPath()}")

    async def list_fields(self, prim: Usd.Prim, *, time_code: Usd.TimeCode):
        del time_code
        return get_sci_field_infos(prim)

    async def get_field(self, prim: Usd.Prim, *, field_names: str | Sequence[str], device: str, time_code: Usd.TimeCode):
        return await get_sci_field(prim, field_names, device=device, time_code=time_code)

    async def _unstructured_to_dataset(self, prim: Usd.Prim, *, device: str, time_code: Usd.TimeCode):
        from pxr import OmniSciVtk

        tokens = OmniSciVtk.Tokens
        points = await _get_points(prim, instance=str(tokens.points), device=device, time_code=time_code)
        cell_offsets = await get_sci_array(prim, str(tokens.connectivityOffsets), time_code, device=device, target_dtype=wp.int32)
        cell_connectivity = await get_sci_array(prim, str(tokens.connectivityArray), time_code, device=device, target_dtype=wp.int32)
        cell_types = await get_sci_array(prim, str(tokens.cellTypes), time_code, device=device, target_dtype=wp.int32)

        return unstructured_grid.create_dataset(
            points=points,
            cell_types=cell_types,
            cell_offsets=cell_offsets,
            cell_connectivity=cell_connectivity,
            faces_offsets=await _get_optional_int32_array(prim, instance=str(tokens.polyhedronFacesOffsets), device=device, time_code=time_code),
            faces_connectivity=await _get_optional_int32_array(prim, instance=str(tokens.polyhedronFacesConnectivityArray), device=device, time_code=time_code),
            face_locations_offsets=await _get_optional_int32_array(prim, instance=str(tokens.polyhedronFaceLocationsOffsets), device=device, time_code=time_code),
            face_locations_connectivity=await _get_optional_int32_array(prim, instance=str(tokens.polyhedronFaceLocationsConnectivityArray), device=device, time_code=time_code),
        )

    async def _structured_to_dataset(self, prim: Usd.Prim, *, device: str, time_code: Usd.TimeCode):
        from pxr import OmniSciVtk

        points = await _get_points(prim, instance=str(OmniSciVtk.Tokens.points), device=device, time_code=time_code)
        api = OmniSciVtk.StructuredGridAPI(prim)
        return structured_grid.create_dataset(
            points=points,
            extent_min=_as_vec3i(api.GetMinExtentAttr().Get(time_code), prim=prim, attr_name="omni:vtk:minExtent"),
            extent_max=_as_vec3i(api.GetMaxExtentAttr().Get(time_code), prim=prim, attr_name="omni:vtk:maxExtent"),
        )

    async def _image_data_to_dataset(self, prim: Usd.Prim, *, device: str, time_code: Usd.TimeCode):
        from pxr import OmniSciVtk

        api = OmniSciVtk.ImageDataAPI(prim)
        return image_data.create_dataset(
            origin=_as_vec3f(api.GetOriginAttr().Get(time_code), prim=prim, attr_name="omni:vtk:origin"),
            spacing=_as_vec3f(api.GetSpacingAttr().Get(time_code), prim=prim, attr_name="omni:vtk:spacing"),
            extent_min=_as_vec3i(api.GetMinExtentAttr().Get(time_code), prim=prim, attr_name="omni:vtk:minExtent"),
            extent_max=_as_vec3i(api.GetMaxExtentAttr().Get(time_code), prim=prim, attr_name="omni:vtk:maxExtent"),
            device=device,
        )

    async def _polydata_to_dataset(self, prim: Usd.Prim, *, device: str, time_code: Usd.TimeCode):
        from pxr import OmniSciVtk

        tokens = OmniSciVtk.Tokens
        strips_offsets = await _get_optional_int32_array(prim, instance=str(tokens.stripsConnectivityOffsets), device=device, time_code=time_code)
        strips_connectivity = await _get_optional_int32_array(prim, instance=str(tokens.stripsConnectivityArray), device=device, time_code=time_code)
        if _has_non_empty_array(strips_offsets) or _has_non_empty_array(strips_connectivity):
            raise USDAdapterError(f"VTK polydata triangle strips are not yet supported for prim {prim.GetPath()}")

        points = await _get_points(prim, instance=str(tokens.points), device=device, time_code=time_code)
        verts_offsets = await _get_optional_int32_array(prim, instance=str(tokens.vertsConnectivityOffsets), device=device, time_code=time_code)
        verts_connectivity = await _get_optional_int32_array(prim, instance=str(tokens.vertsConnectivityArray), device=device, time_code=time_code)
        lines_offsets = await _get_optional_int32_array(prim, instance=str(tokens.linesConnectivityOffsets), device=device, time_code=time_code)
        lines_connectivity = await _get_optional_int32_array(prim, instance=str(tokens.linesConnectivityArray), device=device, time_code=time_code)
        polys_offsets = await _get_optional_int32_array(prim, instance=str(tokens.polysConnectivityOffsets), device=device, time_code=time_code)
        polys_connectivity = await _get_optional_int32_array(prim, instance=str(tokens.polysConnectivityArray), device=device, time_code=time_code)
        return polydata.create_dataset(
            points=points,
            verts_offsets=verts_offsets,
            verts_connectivity=verts_connectivity,
            lines_offsets=lines_offsets,
            lines_connectivity=lines_connectivity,
            polys_offsets=polys_offsets,
            polys_connectivity=polys_connectivity,
        )
