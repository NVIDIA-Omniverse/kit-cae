# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Adapters for OmniSci EnSight schemas."""

from collections.abc import Sequence
from typing import Any

import warp as wp
from pxr import Usd

from dav.data_models.ensight_gold import ensight_shapes, unstructured_part

from ..exceptions import USDAdapterError
from ..utils import get_sci_array, get_sci_field, get_sci_field_infos, get_target_prims
from .base import PrimAdapter

_ENSIGHT_TO_DAV_ELEMENT_TYPE_MAP = {
    "point": ensight_shapes.EN_point,
    "bar2": ensight_shapes.EN_bar2,
    "bar3": ensight_shapes.EN_bar3,
    "tria3": ensight_shapes.EN_tria3,
    "tria6": ensight_shapes.EN_tria6,
    "quad4": ensight_shapes.EN_quad4,
    "quad8": ensight_shapes.EN_quad8,
    "tetra4": ensight_shapes.EN_tetra4,
    "tetra10": ensight_shapes.EN_tetra10,
    "pyramid5": ensight_shapes.EN_pyramid5,
    "pyramid13": ensight_shapes.EN_pyramid13,
    "penta6": ensight_shapes.EN_penta6,
    "penta15": ensight_shapes.EN_penta15,
    "hexa8": ensight_shapes.EN_hexa8,
    "hexa20": ensight_shapes.EN_hexa20,
    "nsided": ensight_shapes.EN_nsided,
    "nfaced": ensight_shapes.EN_nfaced,
}


async def _part_points(part_prim: Usd.Prim, *, device: str, time_code: Usd.TimeCode):
    from pxr import OmniSciEnSight

    tokens = OmniSciEnSight.Tokens
    return await get_sci_array(part_prim, [tokens.coordinatesX, tokens.coordinatesY, tokens.coordinatesZ], time_code, device=device, target_dtype=wp.vec3f)


def _get_dav_element_type(ensight_element_type: Any) -> int:
    token = str(ensight_element_type)
    if token not in _ENSIGHT_TO_DAV_ELEMENT_TYPE_MAP:
        raise USDAdapterError(f"Unsupported EnSight element type '{token}'")
    return _ENSIGHT_TO_DAV_ELEMENT_TYPE_MAP[token]


async def _create_piece_handle(piece_prim: Usd.Prim, *, device: str, time_code: Usd.TimeCode):
    from pxr import OmniSciEnSight

    tokens = OmniSciEnSight.Tokens

    piece_api = OmniSciEnSight.UnstructuredPieceAPI(piece_prim)
    element_type = piece_api.GetElementTypeAttr().Get(time_code)
    dav_element_type = _get_dav_element_type(element_type)

    connectivity = await get_sci_array(piece_prim, tokens.connectivity, time_code, device=device, target_dtype=wp.int32)

    element_node_counts = None
    element_face_counts = None
    face_node_counts = None

    if dav_element_type == ensight_shapes.EN_nsided:
        element_node_counts = await get_sci_array(piece_prim, tokens.elementNodeCounts, time_code, device=device, target_dtype=wp.int32)

    if dav_element_type == ensight_shapes.EN_nfaced:
        element_face_counts = await get_sci_array(piece_prim, tokens.elementFaceCounts, time_code, device=device, target_dtype=wp.int32)
        face_node_counts = await get_sci_array(piece_prim, tokens.faceNodeCounts, time_code, device=device, target_dtype=wp.int32)

    return unstructured_part.create_piece_handle(
        element_type=dav_element_type,
        connectivity=connectivity,
        element_node_counts=element_node_counts,
        element_face_counts=element_face_counts,
        face_node_counts=face_node_counts,
    )


class EnSightPartAdapter(PrimAdapter):
    """Convert OmniSciEnSight unstructured parts into DAV EnSight datasets."""

    def can_handle(self, prim: Usd.Prim) -> bool:
        from pxr import OmniSciEnSight

        return prim.IsValid() and prim.HasAPI(OmniSciEnSight.UnstructuredPartAPI)

    async def to_dataset(self, prim: Usd.Prim, *, device: str, time_code: Usd.TimeCode):
        from pxr import OmniSciEnSight

        points = await _part_points(prim, device=device, time_code=time_code)
        part_api = OmniSciEnSight.UnstructuredPartAPI(prim)
        piece_prims = get_target_prims(prim, part_api.GetPiecesRel().GetName())
        if not piece_prims:
            raise USDAdapterError(f"Missing EnSight pieces for part prim {prim.GetPath()}")

        piece_handles = [await _create_piece_handle(piece_prim, device=device, time_code=time_code) for piece_prim in piece_prims]
        return unstructured_part.create_dataset(points=points, pieces=piece_handles)

    async def list_fields(self, prim: Usd.Prim, *, time_code: Usd.TimeCode):
        del time_code
        return get_sci_field_infos(prim)

    async def get_field(self, prim: Usd.Prim, *, field_names: str | Sequence[str], device: str, time_code: Usd.TimeCode):
        return await get_sci_field(prim, field_names, device=device, time_code=time_code)
