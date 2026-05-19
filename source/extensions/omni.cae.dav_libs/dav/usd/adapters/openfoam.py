# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Adapters for OmniSci OpenFOAM schemas."""

from collections.abc import Sequence

import warp as wp
from pxr import Usd

from dav.data_models.openfoam import boundary_mesh, polymesh

from ..exceptions import USDAdapterError
from ..utils import get_sci_array, get_sci_field, get_sci_field_infos, get_sci_points, get_target_prim
from .base import PrimAdapter


async def _get_polymesh_geometry(prim: Usd.Prim, *, device: str, time_code: Usd.TimeCode):
    points = await get_sci_points(prim, points_instance="points", x_instance="pointsX", y_instance="pointsY", z_instance="pointsZ", time_code=time_code, device=device)
    faces = await get_sci_array(prim, "faces", time_code, device=device, target_dtype=wp.int32)
    face_offsets = await get_sci_array(prim, "facesOffsets", time_code, device=device, target_dtype=wp.int32)
    return points, faces, face_offsets


class OpenFoamPolyMeshAdapter(PrimAdapter):
    """Convert OmniSciOpenFoam polyMesh prims into DAV polyMesh datasets."""

    def can_handle(self, prim: Usd.Prim) -> bool:
        from pxr import OmniSciOpenFoam

        return prim.IsValid() and prim.HasAPI(OmniSciOpenFoam.PolyMeshAPI)

    async def to_dataset(self, prim: Usd.Prim, *, device: str, time_code: Usd.TimeCode):
        points, faces, face_offsets = await _get_polymesh_geometry(prim, device=device, time_code=time_code)
        owner = await get_sci_array(prim, "owner", time_code, device=device, target_dtype=wp.int32)
        neighbour = await get_sci_array(prim, "neighbour", time_code, device=device, target_dtype=wp.int32)
        return polymesh.create_dataset(points=points, faces=faces, owner=owner, neighbour=neighbour, face_offsets=face_offsets)

    async def list_fields(self, prim: Usd.Prim, *, time_code: Usd.TimeCode):
        del time_code
        return get_sci_field_infos(prim)

    async def get_field(self, prim: Usd.Prim, *, field_names: str | Sequence[str], device: str, time_code: Usd.TimeCode):
        return await get_sci_field(prim, field_names, device=device, time_code=time_code)


class OpenFoamBoundaryPatchAdapter(PrimAdapter):
    """Convert OmniSciOpenFoam boundary patch prims into DAV boundary meshes."""

    def can_handle(self, prim: Usd.Prim) -> bool:
        from pxr import OmniSciOpenFoam

        return prim.IsValid() and prim.HasAPI(OmniSciOpenFoam.BoundaryPatchAPI)

    async def to_dataset(self, prim: Usd.Prim, *, device: str, time_code: Usd.TimeCode):
        from pxr import OmniSciOpenFoam

        patch_api = OmniSciOpenFoam.BoundaryPatchAPI(prim)
        mesh_prim = get_target_prim(prim, patch_api.GetMeshRel().GetName())
        if mesh_prim is None:
            raise USDAdapterError(f"Boundary patch prim {prim.GetPath()} is missing its owning mesh relationship")

        points, faces, face_offsets = await _get_polymesh_geometry(mesh_prim, device=device, time_code=time_code)
        start_face = patch_api.GetStartFaceAttr().Get(time_code)
        n_faces = patch_api.GetNFacesAttr().Get(time_code)
        if start_face is None or n_faces is None:
            raise USDAdapterError(f"Boundary patch prim {prim.GetPath()} is missing startFace/nFaces metadata")

        return boundary_mesh.create_dataset(points=points, faces=faces, face_offsets=face_offsets, start_face=int(start_face), n_faces=int(n_faces))

    async def list_fields(self, prim: Usd.Prim, *, time_code: Usd.TimeCode):
        del prim, time_code
        return []

    async def get_field(self, prim: Usd.Prim, *, field_names: str | Sequence[str], device: str, time_code: Usd.TimeCode):
        del field_names, device, time_code
        raise USDAdapterError(f"OpenFOAM boundary patch prim {prim.GetPath()} does not expose DAV fields")
