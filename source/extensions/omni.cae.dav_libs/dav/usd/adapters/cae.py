# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Adapters for marker-style OmniSci CAE schemas."""

from collections.abc import Sequence

import warp as wp
from pxr import Usd

from dav.data_models.custom import point_cloud, surface_mesh

from ..utils import get_sci_array, get_sci_field, get_sci_field_infos, get_sci_points
from .base import PrimAdapter


class CaePointCloudAdapter(PrimAdapter):
    """Convert OmniSciCaePointCloudAPI prims into DAV point clouds."""

    def can_handle(self, prim: Usd.Prim) -> bool:
        from pxr import OmniSciCae

        return prim.IsValid() and prim.HasAPI(OmniSciCae.PointCloudAPI)

    async def to_dataset(self, prim: Usd.Prim, *, device: str, time_code: Usd.TimeCode):
        points = await get_sci_points(prim, points_instance="points", x_instance="pointsX", y_instance="pointsY", z_instance="pointsZ", time_code=time_code, device=device)
        return point_cloud.create_dataset(points)

    async def list_fields(self, prim: Usd.Prim, *, time_code: Usd.TimeCode):
        del time_code
        return get_sci_field_infos(prim)

    async def get_field(self, prim: Usd.Prim, *, field_names: str | Sequence[str], device: str, time_code: Usd.TimeCode):
        return await get_sci_field(prim, field_names, device=device, time_code=time_code)


class CaeMeshAdapter(PrimAdapter):
    """Convert OmniSciCaeMeshAPI prims into DAV surface meshes."""

    def can_handle(self, prim: Usd.Prim) -> bool:
        from pxr import OmniSciCae

        return prim.IsValid() and prim.HasAPI(OmniSciCae.MeshAPI)

    async def to_dataset(self, prim: Usd.Prim, *, device: str, time_code: Usd.TimeCode):
        points = await get_sci_points(prim, points_instance="points", x_instance="pointsX", y_instance="pointsY", z_instance="pointsZ", time_code=time_code, device=device)
        face_vertex_indices = await get_sci_array(prim, "faceVertexIndices", time_code, device=device, target_dtype=wp.int32)
        face_vertex_counts = await get_sci_array(prim, "faceVertexCounts", time_code, device=device, target_dtype=wp.int32)
        return surface_mesh.create_dataset(points=points, face_vertex_indices=face_vertex_indices, face_vertex_counts=face_vertex_counts)

    async def list_fields(self, prim: Usd.Prim, *, time_code: Usd.TimeCode):
        del time_code
        return get_sci_field_infos(prim)

    async def get_field(self, prim: Usd.Prim, *, field_names: str | Sequence[str], device: str, time_code: Usd.TimeCode):
        return await get_sci_field(prim, field_names, device=device, time_code=time_code)
