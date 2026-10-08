# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Private Faces operator implementation."""

from __future__ import annotations

from ..._products import MeshProduct, Primvar
from .. import InvalidRecipe, OperatorContext, operator
from .. import utils as op_utils


@operator(
    required_schemas={"CaeVizFacesAPI", "CaeVizDatasetSelectionAPI:source"},
    optional_schemas={"CaeVizFieldSelectionAPI:*"},
    dataset_components={"source": {"topology", "geometry"}},
)
class Faces:
    """Extract a fresh surface and selected fields whenever Session requires execution.

    Session skips the entire operator when its committed product is still valid
    and its recipe and requested inputs match. No intermediate cache is used.
    """

    def execute(self, context: OperatorContext) -> MeshProduct:
        """Load the source, extract faces, and gather current fields into a new mesh."""
        import warp as wp
        from warp_simdata.operators import element_faces

        prim = context.prim
        if prim.type_name != "Mesh":
            raise InvalidRecipe(f"Faces requires a Mesh prim; got {prim.type_name!r}")

        selections = op_utils.get_field_selections(context, input="source")
        source = op_utils.get_input_dataset(context, role="source")
        external_only = bool(prim.get_attribute("cae:viz:faces:externalOnly").get())
        surface = element_faces.compute(source, external_only=external_only)
        primvars = tuple(self._pass_field(source, surface, selection.output_name) for selection in selections)
        wp.synchronize_device(context.device)
        return op_utils.mesh(
            points=surface.handle.points,
            face_vertex_counts=surface.handle.face_vertex_counts,
            face_vertex_indices=surface.handle.face_vertex_indices,
            primvars=primvars,
        )

    @staticmethod
    def _pass_field(source, surface, name: str) -> Primvar:
        """Gather a node or element field using the extracted surface source indices."""
        import warp_simdata as simdata

        from .._compute import gather_field

        field = source.get_field(name)
        if field.association == simdata.AssociationType.NODE:
            indices = surface.get_field("node_idx")
            interpolation = "vertex"
        elif field.association == simdata.AssociationType.ELEMENT:
            indices = surface.get_field("element_idx")
            interpolation = "uniform"
        else:
            raise InvalidRecipe(f"unsupported field association {field.association!r}")
        return Primvar(name, gather_field(field, indices), interpolation)
