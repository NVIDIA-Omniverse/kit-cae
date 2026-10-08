# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Built-in Iso Surface operator."""

from __future__ import annotations

from .. import InvalidRecipe, OperatorContext, operator
from .. import utils as op_utils


@operator(
    required_schemas={
        "CaeVizIsoSurfaceAPI",
        "CaeVizDatasetSelectionAPI:source",
        "CaeVizFieldSelectionAPI:contour",
        "CaeVizFieldSelectionAPI:colors",
    },
    optional_schemas={"CaeVizFieldSelectionAPI:*"},
    dataset_components={"source": {"topology", "geometry"}},
)
class IsoSurface:
    """Extract a scalar contour when the recipe or declared source inputs change.

    Topology and geometry drive extraction and cell-to-node field conversion.
    Session skips unchanged committed products before scientific materialization.
    """

    def execute(self, context: OperatorContext):
        """Load selected fields, convert cell fields, and extract the current contour."""
        import warp as wp
        import warp_simdata as simdata
        from warp_simdata.operators import iso_surface, node_field

        prim = context.prim
        if prim.type_name != "Mesh":
            raise InvalidRecipe(f"Iso Surface requires a Mesh prim; got {prim.type_name!r}")

        selections = op_utils.get_field_selections(context, input="source")
        source = op_utils.get_input_dataset(context, role="source")
        for selection in selections:
            name = selection.output_name
            if source.get_field(name).association != simdata.AssociationType.NODE:
                source = node_field.compute(source, name, output_field_name=name)

        contour = source.get_field("contour")
        if simdata.utils.is_vector_dtype(contour.dtype) or contour.dtype != wp.float32:
            raise InvalidRecipe(f"contour must be a node-associated float32 scalar; got {contour.dtype}")

        iso_value = float(prim.get_attribute("cae:viz:isoSurface:isoValue").get())
        output_fields = [selection.output_name for selection in selections if selection.output_name != "contour"]
        surface = iso_surface.compute(
            source,
            "contour",
            iso_value,
            field_names=output_fields,
        )
        if surface.get_num_nodes() == 0 or surface.get_num_elems() == 0:
            return op_utils.empty("iso value does not intersect the dataset")

        wp.synchronize_device(context.device)
        return op_utils.mesh_from_surface(
            surface,
            exclude_fields={"contour", "element_idx"},
        )
