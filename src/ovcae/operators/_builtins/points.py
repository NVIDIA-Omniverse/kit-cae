# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Built-in Points operator."""

from __future__ import annotations

from ..._products import Primvar
from .. import InvalidRecipe, OperatorContext, operator
from .. import utils as op_utils


@operator(
    required_schemas={"CaeVizPointsAPI", "CaeVizDatasetSelectionAPI:source"},
    optional_schemas={"CaeVizFieldSelectionAPI:*"},
    dataset_components={"source": {"topology", "geometry"}},
)
class Points:
    """Publish selected scientific nodes when recipe or declared inputs change.

    Topology is required for cell-connected node selection and cell-field conversion.
    Session skips unchanged committed products before loading scientific arrays.
    """

    def execute(self, context: OperatorContext):
        """Load the declared source, select nodes, and publish current fields and widths."""
        import warp as wp

        from ._points_compute import point_dataset

        prim = context.prim
        if prim.type_name != "Points":
            raise InvalidRecipe(f"Points requires a Points prim; got {prim.type_name!r}")

        selections = op_utils.get_field_selections(context, input="source")
        source = op_utils.get_input_dataset(context, role="source")
        max_count = int(prim.get_attribute("cae:viz:points:maxCount").get())
        if max_count < 0:
            raise InvalidRecipe("cae:viz:points:maxCount must be non-negative")
        use_cell_points = bool(prim.get_attribute("cae:viz:points:useCellPoints").get())
        try:
            result = point_dataset(source, max_count=max_count, use_cell_points=use_cell_points)
        except ValueError as exc:
            raise InvalidRecipe(str(exc)) from exc
        if result.get_num_nodes() == 0:
            return op_utils.empty("point selection is empty")

        default_width = float(prim.get_attribute("cae:viz:points:width").get())
        widths = op_utils.resolve_widths(context, result=result, role="widths", default=default_width)
        primvars = tuple(
            Primvar(selection.output_name, result.get_field(selection.output_name).to_array(), "vertex")
            for selection in selections
            if selection.output_name != "widths"
        )
        wp.synchronize_device(context.device)
        return op_utils.points(points=result.handle.points, widths=widths, primvars=primvars)
