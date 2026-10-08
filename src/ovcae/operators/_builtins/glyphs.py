# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Built-in scientific glyphs published as PointInstancer arrays."""

from ..._products import Primvar
from .. import InvalidRecipe, OperatorContext, operator
from .. import utils as op_utils


@operator(
    required_schemas={"CaeVizGlyphsAPI", "CaeVizDatasetSelectionAPI:source"},
    optional_schemas={"CaeVizFieldSelectionAPI:*", "CaeVizFieldMappingAPI:scales"},
    dataset_components={"source": {"topology", "geometry"}},
)
class Glyphs:
    """Instance one application-owned prototype at each selected scientific node.

    Shares Points' cell-to-node conversion and deterministic decimation. Session
    gates unchanged committed inputs and owns generated arrays only; USD population
    owns the prototype hierarchy, including composite prototypes flattened by OVStage.
    """

    def execute(self, context: OperatorContext):
        """Produce current instance transforms and selected scientific primvars."""
        import warp as wp

        from ._glyphs_compute import orientations, scales
        from ._points_compute import point_dataset

        prim = context.prim
        if prim.type_name != "PointInstancer":
            raise InvalidRecipe(f"Glyphs requires a PointInstancer prim; got {prim.type_name!r}")
        max_count = int(prim.get_attribute("cae:viz:glyphs:maxCount").get())
        if max_count < 0:
            raise InvalidRecipe("cae:viz:glyphs:maxCount must be non-negative")
        mode = str(prim.get_attribute("cae:viz:glyphs:orientationsMode").get())
        if mode not in {"eulerAngles", "quaternion"}:
            raise InvalidRecipe(f"unsupported glyph orientations mode {mode!r}")
        source = op_utils.get_input_dataset(context, role="source")
        try:
            result = point_dataset(
                source,
                max_count=max_count,
                use_cell_points=bool(prim.get_attribute("cae:viz:glyphs:useCellPoints").get()),
            )
            scale_values = scales(result, prim)
            rotations = orientations(result, mode)
        except ValueError as exc:
            raise InvalidRecipe(str(exc)) from exc
        primvars = tuple(
            Primvar(s.output_name, result.get_field(s.output_name).to_array(), "vertex")
            for s in op_utils.get_field_selections(context, input="source")
            if s.output_name not in {"scales", "orientations"}
        )
        indices = wp.zeros(result.get_num_nodes(), dtype=wp.int32, device=context.device)
        wp.synchronize_device(context.device)
        # Publish consistently empty arrays as a real product, so an empty source
        # clears old instances and can be reused by the normal committed-product gate.
        return op_utils.instancer(
            positions=result.handle.points,
            proto_indices=indices,
            scales=scale_values,
            orientations=rotations,
            primvars=primvars,
        )
