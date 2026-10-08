# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Built-in Planar Slice operator."""

from __future__ import annotations

from .. import InvalidRecipe, OperatorContext, operator
from .. import utils as op_utils

_MODE_NORMALS = {
    "x": ((1.0, 0.0, 0.0),),
    "y": ((0.0, 1.0, 0.0),),
    "z": ((0.0, 0.0, 1.0),),
    "xy": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    "xz": ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    "yz": ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    "xyz": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
}


@operator(
    required_schemas={
        "CaeVizPlanarSliceAPI",
        "CaeVizDatasetSelectionAPI:source",
        "CaeVizFieldSelectionAPI:colors",
    },
    optional_schemas={"CaeVizFieldSelectionAPI:*"},
    dataset_components={"source": {"topology", "geometry"}},
)
class PlanarSlice:
    """Extract transform-positioned planes when recipe or declared inputs change.

    Session compares declared recipe transforms as well as source topology,
    geometry, and selected fields before deciding whether to execute.
    """

    def execute(self, context: OperatorContext):
        """Slice the declared source and publish surfaces in the recipe local space."""
        import numpy as np
        import warp as wp
        from warp_simdata.operators import slice as planar_slice

        prim = context.prim
        if prim.type_name != "Mesh":
            raise InvalidRecipe(f"Planar Slice requires a Mesh prim; got {prim.type_name!r}")

        selections = op_utils.get_field_selections(context, input="source")
        source = op_utils.get_input_dataset(context, role="source")
        matrix = op_utils.get_local_transform(prim)
        origin = np.asarray(matrix[:3, 3], dtype=np.float64)
        mode = str(prim.get_attribute("cae:viz:planarSlice:mode").get())
        if mode == "free":
            normal = np.asarray(matrix[:3, 1], dtype=np.float64)
            length = np.linalg.norm(normal)
            if not np.isfinite(length) or length == 0:
                raise InvalidRecipe("Planar Slice transform has an invalid local +Y axis")
            normals = (normal / length,)
        else:
            try:
                normals = _MODE_NORMALS[mode]
            except KeyError as exc:
                raise InvalidRecipe(f"unsupported planar-slice mode {mode!r}") from exc

        output_fields = [selection.output_name for selection in selections]
        surfaces = planar_slice.compute_many(
            source,
            origins=[origin] * len(normals),
            normals=normals,
            field_names=output_fields,
        )
        surfaces = tuple(surface for surface in surfaces if surface.get_num_nodes() and surface.get_num_elems())
        if not surfaces:
            return op_utils.empty("planes do not intersect the dataset")

        product = op_utils.mesh_from_surfaces(surfaces, exclude_fields={"element_idx"})
        try:
            inverse = np.linalg.inv(matrix)
        except np.linalg.LinAlgError as exc:
            raise InvalidRecipe("Planar Slice transform is not invertible") from exc
        from .._compute import transform_points

        product = op_utils.mesh(
            points=transform_points(product.points, inverse),
            face_vertex_counts=product.face_vertex_counts,
            face_vertex_indices=product.face_vertex_indices,
            primvars=product.primvars,
        )
        wp.synchronize_device(context.device)
        return product
