# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Built-in Streamlines operator."""

from __future__ import annotations

from .. import InvalidRecipe, OperatorContext, operator
from .. import utils as op_utils


@operator(
    required_schemas={
        "CaeVizStreamlinesAPI",
        "CaeVizDatasetSelectionAPI:source",
        "CaeVizDatasetSelectionAPI:seeds",
        "CaeVizFieldSelectionAPI:velocities",
    },
    optional_schemas={
        "CaeVizFieldSelectionAPI:*",
    },
    dataset_components={"source": {"topology", "geometry"}, "seeds": {"geometry"}},
)
class Streamlines:
    """Integrate selected source fields from declared scene-authored seed points.

    The shared plan validates both input roles before loading. Scene point arrays
    currently lack comparable identity, so Session conservatively executes every
    update. Seed transforms retain their existing operator-local semantics until
    the shared dataset-transformation contract is implemented.

    Like Kit-CAE, use SimData's default stopping conditions. The shared schema's
    velocity threshold remains available for compatibility but is not consumed.
    """

    def execute(self, context: OperatorContext):
        """Load source and seeds, transform seeds, integrate, and probe current fields."""
        import warp as wp
        import warp_simdata as simdata
        from warp_simdata.operators import streamlines

        from ._streamlines_compute import curve_phases, transform_seed_points

        prim = context.prim
        if prim.type_name != "BasisCurves":
            raise InvalidRecipe(f"Streamlines requires a BasisCurves prim; got {prim.type_name!r}")

        selections = op_utils.get_field_selections(context, input="source")
        source = op_utils.get_input_dataset(context, role="source")
        seeds = op_utils.get_input_dataset(context, role="seeds")
        seed_relationship = prim.get_relationship("cae:viz:dataset_selection:seeds:target")
        seed_prim = context.stage.get_prim_at_path(seed_relationship.targets[0])
        if not op_utils.is_point_based(seed_prim):
            raise InvalidRecipe("Streamlines seeds must target a captured UsdGeomPointBased prim")
        seeds = transform_seed_points(seed_prim, seeds)

        velocity = source.get_field("velocities")
        if not simdata.utils.is_vector_dtype(velocity.dtype):
            raise InvalidRecipe(f"velocities must be a vector field; got {velocity.dtype}")

        min_step = float(prim.get_attribute("cae:viz:streamlines:minStepSize").get())
        max_step = float(prim.get_attribute("cae:viz:streamlines:maxStepSize").get())
        initial_step = float(prim.get_attribute("cae:viz:streamlines:initialStepSize").get())
        max_steps = int(prim.get_attribute("cae:viz:streamlines:maxSteps").get())
        tolerance = float(prim.get_attribute("cae:viz:streamlines:tolerance").get())
        direction = str(prim.get_attribute("cae:viz:streamlines:direction").get())
        width = float(prim.get_attribute("cae:viz:streamlines:width").get())
        _validate_integration(min_step, max_step, initial_step, max_steps, tolerance)

        traced = streamlines.compute(
            source,
            "velocities",
            seeds,
            initial_dt=initial_step,
            min_dt=min_step,
            max_dt=max_step,
            max_steps=max_steps,
            tolerance=tolerance,
            direction=direction,
        )
        if traced is None or traced.get_num_nodes() == 0:
            return op_utils.empty("integration produced no streamlines")

        output_selections = tuple(item for item in selections if item.output_name != "velocities")
        selected_primvars = op_utils.probe_fields(source, traced, selections=output_selections)
        primvars = tuple(item for item in selected_primvars if item.name != "widths")
        if traced.has_field("times") and all(item.name != "times" for item in primvars):
            primvars += (op_utils.primvar("times", traced.get_field("times").to_array(), "vertex"),)
        # AnimatedStreaks/Dots read rnd as a cycle offset. Uniform interpolation
        # holds one phase along each curve, avoiding synchronized fronts and
        # preserving smooth motion along its vertex-interpolated integration time.
        if all(item.name != "rnd" for item in primvars):
            phases = curve_phases(traced.handle.curve_vertex_counts.shape[0], context.device)
            primvars += (op_utils.primvar("rnd", phases, "uniform"),)
        widths = op_utils.resolve_widths(
            context,
            result=traced,
            role="widths",
            default=width,
        )
        wp.synchronize_device(context.device)
        return op_utils.curves(
            points=traced.handle.points,
            curve_vertex_counts=traced.handle.curve_vertex_counts,
            widths=widths,
            primvars=primvars,
        )


def _validate_integration(min_step: float, max_step: float, initial_step: float, max_steps: int, tolerance: float):
    if min_step <= 0:
        raise InvalidRecipe("minStepSize must be positive")
    if max_step < min_step:
        raise InvalidRecipe("maxStepSize must be at least minStepSize")
    if not min_step <= initial_step <= max_step:
        raise InvalidRecipe("initialStepSize must be between minStepSize and maxStepSize")
    if max_steps <= 0:
        raise InvalidRecipe("maxSteps must be positive")
    if tolerance <= 0:
        raise InvalidRecipe("tolerance must be positive")
