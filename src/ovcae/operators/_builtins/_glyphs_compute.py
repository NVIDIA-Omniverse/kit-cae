# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Device-side scale and orientation conversion for Glyphs."""

import math

import warp as wp

vec4h = wp.types.vector(length=4, dtype=wp.float16)


@wp.kernel(enable_backward=False)
def _expand_scales(values: wp.array(dtype=wp.float32), output: wp.array(dtype=wp.vec3f)):
    i = wp.tid()
    output[i] = wp.vec3f(values[i])


@wp.kernel(enable_backward=False)
def _map_scales(values: wp.array(dtype=wp.vec3f), domain: wp.vec2f, extent: wp.vec2f, output: wp.array(dtype=wp.vec3f)):
    i = wp.tid()
    v = values[i]
    for c in range(3):
        t = 0.0
        if domain[1] > domain[0]:
            t = wp.clamp((v[c] - domain[0]) / (domain[1] - domain[0]), 0.0, 1.0)
        v[c] = extent[0] + t * (extent[1] - extent[0])
    output[i] = v


@wp.kernel(enable_backward=False)
def _directions(values: wp.array(dtype=wp.vec3f), output: wp.array(dtype=vec4h)):
    i = wp.tid()
    d = wp.normalize(values[i])
    q = wp.vec4f(0.0, 0.0, 0.0, 1.0)
    # atan2 remains stable near -X, where the usual half-vector construction
    # collapses to zero. A zero direction retains the identity rotation.
    transverse = wp.sqrt(d[1] * d[1] + d[2] * d[2])
    if transverse > 0.0:
        half = 0.5 * wp.atan2(transverse, d[0])
        factor = wp.sin(half) / transverse
        q = wp.vec4f(0.0, -d[2] * factor, d[1] * factor, wp.cos(half))
    elif d[0] < 0.0:
        q = wp.vec4f(0.0, 0.0, 1.0, 0.0)
    output[i] = vec4h(q)


def _float_array(field, components):
    """Materialize and cast a scalar/vector field on its producing device."""
    values = field.to_array()
    width = getattr(values.dtype, "_length_", 1)
    if values.ndim != 1 or width not in components:
        raise ValueError(f"expected field with {components} components; got {values.dtype} shape {values.shape}")
    dtype = wp.float32 if width == 1 else wp.types.vector(length=width, dtype=wp.float32)
    if values.dtype != dtype:
        converted = wp.empty(values.shape, dtype=dtype, device=values.device)
        wp.utils.array_cast(values, converted)
        values = converted
    return values, width


def scales(dataset, prim):
    """Expand scalar scales and apply an explicit, clamped mapping per component."""
    count = dataset.get_num_nodes()
    if not dataset.has_field("scales"):
        default = float(prim.get_attribute("cae:viz:glyphs:scale").get())
        if not math.isfinite(default):
            raise ValueError("glyph scale must be finite")
        return wp.full(count, wp.vec3f(default), dtype=wp.vec3f, device=dataset.device)
    values, width = _float_array(dataset.get_field("scales"), (1, 3))
    if width == 1:
        expanded = wp.empty(count, dtype=wp.vec3f, device=dataset.device)
        wp.launch(_expand_scales, dim=count, inputs=[values, expanded], device=dataset.device)
        values = expanded
    if prim.has_api("CaeVizFieldMappingAPI", instance="scales"):
        domain = tuple(prim.get_attribute("cae:viz:field_mapping:scales:domain").get())
        extent = tuple(prim.get_attribute("cae:viz:field_mapping:scales:range").get())
        # The invalid-domain sentinel has the same pass-through behavior as Kit.
        # Automatic range rescaling is not evaluated by OVCAE.
        if domain[0] <= domain[1]:
            if not all(math.isfinite(v) for v in (*domain, *extent)):
                raise ValueError("glyph scale mapping must be finite")
            mapped = wp.empty_like(values)
            wp.launch(
                _map_scales,
                dim=count,
                inputs=[values, wp.vec2f(*domain), wp.vec2f(*extent), mapped],
                device=dataset.device,
            )
            values = mapped
    return values


def orientations(dataset, mode):
    """Return standard quath[] storage in xyzw order, with no host array copy.

    Kit's historical eulerAngles token actually means directions measured from
    local +X. Missing fields produce an empty array (USD identity rotations).
    """
    if not dataset.has_field("orientations"):
        return wp.empty(0, dtype=vec4h, device=dataset.device)
    values, _ = _float_array(dataset.get_field("orientations"), (4,) if mode == "quaternion" else (3,))
    result = wp.empty(dataset.get_num_nodes(), dtype=vec4h, device=dataset.device)
    if mode == "quaternion":
        wp.utils.array_cast(values, result)
    else:
        wp.launch(_directions, dim=len(result), inputs=[values, result], device=dataset.device)
    return result
