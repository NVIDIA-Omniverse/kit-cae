# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Read-only field model that reduces a vector field to a scalar by extracting a
component or computing the vector magnitude."""

__all__ = ["get_field_model_vector_reduced"]

import warp as wp

import dav


@dav.cached
def get_field_model_vector_reduced(inner_field_model, component: int | None = None, magnitude: bool = False):
    """Get a read-only FieldModel that reduces a vector field to a scalar.

    Exactly one of ``component`` or ``magnitude`` must be specified.

    Args:
        inner_field_model: A FieldModel whose value type is a vector dtype.
        component: 0-based index of the component to extract, or ``None``.
        magnitude: If ``True``, compute the L2 norm of each vector value.
            Only supported for floating-point vector types.

    Returns:
        A read-only scalar FieldModel wrapping *inner_field_model*.

    Raises:
        AssertionError: If the inner value type is not a vector, if not exactly
            one of *component*/*magnitude* is specified, if *component* is out
            of range, or if *magnitude* is requested for a non-floating-point type.
    """
    inner_value_type = type(inner_field_model.FieldAPI.zero())
    assert dav.utils.is_vector_dtype(inner_value_type), f"Inner field model must have a vector value type, got {inner_value_type}"
    assert (component is not None) != magnitude, "Exactly one of 'component' or 'magnitude' must be specified"

    vec_length = dav.utils.get_vector_length(inner_value_type)

    if component is not None:
        assert 0 <= component < vec_length, f"Component index {component} out of range for vector of length {vec_length}"

    if magnitude:
        assert dav.utils.is_floating_point_dtype(inner_value_type), "Magnitude is only supported for floating-point vector types"

    ScalarType = dav.utils.get_scalar_dtype(inner_value_type)
    Vec3Type = dav.utils.get_vector_dtype(ScalarType, 3)
    ZeroScalar = ScalarType(0)
    ZeroVec3 = Vec3Type(ZeroScalar)

    @wp.struct
    class FieldHandleVectorReduced:
        association: wp.int32
        inner: inner_field_model.FieldHandle

    if component is not None:

        @dav.func
        def get(field: FieldHandleVectorReduced, idx: wp.int32) -> ScalarType:
            vec = inner_field_model.FieldAPI.get(field.inner, idx)
            return vec[component]
    else:

        @dav.func
        def get(field: FieldHandleVectorReduced, idx: wp.int32) -> ScalarType:
            vec = inner_field_model.FieldAPI.get(field.inner, idx)
            return wp.length(vec)

    class FieldAPIVectorReduced:
        @staticmethod
        @dav.func
        def get_association(field: FieldHandleVectorReduced) -> wp.int32:
            return field.association

        @staticmethod
        @dav.func
        def get_count(field: FieldHandleVectorReduced) -> wp.int32:
            return inner_field_model.FieldAPI.get_count(field.inner)

        @staticmethod
        @dav.func
        def get(field: FieldHandleVectorReduced, idx: wp.int32) -> ScalarType:
            return get(field, idx)

        @staticmethod
        @dav.func
        def set(field: FieldHandleVectorReduced, idx: wp.int32, value: ScalarType):
            pass  # read-only

        @staticmethod
        @dav.func
        def zero() -> ScalarType:
            return ZeroScalar

        @staticmethod
        @dav.func
        def zero_s() -> ScalarType:
            return ZeroScalar

        @staticmethod
        @dav.func
        def zero_vec3() -> Vec3Type:
            return ZeroVec3

    class FieldModelMeta(type):
        def __repr__(cls):
            mode = f"component={component}" if component is not None else "magnitude"
            return f"FieldModelVectorReduced(inner={inner_field_model}, {mode})"

    class FieldModelVectorReduced(metaclass=FieldModelMeta):
        FieldHandle = FieldHandleVectorReduced
        FieldAPI = FieldAPIVectorReduced

    return FieldModelVectorReduced
