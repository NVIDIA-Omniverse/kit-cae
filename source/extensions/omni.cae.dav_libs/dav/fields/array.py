# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


import warp as wp

import dav

# =============================================================================
# FieldHandle Types
# =============================================================================


@dav.cached
def get_field_model_SoA(dtype, length: int):
    # not sure this is a reasonable restriction, but it's here for now
    assert not dav.utils.is_vector_dtype(dtype), "Only scalar dtypes are supported for SoA field handles"
    assert length > 1, "Length must be greater than 1"

    ScalarType = dtype
    Vec3Type = dav.utils.get_vector_dtype(ScalarType, 3)
    ValueType = dav.utils.get_vector_dtype(ScalarType, length)

    ZeroScalar = ScalarType(0)  # seems like 0 works more than 0.0
    ZeroVec3 = Vec3Type(ZeroScalar)
    ZeroValue = ValueType(ZeroScalar)

    @wp.struct
    class ArrayContainer:
        array: wp.array(dtype=ScalarType)

    @wp.struct
    class FieldHandleSoA:
        association: wp.int32
        data: wp.array(dtype=ArrayContainer)

        @staticmethod
        def _set_data(me, data: list[wp.array], device):
            containers = []
            for arr in data:
                c = ArrayContainer()
                c.array = arr
                containers.append(c)
            me.data = wp.array(containers, dtype=ArrayContainer, device=device)

    # Note: we don't use statics to keep hashes from differening
    # between runs.
    class FieldAPISoA:
        @staticmethod
        @dav.func
        def get_association(field: FieldHandleSoA) -> wp.int32:
            """Get the association type of the field."""
            return field.association

        @staticmethod
        @dav.func
        def get_count(field: FieldHandleSoA) -> wp.int32:
            """Get the number of elements in the field."""
            return field.data[0].array.shape[0]

        @staticmethod
        @dav.func
        def get(field: FieldHandleSoA, idx: wp.int32) -> ValueType:
            """Get a value from the field by index."""
            value = ValueType()
            for i in range(length):
                value[i] = field.data[i].array[idx]
            return value

        @staticmethod
        @dav.func
        def set(field: FieldHandleSoA, idx: wp.int32, value: ValueType):
            """Set a value in the field by index."""
            for i in range(length):
                field.data[i].array[idx] = value[i]

        @staticmethod
        @dav.func
        def zero() -> ValueType:
            """Get a zero value of the appropriate type."""
            return ZeroValue

        @staticmethod
        @dav.func
        def zero_s() -> ScalarType:
            """Get a zero scalar value of the appropriate type."""
            return ZeroScalar

        @staticmethod
        @dav.func
        def zero_vec3() -> Vec3Type:
            """Get a zero vec3 value of the appropriate type."""
            return ZeroVec3

    class FieldModelMeta(type):
        def __repr__(cls):
            return f"FieldModelSoA(dtype={dtype.__name__}, length={length})"

    class FieldModelSoA(metaclass=FieldModelMeta):
        FieldHandle = FieldHandleSoA
        FieldAPI = FieldAPISoA

    return FieldModelSoA


@dav.cached
def get_field_model_AoS(dtype, length: int):
    assert length > 0, "Length must be greater than 0"
    assert not dav.utils.is_vector_dtype(dtype), "Only scalar dtypes are supported for AoS field handles"

    ScalarType = dtype
    Vec3Type = dav.utils.get_vector_dtype(ScalarType, 3)
    ValueType = dav.utils.get_vector_dtype(ScalarType, length)

    ZeroScalar = ScalarType(0)  # seems like 0 works more than 0.0
    ZeroVec3 = Vec3Type(ZeroScalar)
    ZeroValue = ValueType(ZeroScalar)

    @wp.struct
    class FieldHandleAoS:
        association: wp.int32
        data: wp.array(dtype=ValueType)

    class FieldAPIAoS:
        @staticmethod
        @dav.func
        def get_association(field: FieldHandleAoS) -> wp.int32:
            """Get the association type of the field."""
            return field.association

        @staticmethod
        @dav.func
        def get_count(field: FieldHandleAoS) -> wp.int32:
            """Get the number of elements in the field."""
            return field.data.shape[0]

        @staticmethod
        @dav.func
        def get(field: FieldHandleAoS, idx: wp.int32) -> ValueType:
            """Get a value from the field by index."""
            return field.data[idx]

        @staticmethod
        @dav.func
        def set(field: FieldHandleAoS, idx: wp.int32, value: ValueType):
            """Set a value in the field by index."""
            field.data[idx] = value

        @staticmethod
        @dav.func
        def zero() -> ValueType:
            """Get a zero value of the appropriate type."""
            return ZeroValue

        @staticmethod
        @dav.func
        def zero_s() -> ScalarType:
            """Get a zero scalar value of the appropriate type."""
            return ZeroScalar

        @staticmethod
        @dav.func
        def zero_vec3() -> Vec3Type:
            """Get a zero vec3 value of the appropriate type."""
            return ZeroVec3

    class FieldModelMeta(type):
        def __repr__(cls):
            return f"FieldModelAoS(dtype={dtype.__name__}, length={length})"

    class FieldModelAoS(metaclass=FieldModelMeta):
        FieldHandle = FieldHandleAoS
        FieldAPI = FieldAPIAoS

    return FieldModelAoS
