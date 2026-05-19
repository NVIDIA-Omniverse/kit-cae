# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Read-only field models that expose selected values from an inner field."""

__all__ = ["get_field_model_indexed_subset", "get_field_model_subrange"]

import warp as wp

import dav


@dav.cached
def get_field_model_subrange(inner_field_model: dav.FieldModel) -> dav.FieldModel:
    """Get a read-only FieldModel that exposes a contiguous range of values.

    Runtime handles carry the ``start`` and ``count`` values.  Logical index
    ``i`` in this field maps to ``start + i`` in the inner field.
    """
    ValueType = type(inner_field_model.FieldAPI.zero())

    @wp.struct
    class FieldHandleSubrange:
        association: wp.int32
        inner: inner_field_model.FieldHandle
        start: wp.int32
        count: wp.int32

    class FieldAPISubrange:
        @staticmethod
        @dav.func
        def get_association(field: FieldHandleSubrange) -> wp.int32:
            return field.association

        @staticmethod
        @dav.func
        def get_count(field: FieldHandleSubrange) -> wp.int32:
            return field.count

        @staticmethod
        @dav.func
        def get(field: FieldHandleSubrange, idx: wp.int32) -> ValueType:
            return inner_field_model.FieldAPI.get(field.inner, field.start + idx)

        @staticmethod
        @dav.func
        def set(field: FieldHandleSubrange, idx: wp.int32, value: ValueType):
            pass  # read-only

        @staticmethod
        @dav.func
        def zero() -> ValueType:
            return inner_field_model.FieldAPI.zero()

        @staticmethod
        @dav.func
        def zero_s():
            return inner_field_model.FieldAPI.zero_s()

        @staticmethod
        @dav.func
        def zero_vec3():
            return inner_field_model.FieldAPI.zero_vec3()

    class FieldModelMeta(type):
        def __repr__(cls):
            return f"FieldModelSubrange(inner={inner_field_model})"

    class FieldModelSubrange(metaclass=FieldModelMeta):
        FieldHandle = FieldHandleSubrange
        FieldAPI = FieldAPISubrange

    return FieldModelSubrange


@dav.cached
def get_field_model_indexed_subset(inner_field_model: dav.FieldModel) -> dav.FieldModel:
    """Get a read-only FieldModel that exposes values by explicit inner indices.

    Runtime handles carry a ``wp.int32`` index array.  Logical index ``i`` in
    this field maps to ``indices[i]`` in the inner field.
    """
    ValueType = type(inner_field_model.FieldAPI.zero())

    @wp.struct
    class FieldHandleIndexedSubset:
        association: wp.int32
        inner: inner_field_model.FieldHandle
        indices: wp.array(dtype=wp.int32)

    class FieldAPIIndexedSubset:
        @staticmethod
        @dav.func
        def get_association(field: FieldHandleIndexedSubset) -> wp.int32:
            return field.association

        @staticmethod
        @dav.func
        def get_count(field: FieldHandleIndexedSubset) -> wp.int32:
            return field.indices.shape[0]

        @staticmethod
        @dav.func
        def get(field: FieldHandleIndexedSubset, idx: wp.int32) -> ValueType:
            return inner_field_model.FieldAPI.get(field.inner, field.indices[idx])

        @staticmethod
        @dav.func
        def set(field: FieldHandleIndexedSubset, idx: wp.int32, value: ValueType):
            pass  # read-only

        @staticmethod
        @dav.func
        def zero() -> ValueType:
            return inner_field_model.FieldAPI.zero()

        @staticmethod
        @dav.func
        def zero_s():
            return inner_field_model.FieldAPI.zero_s()

        @staticmethod
        @dav.func
        def zero_vec3():
            return inner_field_model.FieldAPI.zero_vec3()

    class FieldModelMeta(type):
        def __repr__(cls):
            return f"FieldModelIndexedSubset(inner={inner_field_model})"

    class FieldModelIndexedSubset(metaclass=FieldModelMeta):
        FieldHandle = FieldHandleIndexedSubset
        FieldAPI = FieldAPIIndexedSubset

    return FieldModelIndexedSubset
