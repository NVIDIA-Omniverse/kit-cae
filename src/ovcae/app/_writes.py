# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Typed OVStage attribute writes for application edits.

An application changes a prim's value by writing it at a new ordinal through
an OVStage query. The native write needs the value's USD type to pick the
element dtype, lane count, and semantic; this module maps USD type names to
those and hides the query and path-dictionary boilerplate. ``ovstage`` and
NumPy load on first call, never at import.
"""

from __future__ import annotations

import re
from typing import Any

# USD scalar type names and the NumPy dtype OVStage expects for them.
_SCALARS = {
    "bool": "bool_",
    "uchar": "uint8",
    "int": "int32",
    "uint": "uint32",
    "int64": "int64",
    "uint64": "uint64",
    "half": "float16",
    "float": "float32",
    "double": "float64",
    "timecode": "float64",
}
# Role and matrix types spelled as scalar type plus lane count.
_ALIASES = {
    "point3f": "float3",
    "normal3f": "float3",
    "vector3f": "float3",
    "color3f": "float3",
    "color4f": "float4",
    "texCoord2f": "float2",
    "texCoord3f": "float3",
    "quatf": "float4",
    "point3d": "double3",
    "normal3d": "double3",
    "vector3d": "double3",
    "color3d": "double3",
    "color4d": "double4",
    "quatd": "double4",
    "matrix2d": "double4",
    "matrix3d": "double9",
    "matrix4d": "double16",
}
_SEMANTICS = {
    "point3f": "POINT",
    "point3d": "POINT",
    "normal3f": "NORMAL",
    "normal3d": "NORMAL",
    "vector3f": "VECTOR",
    "vector3d": "VECTOR",
    "color3f": "COLOR",
    "color4f": "COLOR",
    "color3d": "COLOR",
    "color4d": "COLOR",
    "texCoord2f": "TEXTURE_COORDINATE",
    "texCoord3f": "TEXTURE_COORDINATE",
    "quatf": "QUATERNION",
    "quatd": "QUATERNION",
    "matrix2d": "MATRIX",
    "matrix3d": "MATRIX",
    "matrix4d": "MATRIX",
    "timecode": "TIME_CODE",
}
_VECTOR = re.compile(r"^([a-z]+?)(\d+)$")


def element_layout(type_name: str) -> tuple[str, int, str | None]:
    """``(numpy dtype name, lanes, OVStage semantic name)`` for a USD value type.

    ``float2`` is two float32 lanes, ``matrix4d`` sixteen float64 lanes with the
    ``MATRIX`` semantic. Raises ``ValueError`` for a type this module cannot
    write, including ``token`` and ``string``, which have their own writers.
    """

    element = type_name.removesuffix("[]")
    spelled = _ALIASES.get(element, element)
    if spelled in _SCALARS:
        return _SCALARS[spelled], 1, _SEMANTICS.get(element)
    match = _VECTOR.match(spelled)
    if match and match.group(1) in _SCALARS:
        return _SCALARS[match.group(1)], int(match.group(2)), _SEMANTICS.get(element)
    raise ValueError(f"cannot write USD type {type_name!r}")


def write_attribute(stage, path: str, name: str, value: Any, ordinal: int, *, type_name: str) -> None:
    """Write ``value`` to ``name`` on ``path`` at ``ordinal`` as USD type ``type_name``.

    ``token`` and ``string`` values are interned; ``rel`` writes relationship
    targets from a sequence of prim paths; every other type goes through
    :func:`element_layout`. A ``[]`` suffix writes an array, one element per
    row of the value. The prim is created when it does not exist.
    """

    import ovstage

    with ovstage.PathDictionary(stage) as paths:
        path_list = paths.create_path_list_from_strings([path])
        query = stage.query_from_path_list(path_list)
        try:
            kwargs = _tensor_arguments(paths, value, type_name)
            stage.write_attribute(query, name, ordinal=ordinal, prim_mode=ovstage.PrimMode.UPSERT, **kwargs).wait()
        finally:
            stage.release_query(query).wait()
            paths.destroy_path_list(path_list)


def _tensor_arguments(paths, value, type_name: str) -> dict[str, Any]:
    import numpy as np
    import ovstage

    is_array = type_name.endswith("[]")
    element = type_name.removesuffix("[]")
    if element in {"token", "string"}:
        items = [str(item) for item in value] if is_array else [str(value)]
        semantic = ovstage.AttributeSemantic.TOKEN_ID if element == "token" else ovstage.AttributeSemantic.STRING
        tokens = np.asarray([paths.intern_token(item) for item in items], dtype=np.uint64)
        return {"tensors": tokens, "is_array": is_array, "semantic": semantic}
    if element == "rel":
        targets = [value] if isinstance(value, str) else list(value)
        ids = np.asarray([paths.intern_path(str(target)) for target in targets], dtype=np.uint64)
        return {"tensors": ids, "is_array": True, "semantic": ovstage.AttributeSemantic.RELATIONSHIP_PATH_ID}
    dtype_name, lanes, semantic_name = element_layout(type_name)
    dtype = np.dtype(dtype_name)
    flat = np.asarray(value, dtype=dtype).reshape(-1)
    if not is_array and flat.size != lanes:
        raise ValueError(f"{type_name} takes one value with {lanes} component(s); got {flat.size}")
    if flat.size % lanes:
        raise ValueError(f"{type_name} elements have {lanes} component(s); got {flat.size} values")
    data = np.ascontiguousarray(flat.reshape(-1, lanes))
    arguments: dict[str, Any] = {"is_array": is_array}
    if semantic_name is not None:
        arguments["semantic"] = getattr(ovstage.AttributeSemantic, semantic_name)
    if lanes == 1:
        arguments["tensors"] = data.reshape(-1)
    else:
        arguments["tensors"] = ovstage.make_dltensor(
            data, dtype=ovstage.numpy_to_dldatatype(dtype, lanes=lanes), shape=[len(data)], ndim=1
        )
    return arguments


# DLPack type codes OVStage reports, with the bit width, back to NumPy dtype names.
_DL_CODES = {(0, 8): "int8", (0, 32): "int32", (0, 64): "int64", (1, 8): "uint8", (1, 32): "uint32", (1, 64): "uint64"}
_DL_CODES.update({(2, 16): "float16", (2, 32): "float32", (2, 64): "float64", (6, 8): "bool_"})
_DTYPE_NAMES = {name: usd for usd, name in _SCALARS.items() if usd != "timecode"}
_SEMANTIC_TYPES = {
    ("MATRIX", "float64", 16): "matrix4d",
    ("MATRIX", "float64", 9): "matrix3d",
    ("MATRIX", "float64", 4): "matrix2d",
    ("POINT", "float32", 3): "point3f",
    ("POINT", "float64", 3): "point3d",
    ("NORMAL", "float32", 3): "normal3f",
    ("NORMAL", "float64", 3): "normal3d",
    ("VECTOR", "float32", 3): "vector3f",
    ("VECTOR", "float64", 3): "vector3d",
    ("COLOR", "float32", 3): "color3f",
    ("COLOR", "float32", 4): "color4f",
    ("COLOR", "float64", 3): "color3d",
    ("COLOR", "float64", 4): "color4d",
    ("TEXTURE_COORDINATE", "float32", 2): "texCoord2f",
    ("TEXTURE_COORDINATE", "float32", 3): "texCoord3f",
    ("QUATERNION", "float32", 4): "quatf",
    ("QUATERNION", "float64", 4): "quatd",
}


def committed_type(stage, path: str, name: str, ordinal: int) -> str | None:
    """The USD type name of the value committed for ``name`` on ``path`` at ``ordinal``, or None when unset.

    Recovered from the native element dtype, lane count, and semantic, so a
    new value is written with the type the attribute already has; OVStage
    refuses to change an attribute's type in place.
    """

    import ovstage

    with ovstage.PathDictionary(stage) as paths:
        path_list = paths.create_path_list_from_strings([path])
        query = stage.query_from_path_list(path_list)
        try:
            with stage.read_attributes(query, [paths.intern_token(name)], ovstage.OrdinalRange.latest(ordinal)) as read:
                read.wait()
                for group in read.groups():
                    try:
                        if group.is_delete or not group.tensor_count:
                            continue
                        dtype = group.raw.data.tensors[0].dtype
                        semantic = ovstage.AttributeSemantic(group.raw.semantic).name
                        dtype_name = _DL_CODES.get((int(dtype.code), int(dtype.bits)))
                        return _type_from_layout(dtype_name, int(dtype.lanes), semantic, bool(group.is_array))
                    finally:
                        stage.release_group(group)
        finally:
            stage.release_query(query).wait()
            paths.destroy_path_list(path_list)
    return None


def _type_from_layout(dtype: str | None, lanes: int, semantic: str, is_array: bool) -> str | None:
    suffix = "[]" if is_array else ""
    if semantic == "TOKEN_ID":
        return "token" + suffix
    if semantic == "STRING":
        return "string" + suffix
    if semantic in {"RELATIONSHIP_PATH_ID", "CONNECTION_PATH_ID"}:
        return "rel"
    if semantic == "TIME_CODE":
        return "timecode" + suffix
    base = _DTYPE_NAMES.get(dtype)
    if base is None:
        return None
    role = _SEMANTIC_TYPES.get((semantic, dtype, lanes))
    if role is not None:
        return role + suffix
    return (base if lanes == 1 else f"{base}{lanes}") + suffix


__all__ = ["committed_type", "element_layout", "write_attribute"]
