# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

__all__ = ["ElementType", "get_vertex_count", "get_linear_element", "get_face_count", "get_vertex_counts"]

from enum import Enum

import numpy as np


class ElementType(Enum):
    """Corresponds to CGNS ElementType_t enum"""

    ElementTypeNull = 0
    ElementTypeUserDefined = 1
    NODE = 2
    BAR_2 = 3
    BAR_3 = 4
    TRI_3 = 5
    TRI_6 = 6
    QUAD_4 = 7
    QUAD_8 = 8
    QUAD_9 = 9
    TETRA_4 = 10
    TETRA_10 = 11
    PYRA_5 = 12
    PYRA_14 = 13
    PENTA_6 = 14
    PENTA_15 = 15
    PENTA_18 = 16
    HEXA_8 = 17
    HEXA_20 = 18
    HEXA_27 = 19
    MIXED = 20
    PYRA_13 = 21
    NGON_n = 22
    NFACE_n = 23
    BAR_4 = 24
    TRI_9 = 25
    TRI_10 = 26
    QUAD_12 = 27
    QUAD_16 = 28
    TETRA_16 = 29
    TETRA_20 = 30
    PYRA_21 = 31
    PYRA_29 = 32
    PYRA_30 = 33
    PENTA_24 = 34
    PENTA_38 = 35
    PENTA_40 = 36
    HEXA_32 = 37
    HEXA_56 = 38
    HEXA_64 = 39
    BAR_5 = 40
    TRI_12 = 41
    TRI_15 = 42
    QUAD_P4_16 = 43
    QUAD_25 = 44
    TETRA_22 = 45
    TETRA_34 = 46
    TETRA_35 = 47
    PYRA_P4_29 = 48
    PYRA_50 = 49
    PYRA_55 = 50
    PENTA_33 = 51
    PENTA_66 = 52
    PENTA_75 = 53
    HEXA_44 = 54
    HEXA_98 = 55
    HEXA_125 = 56


NofValidElementTypes = 57


def get_vertex_count(etype: ElementType) -> int:
    match etype:
        case ElementType.ElementTypeNull:
            return 0
        case ElementType.ElementTypeUserDefined:
            return 0
        case ElementType.NODE:
            return 0
        case ElementType.MIXED:
            return 0
        case ElementType.NGON_n:
            return 0
        case ElementType.NFACE_n:
            return 0
    txt = etype.name
    return int(txt.rsplit("_", 1)[-1])


def get_linear_element(etype: ElementType) -> ElementType:
    txt = etype.name
    if txt.startswith("BAR_"):
        return ElementType.BAR_2
    elif txt.startswith("TRI_"):
        return ElementType.TRI_3
    elif txt.startswith("QUAD_"):
        return ElementType.QUAD_4
    elif txt.startswith("TETRA_"):
        return ElementType.TETRA_4
    elif txt.startswith("PYRA_"):
        return ElementType.PYRA_5
    elif txt.startswith("PENTA_"):
        return ElementType.PENTA_6
    elif txt.startswith("HEXA_"):
        return ElementType.HEXA_8
    else:
        return ElementType.ElementTypeNull


def get_face_count(etype: ElementType) -> int:
    etype = get_linear_element(etype)
    match etype:
        case ElementType.ElementTypeNull:
            return 0
        case ElementType.TRI_3:
            return 1
        case ElementType.QUAD_4:
            return 1
        case ElementType.TETRA_4:
            return 4
        case ElementType.PYRA_5:
            return 5
        case ElementType.PENTA_6:
            return 5
        case ElementType.HEXA_8:
            return 6
    return 0


def get_face_counts() -> np.ndarray:
    counts = np.zeros(NofValidElementTypes, dtype=np.int32)
    for t in ElementType:
        counts[t.value] = get_face_count(t)
    return counts


def get_face_vertex_count(etype: ElementType) -> int:
    return np.count_nonzero(get_faces(etype))


def get_face_vertex_counts() -> np.ndarray:
    counts = np.zeros(NofValidElementTypes, dtype=np.int32)
    for t in ElementType:
        counts[t.value] = get_face_vertex_count(t)
    return counts


def get_vertex_counts() -> np.ndarray:
    array = np.zeros(NofValidElementTypes, dtype=np.int32)
    for t in ElementType:
        array[t.value] = get_vertex_count(t)
    return array


def get_faces(etype: ElementType) -> np.ndarray:
    etype = get_linear_element(etype)
    match etype:
        case ElementType.TRI_3:
            return np.array([[1, 2, 3]], dtype=np.int32)
        case ElementType.QUAD_4:
            return np.array([[1, 2, 3, 4]], dtype=np.int32)
        case ElementType.TETRA_4:
            return np.array([(1, 3, 2), (1, 2, 4), (2, 3, 4), (3, 1, 4)], dtype=np.int32)
        case ElementType.PYRA_5:
            return np.array([(1, 4, 3, 2), (1, 2, 5, 0), (2, 3, 5, 0), (3, 4, 5, 0), (4, 1, 5, 0)], dtype=np.int32)
        case ElementType.PENTA_6:
            return np.array([(1, 2, 5, 4), (2, 3, 6, 5), (3, 1, 4, 6), (1, 3, 2, 0), (4, 5, 6, 0)], dtype=np.int32)
        case ElementType.HEXA_8:
            return np.array(
                [(1, 4, 3, 2), (1, 2, 6, 5), (2, 3, 7, 6), (3, 4, 8, 7), (1, 5, 8, 4), (5, 6, 7, 8)], dtype=np.int32
            )
    return np.array([], dtype=np.int32)


def get_padded_faces(etype: ElementType, width=4) -> np.ndarray:
    faces = get_faces(etype)
    if faces.ndim < 2:
        return np.zeros(shape=[1, width], dtype=np.int32)
    if faces.shape[1] >= width:
        return faces
    pad_width = width - faces.shape[1]
    return np.pad(faces, ((0, 0), (0, pad_width)), "constant", constant_values=0)


def get_face_map() -> np.ndarray:
    face_lengths = np.array([get_face_count(e) for e in ElementType], dtype=np.int32)
    max_faces = np.amax(face_lengths)
    faces = np.zeros(shape=[NofValidElementTypes, max_faces, 4], dtype=np.int32)
    for e in ElementType:
        pf = get_padded_faces(e, faces.shape[2])
        faces[e.value, 0 : pf.shape[0]] = pf
    return faces


def get_is_uniform(etype: ElementType) -> bool:
    if etype in [
        ElementType.ElementTypeNull,
        ElementType.ElementTypeUserDefined,
        ElementType.MIXED,
        ElementType.NGON_n,
        ElementType.NFACE_n,
    ]:
        return False
    return True


def get_is_volumetric(eType: ElementType) -> bool:
    if eType == ElementType.NFACE_n:
        return True
    if eType == ElementType.MIXED:
        return True
    return get_linear_element(eType).value >= ElementType.TETRA_4.value
