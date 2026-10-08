# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

import numpy as np
import ovstage


def path_query(stage, path: str):
    paths = ovstage.PathDictionary(stage)
    path_list = paths.create_path_list_from_strings([path])
    return paths, path_list, stage.query_from_path_list(path_list)


def release_path_query(paths, path_list, query) -> None:
    query.release().wait()
    paths.destroy_path_list(path_list)
    paths.destroy()


def write_schemas(stage, path: str, schemas: list[str], ordinal: int) -> None:
    paths, path_list, query = path_query(stage, path)
    try:
        values = np.asarray([paths.intern_token(schema) for schema in schemas], dtype=np.uint64)
        stage.write_attribute(
            query,
            "usd-schemas",
            ordinal,
            [values],
            is_array=True,
        ).wait()
    finally:
        release_path_query(paths, path_list, query)


def write_scalar(stage, path: str, attribute: str, value, ordinal: int) -> None:
    paths, path_list, query = path_query(stage, path)
    try:
        stage.write_attribute(
            query,
            attribute,
            ordinal,
            np.asarray([value]),
            is_array=False,
        ).wait()
    finally:
        release_path_query(paths, path_list, query)


def write_token(stage, path: str, attribute: str, value: str, ordinal: int) -> None:
    paths, path_list, query = path_query(stage, path)
    try:
        token = np.asarray([paths.intern_token(value)], dtype=np.uint64)
        stage.write_attribute(
            query,
            attribute,
            ordinal,
            token,
            is_array=False,
            semantic=(
                ovstage.AttributeSemantic.NONE
                if attribute in {"usd-prim-type", "usd-schemas"}
                else ovstage.AttributeSemantic.TOKEN_ID
            ),
        ).wait()
    finally:
        release_path_query(paths, path_list, query)


def write_relationship(stage, path: str, attribute: str, targets: list[str], ordinal: int) -> None:
    paths, path_list, query = path_query(stage, path)
    try:
        values = np.asarray([paths.intern_path(target) for target in targets], dtype=np.uint64)
        stage.write_attribute(
            query,
            attribute,
            ordinal,
            values,
            is_array=True,
            semantic=ovstage.AttributeSemantic.RELATIONSHIP_PATH_ID,
        ).wait()
    finally:
        release_path_query(paths, path_list, query)


def delete_prim(stage, path: str, ordinal: int) -> None:
    paths, path_list, query = path_query(stage, path)
    try:
        stage.delete_attributes(query, [], ordinal).wait()
    finally:
        release_path_query(paths, path_list, query)


def read_scalar(stage, path: str, attribute: str, ordinal: int):
    paths, path_list, query = path_query(stage, path)
    try:
        token = paths.intern_token(attribute)
        with stage.read_attributes(query, [token], ovstage.OrdinalRange.latest(ordinal)) as read:
            read.wait()
            for group in read.groups():
                try:
                    if not group.is_delete and group.tensor_count:
                        values = group.array(0)
                        return values[0].item() if len(values) else None
                finally:
                    stage.release_group(group)
        return None
    finally:
        release_path_query(paths, path_list, query)


def read_array(stage, path: str, attribute: str, ordinal: int) -> np.ndarray | None:
    paths, path_list, query = path_query(stage, path)
    try:
        token = paths.intern_token(attribute)
        with stage.read_attributes(query, [token], ovstage.OrdinalRange.latest(ordinal)) as read:
            read.wait()
            for group in read.groups():
                try:
                    if not group.is_delete and group.tensor_count:
                        import warp as wp

                        return wp.from_dlpack(group.dlpack(0)).numpy().copy()
                finally:
                    stage.release_group(group)
        return None
    finally:
        release_path_query(paths, path_list, query)


def write_transform(stage, path, translation, ordinal):
    """Author a native row-vector translation matrix with MATRIX semantics."""
    matrix = np.eye(4, dtype=np.float64)
    matrix[3, :3] = translation
    paths, plist, query = path_query(stage, path)
    try:
        tensor = ovstage.make_dltensor(
            matrix.reshape(1, 16),
            dtype=ovstage.DLDataType(code=ovstage.DLDataTypeCode.kDLFloat, bits=64, lanes=16),
            shape=[1],
        )
        stage.write_attribute(
            query, "omni:xform", ordinal, tensor, is_array=False, semantic=ovstage.AttributeSemantic.MATRIX
        ).wait()
    finally:
        release_path_query(paths, plist, query)
