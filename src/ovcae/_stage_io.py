# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Small private ovstage access layer used by the core Session."""

from __future__ import annotations

import ctypes
from contextlib import suppress

OPERATOR_SCHEMA = "CaeVizOperatorAPI"
TEST_OPERATOR_SCHEMA = "OvcaeTestOperatorAPI"
TEST_VALUE = "ovcae:test:value"
TEST_PRODUCT_VALUE = "ovcae:test:productValue"
PRODUCT_ATTRIBUTES = "ovcae:product:attributes"


class ProductPublication:
    """An enqueued product write whose inputs stay alive until ``wait()``.

    OVStage's ``Operation`` retains the borrowed DLTensors supplied to a write.
    Keeping the operations here therefore keeps both CPU and device storage
    alive without another ownership wrapper.
    """

    def __init__(self, attributes: frozenset[str], operations, resources):
        self.attributes = attributes
        self._operations = tuple(operations)
        self._resources = resources
        self._waited = False
        self._failure = None

    def wait(self) -> frozenset[str]:
        if self._waited:
            if self._failure is not None:
                raise self._failure
            return self.attributes
        self._waited = True

        failure = None
        if self._operations:
            # The last same-ordinal operation depends on those submitted before
            # it. Waiting for it first lets all product mutations run before we
            # make the earlier, already-complete waits that release their handles.
            ordered = (self._operations[-1], *reversed(self._operations[:-1]))
            for operation in ordered:
                try:
                    operation.wait()
                except Exception as exc:
                    if failure is None:
                        failure = exc

        paths, path_list, query = self._resources
        try:
            _release_path_query(paths, path_list, query)
        except Exception as exc:
            if failure is None:
                failure = exc
        finally:
            self._resources = None

        self._failure = failure
        if failure is not None:
            raise failure
        return self.attributes


def write_floor(stage) -> int:
    with stage.get_attribute_write_floor() as query:
        query.wait()
        return query.fetch()


def discover_operators(stage, input_ordinal: int) -> dict[str, frozenset[str]]:
    """Discover only prims whose Execution Stage metadata marks an operator."""

    import ovstage

    paths = ovstage.PathDictionary(stage)
    schema_token = paths.intern_token("usd-schemas")
    predicate = ovstage.Predicate("usd-schemas", ovstage.FilterOp.CONTAINS, [OPERATOR_SCHEMA])
    discovered: dict[str, frozenset[str]] = {}
    try:
        with stage.query(filter=ovstage.Filter([predicate])) as query:
            query.wait()
            with stage.read_attributes(query, [schema_token], ovstage.OrdinalRange.latest(input_ordinal)) as read:
                read.wait()
                for group in read.groups():
                    try:
                        if group.is_delete:
                            continue
                        prim_paths = paths.get_paths(group.prim_list)
                        for local in range(group.prim_count):
                            path = paths.path_to_string(prim_paths[group.prim_index(local)])
                            row = group.data_row_index(local)
                            values = group.array(row if group.tensor_count > 1 else 0)
                            schemas = frozenset(paths.token_to_string(int(value)) for value in values)
                            if OPERATOR_SCHEMA in schemas:
                                discovered[path] = schemas
                    finally:
                        stage.release_group(group)
    finally:
        paths.destroy()
    return discovered


def read_value(stage, path: str, attribute: str, ordinal: int, *, type_name: str | None = None, resources=None):
    import ovstage

    paths, path_list, query = resources if resources is not None else _path_query(stage, path)
    try:
        token = paths.intern_token(attribute)
        with stage.read_attributes(query, [token], ovstage.OrdinalRange.latest(ordinal)) as read:
            read.wait()
            for group in read.groups():
                try:
                    if not group.is_delete and group.tensor_count:
                        values = group.array(0)
                        if type_name == "point3f[]":
                            # Scene arrays own their storage before native read release.
                            return values.copy().reshape(-1, 3)
                        if len(values):
                            decoded = [
                                _decode_value(paths, group.raw.semantic, attribute, value, type_name)
                                for value in values
                            ]
                            return tuple(decoded) if group.is_array or len(decoded) > 1 else decoded[0]
                finally:
                    stage.release_group(group)
        return None
    finally:
        if resources is None:
            _release_path_query(paths, path_list, query)


def read_scalar(stage, path: str, attribute: str, ordinal: int):
    return read_value(stage, path, attribute, ordinal)


def _decode_value(paths, semantic, attribute: str, value, type_name: str | None):
    import ovstage

    try:
        value = value.item()
    except ValueError:
        value = tuple(value.tolist())
    semantic = ovstage.AttributeSemantic(semantic)
    if semantic in {
        ovstage.AttributeSemantic.RELATIONSHIP_PATH_ID,
        ovstage.AttributeSemantic.CONNECTION_PATH_ID,
    } or attribute.endswith(":target"):
        return paths.path_to_string(int(value))
    if semantic in {
        ovstage.AttributeSemantic.STRING,
        ovstage.AttributeSemantic.TOKEN_ID,
        ovstage.AttributeSemantic.PATH_EXPRESSION_STRING,
    } or type_name in {"string", "token", "token[]"}:
        return paths.token_to_string(int(value))
    # Native reserved metadata stores token IDs but requires semantic NONE.
    if attribute in {"usd-prim-type", "usd-schemas"}:
        return paths.token_to_string(int(value))
    return value


def write_test_product(stage, path: str, value: float, ordinal: int, previous: frozenset[str]) -> ProductPublication:
    import numpy as np

    paths, path_list, query = _path_query(stage, path)
    operations = []
    try:
        current = frozenset({TEST_PRODUCT_VALUE, PRODUCT_ATTRIBUTES})
        operations = _reset_product_attributes(stage, query, previous, current, ordinal)
        writes = [_ownership_manifest(paths, {TEST_PRODUCT_VALUE})]
        import ovstage

        writes.append(ovstage.WriteDesc(TEST_PRODUCT_VALUE, np.asarray([value], dtype=np.float64), is_array=False))
        operations.append(stage.write_attributes(query, writes, ordinal))
        return ProductPublication(current, operations, (paths, path_list, query))
    except Exception:
        _discard_product_publication(operations, paths, path_list, query)
        raise


def write_mesh_product(stage, path: str, product, ordinal: int, *, previous: frozenset[str]) -> ProductPublication:
    import numpy as np
    import ovstage

    paths, path_list, query = _path_query(stage, path)
    operations = []
    try:
        none_token = np.asarray([paths.intern_token("none")], dtype=np.uint64)
        writes = [
            ovstage.WriteDesc(
                "points",
                _point3f_tensor(product.points),
                is_array=True,
                semantic=ovstage.AttributeSemantic.POINT,
            ),
            ovstage.WriteDesc("faceVertexCounts", product.face_vertex_counts, is_array=True),
            ovstage.WriteDesc("faceVertexIndices", product.face_vertex_indices, is_array=True),
            ovstage.WriteDesc(
                "subdivisionScheme",
                none_token,
                is_array=False,
                semantic=ovstage.AttributeSemantic.TOKEN_ID,
            ),
        ]

        def token(value: str):
            return np.asarray([paths.intern_token(value)], dtype=np.uint64)

        for primvar in product.primvars:
            writes.extend(_primvar_writes(ovstage, token, primvar))
        product_attributes = {str(desc.attribute) for desc in writes}
        writes.append(_ownership_manifest(paths, product_attributes))
        current = frozenset(product_attributes | {PRODUCT_ATTRIBUTES})
        operations = _reset_product_attributes(stage, query, previous, current, ordinal)
        operations.append(stage.write_attributes(query, writes, ordinal))
        return ProductPublication(current, operations, (paths, path_list, query))
    except Exception:
        _discard_product_publication(operations, paths, path_list, query)
        raise


def _primvar_values(values):
    """Publish double-precision primvars as their float32 USD counterparts.

    Scientific fields frequently arrive as float64, but a USD ``double[]``
    primvar is invisible to renderer materials: the MDL ``scene::data_lookup_float``
    calls in the CAE library read ``float`` primvars only, so colors silently fall
    back to their default. Cast on the producing device without a host round trip;
    every other dtype passes through untouched.
    """

    import numpy as np
    import warp as wp

    if isinstance(values, wp.array):
        target = {wp.float64: wp.float32, wp.vec2d: wp.vec2f, wp.vec3d: wp.vec3f, wp.vec4d: wp.vec4f}.get(values.dtype)
        if target is None:
            return values
        result = wp.empty(values.shape, dtype=target, device=values.device)
        wp.utils.array_cast(values, result)
        return result
    array = np.asarray(values)
    if array.dtype == np.float64:
        return array.astype(np.float32)
    return values


def _primvar_writes(ovstage, token, primvar):
    """Describe one product primvar and its interpolation token."""

    return [
        ovstage.WriteDesc(f"primvars:{primvar.name}", _primvar_values(primvar.values), is_array=True),
        ovstage.WriteDesc(
            f"primvars:{primvar.name}:interpolation",
            token(primvar.interpolation),
            is_array=False,
            semantic=ovstage.AttributeSemantic.TOKEN_ID,
        ),
    ]


def _widths_writes(ovstage, token, widths):
    """Publish widths as the UsdGeom attribute and as the primvar Kit-CAE materials read.

    Kit-CAE scenes author a placeholder primvars:widths sized to their placeholder
    points; a vertex primvar that no longer matches the published point count makes
    Hydra drop the prim, so the product must overwrite it.
    """
    values = _primvar_values(widths.values)
    return [
        ovstage.WriteDesc("widths", values, is_array=True),
        ovstage.WriteDesc(
            "widths:interpolation",
            token(widths.interpolation),
            is_array=False,
            semantic=ovstage.AttributeSemantic.TOKEN_ID,
        ),
        ovstage.WriteDesc("primvars:widths", values, is_array=True),
        ovstage.WriteDesc(
            "primvars:widths:interpolation",
            token(widths.interpolation),
            is_array=False,
            semantic=ovstage.AttributeSemantic.TOKEN_ID,
        ),
    ]


def write_curves_product(stage, path: str, product, ordinal: int, *, previous: frozenset[str]) -> ProductPublication:
    import numpy as np
    import ovstage

    paths, path_list, query = _path_query(stage, path)
    operations = []
    try:

        def token(value: str):
            return np.asarray([paths.intern_token(value)], dtype=np.uint64)

        writes = [
            ovstage.WriteDesc(
                "points",
                _point3f_tensor(product.points),
                is_array=True,
                semantic=ovstage.AttributeSemantic.POINT,
            ),
            ovstage.WriteDesc("curveVertexCounts", product.curve_vertex_counts, is_array=True),
            ovstage.WriteDesc(
                "type",
                token(product.curve_type),
                is_array=False,
                semantic=ovstage.AttributeSemantic.TOKEN_ID,
            ),
            ovstage.WriteDesc(
                "wrap",
                token(product.wrap),
                is_array=False,
                semantic=ovstage.AttributeSemantic.TOKEN_ID,
            ),
            *_widths_writes(ovstage, token, product.widths),
        ]
        if product.basis is not None:
            writes.append(
                ovstage.WriteDesc(
                    "basis",
                    token(product.basis),
                    is_array=False,
                    semantic=ovstage.AttributeSemantic.TOKEN_ID,
                )
            )
        for primvar in product.primvars:
            writes.extend(_primvar_writes(ovstage, token, primvar))
        product_attributes = {str(desc.attribute) for desc in writes}
        writes.append(_ownership_manifest(paths, product_attributes))
        current = frozenset(product_attributes | {PRODUCT_ATTRIBUTES})
        operations = _reset_product_attributes(stage, query, previous, current, ordinal)
        operations.append(stage.write_attributes(query, writes, ordinal))
        return ProductPublication(current, operations, (paths, path_list, query))
    except Exception:
        _discard_product_publication(operations, paths, path_list, query)
        raise


def write_points_product(stage, path: str, product, ordinal: int, *, previous: frozenset[str]) -> ProductPublication:
    import numpy as np
    import ovstage

    paths, path_list, query = _path_query(stage, path)
    operations = []
    try:

        def token(value: str):
            return np.asarray([paths.intern_token(value)], dtype=np.uint64)

        writes = [
            ovstage.WriteDesc(
                "points",
                _point3f_tensor(product.points),
                is_array=True,
                semantic=ovstage.AttributeSemantic.POINT,
            ),
            *_widths_writes(ovstage, token, product.widths),
        ]
        for primvar in product.primvars:
            writes.extend(_primvar_writes(ovstage, token, primvar))
        product_attributes = {str(desc.attribute) for desc in writes}
        writes.append(_ownership_manifest(paths, product_attributes))
        current = frozenset(product_attributes | {PRODUCT_ATTRIBUTES})
        operations = _reset_product_attributes(stage, query, previous, current, ordinal)
        operations.append(stage.write_attributes(query, writes, ordinal))
        return ProductPublication(current, operations, (paths, path_list, query))
    except Exception:
        _discard_product_publication(operations, paths, path_list, query)
        raise


def write_instancer_product(stage, path: str, product, ordinal: int, *, previous: frozenset[str]) -> ProductPublication:
    import numpy as np
    import ovstage

    paths, path_list, query = _path_query(stage, path)
    operations = []
    try:

        def token(value: str):
            return np.asarray([paths.intern_token(value)], dtype=np.uint64)

        writes = [
            ovstage.WriteDesc(
                "positions", _point3f_tensor(product.positions), is_array=True, semantic=ovstage.AttributeSemantic.POINT
            ),
            ovstage.WriteDesc("protoIndices", product.proto_indices, is_array=True),
            ovstage.WriteDesc("scales", _point3f_tensor(product.scales), is_array=True),
            ovstage.WriteDesc(
                "orientations",
                _vector_tensor(product.orientations, lanes=4, bits=16),
                is_array=True,
                semantic=ovstage.AttributeSemantic.QUATERNION,
            ),
        ]
        for primvar in product.primvars:
            writes.extend(_primvar_writes(ovstage, token, primvar))
        product_attributes = {str(desc.attribute) for desc in writes}
        writes.append(_ownership_manifest(paths, product_attributes))
        current = frozenset(product_attributes | {PRODUCT_ATTRIBUTES})
        # Keep fixed-type instance attributes alive across writes. Deleting and
        # recreating them at one ordinal leaves stale rendered instances when
        # OVRTX 0.5.0.377615 and 0.5.1.385782 consume a nonempty-to-empty transition, despite
        # correct OVStage readback. Primvars still permit replacement/type changes.
        stable = frozenset({"positions", "protoIndices", "scales", "orientations"})
        operations = _reset_product_attributes(stage, query, previous - stable, current - stable, ordinal)
        operations.append(stage.write_attributes(query, writes, ordinal))
        return ProductPublication(current, operations, (paths, path_list, query))
    except Exception:
        _discard_product_publication(operations, paths, path_list, query)
        raise


def clear_instancer_product(stage, path: str, ordinal: int, *, previous: frozenset[str]) -> ProductPublication:
    """Replace instances with an owned empty product while preserving prototypes.

    OVRTX does not reliably clear instances when transform attributes are deleted.
    Retaining neutral arrays also lets a later Session recover their ownership.
    """
    import numpy as np

    from ._products import InstancerProduct

    return write_instancer_product(
        stage,
        path,
        InstancerProduct(
            np.empty((0, 3), np.float32),
            np.empty(0, np.int32),
            np.empty((0, 3), np.float32),
            np.empty((0, 4), np.float16),
        ),
        ordinal,
        previous=previous,
    )


def delete_product_attributes(stage, path: str, attributes: frozenset[str], ordinal: int) -> None:
    if not attributes:
        return
    paths, path_list, query = _path_query(stage, path)
    try:
        stage.delete_attributes(query, sorted(attributes), ordinal=ordinal).wait()
    finally:
        _release_path_query(paths, path_list, query)


def _point3f_tensor(values):
    """Describe a contiguous point buffer as USD ``point3f[]`` elements.

    Warp exports vector arrays through DLPack as a flat scalar tensor. OVStage
    intentionally preserves that producer metadata, but a USD points attribute
    is a one-dimensional array whose elements each have three float32 lanes.
    Re-describe the same device pointer without copying it so downstream OVStage
    consumers see the layout declared by USD population.
    """

    return _vector_tensor(values, lanes=3, bits=32)


def _vector_tensor(values, *, lanes: int, bits: int):
    """Re-describe contiguous vector storage without copying CPU or GPU data."""
    import ovstage

    tensor = ovstage.make_dltensor(values)
    dtype = tensor.dtype
    if dtype.code != ovstage.DLDataTypeCode.kDLFloat or dtype.bits != bits:
        raise TypeError(f"vector must contain float{bits} values; got {dtype!r}")
    if tensor.strides:
        raise ValueError("vector must be contiguous")

    scalar_count = int(dtype.lanes)
    for axis in range(tensor.ndim):
        scalar_count *= int(tensor.shape[axis])
    if scalar_count % lanes:
        raise ValueError(f"vector must contain a multiple of {lanes} values; got {scalar_count}")

    tensor.ndim = 1
    tensor.dtype = ovstage.DLDataType(
        code=ovstage.DLDataTypeCode.kDLFloat,
        bits=bits,
        lanes=lanes,
    )
    tensor._shape_storage = (ctypes.c_int64 * 1)(scalar_count // lanes)
    tensor.shape = ctypes.cast(tensor._shape_storage, ctypes.POINTER(ctypes.c_int64))
    tensor.strides = None
    return tensor


def _reset_product_attributes(stage, query, previous: frozenset[str], current: frozenset[str], ordinal: int) -> list:
    attributes = previous | current
    if attributes:
        return [stage.delete_attributes(query, sorted(attributes), ordinal=ordinal)]
    return []


def _discard_product_publication(operations, paths, path_list, query) -> None:
    """Best-effort cleanup when publication construction fails."""

    with suppress(Exception):
        ProductPublication(frozenset(), operations, (paths, path_list, query)).wait()


def _ownership_manifest(paths, attributes: set[str]):
    import numpy as np
    import ovstage

    values = np.asarray([paths.intern_token(attribute) for attribute in sorted(attributes)], dtype=np.uint64)
    return ovstage.WriteDesc(
        PRODUCT_ATTRIBUTES,
        values,
        is_array=True,
        semantic=ovstage.AttributeSemantic.TOKEN_ID,
    )


def _path_query(stage, path: str | list[str]):
    import ovstage

    paths = ovstage.PathDictionary(stage)
    path_list = paths.create_path_list_from_strings([path] if isinstance(path, str) else path)
    query = stage.query_from_path_list(path_list)
    return paths, path_list, query


def _release_path_query(paths, path_list, query) -> None:
    query.release().wait()
    paths.destroy_path_list(path_list)
    paths.destroy()
