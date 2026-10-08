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

import gc
import weakref

import numpy as np
import pytest

from ovcae._stage_io import _point3f_tensor, write_test_product


@pytest.mark.parametrize(
    "values",
    [
        np.asarray([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float32),
        pytest.param("warp", id="warp-vec3f"),
    ],
)
def test_point_products_publish_the_usd_point3f_layout(values) -> None:
    if isinstance(values, str):
        import warp as wp

        values = wp.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=wp.vec3f, device="cpu")

    tensor = _point3f_tensor(values)

    assert tensor.ndim == 1
    assert tensor.shape_tuple == (2,)
    assert tensor.dtype.code == 2
    assert tensor.dtype.bits == 32
    assert tensor.dtype.lanes == 3


def test_product_publication_retains_inputs_and_waits_at_the_boundary(monkeypatch) -> None:
    events = []

    class Operation:
        def __init__(self, name, keepalive=None):
            self.name = name
            self.keepalive = keepalive

        def wait(self):
            events.append(f"wait:{self.name}")
            self.keepalive = None

    class Paths:
        def __init__(self):
            self.tokens = {}

        def intern_token(self, value):
            return self.tokens.setdefault(value, len(self.tokens) + 1)

        def destroy_path_list(self, path_list):
            events.append(f"destroy:{path_list}")

        def destroy(self):
            events.append("destroy:paths")

    class Query:
        def release(self):
            events.append("release:query")
            return Operation("query-release")

    class Stage:
        def delete_attributes(self, query, attributes, ordinal):
            events.append("enqueue:delete")
            return Operation("delete")

        def write_attributes(self, query, writes, ordinal):
            events.append("enqueue:write")
            self.write_operation = Operation("write", writes)
            return self.write_operation

    paths = Paths()
    query = Query()
    monkeypatch.setattr(
        "ovcae._stage_io._path_query",
        lambda stage, path: (paths, "path-list", query),
    )
    stage = Stage()

    publication = write_test_product(stage, "/World/Recipe", 3.5, 2, frozenset())
    value = stage.write_operation.keepalive[-1].tensors
    value_ref = weakref.ref(value)
    del value
    del stage.write_operation
    gc.collect()

    assert events == ["enqueue:delete", "enqueue:write"]
    assert value_ref() is not None

    publication.wait()
    gc.collect()

    assert value_ref() is None
    assert events == [
        "enqueue:delete",
        "enqueue:write",
        "wait:write",
        "wait:delete",
        "release:query",
        "wait:query-release",
        "destroy:path-list",
        "destroy:paths",
    ]


def test_double_primvars_publish_as_float32() -> None:
    from ovcae._stage_io import _primvar_values

    wp = pytest.importorskip("warp")

    # CGNS and similar sources hand operators float64 fields; MDL data lookups read float.
    scalars = _primvar_values(wp.array([1.5, 2.5], dtype=wp.float64, device="cpu"))
    assert scalars.dtype == wp.float32 and scalars.numpy().tolist() == [1.5, 2.5]
    vectors = _primvar_values(wp.array([[1.0, 2.0, 3.0]], dtype=wp.vec3d, device="cpu"))
    assert vectors.dtype == wp.vec3f and vectors.numpy().tolist() == [[1.0, 2.0, 3.0]]
    assert _primvar_values(np.asarray([1.5], dtype=np.float64)).dtype == np.float32

    # Everything already renderable passes through untouched.
    float32 = wp.array([1.0], dtype=wp.float32, device="cpu")
    assert _primvar_values(float32) is float32
    ints = np.asarray([1, 2], dtype=np.int32)
    assert _primvar_values(ints) is ints


def test_mesh_product_round_trips_double_colors_as_float32() -> None:
    import ovstage

    from ovcae._products import MeshProduct, Primvar
    from ovcae._stage_io import write_mesh_product

    from ._ovstage_helpers import read_array

    stage = ovstage.Stage("ovcae.tests.stage_io.double-primvars")
    try:
        product = MeshProduct(
            np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
            np.asarray([3], dtype=np.int32),
            np.asarray([0, 1, 2], dtype=np.int32),
            (Primvar("colors", np.asarray([0.25, 0.5, 0.75], dtype=np.float64), "vertex"),),
        )
        publication = write_mesh_product(stage, "/World/Mesh", product, 1, previous=frozenset())
        publication.wait()
        stage.advance_write_floor(1).wait()
        colors = read_array(stage, "/World/Mesh", "primvars:colors", 1)
        assert colors.dtype == np.float32
        assert colors.tolist() == pytest.approx([0.25, 0.5, 0.75])
    finally:
        stage.destroy()
