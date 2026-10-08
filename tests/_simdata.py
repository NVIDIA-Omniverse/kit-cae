# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Build Warp SimData datasets for tests from plain arrays."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def dataset_from_bundle(bundle: Mapping[str, Any], device: str = "cpu"):
    """Turn ``points`` plus optional VTK connectivity and ``fields`` into a SimData dataset.

    ``fields`` maps a field name to ``{"values": array, "association": "node" | "element"}``;
    each becomes a dataset field of the same name.
    """
    from ovcae.data import add_fields, dataset_from_arrays

    dataset = dataset_from_arrays(bundle, device)
    add_fields(dataset, bundle, {name: (name,) for name in bundle.get("fields", {})}, device)
    return dataset


def tetrahedron(device: str = "cpu", **fields):
    """One tetrahedral cell with optional node fields given as ``name=values``."""
    return dataset_from_bundle(
        {
            "points": [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "cell_types": [10],
            "connectivity_offsets": [0, 4],
            "connectivity": [0, 1, 2, 3],
            "fields": {name: {"values": values, "association": "node"} for name, values in fields.items()},
        },
        device,
    )
