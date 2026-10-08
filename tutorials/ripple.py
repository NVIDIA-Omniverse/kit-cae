# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""The rippling plate that tutorials 4 and 5 share.

Tutorial 4 builds the plate and its time samples cell by cell, with the
explanations; this module holds the same code so tutorial 5 can start from
the finished dataset without repeating them. Keep the two in step.
"""

from __future__ import annotations

import numpy as np
import warp as wp
import warp_simdata as simdata
from warp_simdata.data_models.ensight_gold import ensight_shapes
from warp_simdata.data_models.ensight_gold.unstructured_part import create_dataset, create_piece_handle

AMPLITUDE = 0.15  # peak height at the center
HEIGHT_RANGE = (-0.1, 0.1)  # colormap range that leaves the outer rings visible


def make_plate(n: int, device: str):
    """Return an EnSight Gold quad plate spanning [-1, 1]^2 with ``n`` nodes per side, and its points."""
    xs, ys = np.meshgrid(np.linspace(-1.0, 1.0, n), np.linspace(-1.0, 1.0, n), indexing="ij")
    points = np.stack([xs.ravel(), ys.ravel(), np.zeros(n * n)], axis=1).astype(np.float32)

    i, j = np.meshgrid(np.arange(n - 1), np.arange(n - 1), indexing="ij")
    corner = (i * n + j).ravel()  # 0-based id of each quad's first node
    quads = np.stack([corner, corner + n, corner + n + 1, corner + 1], axis=1) + 1  # 1-based, counter-clockwise

    plate = create_dataset(
        wp.array(points, dtype=wp.vec3f, device=device),
        [create_piece_handle(ensight_shapes.EN_quad4, wp.array(quads.ravel().astype(np.int32), device=device))],
    )
    return plate, points


def ripple_height(points: np.ndarray, t: float) -> np.ndarray:
    """A damped sine wave in the distance from the center, travelling outward with ``t``."""
    radius = np.hypot(points[:, 0], points[:, 1])
    return (AMPLITUDE * np.sin(12.0 * radius - 6.0 * t) * np.exp(-2.0 * radius)).astype(np.float32)


def ripple_sample(plate, points: np.ndarray, t: float, device: str, *, displacement: bool = False):
    """A shallow copy of ``plate`` with ``height`` at time ``t``; ``displacement`` adds the vector (0, 0, height)."""
    height = ripple_height(points, t)
    sample = plate.shallow_copy(pass_fields=False)
    sample.add_field("height", simdata.Field.from_array(wp.array(height, device=device), simdata.AssociationType.NODE))
    if displacement:
        vectors = np.zeros((len(height), 3), np.float32)
        vectors[:, 2] = height
        sample.add_field(
            "displacement",
            simdata.Field.from_array(wp.array(vectors, dtype=wp.vec3f, device=device), simdata.AssociationType.NODE),
        )
    return sample
