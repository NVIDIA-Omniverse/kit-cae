# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""An orbit camera expressed as a USD camera transform."""

from __future__ import annotations

import math

import numpy as np

_UP = {"Z": np.array([0.0, 0.0, 1.0]), "Y": np.array([0.0, 1.0, 0.0])}


class OrbitCamera:
    """Yaw, pitch, and distance around a fixed target, as a camera-to-world matrix with row vectors.

    ``eye`` and ``target`` fix the initial pose; :attr:`yaw` and :attr:`pitch`
    are offsets in radians and :attr:`zoom` an offset in scene units along the
    horizontal distance. The matrix is what ``omni:xform`` on a USD camera
    prim takes: the camera looks down its local -Z with +Y up, and the last
    row holds the eye position. ``up_axis`` is the stage's, ``"Z"`` or ``"Y"``.
    """

    def __init__(self, eye, target, *, up_axis: str = "Z"):
        if up_axis not in _UP:
            raise ValueError(f"up_axis must be 'Z' or 'Y', not {up_axis!r}")
        self.up_axis = up_axis
        self.target = np.asarray(target, dtype=np.float64)
        offset = np.asarray(eye, dtype=np.float64) - self.target
        # Work in a frame whose third component is up, whichever axis that is.
        local = self._to_local(offset)
        self._horizontal = float(np.hypot(local[0], local[1]))
        self._height = float(local[2])
        self._theta = float(np.arctan2(local[1], local[0]))
        self.yaw = self.pitch = self.zoom = 0.0

    def _to_local(self, vector):
        return vector if self.up_axis == "Z" else np.array([vector[0], -vector[2], vector[1]])

    def _from_local(self, vector):
        return vector if self.up_axis == "Z" else np.array([vector[0], vector[2], -vector[1]])

    def matrix(self) -> np.ndarray:
        """The camera-to-world matrix for the current yaw, pitch, and zoom."""
        theta = self._theta + self.yaw
        radius = max(0.5, self._horizontal + self.zoom)
        local_offset = np.array(
            [
                radius * math.cos(self.pitch) * math.cos(theta),
                radius * math.cos(self.pitch) * math.sin(theta),
                radius * math.sin(self.pitch) + self._height,
            ]
        )
        eye = self.target + self._from_local(local_offset)
        forward = self.target - eye
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, _UP[self.up_axis])
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        matrix = np.eye(4)
        matrix[0, :3], matrix[1, :3], matrix[2, :3], matrix[3, :3] = right, up, -forward, eye
        return matrix


__all__ = ["OrbitCamera"]
