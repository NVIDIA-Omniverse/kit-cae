# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

__all__ = ["Range3i", "IJKExtents"]

import numpy as np
from pxr import Gf

r"""
This file defines some basic types.
"""


class Range3i:

    def __init__(self, min: Gf.Vec3i, max: Gf.Vec3i):
        self._min = Gf.Vec3i(int(min[0]), int(min[1]), int(min[2]))
        self._max = Gf.Vec3i(int(max[0]), int(max[1]), int(max[2]))

    @property
    def min(self) -> Gf.Vec3i:
        return self._min

    @property
    def max(self) -> Gf.Vec3i:
        return self._max

    def __eq__(self, value: "Range3i") -> bool:
        return self._min == value._min and self._max == value._max

    def __str__(self) -> str:
        return f"{self._min} - {self._max}"

    def isEmpty(self) -> bool:
        return np.any(np.greater(self.min, self.max))

    @staticmethod
    def empty() -> "Range3i":
        return Range3i(Gf.Vec3i(0, 0, 0), Gf.Vec3i(-1, -1, -1))

    def numpy(self) -> np.ndarray:
        return np.array(
            [(self.min[0], self.min[1], self.min[2]), (self.max[0], self.max[1], self.max[2])], dtype=np.int32
        )


class IJKExtents:

    def __init__(self, min: Gf.Vec3i, max: Gf.Vec3i, spacing: Gf.Vec3d = Gf.Vec3d(1.0, 1.0, 1.0)):
        self._min = min
        self._max = max
        self._spacing = spacing

    def __eq__(self, value: "IJKExtents") -> bool:
        return self._min == value._min and self._max == value._max and self._spacing == value._spacing

    @property
    def min(self) -> Gf.Vec3i:
        return self._min

    @property
    def max(self) -> Gf.Vec3i:
        return self._max

    @property
    def spacing(self) -> Gf.Vec3d:
        return self._spacing

    def isEmpty(self) -> bool:
        return np.any(np.greater(self.min, self.max))

    def __str__(self) -> str:
        return f"min: {self._min} max: {self._max} spacing: {self._spacing}"

    def intersect(self, other: "IJKExtents") -> bool:
        if self.spacing != other.spacing:
            raise ValueError(
                "Intersection is only supported with spacings match (%s != %s)", self.spacing, other.spacing
            )

        self._min = Gf.Vec3i(np.maximum(self.min, other.min).tolist())
        self._max = Gf.Vec3i(np.minimum(self.max, other.max).tolist())
        return not self.isEmpty()

    @property
    def dims(self) -> Gf.Vec3i:
        return Gf.Vec3i((np.substract(self.max, self.min) + 1).tolist()) if not self.isEmpty() else Gf.Vec3i(0, 0, 0)

    def getRange(self) -> Range3i:
        return Range3i(self.min, self.max)

    @staticmethod
    def empty() -> "IJKExtents":
        return IJKExtents(Gf.Vec3i(0, 0, 0), Gf.Vec3i(-1, -1, -1))

    @staticmethod
    def fromBounds(bbox: Gf.Range3d, spacing: Gf.Vec3d) -> "IJKExtents":
        if bbox.IsEmpty() or np.any(np.asarray(spacing) <= 0):
            return IJKExtents.empty()

        min_bds = np.asarray(bbox.GetMin())
        max_bds = np.asarray(bbox.GetMax())
        voxel_size = np.asarray(spacing)
        ijk_min = np.around(min_bds / voxel_size).astype(np.int32)
        ijk_max = np.around(max_bds / voxel_size).astype(np.int32)
        return IJKExtents(Gf.Vec3i(ijk_min.tolist()), Gf.Vec3i(ijk_max.tolist()), spacing)
