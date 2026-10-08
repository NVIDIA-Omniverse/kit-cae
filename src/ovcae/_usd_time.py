# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Held-sample selection shared by time-aware Data Sources."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSample:
    """Selected dataset-wide sample, rather than an individual array's native time.

    Time is in seconds, or None for default-only data. All described components
    share this sample even when their underlying attributes have different schedules.
    Source identity is separate; this description is not a cache key.
    """

    time: float | None
    granularity: str = "dataset"


def held_time(samples: tuple[float, ...], requested: float) -> float | None:
    """Select the preceding sample, clamp outside the range, or identify static data.

    Samples must be a sorted tuple of the dataset's authored sample times. Materialization
    must also use held USD interpolation: snapping to a union time alone does not
    prevent interpolation of attributes with sparser schedules.
    """
    return samples[max(0, bisect_right(samples, requested) - 1)] if samples else None
