# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import numpy as np
import pytest

from ._memory_series import MemorySeries, open_series


def series():
    result = MemorySeries()
    result.publish("/data", component="topology", value={})
    result.publish("/data", component="geometry", value=np.zeros((2, 3), np.float32), time=2.0)
    result.publish(
        "/data", component="field:U", value={"values": np.ones(2, np.float32), "association": "node"}, time=2.4
    )
    return result


def describe(frame, component):
    return frame._provider.describe_component("/data", "", component)


def test_independent_samples_and_held_selection():
    data = series()
    with open_series(data) as source, source.at_time(2.4) as first:
        data.publish(
            "/data", component="field:U", value={"values": np.full(2, 5, np.float32), "association": "node"}, time=2.5
        )
        with source.at_time(2.49) as held, source.at_time(2.5) as next_frame:
            assert describe(first, "field:U") == describe(held, "field:U")
            assert describe(first, "geometry") == describe(next_frame, "geometry")
            assert describe(first, "topology") == describe(next_frame, "topology")
            assert describe(first, "field:U") != describe(next_frame, "field:U")
            assert describe(next_frame, "field:U")[1].time == 2.5


def test_replacement_keeps_active_lease_and_copies_producer_storage():
    data = series()
    with open_series(data) as source, source.at_time(2.4) as old:
        points = np.ones((2, 3), np.float32)
        data.publish("/data", component="geometry", value=points, time=2.0)
        points[:] = 99
        with source.at_time(2.4) as new:
            assert describe(old, "geometry") != describe(new, "geometry")
            for frame, expected in [(old, 0), (new, 1)]:
                dataset = frame._resolve_dataset("/data", identifier="", device="cpu", fields={})
                np.testing.assert_array_equal(dataset.handle.points.numpy(), np.full((2, 3), expected))


def test_namespace_and_schedule_validation():
    one, two = series(), series()
    with open_series(one) as a, open_series(two) as b, a.at_time(0) as fa, b.at_time(0) as fb:
        assert describe(fa, "geometry") != describe(fb, "geometry")
    with pytest.raises(ValueError, match="mix static"):
        one.publish("/data", component="geometry", value=np.zeros((2, 3)), time=None)
    with pytest.raises(ValueError, match="finite"):
        one.publish("/data", component="geometry", value=[], time=float("nan"))
