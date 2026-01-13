# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

r"""

This module provides a simple way to provide status bar progress and activity
messages to the user.

Usage:

    from omni.cae.data import progress

    with progress.ProgressContext("Doing something") as pc:
        ... do 1/2 of something ...
        pc.notify(0.5)
        ... do 1/2 of something ...

It's also possible to nest contexts:

    with progress.ProgressContext("Doing something") as segment:
        count = 10
        for i in range(count):
            with progress.ProgressContext("Doing something %d of %d" % (i + 1, count), shift=i / count, scale=1.0 / count) as pc:
                ... do 1/2 of segment
                pc.notify(0.5)
                ... do 1/2 of segment

Thus, each segment can be shifted and scaled independently. Thus, when reporting progress for that segment of work,
we just need to report the progress within the range [0, 1] for that segment. The total progress will be
computed based on the sum of the progress of all segments in the stack.

`Segment.finish` should be called when you're done with the segment. It's not strictly necessary to call `Segment.finish`
for each segment, but it's a good practice. Destroying the segment will also call `Segment.finish` for you. This ensures
the stack survives exceptions, etc.

"""

from .impl.progress import ProgressContext, interrupt, progress_context
