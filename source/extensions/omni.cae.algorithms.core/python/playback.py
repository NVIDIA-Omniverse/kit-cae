# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
#  its affiliates is strictly prohibited.

import asyncio
import weakref
from logging import getLogger

from omni.usd import get_context
from pxr import Usd

from .controller import UsdRtController

logger = getLogger(__name__)

_controller: weakref.ReferenceType = None


def set_controller(controller: UsdRtController):
    global _controller
    _controller = weakref.ref(controller) if controller else None


def get_controller() -> UsdRtController:
    global _controller
    return _controller() if _controller else None


def get_all_authored_time_samples(stage) -> list:
    time_samples = set()
    for prim in stage.Traverse():
        for attr in prim.GetAttributes():
            samples = attr.GetTimeSamples()
            time_samples.update(samples)
    time_samples = list(time_samples)
    time_samples.sort()
    return time_samples


async def iterate_over_all_samples():
    controller = get_controller()
    if controller is None:
        logger.error("Controller not set. Cannot iterate over samples.")
        return False

    stage: Usd.Stage = get_context().get_stage()

    # get all authored time samples
    time_samples: list = get_all_authored_time_samples(stage)
    if len(time_samples) == 0:
        logger.error("No time samples found.")
        return False

    # iterate over all time samples
    for t in time_samples:
        logger.warning(f"Time sample: {t}")
        tc = Usd.TimeCode(float(t))
        await controller.sync(tc)
    return True


def do_playback():
    asyncio.ensure_future(iterate_over_all_samples())


# from omni.cae.algorithms.core import playback
# playback.do_playback()
