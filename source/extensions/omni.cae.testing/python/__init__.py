# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
__all__ = [
    "get_test_data_root",
    "get_test_stage_root",
    "get_test_data_path",
    "get_test_stage_path",
    "get_vtrt_array_as_numpy",
    "wait_for_update",
    "new_stage",
    "frame_prims",
]


import pathlib
from logging import getLogger

import numpy as np
import warp as wp
from omni.usd import get_context

logger = getLogger(__name__)

_test_data_root = pathlib.Path(__file__).parent.parent.parent.parent / "shared" / "data"
_test_stage_root = pathlib.Path(__file__).parent.parent.parent.parent / "shared" / "stages"


def get_test_data_root() -> str:
    return str(_test_data_root)


def get_test_stage_root() -> str:
    return str(_test_stage_root)


def get_test_data_path(relative_path: str) -> str:
    if relative_path is None:
        return _test_data_root
    elif pathlib.Path(relative_path).is_absolute():
        #  check if path is absolute
        return pathlib.Path(relative_path)
    else:
        path = str(_test_data_root / relative_path)
        logger.info("Using test data %s", path)
        return path


def get_test_stage_path(relative_path: str) -> str:
    if relative_path is None:
        return _test_stage_root
    elif pathlib.Path(relative_path).is_absolute():
        #  check if path is absolute
        return pathlib.Path(relative_path)
    else:
        path = str(_test_stage_root / relative_path)
        logger.info("Using test stage %s", path)
        return path


def get_vtrt_array_as_numpy(rt_attr) -> np.ndarray:
    """
    Converts a UsdRT.Array to a numpy array.
    """
    if not rt_attr.IsValid():
        raise ValueError(f"Attribute is not valid")
    if rt_attr.IsGpuDataValid():
        return wp.array(rt_attr.Get()).numpy()
    elif rt_attr.IsCpuDataValid():
        rt_attr.SyncDataToGpu()
        return wp.array(rt_attr.Get()).numpy()


async def wait_for_update(cycles: int = 10):
    """
    Wait for update cycles to ensure async operations complete.

    Parameters
    ----------
    cycles : int, optional
        Number of update cycles to wait for, by default 10.
        - If None or <= 0: Does a single update async (brief wait)
        - If > 0: Waits for the specified number of cycles with small delays
    """
    import asyncio

    from omni.kit.app import get_app

    if cycles is None or cycles <= 0:
        await get_app().next_update_async()
    else:
        for i in range(cycles):
            await get_app().next_update_async()
            await asyncio.sleep(0.01)


async def frame_prims(prim_paths: list[str], zoom: float = 1.0):
    """
    Frame the camera on the specified prims.

    Parameters
    ----------
    prim_paths : list[str]
        List of prim paths to frame
    zoom : float, optional
        Zoom factor, by default 1.0
    """
    from carb.settings import get_settings
    from omni.cae.data.commands import execute_command
    from omni.kit.viewport.utility import get_active_viewport

    settings = get_settings()
    if settings.get_as_bool("/app/isTestRun"):
        logger.warning("Skipping frame prims in test run")
        return

    viewport = get_active_viewport()
    if viewport is None:
        logger.warning("No active viewport found, cannot frame prims")
        return

    camera_path = viewport.camera_path
    await execute_command("FramePrimsCommand", prim_to_move=camera_path, prims_to_frame=prim_paths, zoom=zoom)


class new_stage:
    """
    Context manager that creates a new stage on entry and tears it down on exit.

    Usage:
        async with new_stage():
            # work with the new stage
            pass
    """

    def __init__(self, path: str = None):
        self.usd_context = get_context()
        self.path = path

    async def __aenter__(self):
        if self.path:
            logger.info("Opening stage %s", self.path)
            if not pathlib.Path(self.path).exists():
                logger.error("Stage %s does not exist", self.path)
                raise FileNotFoundError(f"Stage {self.path} does not exist")
            await self.usd_context.open_stage_async(self.path)
        else:
            await self.usd_context.new_stage_async()
        await wait_for_update(10)
        return self.usd_context.get_stage()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.usd_context.close_stage_async()
        return False
