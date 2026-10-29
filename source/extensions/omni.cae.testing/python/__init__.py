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
    "wait_for_rtx_renderer_ready",
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


async def wait_for_rtx_renderer_ready(timeout: float = 120.0, settle_cycles: int = 2):
    """
    Wait until RTX can be used by headless tests.

    This deliberately avoids viewport frame events: bundle tests run with
    --no-window, so there may never be a viewport NEW_FRAME signal.
    """
    import time

    import omni.kit.app
    import omni.usd

    app = omni.kit.app.get_app()
    deadline = time.monotonic() + timeout

    async def wait_for_next_update(message: str):
        if time.monotonic() >= deadline:
            raise TimeoutError(message)
        await app.next_update_async()

    while not app.is_app_ready():
        await wait_for_next_update("Timed out waiting for Kit app readiness before RTX setup.")

    ext_manager = app.get_extension_manager()
    while not ext_manager.is_extension_enabled("omni.hydra.rtx"):
        await wait_for_next_update("Timed out waiting for omni.hydra.rtx to be enabled.")

    try:
        import omni.mdl.neuraylib
    except ImportError:
        logger.debug("omni.mdl.neuraylib is not available while waiting for RTX readiness.", exc_info=True)
    else:
        omni.mdl.neuraylib.ensure_running()

    usd_context = omni.usd.get_context()
    last_attach_error = None

    def is_rtx_attached():
        return "rtx" in usd_context.get_attached_hydra_engine_names()

    create_requested = False
    invalid_uid = getattr(omni.usd, "HydraEngineInvalidUniqueId", None)

    while not is_rtx_attached():
        if not create_requested:
            try:
                engine_uid = omni.usd.create_hydra_engine("rtx", usd_context)
                create_requested = invalid_uid is None or engine_uid != invalid_uid
            except Exception as exc:
                last_attach_error = exc
                logger.debug("RTX Hydra engine is not attachable yet.", exc_info=True)

        if is_rtx_attached():
            break

        if time.monotonic() >= deadline:
            raise TimeoutError("Timed out waiting for the RTX Hydra engine to attach.") from last_attach_error
        await app.next_update_async()

    for _ in range(settle_cycles):
        await app.next_update_async()


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
