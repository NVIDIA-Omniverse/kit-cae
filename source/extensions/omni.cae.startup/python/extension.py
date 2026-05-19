# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
Extension that opens a USD file at startup after RTX delivers its first frame.
"""

import logging

import carb.settings
import carb.tokens
import omni.ext

logger = logging.getLogger(__name__)

SETTING_USD_FILE = "/exts/omni.cae.startup/usdFile"


class Extension(omni.ext.IExt):
    """Extension class for omni.cae.startup"""

    def on_startup(self, ext_id):

        self._new_frame_sub = None

        usd_file = carb.settings.get_settings().get_as_string(SETTING_USD_FILE)
        if usd_file:
            usd_file = carb.tokens.get_tokens_interface().resolve(usd_file)
        if not usd_file:
            logger.debug("No usdFile setting found; skipping startup stage load.")
            return

        logger.info(f"Will open '{usd_file}' after the first RTX frame.")
        self._schedule_open(usd_file)

    def on_shutdown(self):
        self._new_frame_sub = None

    def _schedule_open(self, usd_file: str):
        import carb.eventdispatcher
        import omni.usd

        usd_context = omni.usd.get_context()
        event_name = usd_context.stage_rendering_event_name(omni.usd.StageRenderingEventType.NEW_FRAME, immediate=True)

        def _on_new_frame(_event):
            self._new_frame_sub = None
            logger.info(f"RTX ready; opening '{usd_file}'.")
            from omni.kit.async_engine import run_coroutine

            run_coroutine(_open_stage(usd_file))

        self._new_frame_sub = carb.eventdispatcher.get_eventdispatcher().observe_event(
            event_name=event_name,
            on_event=_on_new_frame,
            observer_name="omni.cae.startup.Extension",
        )


async def _open_stage(usd_file: str):
    import omni.usd

    usd_context = omni.usd.get_context()
    await usd_context.close_stage_async()
    success, error = await usd_context.open_stage_async(usd_file)
    if not success:
        logger.error(f"Failed to open '{usd_file}': {error}")
    else:
        logger.info(f"Successfully opened '{usd_file}'.")
