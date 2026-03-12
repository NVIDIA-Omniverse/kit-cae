# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

__all__ = ["Listener"]

import logging

from carb.settings import get_settings
from omni.stageupdate import get_stage_update_interface
from pxr import UsdUtils

logger = logging.getLogger(__name__)

_EX_VARS_SETTINGS_PATH = "/exts/omni.cae.exVars/variables"


class Listener:
    """Listens to USD stage lifecycle events for the exVars extension."""

    def __init__(self):
        stage_update_iface = get_stage_update_interface()
        self._stage_subscription = stage_update_iface.create_stage_update_node(
            "cae.exVars.listener",
            on_attach_fn=self.on_attach,
        )

    def finalize(self):
        if self._stage_subscription:
            del self._stage_subscription
            self._stage_subscription = None

    def on_attach(self, stageId: int, metersPerUnit: float):
        logger.info("Stage attached: ID=%s, metersPerUnit=%s", stageId, metersPerUnit)

        ex_vars = get_settings().get(_EX_VARS_SETTINGS_PATH)
        if not ex_vars:
            logger.info("No expressionVariables configured at %s", _EX_VARS_SETTINGS_PATH)
            return

        cache = UsdUtils.StageCache.Get()
        stage = cache.Find(cache.Id.FromLongInt(stageId))
        if not stage:
            logger.error("Could not find stage with ID %s in cache", stageId)
            return

        session_layer = stage.GetSessionLayer()
        current = dict(session_layer.expressionVariables)
        current.update(ex_vars)
        session_layer.expressionVariables = current

        logger.info("Applied expressionVariables to session layer:")
        print("Applied expressionVariables to session layer:")

        for name, value in ex_vars.items():
            logger.info("  %s = %r", name, value)
            print("  %s = %r" % (name, value))
