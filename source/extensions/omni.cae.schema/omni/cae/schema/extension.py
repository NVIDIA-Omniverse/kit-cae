# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
#  its affiliates is strictly prohibited.

__all__ = ["Extension"]

import os.path
from logging import getLogger

import omni.ext
from pxr import Plug

logger = getLogger(__name__)


def _load_usd_plugins():
    schemas = ["OmniCae", "OmniCaeSids", "OmniCaeVtk", "OmniCaeEnSight"]

    def get_parent_dir(path, level=1):
        for i in range(level):
            path = os.path.dirname(path)
        return path

    extRoot = get_parent_dir(__file__, 4)
    for name in schemas:
        schemaPath = f"{extRoot}/plugins/{name}/resources"
        logger.info("loading USD plugin from '%s'", schemaPath)
        result = Plug.Registry().RegisterPlugins(schemaPath)
        if not result:
            logger.error("Failed to load USD plugin from '%s'", schemaPath)


class Extension(omni.ext.IExt):
    def on_startup(self, extId):
        logger.info("starting extension %s", extId)
        _load_usd_plugins()

    def on_shutdown(self):
        logger.info("shutting down")
