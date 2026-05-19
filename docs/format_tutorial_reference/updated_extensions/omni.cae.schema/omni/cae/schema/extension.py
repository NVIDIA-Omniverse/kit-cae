# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

__all__ = ["Extension"]

import os
import sys
from pathlib import Path
from logging import getLogger

import omni.ext
from omni.kit.app import get_app
from pxr import Plug, Usd  # noqa: F401 -- Usd import ensures base class wrappers exist for schema modules

logger = getLogger(__name__)


def _load_usd_plugins(ext_id):
    plugin_dir = Path(get_app().get_extension_manager().get_extension_path(ext_id)) / "usd" / "plugin"

    resource_dirs = sorted(path for path in plugin_dir.glob("*/resources") if (path / "plugInfo.json").is_file())
    if not resource_dirs:
        logger.warning("No USD schema plugins found under '%s'", plugin_dir)
        return

    dll_dir_handles = []

    # On Windows, importing the generated Python bindings (for example
    # `usd/python/OmniCaeViz/_omniCaeViz.pyd`) does not automatically make the
    # matching schema DLL in `usd/plugin/OmniCaeViz/` discoverable. Register each
    # plugin directory with the process DLL search path before any schema Python
    # modules are imported so dependent DLLs such as `omniCaeViz.dll` and
    # `omniCae.dll` can be resolved.
    #
    # Keep the returned handles alive for the lifetime of the extension;
    # dropping them removes the directories from the DLL search path.
    if sys.platform == "win32":
        for resource_dir in resource_dirs:
            dll_dir = resource_dir.parent
            logger.info("Adding schema DLL search path '%s'", dll_dir)
            dll_dir_handles.append(os.add_dll_directory(str(dll_dir)))

    for resource_dir in resource_dirs:
        logger.info("Registering USD plugin from '%s'", resource_dir)
        result = Plug.Registry().RegisterPlugins(str(resource_dir))
        if not result:
            logger.error("Failed to register USD plugin from '%s'", resource_dir)

    return dll_dir_handles


class Extension(omni.ext.IExt):
    def on_startup(self, extId):
        logger.info("starting extension %s", extId)
        self._dll_dir_handles = _load_usd_plugins(extId)

    def on_shutdown(self):
        self._dll_dir_handles = []
        logger.info("shutting down")
