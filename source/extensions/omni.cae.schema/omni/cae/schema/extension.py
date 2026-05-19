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

import ctypes
import os
import sys
from logging import getLogger
from pathlib import Path

import omni.ext
from omni.kit.app import get_app
from pxr import Plug, Usd  # noqa: F401 -- Usd import ensures base class wrappers exist for schema modules

logger = getLogger(__name__)


def _get_schema_plugins(plugin_dir, registered_plugins):
    """Return the USD schema plugins that belong to this extension.

    RegisterPlugins() returns the plugins it registered during this call, but
    it can return an empty list if the same plugInfo files were registered
    earlier. Preloading is still required in that case, so fall back to the USD
    registry and select plugins whose resolved library path lives under this
    extension's usd/plugin directory.
    """
    plugins = list(registered_plugins)
    if plugins:
        return plugins

    # RegisterPlugins() may return an empty list if another path already
    # registered these plugins. In that case, recover the plugins that live
    # under this extension so startup still preloads them.
    try:
        return [plugin for plugin in Plug.Registry().GetAllPlugins() if Path(plugin.path).is_relative_to(plugin_dir)]
    except AttributeError:
        plugin_dir = plugin_dir.resolve()
        return [
            plugin for plugin in Plug.Registry().GetAllPlugins() if plugin_dir in Path(plugin.path).resolve().parents
        ]


def _preload_registered_plugin_libraries(plugins):
    """Preload schema plugin libraries without invoking USD's plugin loader.

    Registering a USD plugin only makes its plugInfo metadata discoverable. The
    actual schema library is normally loaded lazily by USD when a type from that
    plugin is needed. That is too late for native Kit plugins such as
    omni.cae.data.plugin, which have direct dynamic-library dependencies on
    generated schema libraries like libomniCae.so.

    Do not call PlugPlugin.Load() here. In this Kit packaging layout USD tries
    to import generated Python modules as usd.python.pxr.<SchemaName>, which is
    not the import path used by these extensions and produces noisy warnings.
    Loading the library by absolute path gives the platform loader the resident
    shared library it needs without asking USD to import Python wrappers.

    On Linux, RTLD_GLOBAL is important: it lets later native plugin loads
    resolve DT_NEEDED entries such as libomniCae.so by SONAME against the schema
    library already loaded here. On Windows, these flags are ignored by ctypes,
    but loading the DLL by absolute path is still useful as an explicit preload.
    Keep the returned CDLL handles alive for the lifetime of the extension so
    the loader does not release the libraries early.
    """

    def _sort_key(plugin):
        name = plugin.name
        return (0 if name == "omniCae" else 1, name)

    handles = []
    mode = getattr(os, "RTLD_NOW", 0) | getattr(os, "RTLD_GLOBAL", 0)
    for plugin in sorted(plugins, key=_sort_key):
        if plugin.isLoaded:
            logger.info("USD schema plugin '%s' is already loaded from '%s'", plugin.name, plugin.path)
            continue

        logger.info("Preloading USD schema plugin library '%s' from '%s'", plugin.name, plugin.path)
        try:
            handles.append(ctypes.CDLL(plugin.path, mode=mode))
        except OSError as exc:
            logger.error(
                "Failed to preload USD schema plugin library '%s' from '%s': %s", plugin.name, plugin.path, exc
            )

    return handles


def _load_usd_plugins(ext_id):
    plugin_dir = Path(get_app().get_extension_manager().get_extension_path(ext_id)) / "usd" / "plugin"

    handles = []
    # On Windows, loading a schema DLL by absolute path does not automatically
    # make sibling schema DLL directories available for dependency resolution.
    # Add each USD plugin directory to the process DLL search path before the
    # explicit preloads below and before any generated Python bindings are
    # imported. This lets dependencies such as `omniCaeViz.dll -> omniCae.dll`
    # and bindings such as `usd/python/pxr/OmniCaeViz/_omniCaeViz.pyd ->
    # omniCaeViz.dll` resolve against the packaged schema libraries.
    #
    # Keep the returned add_dll_directory() handles alive for the lifetime of
    # the extension; dropping them removes the directories from the DLL search
    # path.
    if sys.platform == "win32":
        resource_dirs = sorted(path for path in plugin_dir.glob("*/resources") if (path / "plugInfo.json").is_file())
        if not resource_dirs:
            logger.error("No USD schema plugins found under '%s'", plugin_dir)
            return handles
        for resource_dir in resource_dirs:
            dll_dir = resource_dir.parent
            logger.info("Adding schema DLL search path '%s'", dll_dir)
            handles.append(os.add_dll_directory(str(dll_dir)))

    logger.info("Registering USD plugin from '%s'", plugin_dir)
    result = Plug.Registry().RegisterPlugins(str(plugin_dir))
    plugins = _get_schema_plugins(plugin_dir, result)
    if not plugins:
        logger.error("Failed to find registered USD schema plugins under '%s'", plugin_dir)

    handles.extend(_preload_registered_plugin_libraries(plugins))

    return handles


class Extension(omni.ext.IExt):
    def on_startup(self, extId):
        logger.info("starting extension %s", extId)
        # Keep both DLL directory handles and preloaded library handles alive
        # for the extension lifetime.
        self._handles = _load_usd_plugins(extId)

    def on_shutdown(self):
        self._handles = []
        logger.info("shutting down")
