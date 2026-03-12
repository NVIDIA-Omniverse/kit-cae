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

from logging import getLogger

import carb.settings
from omni.cae.data import get_data_delegate_registry
from omni.ext import IExt

logger = getLogger(__name__)


class Extension(IExt):
    def on_startup(self, ext_id):
        try:
            from .delegate import VTKDataDelegate
            from .vtu_delegate import VTUDataDelegate
        except ImportError:
            logger.error(
                "VTK packages not available. VTK data delegate will not function.\n"
                "To install optional dependencies and relaunch:\n"
                "  Linux:   ./repo.sh pip_download\n"
                "  Windows: repo.bat pip_download\n"
                "See docs/Build.md for details."
            )
            self._delegate = None
            return

        self._registry = get_data_delegate_registry()
        self._delegate = VTKDataDelegate(ext_id)
        self._registry.register_data_delegate(self._delegate, 0)  # base VTK delegate with lower priority

        self._vtu_delegate = None
        if carb.settings.get_settings().get_as_bool("/exts/omni.cae.delegate.vtk/useCustomVtuReader"):
            self._vtu_delegate = VTUDataDelegate(ext_id)
            self._registry.register_data_delegate(self._vtu_delegate, 1)  # higher priority than the base VTK delegate
        else:
            logger.info("Custom VTU reader disabled via /exts/omni.cae.delegate.vtk/useCustomVtuReader")

    def on_shutdown(self):
        if self._delegate is None:
            return

        if self._vtu_delegate is not None:
            self._registry.deregister_data_delegate(self._vtu_delegate)
            del self._vtu_delegate

        self._registry.deregister_data_delegate(self._delegate)
        del self._delegate
