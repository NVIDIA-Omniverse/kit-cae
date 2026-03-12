# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from logging import getLogger

import omni.ext

logger = getLogger(__name__)


class Extension(omni.ext.IExt):
    def on_startup(self, ext_id):
        try:
            from .delegate import TrimeshDataDelegate
        except ImportError:
            logger.error(
                "trimesh package not available. Trimesh data delegate will not function.\n"
                "To install optional dependencies and relaunch:\n"
                "  Linux:   ./repo.sh pip_download\n"
                "  Windows: repo.bat pip_download\n"
                "See docs/Build.md for details."
            )
            self._delegate = None
            return

        from omni.cae.data import get_data_delegate_registry

        self._registry = get_data_delegate_registry()
        self._delegate = TrimeshDataDelegate(ext_id)
        self._registry.register_data_delegate(self._delegate)

    def on_shutdown(self):
        if self._delegate is None:
            return
        self._registry.deregister_data_delegate(self._delegate)
        del self._delegate
