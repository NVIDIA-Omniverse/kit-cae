# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import warnings
from logging import getLogger

import omni.ext

logger = getLogger(__name__)

# Suppress h5py warning: runtime HDF5 (2.1.0) differs from build-time (2.0.0).
# Safe because both share soversion 320 (ABI-compatible).
warnings.filterwarnings("ignore", message="h5py is running against HDF5")
try:
    import h5py
except ImportError:
    h5py = None


class Extension(omni.ext.IExt):
    def on_startup(self, _ext_id):
        import omni.kit.tool.asset_importer as ai

        if h5py is None:
            logger.error(
                "h5py not available. EDEM file importer will not function.\n"
                "To install optional dependencies and relaunch:\n"
                "  Linux:   ./repo.sh pip_download\n"
                "  Windows: repo.bat pip_download\n"
                "See docs/Build.md for details."
            )
            self._importer = None
            return

        from .importer import EDEMImporter

        self._importer = EDEMImporter()
        ai.register_importer(self._importer)

    def on_shutdown(self):
        import omni.kit.tool.asset_importer as ai

        if self._importer is not None:
            ai.remove_importer(self._importer)
            self._importer = None
