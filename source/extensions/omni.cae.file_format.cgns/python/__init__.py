# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

__all__ = []

# import Usd fileformat plugin
import logging
import os
from pathlib import Path

from pxr import Plug

scriptDir = Path(os.path.dirname(__file__))
pluginsRoot = scriptDir.parent.parent.parent.parent / "plugin"
schemaPath = pluginsRoot / "omni.cae.file_format.cgns.plugin" / "resources"
logging.info("loading CGNS USD file format plugin from '%s'", str(schemaPath))
result = Plug.Registry().RegisterPlugins(str(schemaPath))
if not result:
    logging.error("Failed to load CGNS USD file format plugin from '%s'!!!", str(schemaPath))
