# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
from pathlib import Path

# Very selective repo deps that do not depend on 3rd party libs.
from omni.repo.man.deps import validate_dependencies

TOOL_NAME = "repo_cae_tools"
ROOT_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "..", "..")
VERSION = Path(f"{ROOT_DIR}/VERSION").read_text().strip()

# vendor_directory = validate_dependencies(Path(__file__), tool_name=TOOL_NAME, strict_deps=True)
