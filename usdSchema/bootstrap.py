# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import contextlib
import io
import os
import sys

import packmanapi

REPO_ROOT = os.path.dirname(os.path.realpath(__file__))
REPO_DEPS_FILE = os.path.join(REPO_ROOT, "deps", "repo-deps.packman.xml")

if __name__ == "__main__":
    # pull all repo dependencies first
    # and add them to the python path
    with contextlib.redirect_stdout(io.StringIO()):
        deps = packmanapi.pull(REPO_DEPS_FILE)

    for dep_path in deps.values():
        if dep_path not in sys.path:
            sys.path.append(dep_path)

    sys.path.append(REPO_ROOT)

    import omni.repo.usd

    omni.repo.usd.bootstrap(REPO_ROOT)
