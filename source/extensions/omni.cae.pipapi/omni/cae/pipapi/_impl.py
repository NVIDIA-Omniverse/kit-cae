# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from importlib import import_module
from logging import getLogger

from omni.kit.pipapi import add_archive_directory
from omni.kit.pipapi import install as pip_install_api
from omni.kit.pipapi import remove_archive_directory

logger = getLogger(__name__)


def pip_install(package: str, version: str, module: str = None, required: bool = True):
    """
    Install a package using the omni.kit.pipapi interface.

    Args:
        package (str): The name of the package to install.
        version (str): The version of the package to install.
    """

    status = pip_install_api(
        package=package,
        version=version,
        module=module,
        ignore_import_check=False,
        use_online_index=False,
        extra_args=["--no-deps"],
    )

    if not status and required:
        logger.error(
            f"""


=====================================================================================================

    ***********  MISSING REQUIRED PYTHON PACKAGE  ***********

    Package: "{package}" version: "{version}" is not available.

    Please use "--/exts/omni.kit.pipapi/archiveDirs" setting to specify the directory where the package
    archives are located to install the package.

    You can manually download the package archive from PyPI or other sources and place it in the specified directory
    as follows (see documentation for more details). After placing the archive in the directory,
    restart the application to apply the changes.

      ./repo.sh launch -n <app-name> -- --/exts/omni.kit.pipapi/archiveDirs=[<archive-dir>]

=====================================================================================================


"""
        )
        return False
    else:
        logger.warning("Package installation completed successfully.")
        return True
