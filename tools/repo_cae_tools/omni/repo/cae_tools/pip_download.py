# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

r"""
Tool to install optional pip packages into a target directory.

Packages are installed using `pip install --target` so they are immediately
importable at runtime without any additional configuration.
"""
import os
import pathlib
import shutil
import sys
from logging import getLogger

import omni.repo.man
from omni.repo.man import resolve_tokens
from omni.repo.man.exceptions import QuietExpectedError

from .helpers import run_command

logger = getLogger(__name__)


# These dependencies come from repo_kit_template
from rich.console import Console
from rich.theme import Theme

# This should match repo_kit_template.palette
INFO_COLOR = "#3A96D9"
WARN_COLOR = "#FFD700"

theme = Theme()
console = Console(theme=theme)


def _get_python_cmd(tool_config):
    """Get the python executable path from config."""
    python_root = resolve_tokens(tool_config.get("python_root", "${root}/_build/target-deps/python"))

    if os.name == "nt":
        python_exe = os.path.join(python_root, "python.exe")
    else:
        python_exe = os.path.join(python_root, "bin", "python3")

    return python_exe


def _install_package(py_cmd, target_dir, package_or_requirements, is_requirements=False):
    """Install a package or requirements file into the target directory."""
    command = [py_cmd, "-m", "pip", "install", "--no-deps", "--target", target_dir]

    if is_requirements:
        if not pathlib.Path(package_or_requirements).exists():
            console.print(f"[pip_download] Requirements file not found: {package_or_requirements}", style=WARN_COLOR)
            raise QuietExpectedError(f"Requirements file not found: {package_or_requirements}")
        command += ["-r", package_or_requirements]
        console.print(f"[pip_download] Installing from requirements: {package_or_requirements}", style=INFO_COLOR)
    else:
        command += [package_or_requirements]
        console.print(f"[pip_download] Installing package: {package_or_requirements}", style=INFO_COLOR)

    run_command(command, silent=True)


def run_repo_tool(options, config):
    console.print("[pip_download] Installing optional PIP dependencies", style=INFO_COLOR)

    # Get tool configuration
    tool_config = config.get("repo_pip_download", {})

    # Get python command (CLI --python-root overrides config)
    if options.python_root:
        py_cmd = _get_python_cmd({"python_root": options.python_root})
    else:
        py_cmd = _get_python_cmd(tool_config)
    if not pathlib.Path(py_cmd).exists():
        console.print(
            f"[pip_download] Python command not found: {py_cmd}. Did you run the `build` step before calling this tool?",
            style=WARN_COLOR,
        )
        raise QuietExpectedError(f"Python command not found: {py_cmd}")

    # Get target directory
    if options.dest:
        target_dir = resolve_tokens(options.dest)
    else:
        target_dir = resolve_tokens(tool_config.get("install_dir", "${root}/_build/target-deps/pip_prebundle"))

    # Clean target if requested
    if options.clean:
        if os.path.exists(target_dir):
            console.print(f"[pip_download] Cleaning install directory: {target_dir}", style=INFO_COLOR)
            shutil.rmtree(target_dir)
        else:
            console.print(
                f"[pip_download] Install directory does not exist (nothing to clean): {target_dir}", style=INFO_COLOR
            )

    # Create target directory
    os.makedirs(target_dir, exist_ok=True)
    console.print(f"[pip_download] Install directory: {target_dir}", style=INFO_COLOR)

    # Install packages
    if options.package:
        # Command-line package/requirements specified
        _install_package(py_cmd, target_dir, options.package, is_requirements=options.requirements)
    else:
        # Use configured requirements files
        requirements_files = tool_config.get("requirements_files", [])
        if not requirements_files:
            console.print(
                "[pip_download] No requirements files configured and no package specified.\n"
                "Either:\n"
                "  - Specify a package: repo.sh pip_download <package>\n"
                "  - Or configure 'requirements_files' in repo_tools.toml",
                style=WARN_COLOR,
            )
            return

        console.print(f"[pip_download] Processing {len(requirements_files)} requirements file(s)", style=INFO_COLOR)
        for req_file in requirements_files:
            req_file_resolved = resolve_tokens(req_file)
            _install_package(py_cmd, target_dir, req_file_resolved, is_requirements=True)

    console.print(f"[pip_download] Install complete. Packages installed to: {target_dir}", style=INFO_COLOR)


def setup_repo_tool(parser, config):
    tool_config = config.get("repo_pip_download", {})
    parser.description = (
        "Install optional pip packages into a target directory. "
        "If no package is specified, installs from configured requirements files."
    )
    parser.add_argument(
        "--python-root",
        help="Path to Python root directory (overrides config; used by CI test runners)",
    )
    parser.add_argument(
        "--dest",
        help="Install packages into <dir> (default: _build/target-deps/pip_prebundle)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        default=False,
        help="Clean the target directory before installing",
    )
    parser.add_argument(
        "-r",
        "--requirements",
        action="store_true",
        help="Treat 'package' argument as a path to requirements.txt file",
    )
    parser.add_argument(
        "package",
        nargs="?",
        help="Package name (e.g. 'numpy' or 'numpy==1.21.0') or requirements file path (with -r). "
        "If omitted, uses configured requirements_files.",
    )

    if tool_config.get("enabled", False):
        return run_repo_tool
