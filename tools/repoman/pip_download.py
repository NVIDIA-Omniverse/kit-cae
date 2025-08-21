# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
#  its affiliates is strictly prohibited.

r"""
Tools to download pip package archives.
"""
import pathlib
import sys
import os
from logging import getLogger

import omni.repo.man
from omni.repo.kit_template.frontend import CLIInputColorPalette, Separator
from omni.repo.man import resolve_tokens
from omni.repo.man.exceptions import QuietExpectedError, StorageError

logger = getLogger(__name__)


# These dependencies come from repo_kit_template
from rich.console import Console
from rich.theme import Theme

# This should match repo_kit_template.palette
INFO_COLOR = "#3A96D9"
WARN_COLOR = "#FFD700"

theme = Theme()
console = Console(theme=theme)


def _get_python_cmd():
    if os.name == "nt":
        repo_cmd = "${root}/_build/target-deps/python/python.exe"
    else:
        repo_cmd = "${root}/_build/target-deps/python/bin/python3"

    return omni.repo.man.resolve_tokens(repo_cmd)


def _run_command(command):
    console.print("\[ctrl+c to Exit]", style=INFO_COLOR)
    try:
        omni.repo.man.run_process(resolve_tokens(command), exit_on_error=True)
    except (KeyboardInterrupt, SystemExit):
        console.print("Exiting", style=INFO_COLOR)
        # exit(0) for now due to non-zero exit reporting.
        sys.exit(0)


def run_repo_tool(options, config):
    console.print("\[pip_download] Downloading PIP archives", style=INFO_COLOR)
    py_cmd = _get_python_cmd()
    if not pathlib.Path(py_cmd).exists():
        console.print(f"\[pip_download] Python command not found: {py_cmd}. Did you run the `build` step before calling this tool?", style=WARN_COLOR)
        raise QuietExpectedError(f"Python command not found: {py_cmd}")

    command = [py_cmd, "-m", "pip", "download", "--no-deps", "--dest", options.dest]
    if options.requirements:
        if not pathlib.Path(options.package).exists():
            console.print(f"\[pip_download] Requirements file not found: {options.package}", style=WARN_COLOR)
            raise QuietExpectedError(f"Requirements file not found: {options.package}")
        command += ["-r", options.package]
    else:
        command += [options.package]
    _run_command(command)


def setup_repo_tool(parser, config):
    tool_config = config.get("repo_pip_download", {})
    parser.description = "Tool to download pip package archives."
    omni.repo.man.add_config_arg(parser)
    parser.add_argument(
        "--dest",
        help="Download packages into <dir> ",
    )
    parser.add_argument(
        "-r",
        "--requirements",
        action="store_true",
        help="'package' is a path to requirements.txt file to download all packages listed in it.",
    )
    parser.add_argument("package", help="Package name to download, e.g. 'numpy' or 'numpy==1.21.0'.")

    if tool_config.get("enabled", False):
        return run_repo_tool
