# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
#  its affiliates is strictly prohibited.

import pathlib
import sys
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


def _get_repo_cmd():
    repo_cmd = "${root}/repo${shell_ext}"
    return omni.repo.man.resolve_tokens(repo_cmd)


def _quiet_error(err_msg: str):
    # Need something like QuietExpectedError that just prints and exits 1.
    print(err_msg)
    raise QuietExpectedError(err_msg)


def _select(query: str, apps: list) -> str:
    cli_input = CLIInputColorPalette()
    return cli_input.select(message=query, choices=apps, default=apps[0])


def _run_command(command):
    console.print("\[ctrl+c to Exit]", style=INFO_COLOR)
    try:
        omni.repo.man.run_process(resolve_tokens(command), exit_on_error=True)
    except (KeyboardInterrupt, SystemExit):
        console.print("Exiting", style=INFO_COLOR)
        # exit(0) for now due to non-zero exit reporting.
        sys.exit(0)


def run_repo_tool(options, config):
    console.print("\[schema] Generating and building CAE USD schemas", style=INFO_COLOR)
    repo_folders = config["repo"]["folders"]
    root = pathlib.Path(repo_folders["root"])
    source_path = root / pathlib.Path(repo_folders.get("source", "usdSchema"))
    console.print("\[schema] root=%s" % root, style=INFO_COLOR)
    console.print("\[schema] source_path=%s" % source_path, style=INFO_COLOR)
    console.print("\[schema] config=%s" % options.config, style=INFO_COLOR)
    if not source_path.exists():
        _quiet_error(f"Missing schema source dir {source_path}.")

    command = ["%s/build${shell_ext}" % source_path]

    if options.clean:
        command += ["--clean"]

    if options.generate:
        command += ["--generate"]
    else:
        command += ["--generate", "--build", "--configure"]

    if options.config == "debug":
        command += ["--debug"]

    if options.vs2022:
        # command += ["--vs2019"]
        pass
    else:
        # for now, we require 2019
        command += ["--vs2019"]
    _run_command(command)


def setup_repo_tool(parser, config):
    tool_config = config.get("repo_schema", {})
    parser.description = "Tool to generate USD CAE schemas."
    omni.repo.man.add_config_arg(parser)
    parser.add_argument(
        "-g",
        "--generate",
        action="store_true",
        default=False,
        help="Generate projects, skip build and configure stages.",
    )
    parser.add_argument("--clean", action="store_true", default=False, help="Clean before executing any steps.")
    parser.add_argument(
        "--vs2022", action="store_true", default=False, help="Use VS2022 instead of the default VS2019."
    )

    if tool_config.get("enabled", False):
        return run_repo_tool
