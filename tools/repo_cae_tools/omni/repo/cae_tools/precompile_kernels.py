# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Precompile WARP kernels tool for Kit-CAE.

Launches kit with omni.cae.dav_libs and omni.cae.data enabled, then executes
dav.aot_compile inside kit to compile Warp kernels ahead of time.

Usage:
    ./repo.sh precompile_kernels --json aot_config.json
    ./repo.sh precompile_kernels --json aot_config.json --devices cpu cuda:0
    ./repo.sh precompile_kernels --json aot_config.json --kernel-cache-dir /path/to/cache

All arguments can also be set in [repo_precompile_kernels] in repo_tools.toml (or repo.toml) and
will be used as defaults when the corresponding command-line flag is omitted.
"""

import os
from logging import getLogger

import omni.repo.man
from omni.repo.man import resolve_tokens

from .helpers import run_command

logger = getLogger(__name__)

from rich.console import Console
from rich.theme import Theme

INFO_COLOR = "#3A96D9"
WARN_COLOR = "#FFD700"

console = Console(theme=Theme())


def run_repo_tool(options, config):
    """Launch kit with omni.cae.dav_libs + omni.cae.data and run aot_compile."""
    console.print("[precompile_kernels] Precompiling kernels", style=INFO_COLOR)

    tool_config = config.get("repo_precompile_kernels", {})

    build_dir = resolve_tokens(tool_config.get("build_dir", "${root}/_build/${platform}/${config}"))
    kit_exe = os.path.join(build_dir, "kit", "kit")
    script = resolve_tokens(
        tool_config.get("script", "${root}/tools/repo_cae_tools/scripts/precompile_kernels_script.py")
    )

    ext_folders = tool_config.get(
        "ext_folders",
        [
            "${root}/_build/${platform}/${config}/exts",
            "${root}/_build/${platform}/${config}/extscache",
            "${root}/_build/${platform}/${config}/apps",
        ],
    )

    # Resolve each argument: CLI flag takes precedence, then toml config, then built-in default.
    json_path = options.json or resolve_tokens(tool_config.get("json", ""))
    if not json_path:
        console.print(
            "[precompile_kernels] No AOT config JSON specified — skipping. "
            "Use --json PATH or set 'json' in [repo_precompile_kernels].",
            style=WARN_COLOR,
        )
        return
    if not os.path.exists(json_path):
        console.print(
            f"[precompile_kernels] AOT config JSON not found: {json_path} — skipping.",
            style=WARN_COLOR,
        )
        return

    devices = options.devices or tool_config.get("devices", ["cuda"])

    kernel_cache_dir = options.kernel_cache_dir or resolve_tokens(tool_config.get("kernel_cache_dir", "")) or None

    # Pass arguments to the kit-side script via environment variables.
    # kit's --exec does not forward argv to the executed script.
    os.environ["DAV_COMPILE_KERNELS_AOT"] = "1"
    os.environ["PRECOMPILE_KERNELS_JSON"] = json_path
    os.environ["PRECOMPILE_KERNELS_DEVICES"] = " ".join(devices)
    if kernel_cache_dir:
        os.environ["WARP_CACHE_PATH"] = kernel_cache_dir

    cmd = [kit_exe]
    cmd += ["--enable", "omni.cae.dav_libs"]
    cmd += ["--enable", "omni.cae.data"]
    for folder in ext_folders:
        cmd += ["--ext-folder", resolve_tokens(folder)]
    cmd += ["--portable-root", build_dir + "/"]
    cmd += ["--no-window"]
    cmd += ["--/telemetry/mode=test"]
    cmd += ["-v"]  # show info messages
    cmd += ["--exec", script]

    run_command(cmd)

    console.print("[precompile_kernels] Done", style=INFO_COLOR)


def setup_repo_tool(parser, config):
    """Set up the precompile_kernels command."""
    tool_config = config.get("repo_precompile_kernels", {})
    parser.description = "Precompile Warp kernels by launching kit with omni.cae.dav_libs and omni.cae.data."
    omni.repo.man.add_config_arg(parser)
    parser.add_argument(
        "--json",
        default=None,
        metavar="PATH",
        help="Path to the AOT configuration JSON file (produced by dav.recorder). "
        "Falls back to 'json' in [repo_precompile_kernels] if omitted.",
    )
    parser.add_argument(
        "--devices",
        nargs="+",
        default=None,
        metavar="DEVICE",
        help="One or more Warp device strings to compile for (e.g. cuda cpu cuda:0). "
        "Falls back to 'devices' in [repo_precompile_kernels], or cuda if omitted.",
    )
    parser.add_argument(
        "--kernel-cache-dir",
        default=None,
        metavar="DIR",
        help="Directory where compiled kernels are cached. "
        "Falls back to 'kernel_cache_dir' in [repo_precompile_kernels], "
        "or Warp's automatic location if omitted.",
    )

    if tool_config.get("enabled", False):
        return run_repo_tool
