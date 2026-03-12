# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""USD schema generation tool for Kit-CAE.

This tool generates USD schema source code from .usda schema definitions using
repo_usd. The actual compilation of schemas is handled by premake via repo_build.

Usage:
    ./repo.sh schema               # Generate schema source (runs during fetch)
    ./repo.sh schema --clean-only  # Clean generated files
    ./repo.sh schema --pluginfo-only  # Configure plugInfo.json (runs post-build)
"""

import os
import shutil
import sys
from logging import getLogger

import omni.repo.man
from omni.repo.man import resolve_tokens

from .helpers import invoke_tool

logger = getLogger(__name__)


# These dependencies come from repo_kit_template
from rich.console import Console
from rich.theme import Theme

# This should match repo_kit_template.palette
INFO_COLOR = "#3A96D9"
WARN_COLOR = "#FFD700"

theme = Theme()
console = Console(theme=theme)


def fetch_dependencies(options, tool_config):
    console.print(f"\[schema] Fetching dependencies for config: {options.config}", style=INFO_COLOR)

    # Get packman files from config
    packman_files = tool_config.get("dependencies_packman_xml", [])
    if not packman_files:
        # print a warning
        console.print(
            f"\[schema] No packman files found in config. Use 'dependencies_packman_xml' to specify the packman files to fetch in repo_tools.toml.",
            style=WARN_COLOR,
        )
        return

    # Build packman arguments (these are packman -t tokens, not repo --set-token)
    for packman_file in packman_files:
        packman_args = [
            "pull",
            packman_file,
            "-p",
            "${platform}",
            "-t",
            "platform_target=${platform}",
            "-t",
            "platform_target_abi=${platform_target_abi}",
            "-t",
            f"config={options.config}",
            "-t",
            "root=${root}",
        ]

        invoke_tool("packman", args=packman_args, silent=True)

    console.print(f"\[schema] Dependencies fetched for config: {options.config}", style=INFO_COLOR)


def generate_schemas(options, _tool_config, repo_usd_config):
    """Generate USD schema source code using repo_usd.

    This generates C++ and Python source files from .usda schema definitions.
    The actual compilation is handled by premake via repo_build.
    """
    console.print(f"\[schema] Generating schemas", style=INFO_COLOR)

    # Create directories based on each plugin's generate_dir
    # This ensures the directories exist before repo_usd tries to write to them
    plugins = repo_usd_config.get("plugin", {})
    for _plugin_name, plugin_config in plugins.items():
        generate_dir = plugin_config.get("generate_dir")
        if generate_dir:
            resolved_dir = resolve_tokens(generate_dir)
            os.makedirs(resolved_dir, exist_ok=True)
            console.print(f"\[schema] Created directory: {resolved_dir}", style=INFO_COLOR)

    # Invoke repo_usd to generate schema source code
    # repo_usd uses generate_dir from each plugin config in repo_schemas.toml
    invoke_tool("usd", args=["--configuration", options.config], tokens=[], silent=True)

    console.print(f"\[schema] Schemas generated (will be built via repo_build)", style=INFO_COLOR)


def clean_schemas(options, tool_config):
    """Clean generated schema source files."""
    console.print(f"\[schema] Cleaning schemas", style=INFO_COLOR)

    # Get the root schema directory (parent of all schema directories)
    schema_generated_source_root = tool_config.get("schema_generated_source_root", "${root}/_schemas/source")
    # Extract the parent directory (e.g., ${root}/_schemas from ${root}/_schemas/source)
    schemas_parent = resolve_tokens(schema_generated_source_root.rsplit("/", 1)[0])

    if os.path.exists(schemas_parent):
        shutil.rmtree(schemas_parent)
        console.print(f"\[schema] Removed {schemas_parent}", style=INFO_COLOR)
    else:
        console.print(f"\[schema] Nothing to clean, {schemas_parent} does not exist", style=INFO_COLOR)


def configure_pluginfo(options, repo_usd_config):
    """Configure plugInfo.json files using repo_usd.

    Uses repo_usd's built-in --configure-pluginfo to handle @PLUG_INFO_*@ token
    replacement in the install_root (schemas/ directory). Then copies the configured
    plugInfo.json files to the extension directory (premake postbuildcommands already
    copied the unconfigured versions during the build phase).
    """
    console.print(f"\[schema] Configuring plugInfo.json for config: {options.config}", style=INFO_COLOR)

    # Use repo_usd's built-in --configure-pluginfo to handle token replacement
    # in the install_root (schemas/ directory). repo_usd reads install_root and
    # other paths from repo_schemas.toml [repo_usd.plugin.*] configuration.
    invoke_tool("usd", args=["--configuration", options.config, "--configure-pluginfo"], tokens=[], silent=True)
    console.print(f"\[schema] PlugInfo.json configured for config: {options.config}", style=INFO_COLOR)

    # Copy schema outputs (plugins, Python modules, native libs) to extension directory.
    # This replaces premake postbuildcommands which can't be used because:
    #   - omni.cae.schema is Python-only (no premake build target)
    #   - dependson with forward references corrupts premake5 on Ubuntu 24.04 / GLIBC 2.39
    root = resolve_tokens("${root}")
    platform = "linux-x86_64" if sys.platform == "linux" else "windows-x86_64"
    config = options.config

    schemas_root = os.path.join(root, "_build", platform, config, "schemas")
    ext_dir = os.path.join(root, "_build", platform, config, "exts", "omni.cae.schema")

    plugins = repo_usd_config.get("plugin", {})
    copy_count = 0
    for _plugin_name, plugin_cfg in plugins.items():
        library_prefix = plugin_cfg.get("library_prefix", _plugin_name)
        display_name = library_prefix[0].upper() + library_prefix[1:]

        # Copy plugin resources (includes configured plugInfo.json)
        src_plugins = os.path.join(schemas_root, "plugins", display_name)
        dst_plugins = os.path.join(ext_dir, "plugins", display_name)
        if os.path.isdir(src_plugins):
            if os.path.exists(dst_plugins):
                shutil.rmtree(dst_plugins)
            shutil.copytree(src_plugins, dst_plugins)

        # Copy Python module directory
        src_module = os.path.join(schemas_root, display_name)
        dst_module = os.path.join(ext_dir, display_name)
        if os.path.isdir(src_module):
            if os.path.exists(dst_module):
                shutil.rmtree(dst_module)
            shutil.copytree(src_module, dst_module)

        # Copy native library (lib name uses lowercase first char: OmniCae -> omniCae)
        lib_base = library_prefix[0].lower() + library_prefix[1:]
        if sys.platform == "linux":
            lib_name = f"lib{lib_base}.so"
            src_lib = os.path.join(schemas_root, "lib", lib_name)
            dst_lib_dir = os.path.join(ext_dir, "lib")
        else:
            src_lib = os.path.join(schemas_root, "bin", f"{lib_base}.dll")
            dst_lib_dir = os.path.join(ext_dir, "bin")
            # Also copy import lib on Windows
            imp_src = os.path.join(schemas_root, "lib", f"{lib_base}.lib")
            imp_dst_dir = os.path.join(ext_dir, "lib")
            if os.path.isfile(imp_src):
                os.makedirs(imp_dst_dir, exist_ok=True)
                shutil.copy2(imp_src, imp_dst_dir)

        if os.path.isfile(src_lib):
            os.makedirs(dst_lib_dir, exist_ok=True)
            shutil.copy2(src_lib, dst_lib_dir)

        copy_count += 1

    console.print(f"\[schema] Copied {copy_count} configured plugInfo.json to extension", style=INFO_COLOR)


def run_repo_tool(options, config):
    """Generate USD schema source code.

    Schema compilation is handled by premake via repo_build, not by this tool.
    This tool generates schema source code, which is then compiled during the
    normal build process.
    """
    if not options.no_warn:
        console.print(
            "\[schema] WARNING: `repo schema` is no longer needed. Use `repo build` instead, which generates and builds schemas automatically.",
            style=WARN_COLOR,
        )

    console.print("\[schema] Generating CAE USD schemas", style=INFO_COLOR)

    # Get tool configuration
    tool_config = config.get("repo_schema", {})
    repo_usd_config = config.get("repo_usd", {})

    # Ensure Kit version is configured and packman XML files are generated.
    # This is only needed for standalone ./repo.sh schema calls.
    # When called via ./repo.sh build, before_pull_commands already handles this.
    kit_sdk_packman = resolve_tokens("${root}/tools/deps/kit-sdk.packman.xml")
    if not os.path.exists(kit_sdk_packman):
        console.print("[schema] Ensuring Kit configuration is set up...", style=INFO_COLOR)
        invoke_tool("select_kit_version", args=["--auto", "--quiet"], silent=True)

    do_clean = options.clean or options.clean_only
    do_fetch = options.fetch_only or not (options.clean_only or options.generate_only or options.pluginfo_only)
    do_generate = options.generate_only or not (options.clean_only or options.fetch_only or options.pluginfo_only)
    do_pluginfo = options.pluginfo_only or not (options.clean_only or options.generate_only or options.fetch_only)

    if do_clean:
        clean_schemas(options, tool_config)
    if do_fetch:
        fetch_dependencies(options, tool_config)
    if do_generate:
        generate_schemas(options, tool_config, repo_usd_config)
    if do_pluginfo:
        configure_pluginfo(options, repo_usd_config)


def setup_repo_tool(parser, config):
    """Set up the schema command arguments.

    Note: Schema compilation is handled by premake via repo_build. This tool
    only generates source code and configures plugInfo.json files.
    """
    tool_config = config.get("repo_schema", {})
    parser.description = "Tool to generate USD CAE schema source code."
    omni.repo.man.add_config_arg(parser)
    parser.add_argument("--clean", action="store_true", default=False, help="Clean before executing any steps.")
    parser.add_argument("--clean-only", action="store_true", default=False, help="Only clean generated schema files.")
    parser.add_argument(
        "--fetch-only", action="store_true", default=False, help="Only fetch schema generation dependencies."
    )
    parser.add_argument("--generate-only", action="store_true", default=False, help="Only generate schema source code.")
    parser.add_argument(
        "--pluginfo-only",
        action="store_true",
        default=False,
        help="Only configure plugInfo.json files (run after build).",
    )

    parser.add_argument(
        "--no-warn",
        action="store_true",
        default=False,
    )

    if tool_config.get("enabled", False):
        return run_repo_tool
