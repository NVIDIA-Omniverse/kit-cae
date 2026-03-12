# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
Helper utilities for CAE tools to interact with the repo tool ecosystem.
"""

import sys
from typing import Dict, List, Optional

import omni.repo.man
from omni.repo.man import get_tokens, resolve_tokens
from omni.repo.man.exceptions import QuietExpectedError


def invoke_tool(
    tool: str,
    args: Optional[List[str]] = None,
    tokens: Optional[List[str]] = None,
    exit_on_error: bool = True,
    silent: bool = False,
) -> int:
    """
    Invoke a repo tool as a subprocess.

    This is the standard way to invoke other repo tools from within a tool.
    It handles token resolution and proper command construction.

    Args:
        tool: The repo tool command name (e.g., "build", "packman", "usd")
        args: Optional list of arguments to pass to the tool
        tokens: Optional list of --set-token arguments in format ["key:value", ...]
                These will NOT be resolved (as they shouldn't be)
        exit_on_error: If True, exit on non-zero return code (default: True)
        silent: If True, suppress the "ctrl+c to Exit" message (default: False)

    Returns:
        The return code from the subprocess

    Example:
        >>> # Call packman pull
        >>> invoke_tool("packman", [
        ...     "pull",
        ...     "${root}/tools/deps/kit-sdk.packman.xml",
        ...     "-p", "${platform}",
        ...     "-t", "config=${config}"
        ... ])

        >>> # Call build with tokens
        >>> invoke_tool("build",
        ...     args=["--config", "release"],
        ...     tokens=["custom_var:value"]
        ... )

        >>> # Call usd tool
        >>> invoke_tool("usd", ["--configuration", "release"])
    """
    # Build the command
    command = [resolve_tokens("${root}/repo${shell_ext}")]

    # Add tokens first (if any)
    if tokens:
        for token in tokens:
            command.append("--set-token")
            command.append(token)  # Don't resolve token values

    # Add the tool name
    command.append(tool)

    # Add additional arguments and resolve tokens in them
    if args:
        for arg in args:
            # Resolve tokens in arguments
            command.append(resolve_tokens(arg))

    # Run the command
    if not silent:
        try:
            from rich.console import Console
            from rich.theme import Theme

            console = Console(theme=Theme())
            console.print("[ctrl+c to Exit]", style="#3A96D9")
        except ImportError:
            # Fall back to simple print if rich is not available
            print("[ctrl+c to Exit]")

    try:
        return omni.repo.man.run_process(command, exit_on_error=exit_on_error)
    except (KeyboardInterrupt, SystemExit):
        if not silent:
            try:
                from rich.console import Console
                from rich.theme import Theme

                console = Console(theme=Theme())
                console.print("Exiting", style="#3A96D9")
            except ImportError:
                print("Exiting")
        sys.exit(0)


def run_command(command: List[str], exit_on_error: bool = True, silent: bool = False) -> int:
    """
    Run an arbitrary command (not necessarily a repo tool).

    This is useful for running external commands like cmake, make, etc.
    All arguments in the command list will have tokens resolved.

    Args:
        command: List of command parts, e.g., ["cmake", "-S", "${root}/source"]
        exit_on_error: If True, exit on non-zero return code (default: True)
        silent: If True, suppress the "ctrl+c to Exit" message (default: False)

    Returns:
        The return code from the subprocess

    Example:
        >>> # Run cmake
        >>> run_command([
        ...     "cmake",
        ...     "-S", "${root}/source",
        ...     "-B", "${root}/_build"
        ... ])
    """
    # Resolve tokens in all arguments
    resolved_command = [resolve_tokens(arg) for arg in command]

    if not silent:
        try:
            from rich.console import Console
            from rich.theme import Theme

            console = Console(theme=Theme())
            console.print("[ctrl+c to Exit]", style="#3A96D9")
        except ImportError:
            print("[ctrl+c to Exit]")

    try:
        return omni.repo.man.run_process(resolved_command, exit_on_error=exit_on_error)
    except (KeyboardInterrupt, SystemExit):
        if not silent:
            try:
                from rich.console import Console
                from rich.theme import Theme

                console = Console(theme=Theme())
                console.print("Exiting", style="#3A96D9")
            except ImportError:
                print("Exiting")
        sys.exit(0)


def quiet_error(message: str):
    """
    Raise a QuietExpectedError with the given message.

    This is useful for signaling expected errors that should be
    displayed to the user without a full traceback.

    Args:
        message: The error message to display

    Raises:
        QuietExpectedError
    """
    print(message)
    raise QuietExpectedError(message)


def unresolve_tokens(path: str) -> str:
    """
    Convert a resolved path back to token format by replacing token values with ${token-name}.

    This is the reverse operation of resolve_tokens. It takes a string with resolved
    values and replaces them with their token placeholders.

    Args:
        path: A path string with resolved token values

    Returns:
        The path with resolved values replaced by token placeholders

    Example:
        >>> # If root=/home/user/project
        >>> unresolve_tokens("/home/user/project/_schemas/source")
        "${root}/_schemas/source"

        >>> # If platform=linux-x86_64
        >>> unresolve_tokens("/build/linux-x86_64/release")
        "/build/${platform}/release"
    """
    # Get the current token values
    tokens = get_tokens()

    if not tokens:
        return path

    # Sort tokens by value length (longest first) to avoid partial replacements
    # For example, if root=/home/user and root_build=/home/user/build,
    # we want to match root_build first
    sorted_tokens = sorted(tokens.items(), key=lambda x: len(str(x[1])), reverse=True)

    result = path
    for token_name, token_value in sorted_tokens:
        if token_value and str(token_value) in result:
            # Replace the value with the token placeholder
            result = result.replace(str(token_value), f"${{{token_name}}}")

    return result
