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
Kit Version Configuration Tool

This tool manages Kit SDK version selection and generates files from templates
based on the selected version.

Configuration (in repo.toml under [repo_select_kit_version]):
    versions_config_file: Path to kit-versions.json (default: ${root}/tools/kit-versions.json)
    tracking_file: File to track selected version and mode (default: ${root}/.kit_selection.json)

Kit Versions Configuration File Structure:
    {
        "common": {
            "templates": [
                {"src": "path/to/template", "dest": "path/to/output"}
            ]
        },
        "versions": {
            "version_label": {
                "variables": {
                    "VAR_NAME": "value"
                },
                "templates": [
                    {"src": "path/to/template", "dest": "path/to/output"}
                ]
            }
        },
        "default": "version_label"
    }

Template Processing:
    - .template files: Performs @@VARIABLE@@ substitution using version variables
    - Jinja2 files: Processes {{ version }} and comments out other Jinja2 constructs
    - Plain files: Creates symlinks (development) or copies (packaging with --no-use-symlinks)

Usage Modes:
    1. Interactive selection (no args):
       - Prompts user to select a version
       - Saves selection to tracking file (git-ignored)
       - Generates files immediately after selection

    2. Auto mode (--auto):
       - Uses tracked version if available, otherwise default
       - Generates files with the selected version
       - Ideal for automation and CI/CD
       - If no version was previously selected, uses default and tracks it

    3. Explicit version selection (--version <label>):
       - Selects and generates files for a specific version
       - Saves to tracking file
       - Files are always overwritten to ensure correct version

    4. Force default version (--default):
       - Uses default version from config
       - Generates files immediately
       - Updates tracking file

    5. Dry-run mode (--dry-run):
       - Standalone: Validates current files and shows what needs fixing
       - With --auto/--version/--default: Previews changes without modifying files
       - Shows each template and its action type
       - Shows which files would be overwritten
       - Identifies errors before actual generation
       - Exits with code 0 if OK, 1 if errors found

    6. List available versions (--list-versions):
       - Displays all available Kit versions in a table
       - Shows version variables for each version
       - Shows which version is default and selected
       - Use --json for machine-readable JSON output
       - Never makes changes, just displays information

    7. Show configuration (--show-config):
       - Displays current tool configuration
       - Shows resolved file paths
       - Shows configuration overrides from repo.toml
       - Checks if configuration files exist
       - Never makes changes, just displays information

Options:
    --list-versions: Display available versions and current selection status
    --json: Output in JSON format (use with --list-versions)
    --show-config: Display current tool configuration and file paths
    --dry-run: Validate files (standalone) or preview changes (with version options)
    --auto: Use tracked/default version and generate files (automation-friendly)
    --default: Use default version and generate files
    --version <label>: Select and generate files for a specific version (e.g., "108.0.0")
    --use-symlinks / --no-use-symlinks: Toggle symlink vs copy mode for plain files (sticky, persisted in .kit_selection.json)
    --quiet: Suppress non-essential output for automation/CI (only shows errors)

Exit Codes:
    0: Success (operation completed successfully)
    1: Error (validation failed, file errors, or configuration issues)
    2: User cancelled operation (Ctrl+C in interactive mode)

The tool processes:
    - Common templates: Applied to all versions
    - Version-specific templates: Applied only to the selected version
    - Templates can be .template files, Jinja2 files, or plain files
"""

import argparse
import json
import os
import re
import shutil
import sys
from logging import getLogger
from pathlib import Path

import omni.repo.man
from omni.repo.man import resolve_tokens

# These dependencies come from repo_kit_template
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table
from rich.theme import Theme

logger = getLogger(__name__)

# This should match repo_kit_template.palette
INFO_COLOR = "#3A96D9"
WARN_COLOR = "#FFD700"
SUCCESS_COLOR = "#00FF00"
ERROR_COLOR = "#FF0000"

# Kit selection state file (excluded from git)
KIT_SELECTION_FILE = ".kit_selection.json"

# Global quiet mode flag (set by --quiet)
_quiet_mode = False

theme = Theme()
console = Console(theme=theme)


def set_quiet_mode(quiet):
    """Set global quiet mode for suppressing non-essential output."""
    global _quiet_mode
    _quiet_mode = quiet


def print_info(*args, **kwargs):
    """Print info message only if not in quiet mode."""
    if not _quiet_mode:
        console.print(*args, **kwargs)


def print_error(*args, **kwargs):
    """Print error message (always shown, even in quiet mode)."""
    console.print(*args, **kwargs)


def get_config_paths(tool_config):
    """
    Get configured paths from tool configuration.

    Args:
        tool_config: The tool configuration dict from repo.toml

    Returns:
        Dictionary with resolved paths
    """
    # Get paths from config with defaults
    versions_config_file = tool_config.get("versions_config_file", "${root}/tools/kit-versions.json")
    tracking_file = tool_config.get("tracking_file", "${root}/.kit_selection.json")

    # Resolve tokens
    return {
        "versions_config_file": resolve_tokens(versions_config_file),
        "tracking_file": resolve_tokens(tracking_file),
    }


def validate_kit_versions_config(config, config_file_path):
    """
    Validate the structure of kit-versions.json configuration.

    Args:
        config: The loaded configuration dictionary
        config_file_path: Path to the config file (for error messages)

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    # Check top-level structure
    if not isinstance(config, dict):
        errors.append(f"Configuration must be a JSON object, got {type(config).__name__}")
        return errors

    # Check for required keys
    if "versions" not in config:
        errors.append("Missing required 'versions' key")
    elif not isinstance(config["versions"], dict):
        errors.append(f"'versions' must be an object, got {type(config['versions']).__name__}")
    elif len(config["versions"]) == 0:
        errors.append("'versions' object is empty - at least one version must be defined")

    if "default" not in config:
        errors.append("Missing required 'default' key")
    elif not isinstance(config.get("default"), str):
        errors.append(f"'default' must be a string, got {type(config.get('default')).__name__}")

    # Validate default version exists
    if "default" in config and "versions" in config:
        default_version = config["default"]
        if default_version not in config["versions"]:
            available = ", ".join(f"'{v}'" for v in config["versions"].keys())
            errors.append(
                f"Default version '{default_version}' not found in versions. " f"Available versions: {available}"
            )

    # Validate each version structure
    if "versions" in config and isinstance(config["versions"], dict):
        for version_label, version_info in config["versions"].items():
            if not isinstance(version_info, dict):
                errors.append(f"Version '{version_label}' must be an object, got {type(version_info).__name__}")
                continue

            # Check variables structure
            if "variables" in version_info:
                if not isinstance(version_info["variables"], dict):
                    errors.append(
                        f"Version '{version_label}': 'variables' must be an object, "
                        f"got {type(version_info['variables']).__name__}"
                    )

            # Check templates structure
            if "templates" in version_info:
                templates = version_info["templates"]
                if not isinstance(templates, list):
                    errors.append(
                        f"Version '{version_label}': 'templates' must be an array, " f"got {type(templates).__name__}"
                    )
                else:
                    for idx, template in enumerate(templates):
                        if not isinstance(template, dict):
                            errors.append(
                                f"Version '{version_label}': template #{idx} must be an object, "
                                f"got {type(template).__name__}"
                            )
                        elif "src" not in template or "dest" not in template:
                            errors.append(
                                f"Version '{version_label}': template #{idx} missing required 'src' or 'dest' key"
                            )

    # Validate common templates structure
    if "common" in config:
        common = config["common"]
        if not isinstance(common, dict):
            errors.append(f"'common' must be an object, got {type(common).__name__}")
        elif "templates" in common:
            templates = common["templates"]
            if not isinstance(templates, list):
                errors.append(f"'common.templates' must be an array, got {type(templates).__name__}")
            else:
                for idx, template in enumerate(templates):
                    if not isinstance(template, dict):
                        errors.append(f"common template #{idx} must be an object, got {type(template).__name__}")
                    elif "src" not in template or "dest" not in template:
                        errors.append(f"common template #{idx} missing required 'src' or 'dest' key")

    return errors


def load_kit_versions_config(config_file_path):
    """
    Load and validate the kit-versions.json configuration file.

    Args:
        config_file_path: Path to the versions config file
    """
    config_file = Path(config_file_path)

    if not config_file.exists():
        print_error(f"Error: Kit versions config not found: {config_file}", style=ERROR_COLOR)
        print_error()
        print_error("Expected location:", style=INFO_COLOR)
        print_error(f"  {config_file}")
        print_error()
        print_error("You can configure this path in repo.toml under [repo_select_kit_version]:", style=INFO_COLOR)
        print_error('  versions_config_file = "${root}/path/to/kit-versions.json"')
        sys.exit(1)

    try:
        with open(config_file) as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        print_error(f"Error: Invalid JSON in {config_file}", style=ERROR_COLOR)
        print_error(f"  {e}")
        sys.exit(1)

    # Validate configuration structure
    validation_errors = validate_kit_versions_config(config, config_file_path)
    if validation_errors:
        print_error(f"Error: Invalid configuration in {config_file}", style=ERROR_COLOR)
        print_error()
        print_error("Configuration errors found:", style=ERROR_COLOR)
        for error in validation_errors:
            print_error(f"  • {error}")
        print_error()
        print_error("Expected structure:", style=INFO_COLOR)
        print_error(
            """{
  "common": {
    "templates": [{"src": "...", "dest": "..."}]
  },
  "versions": {
    "version_label": {
      "variables": {"VAR": "value"},
      "templates": [{"src": "...", "dest": "..."}]
    }
  },
  "default": "version_label"
}"""
        )
        sys.exit(1)

    return config


def get_version_tracking_file_path(tracking_file_path):
    """
    Get the path to the version tracking file.

    Args:
        tracking_file_path: Configured tracking file path
    """
    return Path(tracking_file_path)


def read_kit_state(tracking_file_path):
    """
    Read kit selection state (version + use_symlinks) from the JSON state file.

    Args:
        tracking_file_path: Configured tracking file path

    Returns:
        Dict with keys 'version' (str or None) and 'use_symlinks' (bool, default True)
    """
    tracking_file = get_version_tracking_file_path(tracking_file_path)
    if not tracking_file.exists():
        return {"version": None, "use_symlinks": True}
    try:
        data = json.loads(tracking_file.read_text())
        return {
            "version": data.get("version") or None,
            "use_symlinks": data.get("use_symlinks", True),
        }
    except Exception as e:
        console.print(f"Warning: Could not read {tracking_file.name}: {e}", style=WARN_COLOR)
        return {"version": None, "use_symlinks": True}


def read_selected_version(tracking_file_path):
    """
    Read the currently selected version from the state file.

    Args:
        tracking_file_path: Configured tracking file path

    Returns:
        The version label string if file exists, None otherwise
    """
    return read_kit_state(tracking_file_path)["version"]


def write_kit_state(version_label, use_symlinks, tracking_file_path):
    """
    Write kit selection state (version + use_symlinks) to the JSON state file.

    Args:
        version_label: The version label to write
        use_symlinks: Whether to use symlinks for plain files
        tracking_file_path: Configured tracking file path
    """
    tracking_file = get_version_tracking_file_path(tracking_file_path)
    try:
        tracking_file.write_text(json.dumps({"version": version_label, "use_symlinks": use_symlinks}))
        console.print(f"Saved configuration to {tracking_file.name}", style=INFO_COLOR)
    except Exception as e:
        console.print(f"Warning: Could not write {tracking_file.name}: {e}", style=WARN_COLOR)


def write_selected_version(version_label, tracking_file_path):
    """
    Write the selected version to the state file, preserving existing use_symlinks setting.

    Args:
        version_label: The version label to write
        tracking_file_path: Configured tracking file path
    """
    state = read_kit_state(tracking_file_path)
    write_kit_state(version_label, state["use_symlinks"], tracking_file_path)


def process_template_file(src_path, dest_path, variables, version_label, use_symlinks=True, dry_run=False):
    """
    Process a single template file according to its type.

    For .template files: Performs variable substitution using @@VARIABLE@@ syntax
    For Jinja2 files: Processes {{ version }} and comments out other Jinja2 constructs
    For plain files: Creates symlinks (development) or copies (packaging)

    Args:
        src_path: Path to the source template file
        dest_path: Path to the destination file
        variables: Dictionary of variables for @@VARIABLE@@ substitution
        version_label: Version label for {{ version }} substitution
        use_symlinks: If True, create symlinks for plain files; if False, copy
        dry_run: If True, don't modify files, just return what would be done

    Returns:
        If dry_run: Dict with action details
        If not dry_run: Path to the processed file
    """
    src = Path(src_path)
    dest = Path(dest_path)

    # Check if it's a .template file (needs @@VARIABLE@@ substitution)
    if src.suffix == ".template":
        action = "Process .template file (@@VARIABLE@@ substitution)"
        used_vars = []

        if dry_run:
            # Analyze what variables would be used
            template_content = src.read_text(encoding="utf-8")
            for var in variables:
                if f"@@{var}@@" in template_content:
                    used_vars.append(var)

            return {
                "src": src,
                "dest": dest,
                "action": action,
                "type": "template",
                "variables": used_vars,
                "would_overwrite": dest.exists() or dest.is_symlink(),
            }

        # Real processing
        console.print(f"  Processing {dest.name} from {src.name}...")

        # Ensure destination directory exists
        dest.parent.mkdir(parents=True, exist_ok=True)

        # Remove existing file/symlink
        if dest.exists() or dest.is_symlink():
            dest.unlink()

        template_content = src.read_text(encoding="utf-8")
        output_content = template_content

        # Replace all @@VARIABLE@@ occurrences
        for var, value in variables.items():
            if isinstance(value, str):
                pattern = f"@@{var}@@"
                output_content = output_content.replace(pattern, value)

        # Check for unresolved tags
        remaining_tags = re.findall(r"@@\w+@@", output_content)
        if remaining_tags:
            console.print(f"  Warning: Unresolved tags in {src.name}: {remaining_tags}", style=WARN_COLOR)

        dest.write_text(output_content, encoding="utf-8")
        return dest

    # Check if it has Jinja2 syntax (needs {{ version }} processing)
    elif has_jinja_syntax(src):
        action = f'Process Jinja2 file ({{{{ version }}}} → "{version_label}")'

        if dry_run:
            return {
                "src": src,
                "dest": dest,
                "action": action,
                "type": "jinja2",
                "version_label": version_label,
                "would_overwrite": dest.exists() or dest.is_symlink(),
            }

        # Real processing
        console.print(f"  Processing {dest.name} from {src.name}...")

        dest.parent.mkdir(parents=True, exist_ok=True)

        if dest.exists() or dest.is_symlink():
            dest.unlink()

        template_content = src.read_text(encoding="utf-8")
        output_lines = []

        for line in template_content.splitlines(keepends=True):
            # Replace {{ version }} with actual version
            if "version" in line and "{{" in line:
                try:
                    if "version" in line[line.index("{{") : line.index("}}") + 2]:
                        processed_line = re.sub(r'(["\'])?\{\{\s*version\s*\}\}(["\'])?', f'"{version_label}"', line)
                        output_lines.append(processed_line)
                        continue
                except ValueError:
                    pass

            # Comment out other Jinja2 constructs
            if "{{" in line or "{%" in line or "{#" in line:
                output_lines.append(f"# {line}" if not line.lstrip().startswith("#") else line)
            else:
                output_lines.append(line)

        dest.write_text("".join(output_lines), encoding="utf-8")
        return dest

    # Plain file - symlink or copy
    else:
        if use_symlinks:
            action = f"Create symlink (plain file)"
        else:
            action = f"Copy file (plain file)"

        if dry_run:
            return {
                "src": src,
                "dest": dest,
                "action": action,
                "type": "symlink" if use_symlinks else "copy",
                "would_overwrite": dest.exists() or dest.is_symlink(),
            }

        # Real processing
        dest.parent.mkdir(parents=True, exist_ok=True)

        if dest.exists() or dest.is_symlink():
            dest.unlink()

        if use_symlinks:
            console.print(f"  Creating symlink for {dest.name}...")

            try:
                # Use absolute path for the symlink target
                absolute_src_path = src.resolve()
                dest.symlink_to(absolute_src_path, target_is_directory=False)
                return dest

            except OSError as e:
                # Symlink creation might fail on Windows without proper privileges
                console.print(
                    f"  Warning: Could not create symlink for {dest.name}, copying file instead",
                    style=WARN_COLOR,
                )
                if sys.platform == "win32":
                    console.print(
                        f"  (Windows tip: Enable Developer Mode or run as Administrator for symlink support)",
                        style=WARN_COLOR,
                    )

                # Copy the file as fallback
                shutil.copy2(src, dest)
                return dest
        else:
            # Copy mode - needed for packaging
            console.print(f"  Copying {dest.name}...")
            shutil.copy2(src, dest)
            return dest


def process_templates(version_label, common_templates, version_templates, variables, use_symlinks=True, dry_run=False):
    """
    Process all templates (both common and version-specific).

    Args:
        version_label: The version label (e.g., "108.0.0", "109.0.1")
        common_templates: List of common template dicts with 'src' and 'dest' keys
        version_templates: List of version-specific template dicts with 'src' and 'dest' keys
        variables: Dictionary of variables for substitution
        use_symlinks: If True, create symlinks for plain files; if False, copy
        dry_run: If True, don't modify files, return list of actions that would be taken

    Returns:
        If dry_run: List of action dicts
        If not dry_run: List of processed file paths
    """
    if dry_run:
        actions = []

        # Analyze common templates
        if common_templates:
            for template in common_templates:
                src = resolve_tokens(template["src"])
                dest = resolve_tokens(template["dest"])

                src_path = Path(src)
                if not src_path.exists():
                    actions.append(
                        {
                            "src": src_path,
                            "dest": Path(dest),
                            "action": "ERROR: Source template not found",
                            "type": "error",
                            "is_common": True,
                        }
                    )
                    continue

                action = process_template_file(src, dest, variables, version_label, use_symlinks, dry_run=True)
                action["is_common"] = True
                actions.append(action)

        # Analyze version-specific templates
        if version_templates:
            for template in version_templates:
                src = resolve_tokens(template["src"])
                dest = resolve_tokens(template["dest"])

                src_path = Path(src)
                if not src_path.exists():
                    actions.append(
                        {
                            "src": src_path,
                            "dest": Path(dest),
                            "action": "ERROR: Source template not found",
                            "type": "error",
                            "is_common": False,
                        }
                    )
                    continue

                action = process_template_file(src, dest, variables, version_label, use_symlinks, dry_run=True)
                action["is_common"] = False
                actions.append(action)

        return actions

    # Real processing (existing code)
    processed_files = []

    # Process common templates first
    if common_templates:
        console.print("Processing common templates...", style=INFO_COLOR)
        for template in common_templates:
            src = resolve_tokens(template["src"])
            dest = resolve_tokens(template["dest"])

            src_path = Path(src)
            if not src_path.exists():
                console.print(f"  Warning: Template not found: {src}", style=WARN_COLOR)
                console.print(f"           Check that the path exists or update kit-versions.json", style="dim")
                continue

            processed_file = process_template_file(src, dest, variables, version_label, use_symlinks)
            processed_files.append(processed_file)

    # Process version-specific templates
    if version_templates:
        if common_templates:
            console.print()
        console.print("Processing version-specific templates...", style=INFO_COLOR)
        for template in version_templates:
            src = resolve_tokens(template["src"])
            dest = resolve_tokens(template["dest"])

            src_path = Path(src)
            if not src_path.exists():
                console.print(f"  Warning: Template not found: {src}", style=WARN_COLOR)
                console.print(f"           Check that the path exists or update kit-versions.json", style="dim")
                continue

            processed_file = process_template_file(src, dest, variables, version_label, use_symlinks)
            processed_files.append(processed_file)

    return processed_files


def display_dry_run_results(actions, version_label):
    """
    Display the results of a dry-run.

    Args:
        actions: List of action dicts from process_templates(..., dry_run=True)
        version_label: The version label being processed
    """
    console.print()
    console.print("=" * 80, style=INFO_COLOR)
    console.print("DRY-RUN MODE - No files will be modified", style=WARN_COLOR)
    console.print("=" * 80, style=INFO_COLOR)
    console.print()

    # Separate common and version-specific
    common_actions = [a for a in actions if a.get("is_common", False)]
    version_actions = [a for a in actions if not a.get("is_common", False)]

    # Count by type
    total = len(actions)
    errors = len([a for a in actions if a.get("type") == "error"])
    overwrites = len([a for a in actions if a.get("would_overwrite", False)])

    console.print(f"Would process {total} template(s) for version {version_label}:", style=INFO_COLOR)
    if overwrites > 0:
        console.print(f"  {overwrites} file(s) would be overwritten", style=WARN_COLOR)
    if errors > 0:
        console.print(f"  {errors} error(s) found", style=ERROR_COLOR)
    console.print()

    # Display common templates
    if common_actions:
        console.print("Common Templates:", style=INFO_COLOR)
        console.print()

        for action in common_actions:
            dest_name = action["dest"].name

            if action.get("type") == "error":
                console.print(f"  ✗ {dest_name}", style=ERROR_COLOR)
                console.print(f"    {action['action']}", style=ERROR_COLOR)
                console.print(f"    Source: {action['src']}", style="dim")
            else:
                symbol = "⚠" if action.get("would_overwrite") else "✓"
                color = WARN_COLOR if action.get("would_overwrite") else SUCCESS_COLOR

                console.print(f"  {symbol} {dest_name}", style=color)
                console.print(f"    Action: {action['action']}", style="cyan")

                if action.get("type") == "template" and action.get("variables"):
                    console.print(f"    Variables: {', '.join(action['variables'])}", style="dim")
                elif action.get("type") == "jinja2":
                    console.print(f"    Version: {action.get('version_label')}", style="dim")

                console.print(f"    Source: {action['src']}", style="dim")
                console.print(f"    Dest:   {action['dest']}", style="dim")

                if action.get("would_overwrite"):
                    console.print(f"    Would overwrite existing file", style=WARN_COLOR)

        console.print()

    # Display version-specific templates
    if version_actions:
        console.print(f"Version-Specific Templates ({version_label}):", style=INFO_COLOR)
        console.print()

        for action in version_actions:
            dest_name = action["dest"].name

            if action.get("type") == "error":
                console.print(f"  ✗ {dest_name}", style=ERROR_COLOR)
                console.print(f"    {action['action']}", style=ERROR_COLOR)
                console.print(f"    Source: {action['src']}", style="dim")
            else:
                symbol = "⚠" if action.get("would_overwrite") else "✓"
                color = WARN_COLOR if action.get("would_overwrite") else SUCCESS_COLOR

                console.print(f"  {symbol} {dest_name}", style=color)
                console.print(f"    Action: {action['action']}", style="cyan")

                if action.get("type") == "template" and action.get("variables"):
                    console.print(f"    Variables: {', '.join(action['variables'])}", style="dim")
                elif action.get("type") == "jinja2":
                    console.print(f"    Version: {action.get('version_label')}", style="dim")

                console.print(f"    Source: {action['src']}", style="dim")
                console.print(f"    Dest:   {action['dest']}", style="dim")

                if action.get("would_overwrite"):
                    console.print(f"    Would overwrite existing file", style=WARN_COLOR)

        console.print()

    # Summary
    console.print("=" * 80, style=INFO_COLOR)
    if errors > 0:
        console.print(f"Dry-run completed with {errors} error(s)", style=ERROR_COLOR)
        console.print()
        console.print("To fix these errors:", style=INFO_COLOR)
        console.print("  • Run with --auto or select a version to create/update files")
        console.print("  • Use --no-use-symlinks if you need copies instead of symlinks")
        console.print("  • Check kit-versions.json if template sources are missing")
    else:
        console.print("Dry-run completed successfully", style=SUCCESS_COLOR)
        console.print()
        console.print("Next steps:", style=INFO_COLOR)
        console.print("  • Run with --auto or select a version to apply these changes")
        console.print("  • All files are correctly configured")
    console.print("=" * 80, style=INFO_COLOR)


def has_jinja_syntax(file_path):
    """
    Check if a file contains Jinja2 template syntax.

    Args:
        file_path: Path to the file to check

    Returns:
        True if the file contains {{ }}, {% %}, or {# #} syntax
    """
    try:
        content = file_path.read_text(encoding="utf-8")
        return bool(re.search(r"\{\{|\{\%|\{\#", content))
    except Exception:
        return False


def check_template_file(src_path, dest_path, variables, version_label, use_symlinks):
    """
    Check if a template file has been processed correctly.

    Args:
        src_path: Path to the source template file
        dest_path: Path to the destination file
        variables: Dictionary of variables for substitution
        version_label: Version label for Jinja2 processing
        use_symlinks: Whether symlinks are expected for plain files

    Returns:
        Tuple of (exists, correct_type, matches, issues) where:
        - exists: bool indicating if file exists
        - correct_type: bool indicating if it's the right type
        - matches: bool indicating if content/target matches expected
        - issues: list of issue descriptions
    """
    src = Path(src_path)
    dest = Path(dest_path)
    issues = []

    if not dest.exists() and not dest.is_symlink():
        return False, None, None, ["File does not exist - run tool to create it"]

    # Check .template files (need @@VARIABLE@@ substitution)
    if src.suffix == ".template":
        if dest.is_symlink():
            issues.append("Expected regular file but found symlink - run tool to fix")
            return True, False, None, issues

        # Generate expected content
        template_content = src.read_text(encoding="utf-8")
        expected_content = template_content

        for var, value in variables.items():
            if isinstance(value, str):
                pattern = f"@@{var}@@"
                expected_content = expected_content.replace(pattern, value)

        # Compare with actual content
        try:
            actual_content = dest.read_text(encoding="utf-8")
        except Exception as e:
            issues.append(f"Could not read file: {e}")
            return True, True, False, issues

        if actual_content != expected_content:
            remaining_tags = re.findall(r"@@\w+@@", actual_content)
            if remaining_tags:
                issues.append(f"Contains unresolved tags: {remaining_tags} - run tool to update")
            else:
                issues.append("Content does not match expected output - run tool to update")
            return True, True, False, issues

        return True, True, True, []

    # Check Jinja2 files (need {{ version }} processing)
    elif has_jinja_syntax(src):
        if dest.is_symlink():
            issues.append("Expected regular file but found symlink - run tool to fix")
            return True, False, None, issues

        # Generate expected content
        template_content = src.read_text(encoding="utf-8")
        expected_lines = []

        for line in template_content.splitlines(keepends=True):
            if "version" in line and "{{" in line:
                try:
                    if "version" in line[line.index("{{") : line.index("}}") + 2]:
                        processed_line = re.sub(r'(["\'])?\{\{\s*version\s*\}\}(["\'])?', f'"{version_label}"', line)
                        expected_lines.append(processed_line)
                        continue
                except ValueError:
                    pass

            if "{{" in line or "{%" in line or "{#" in line:
                expected_lines.append(f"# {line}" if not line.lstrip().startswith("#") else line)
            else:
                expected_lines.append(line)

        expected_content = "".join(expected_lines)

        try:
            actual_content = dest.read_text(encoding="utf-8")
        except Exception as e:
            issues.append(f"Could not read file: {e}")
            return True, True, False, issues

        if actual_content != expected_content:
            issues.append("Processed content does not match expected output - run tool to update")
            return True, True, False, issues

        return True, True, True, []

    # Check plain files (should be symlink or copy)
    else:
        if use_symlinks:
            if not dest.is_symlink():
                issues.append("Expected symlink but found regular file - run tool to fix")
                return True, False, None, issues

            # Check symlink target
            try:
                actual_target = dest.resolve()
                expected_target = src.resolve()

                if actual_target != expected_target:
                    issues.append(f"Symlink points to wrong target - run tool to fix")
                    issues.append(f"  Current: {actual_target}")
                    issues.append(f"  Expected: {expected_target}")
                    return True, True, False, issues

                return True, True, True, []
            except Exception as e:
                issues.append(f"Error checking symlink: {e}")
                return True, True, False, issues
        else:
            # Copy mode - should be regular file
            if dest.is_symlink():
                issues.append("Expected regular file but found symlink - run tool with --no-use-symlinks to fix")
                return True, False, None, issues

            # Check if content matches template
            try:
                expected_content = src.read_text(encoding="utf-8")
                actual_content = dest.read_text(encoding="utf-8")

                if actual_content != expected_content:
                    issues.append("File content does not match template - run tool with --no-use-symlinks to update")
                    return True, True, False, issues

                return True, True, True, []
            except Exception as e:
                issues.append(f"Error checking file content: {e}")
                return True, True, False, issues


def run_check_mode(version_label, common_templates, version_templates, variables, use_symlinks):
    """
    Check the status of generated files without making changes.

    Args:
        version_label: The version label to check against
        common_templates: List of common template dicts with 'src' and 'dest' keys
        version_templates: List of version-specific template dicts with 'src' and 'dest' keys
        variables: Dictionary of variables for substitution
        use_symlinks: Whether to expect symlinks (vs copies) for plain files
    """
    console.print()
    console.print("=" * 70, style=INFO_COLOR)
    console.print("File Status Check", style=INFO_COLOR)
    console.print("=" * 70, style=INFO_COLOR)
    console.print()

    all_ok = True
    mode_description = "symlinks" if use_symlinks else "copies"

    # Check common templates
    if common_templates:
        console.print(f"Common Templates (expecting {mode_description}):", style=INFO_COLOR)
        console.print()

        for template in common_templates:
            src = resolve_tokens(template["src"])
            dest = resolve_tokens(template["dest"])

            src_path = Path(src)
            dest_path = Path(dest)

            if not src_path.exists():
                console.print(f"  ✗ {dest_path.name}", style=ERROR_COLOR)
                console.print(f"    - Template source not found: {src}", style=WARN_COLOR)
                console.print(f"    - Action: Check that the file exists or update kit-versions.json", style="dim")
                all_ok = False
                continue

            exists, correct_type, matches, issues = check_template_file(
                src, dest, variables, version_label, use_symlinks
            )

            if exists and correct_type and matches:
                console.print(f"  ✓ {dest_path.name}", style=SUCCESS_COLOR)
            elif not exists:
                console.print(f"  ✗ {dest_path.name}", style=ERROR_COLOR)
                for issue in issues:
                    console.print(f"    - {issue}", style=WARN_COLOR)
                all_ok = False
            else:
                console.print(f"  ✗ {dest_path.name}", style=ERROR_COLOR)
                for issue in issues:
                    console.print(f"    - {issue}", style=WARN_COLOR)
                all_ok = False

        console.print()

    # Check version-specific templates
    if version_templates:
        console.print(f"Version-Specific Templates (expecting {mode_description}):", style=INFO_COLOR)
        console.print()

        for template in version_templates:
            src = resolve_tokens(template["src"])
            dest = resolve_tokens(template["dest"])

            src_path = Path(src)
            dest_path = Path(dest)

            if not src_path.exists():
                console.print(f"  ✗ {dest_path.name}", style=ERROR_COLOR)
                console.print(f"    - Template source not found: {src}", style=WARN_COLOR)
                console.print(f"    - Action: Check that the file exists or update kit-versions.json", style="dim")
                all_ok = False
                continue

            exists, correct_type, matches, issues = check_template_file(
                src, dest, variables, version_label, use_symlinks
            )

            if exists and correct_type and matches:
                console.print(f"  ✓ {dest_path.name}", style=SUCCESS_COLOR)
            elif not exists:
                console.print(f"  ✗ {dest_path.name}", style=ERROR_COLOR)
                for issue in issues:
                    console.print(f"    - {issue}", style=WARN_COLOR)
                all_ok = False
            else:
                console.print(f"  ✗ {dest_path.name}", style=ERROR_COLOR)
                for issue in issues:
                    console.print(f"    - {issue}", style=WARN_COLOR)
                all_ok = False

        console.print()

    # Summary
    console.print("=" * 70, style=INFO_COLOR)
    if all_ok:
        console.print("All files are present and correct!", style=SUCCESS_COLOR)
    else:
        console.print("Some files have issues. Select a version or use --auto to fix.", style=WARN_COLOR)
    console.print("=" * 70, style=INFO_COLOR)

    return all_ok


def display_configuration(tool_config, paths):
    """
    Display the current tool configuration.

    Args:
        tool_config: The tool configuration dict from repo.toml
        paths: The resolved paths dictionary
    """
    console.print()
    console.print("Kit Version Tool Configuration", style=INFO_COLOR)
    console.print("=" * 80, style=INFO_COLOR)
    console.print()

    # Show resolved paths
    console.print("Resolved Paths:", style=INFO_COLOR)
    console.print(f"  Versions config file: {paths['versions_config_file']}", style="green")
    console.print(f"  Tracking file:        {paths['tracking_file']}", style="green")
    console.print()

    # Show raw configuration (what's in repo.toml)
    console.print("Configuration Settings:", style=INFO_COLOR)
    if tool_config:
        for key, value in tool_config.items():
            console.print(f"  {key}: {value}", style="yellow")
    else:
        console.print("  Using all defaults (no overrides in repo.toml)", style="dim")
    console.print()

    # Check if files exist
    console.print("File Status:", style=INFO_COLOR)

    versions_file = Path(paths["versions_config_file"])
    if versions_file.exists():
        console.print(f"  ✓ Versions config exists: {versions_file}", style=SUCCESS_COLOR)
    else:
        console.print(f"  ✗ Versions config missing: {versions_file}", style=ERROR_COLOR)

    tracking_file = Path(paths["tracking_file"])
    if tracking_file.exists():
        try:
            version = tracking_file.read_text().strip()
            console.print(f"  ✓ Tracking file exists: {tracking_file}", style=SUCCESS_COLOR)
            console.print(f"    Current selection: {version}", style="cyan")
        except Exception as e:
            console.print(f"  ⚠ Tracking file exists but unreadable: {tracking_file}", style=WARN_COLOR)
            console.print(f"    Error: {e}", style="dim")
    else:
        console.print(f"  ○ Tracking file not yet created: {tracking_file}", style="dim")
        console.print(f"    Will be created when a version is selected", style="dim")

    console.print()


def display_available_versions(config, tracking_file_path, json_output=False):
    """
    Display available Kit versions in a formatted table or JSON.

    Args:
        config: The loaded kit-versions.json config
        tracking_file_path: Path to the tracking file
        json_output: If True, output JSON instead of formatted table
    """
    versions = config["versions"]
    default_version = config.get("default")
    tracked_version = read_selected_version(tracking_file_path)

    if json_output:
        # Build JSON structure
        version_list = []
        for label in sorted(versions.keys(), reverse=True):
            version_info = versions[label]
            variables = version_info.get("variables", {})
            templates = version_info.get("templates", [])
            common_templates = config.get("common", {}).get("templates", [])

            version_list.append(
                {
                    "label": label,
                    "variables": variables,
                    "is_default": label == default_version,
                    "is_selected": label == tracked_version,
                    "template_count": {"common": len(common_templates), "version_specific": len(templates)},
                }
            )

        output = {"current": tracked_version, "default": default_version, "available_versions": version_list}

        console.print(json.dumps(output, indent=2))
        return

    # Regular formatted output
    console.print()
    console.print("Available Kit Versions", style=INFO_COLOR)
    console.print("=" * 80, style=INFO_COLOR)
    console.print()

    # Create a nice table
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Version Label", style="green", width=15)
    table.add_column("Variables", style="yellow", width=45)
    table.add_column("Status", style="magenta", width=20)

    version_labels = sorted(versions.keys(), reverse=True)  # Show newest first

    for idx, label in enumerate(version_labels):
        version_info = versions[label]
        variables = version_info.get("variables", {})

        # Show key variables (e.g., KIT_VERSION_TAG)
        var_display = []
        for key in ["KIT_VERSION_TAG", "INDEX_HEADERS_TAG"]:
            if key in variables:
                var_display.append(f"{key}={variables[key]}")

        var_str = "\n".join(var_display) if var_display else "N/A"

        # Build status indicators
        status_parts = []
        if label == default_version:
            status_parts.append("default")
        if label == tracked_version:
            status_parts.append("selected")

        status = ", ".join(status_parts) if status_parts else ""

        table.add_row(label, var_str, status)

        # Add separator between versions (but not after the last one)
        if idx < len(version_labels) - 1:
            table.add_section()

    console.print(table)
    console.print()

    # Show tracking file status
    tracking_file_name = Path(tracking_file_path).name
    if tracked_version:
        console.print(f"Currently selected: {tracked_version}", style=SUCCESS_COLOR)
    else:
        console.print(f"No version selected (tracked in {tracking_file_name})", style=WARN_COLOR)

    console.print(f"Default version: {default_version}", style=INFO_COLOR)
    console.print()


def select_version_interactive(config):
    """
    Interactively prompt user to select a Kit version.

    Args:
        config: The loaded kit-versions.json config

    Returns:
        Selected version label string, or None if user chose to exit
    """
    versions = config["versions"]
    default_version = config.get("default")

    # Create a nice table showing available versions
    table = Table(title="Available Kit Versions", show_header=True, header_style="bold cyan")
    table.add_column("Option", style="cyan", justify="right")
    table.add_column("Version Label", style="green")
    table.add_column("Variables", style="yellow", width=40)
    table.add_column("Default", style="magenta")

    version_labels = sorted(versions.keys(), reverse=True)  # Show newest first

    for idx, label in enumerate(version_labels, 1):
        version_info = versions[label]
        variables = version_info.get("variables", {})

        # Show key variables
        var_display = []
        for key in ["KIT_VERSION_TAG", "INDEX_HEADERS_TAG"]:
            if key in variables:
                var_display.append(f"{key}={variables[key]}")

        var_str = "\n".join(var_display) if var_display else "N/A"
        is_default = "*" if label == default_version else ""
        table.add_row(str(idx), label, var_str, is_default)

        # Add separator between versions
        table.add_section()

    # Add exit option as 0
    table.add_row("0", "Exit", "(no changes)", "")

    console.print(table)
    console.print()
    console.print("[dim]Press Ctrl+C or select 0 to exit without changes[/dim]")
    console.print()

    # Prompt for selection
    default_option = str(version_labels.index(default_version) + 1) if default_version else "1"

    # Build list of valid choices: 0 for exit, 1-N for versions
    valid_choices = ["0"] + [str(i) for i in range(1, len(version_labels) + 1)]

    try:
        choice = Prompt.ask(f"Select Kit version", choices=valid_choices, default=default_option)

        # Check if user chose to exit
        if choice == "0":
            return None

        selected_label = version_labels[int(choice) - 1]
        return selected_label

    except (KeyboardInterrupt, EOFError):
        # User pressed Ctrl+C or Ctrl+D - treat as exit
        console.print()
        return None


def get_tracked_or_default_version(paths, kit_config):
    """
    Get the tracked version, or fall back to default if not tracked or invalid.

    Args:
        paths: Dictionary with tracking_file path
        kit_config: Kit versions configuration

    Returns:
        str: The version to use (tracked or default)
    """
    versions = kit_config["versions"]
    default_version = kit_config.get("default")
    selected_version = read_selected_version(paths["tracking_file"])

    if selected_version:
        if selected_version not in versions:
            console.print(
                f"Warning: Previously selected version '{selected_version}' not found in config",
                style=WARN_COLOR,
            )
            console.print(f"Falling back to default version: {default_version}", style=INFO_COLOR)
            return default_version
        return selected_version

    return default_version


def choose_version(options, config_dict, paths):
    """
    Determine which Kit version to use based on options.

    Returns:
        str or None: The version label to use, or None if user cancelled
    """
    tool_config = config_dict.get("repo_select_kit_version", {})
    kit_config = load_kit_versions_config(paths["versions_config_file"])
    versions = kit_config["versions"]
    default_version = kit_config.get("default")

    # --auto mode: use tracked or default version
    if options.auto:
        selected_version = get_tracked_or_default_version(paths, kit_config)

        # Update tracking file if we fell back to default
        tracked_version = read_selected_version(paths["tracking_file"])
        if not tracked_version or tracked_version not in versions:
            write_selected_version(selected_version, paths["tracking_file"])
            print_info(f"No tracked version, using default: {selected_version}")
        else:
            print_info(f"Using tracked version: {selected_version}")

        return selected_version

    # --default mode: force use of default version
    elif options.use_default:
        selected_version = default_version
        print_info(f"Using default version: {selected_version}")
        return selected_version

    # --version mode: use specified version
    elif options.version:
        selected_version = options.version
        if selected_version not in versions:
            print_error(f"Error: Version '{selected_version}' not found in configuration")
            print_error()
            available = sorted(versions.keys(), reverse=True)
            print_error("Available versions:")
            for v in available:
                marker = " (default)" if v == default_version else ""
                print_error(f"  • {v}{marker}")
            print_error()
            print_error("To see detailed version information, run:")
            print_error("  repo select_kit_version --list-versions")
            sys.exit(1)
        print_info(f"Selecting version: {selected_version}")
        return selected_version

    # Interactive mode: prompt user to select
    else:
        selected_version = select_version_interactive(kit_config)

        # Check if user chose to exit
        if selected_version is None:
            console.print()
            console.print("Exiting without making changes.", style=INFO_COLOR)
            return None

        console.print()
        console.print(f"Selected version: {selected_version}", style=SUCCESS_COLOR)
        return selected_version


def run_repo_tool(options, config_dict):
    """Main entry point for the kit_configure tool."""

    # Set quiet mode if requested
    set_quiet_mode(options.quiet)

    print_info("Kit Version Configuration Tool")
    print_info()

    # Get tool configuration and resolve paths
    tool_config = config_dict.get("repo_select_kit_version", {})
    paths = get_config_paths(tool_config)

    # Handle show-config mode first (just display and exit)
    if options.show_config:
        display_configuration(tool_config, paths)
        sys.exit(0)

    # Load kit versions configuration
    kit_config = load_kit_versions_config(paths["versions_config_file"])
    versions = kit_config["versions"]
    default_version = kit_config.get("default")

    # Handle list-versions mode (just display and exit)
    if options.list_versions:
        display_available_versions(kit_config, paths["tracking_file"], json_output=options.json_output)
        sys.exit(0)

    # Check if we're in standalone dry-run mode (without any version selection)
    is_standalone_dry_run = options.dry_run and not (options.auto or options.use_default or options.version)

    # Handle standalone dry-run mode (validation without changing anything)
    if is_standalone_dry_run:
        # Use tracked version (or default if not tracked) for validation
        selected_version = get_tracked_or_default_version(paths, kit_config)

        tracked_version = read_selected_version(paths["tracking_file"])
        if tracked_version and tracked_version == selected_version:
            console.print(f"Validating with tracked version: {selected_version}", style=INFO_COLOR)
        else:
            console.print(f"No tracked version, validating with default: {selected_version}", style=INFO_COLOR)

        # Get version configuration
        version_config = versions[selected_version]
        variables = version_config.get("variables", {})
        common_templates = kit_config.get("common", {}).get("templates", [])
        version_templates = version_config.get("templates", [])

        # Display current tracking file status
        tracked_version = read_selected_version(paths["tracking_file"])
        tracking_file_name = Path(paths["tracking_file"]).name
        console.print()
        console.print("Current Status:", style=INFO_COLOR)
        if tracked_version:
            console.print(
                f"  Tracked version: {tracked_version}",
                style=SUCCESS_COLOR if tracked_version == selected_version else WARN_COLOR,
            )
            if tracked_version != selected_version:
                console.print(f"  (Validating against: {selected_version})", style=INFO_COLOR)
        else:
            console.print(f"  No version tracked in {tracking_file_name}", style=WARN_COLOR)
            console.print(f"  (Validating against: {selected_version})", style=INFO_COLOR)

        # Display version config
        console.print()
        console.print("Version Configuration:", style=INFO_COLOR)
        console.print(f"  Variables:")
        for key, value in variables.items():
            console.print(f"    {key}: {value}")
        console.print(f"  Common templates: {len(common_templates)}")
        console.print(f"  Version-specific templates: {len(version_templates)}")

        # Check file status
        kit_state = read_kit_state(paths["tracking_file"])
        use_symlinks = kit_state["use_symlinks"] if options.use_symlinks is None else options.use_symlinks
        all_ok = run_check_mode(selected_version, common_templates, version_templates, variables, use_symlinks)

        # Exit with appropriate code
        sys.exit(0 if all_ok else 1)

    # Not standalone dry-run, so choose a version and generate files
    selected_version = choose_version(options, config_dict, paths)

    # If user cancelled (interactive mode), exit gracefully
    if selected_version is None:
        sys.exit(0)

    # Get version configuration
    version_config = versions[selected_version]
    variables = version_config.get("variables", {})
    common_templates = kit_config.get("common", {}).get("templates", [])
    version_templates = version_config.get("templates", [])

    # Resolve effective use_symlinks: CLI flag wins; otherwise fall back to persisted value
    kit_state = read_kit_state(paths["tracking_file"])
    use_symlinks = kit_state["use_symlinks"] if options.use_symlinks is None else options.use_symlinks

    # Save the selected version and mode to state file (if not dry-run)
    if not options.dry_run:
        write_kit_state(selected_version, use_symlinks, paths["tracking_file"])

    # Display version details
    console.print()
    console.print("Version Configuration:", style=INFO_COLOR)
    console.print(f"  Variables:")
    for key, value in variables.items():
        console.print(f"    {key}: {value}")
    console.print(f"  Common templates: {len(common_templates)}")
    console.print(f"  Version-specific templates: {len(version_templates)}")
    console.print()

    # Generate files (always happens when we have a valid version)
    if options.dry_run:
        # Dry-run mode: show what would be done
        actions = process_templates(
            selected_version, common_templates, version_templates, variables, use_symlinks, dry_run=True
        )
        display_dry_run_results(actions, selected_version)

        # Exit with error code if there are errors in the dry-run
        has_errors = any(a.get("type") == "error" for a in actions)
        sys.exit(1 if has_errors else 0)
    else:
        # Real generation
        console.print("Generating files from templates...", style=INFO_COLOR)
        if not use_symlinks:
            console.print("Using copy mode (--no-use-symlinks, symlinks disabled)", style=INFO_COLOR)
        console.print()

        # Process all templates
        all_generated_files = process_templates(
            selected_version, common_templates, version_templates, variables, use_symlinks
        )

        if all_generated_files:
            console.print()
            console.print(f"Generated {len(all_generated_files)} file(s):", style=SUCCESS_COLOR)
            for file_path in all_generated_files:
                console.print(f"  * {file_path.relative_to(resolve_tokens('${root}'))}")
        else:
            console.print("No files were generated.", style=WARN_COLOR)

        console.print()
        console.print("Kit configuration complete!", style=SUCCESS_COLOR)


def setup_repo_tool(parser, config):
    """Setup argument parser for the kit_configure tool."""

    parser.description = "Select Kit SDK version and generate files from templates."

    # Version selection options (mutually exclusive)
    version_group = parser.add_mutually_exclusive_group()
    version_group.add_argument(
        "--version",
        type=str,
        help="Specify Kit version label to select and generate files (e.g., '108.0.0', '109.0.1')",
    )
    version_group.add_argument(
        "--default",
        action="store_true",
        dest="use_default",
        help="Use the default version and generate files",
    )
    version_group.add_argument(
        "--auto",
        action="store_true",
        help="Use tracked (or default if not tracked) version and generate files (automation-friendly)",
    )

    # Information display
    parser.add_argument(
        "--list-versions",
        action="store_true",
        dest="list_versions",
        help="Display available Kit versions and current selection status",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output in JSON format (use with --list-versions)",
    )

    parser.add_argument(
        "--show-config",
        action="store_true",
        dest="show_config",
        help="Display current tool configuration and file paths",
    )

    # Dry-run mode
    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Standalone: validate files and show what needs fixing. With --auto/--version/--default: preview changes without modifying files",
    )

    # Symlink vs copy mode (sticky — persisted in .kit_selection.json)
    parser.add_argument(
        "--use-symlinks",
        action=argparse.BooleanOptionalAction,
        dest="use_symlinks",
        default=None,
        help=(
            "Use symlinks for plain files (default on first run). "
            "--no-use-symlinks copies files instead (required for packaging). "
            "This setting is sticky — persisted in .kit_selection.json and reused on subsequent invocations."
        ),
    )

    # Output control
    parser.add_argument(
        "--quiet",
        action="store_true",
        dest="quiet",
        help="Suppress non-essential output (useful for automation/CI). Only shows errors and JSON output.",
    )

    tool_config = config.get("repo_select_kit_version", {})

    if tool_config.get("enabled", False):
        return run_repo_tool
