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
Extends omni.repo.kit_tools.bump to support bumping the package version.
"""
import os

import omni.repo.kit_tools.bump
import omni.repo.man as repo_man


def bump_version(version, component) -> str:
    import semver

    ver = semver.VersionInfo.parse(version)
    if component == "prerelease":
        new_ver = ver.bump_prerelease()
    elif component == "patch":
        new_ver = ver.bump_patch()
    elif component == "minor":
        new_ver = ver.bump_minor()
    elif component == "major":
        new_ver = ver.bump_major()

    return str(new_ver)


def bump(options, config):
    from InquirerPy import inquirer
    from InquirerPy.base.control import Choice

    tool_config = config.get("repo_bump", {})
    pkg_version_md = repo_man.resolve_tokens(tool_config.get("pkg_version_md", "${root}/VERSION.md"))
    if not os.path.exists(pkg_version_md):
        print(f"[error] {pkg_version_md} not found!")
        return False

    with open(pkg_version_md, "rt") as f:
        version = f.readline().strip()

    proceed = inquirer.confirm(message=f"Proceed to bump package version from {version}?", default=False).execute()

    if not proceed:
        return

    component = inquirer.select(
        message="Which package version component (X) to bump?",
        choices=[
            Choice(value="prerelease", name="Prerelease (1.0.0-X)"),
            Choice(value="patch", name="Patch (1.0.0-X)"),
            Choice(value="minor", name="Minor (1.0.0-X)"),
            Choice(value="major", name="Major (1.0.0-X)"),
        ],
        default=None,
    ).execute()

    ver = bump_version(version, component)
    with open(pkg_version_md, "wt") as f:
        f.write(ver)
        print(f"Bumped {pkg_version_md} {version} -> {ver}")


def setup_repo_tool(parser, config):
    og_bump = omni.repo.kit_tools.bump.setup_repo_tool(parser, config)

    def run_repo_tool(options, config):
        og_bump(options, config)
        bump(options, config)

    return run_repo_tool
