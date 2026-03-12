#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
Format Tutorial Setup Script

Copies all Scae tutorial reference files from docs/format_tutorial_reference/
into their correct locations in the Kit-CAE repository tree.

Usage (from repo root):
    python docs/format_tutorial_reference/setup_tutorial.py

This script uses only the Python standard library.  Safe to run multiple times.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path


def _find_repo_root() -> Path:
    """Walk up from this script to find the repo root (contains repo.toml)."""
    candidate = Path(__file__).resolve().parent
    while candidate != candidate.parent:
        if (candidate / "repo.toml").exists():
            return candidate
        candidate = candidate.parent
    print("ERROR: Could not find repo root (no repo.toml found).", file=sys.stderr)
    sys.exit(1)


def _copy_file(src: Path, dst: Path, *, label: str = "") -> None:
    """Copy a single file, creating parent directories as needed."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    display = label or str(dst)
    if dst.exists():
        if src.read_bytes() == dst.read_bytes():
            print(f"  [skip]    {display}  (already up to date)")
            return
        print(f"  [update]  {display}")
    else:
        print(f"  [create]  {display}")
    shutil.copy2(src, dst)


def _copy_tree(src_dir: Path, dst_dir: Path, *, label: str = "") -> None:
    """Recursively copy a directory, skipping __pycache__."""
    for src_file in sorted(src_dir.rglob("*")):
        if "__pycache__" in src_file.parts:
            continue
        if not src_file.is_file():
            continue
        relative = src_file.relative_to(src_dir)
        dst_file = dst_dir / relative
        _copy_file(src_file, dst_file, label=str(Path(label) / relative) if label else "")


def main() -> None:
    repo_root = _find_repo_root()
    ref_dir = repo_root / "docs" / "format_tutorial_reference"

    if not ref_dir.exists():
        print(f"ERROR: Reference directory not found: {ref_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Repo root:  {repo_root}")
    print(f"Reference:  {ref_dir}")
    print()

    # 1. repo_schemas.toml
    print("1/6  Updating repo_schemas.toml ...")
    _copy_file(
        ref_dir / "repo_schemas.toml",
        repo_root / "repo_schemas.toml",
        label="repo_schemas.toml",
    )
    print()

    # 2. source/schemas/premake5.lua
    print("2/6  Updating source/schemas/premake5.lua ...")
    _copy_file(
        ref_dir / "premake5.lua",
        repo_root / "source" / "schemas" / "premake5.lua",
        label="source/schemas/premake5.lua",
    )
    print()

    # 3. New schema: omniCaeScae
    print("3/6  Installing new schema: omniCaeScae ...")
    _copy_tree(
        ref_dir / "new_schemas" / "formats" / "omniCaeScae",
        repo_root / "source" / "schemas" / "formats" / "omniCaeScae",
        label="source/schemas/formats/omniCaeScae",
    )
    print()

    # 4. Updated / new extensions
    print("4/6  Installing extensions ...")
    ext_ref = ref_dir / "updated_extensions"
    ext_dst = repo_root / "source" / "extensions"
    for ext_dir in sorted(ext_ref.iterdir()):
        if not ext_dir.is_dir():
            continue
        ext_name = ext_dir.name
        print(f"     -> {ext_name}")
        _copy_tree(ext_dir, ext_dst / ext_name, label=f"source/extensions/{ext_name}")
    print()

    # 5. Kit launch script
    print("5/6  Installing scripts/generate_scae_data.py ...")
    _copy_file(
        ref_dir / "generate_scae_data.py",
        repo_root / "scripts" / "generate_scae_data.py",
        label="scripts/generate_scae_data.py",
    )
    print()

    # 6. Summary
    print("6/6  Done!")
    print()
    print("Next steps:")
    print("  1. Build (clean rebuild to generate new schemas):")
    print("       ./repo.sh build -xr")
    print()
    print("  2. Launch with the smoke-test script:")
    print("       ./repo.sh launch -n omni.cae.kit -- --exec scripts/generate_scae_data.py")
    print()
    print("  The script generates sample data into data/ and imports it into the stage.")


if __name__ == "__main__":
    main()
