# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

import os

import ovcae.materials


def test_material_library_bundle_is_complete() -> None:
    root = ovcae.materials.material_library_path()
    module = root / ovcae.materials.MDL_MODULE
    text = module.read_text()
    # Kit-CAE scenes bind these exports by sub-identifier; the names are the contract.
    for name in ("ScalarColor", "UnlitScalarColor", "AnimatedStreaks", "AnimatedDot"):
        assert f"export material {name}(" in text
    colormaps = {path.name for path in ovcae.materials.colormap_paths()}
    assert {"gist_rainbow.png", "gist_gray.png"} <= colormaps
    assert all(path.parent == root / "cae" / "colormaps" for path in ovcae.materials.colormap_paths())


def test_register_prepends_once_and_keeps_existing_entries() -> None:
    environ = {ovcae.materials.MDL_SEARCH_PATH_VARIABLE: os.pathsep.join(["/site/mdl", "/user/mdl"])}
    library = ovcae.materials.register_material_library(environ)
    assert library == ovcae.materials.material_library_path()
    assert environ[ovcae.materials.MDL_SEARCH_PATH_VARIABLE].split(os.pathsep) == [
        str(library),
        "/site/mdl",
        "/user/mdl",
    ]

    ovcae.materials.register_material_library(environ)
    assert environ[ovcae.materials.MDL_SEARCH_PATH_VARIABLE].count(str(library)) == 1


def test_register_handles_unset_and_empty_variable() -> None:
    for environ in ({}, {ovcae.materials.MDL_SEARCH_PATH_VARIABLE: ""}):
        library = ovcae.materials.register_material_library(environ)
        assert environ[ovcae.materials.MDL_SEARCH_PATH_VARIABLE] == str(library)


def test_ovcae_adapter_registers_in_process_environment(monkeypatch) -> None:
    import ovcae

    monkeypatch.delenv(ovcae.materials.MDL_SEARCH_PATH_VARIABLE, raising=False)
    library = ovcae.register_material_library()
    assert os.environ[ovcae.materials.MDL_SEARCH_PATH_VARIABLE] == str(library)
    assert ovcae.materials.material_library_path() == library
