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
import subprocess
import sys
from importlib.metadata import EntryPoint, entry_points

import pytest

import ovcae
import ovcae.materials
from ovcae import _setup


def test_setup_runs_every_registration_in_order(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(_setup, "load_readers", lambda *, strict: calls.append("readers"))
    monkeypatch.setattr(_setup, "register_schema_paths", lambda: calls.append("schemas"))
    monkeypatch.setattr(_setup, "register_material_library", lambda: calls.append("materials"))
    ovcae.setup()
    assert calls == ["readers", "schemas", "materials"]


def test_setup_is_idempotent_against_real_runtimes(monkeypatch) -> None:
    pytest.importorskip("ovstage")
    monkeypatch.delenv(ovcae.materials.MDL_SEARCH_PATH_VARIABLE, raising=False)
    ovcae.setup()
    ovcae.setup()
    library = str(ovcae.materials.material_library_path())
    assert os.environ[ovcae.materials.MDL_SEARCH_PATH_VARIABLE].split(os.pathsep).count(library) == 1


def test_first_party_reader_is_declared_as_an_entry_point() -> None:
    names = {entry.value for entry in entry_points(group=_setup.READER_ENTRY_POINT_GROUP)}
    assert "ovcae.readers" in names


def test_load_readers_skips_missing_dependencies_unless_strict(monkeypatch) -> None:
    fake = [
        EntryPoint(name="missing", value="ovcae_tests_no_such_reader", group=_setup.READER_ENTRY_POINT_GROUP),
        EntryPoint(name="core", value="ovcae.data", group=_setup.READER_ENTRY_POINT_GROUP),
    ]
    monkeypatch.setattr(_setup, "entry_points", lambda *, group: fake)
    with pytest.warns(RuntimeWarning, match="'missing' is not available"):
        assert ovcae.load_readers() == ("core",)
    with pytest.raises(ImportError):
        ovcae.load_readers(strict=True)


def test_load_readers_propagates_process_state_errors(monkeypatch) -> None:
    class Broken:
        name = "broken"

        def load(self):
            raise RuntimeError("stage created before the reader registered")

    monkeypatch.setattr(_setup, "entry_points", lambda *, group: [Broken()])
    with pytest.raises(RuntimeError, match="stage created before"):
        ovcae.load_readers()


def test_setup_makes_import_order_irrelevant() -> None:
    pytest.importorskip("ovcae.readers")
    code = (
        "import ovcae\n"
        "ovcae.setup()\n"
        "from pxr import Sdf, Usd\n"
        "stage = Usd.Stage.CreateInMemory()\n"
        "assert Sdf.FileFormat.FindByExtension('cgns'), 'CAE file formats missing after setup()'\n"
        "import ovcae.authoring as author\n"
        "import ovcae.readers\n"
        "holder = ovcae.readers.import_dataset(stage, '/Data', './sim.cgns', zoneName='Zone1')\n"
        "faces = author.create_faces(stage, '/Faces', source=holder)\n"
        "assert faces.HasAPI('CaeVizFacesAPI')\n"
    )
    env = {key: value for key, value in os.environ.items() if key != "PXR_PLUGINPATH_NAME"}
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
