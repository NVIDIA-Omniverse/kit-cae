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

import ast
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[1] / "src"
_HEAVY = "{'ovstage', 'numpy', 'warp', 'pxr', 'cae_openusd_plugins'}"


def _run(code: str, **kwargs) -> None:
    subprocess.run([sys.executable, "-c", code], check=True, **kwargs)


def test_import_is_lightweight() -> None:
    _run(f"import sys; import ovcae, ovcae.data; assert not {_HEAVY} & set(sys.modules), sorted(sys.modules)")


@pytest.mark.parametrize("package", ["ovcae.schemas", "ovcae.materials"])
def test_neutral_subpackage_import_stays_standard_library(package: str) -> None:
    _run(f"import sys; import {package}; assert not {_HEAVY} & set(sys.modules), sorted(sys.modules)")


def test_app_and_stream_imports_load_no_renderer() -> None:
    _run(
        "import sys; import ovcae.app, ovcae.stream; "
        "assert not {'ovrtx', 'ovstream', 'ovstream_utils', 'warp', 'ovstage', 'pxr'} & set(sys.modules), "
        "sorted(sys.modules)"
    )


def test_core_import_never_loads_a_reader() -> None:
    _run(
        "import sys; import ovcae, ovcae.data; "
        "assert not {'pxr', 'cae_openusd_plugins', 'ovcae.readers', 'ovcae.authoring'} & set(sys.modules), "
        "sorted(sys.modules)"
    )


def test_authoring_import_never_loads_the_runtime_dependencies() -> None:
    pytest.importorskip("pxr")
    _run(
        "import sys; import ovcae.authoring; "
        "assert not {'ovstage', 'numpy', 'warp', 'ovcae.readers'} & set(sys.modules), sorted(sys.modules)"
    )


def test_readers_import_registers_plugins_without_authoring() -> None:
    pytest.importorskip("ovcae.readers")
    _run(
        "import sys; import ovcae.readers; "
        "assert 'cae_openusd_plugins' in sys.modules and 'ovcae.authoring' not in sys.modules, sorted(sys.modules)"
    )


def test_readers_import_after_first_stage_raises_runtime_error() -> None:
    pytest.importorskip("ovcae.readers")
    code = (
        "import ovcae.authoring\n"
        "from pxr import Usd\n"
        "stage = Usd.Stage.CreateInMemory()\n"
        "try:\n    import ovcae.readers\nexcept RuntimeError as exc:\n"
        "    assert 'before creating any pxr.Usd.Stage' in str(exc), exc\n"
        "else:\n    raise SystemExit('expected RuntimeError')"
    )
    # A parent that already registered the plug-ins exports PXR_PLUGINPATH_NAME,
    # which would make the child's first stage find the formats on its own.
    env = {key: value for key, value in os.environ.items() if key != "PXR_PLUGINPATH_NAME"}
    _run(code, env=env)


def test_authoring_after_readers_and_a_stage_still_authors_recipes() -> None:
    pytest.importorskip("ovcae.readers")
    _run(
        "import ovcae.readers\n"
        "from pxr import Usd\n"
        "stage = Usd.Stage.CreateInMemory()\n"
        "import ovcae.authoring as author\n"
        "holder = ovcae.readers.import_dataset(stage, '/Data', './sim.cgns', zoneName='Zone1')\n"
        "faces = author.create_faces(stage, '/Faces', source=holder, materials=False)\n"
        "assert faces.HasAPI('CaeVizFacesAPI') and 'OmniSciFileFormatArgsCgnsAPI' in holder.GetAppliedSchemas()\n"
    )


def test_readers_import_with_mismatched_plugins_raises_import_error() -> None:
    pytest.importorskip("cae_openusd_plugins")
    _run(
        "import cae_openusd_plugins\n"
        "def broken(**kwargs): raise RuntimeError('pxr 0.0 does not match')\n"
        "cae_openusd_plugins.check_runtime = broken\n"
        "try:\n    import ovcae.readers\nexcept ImportError as exc:\n"
        "    assert 'does not match' in str(exc), exc\n"
        "else:\n    raise SystemExit('expected ImportError')"
    )


def test_readers_import_without_plugins_raises_clear_error() -> None:
    _run(
        "import sys; sys.modules['cae_openusd_plugins'] = None\n"
        "try:\n    import ovcae.readers\nexcept ImportError as exc:\n"
        "    assert 'ovcae[usd]' in str(exc), exc\nelse:\n    raise SystemExit('expected ImportError')"
    )


# Layering inside the one import package. The runtime (every module directly
# under ``ovcae`` plus ``ovcae.scene`` and ``ovcae.operators``) defines the
# contracts a reader builds on and knows no dataset representation. The
# schema and material bundles depend on nothing else in the tree. Authoring
# depends on the bundles only, never on the runtime, so a scene-preparation
# tool runs without OVStage or Warp. Readers depend on the runtime only. A
# subpackage is named by its dotted module prefix; "ovcae" alone means the
# runtime modules that are not in any other listed subpackage. The application
# loop sits on the runtime and the schema bundle and knows no renderer; the
# stream package sits on the loop and is the only layer that may name OVRTX
# and OVStream, and only inside functions.
_SUBPACKAGES = ("ovcae.schemas", "ovcae.materials", "ovcae.authoring", "ovcae.readers", "ovcae.app", "ovcae.stream")
_ALLOWED_INTERNAL_IMPORTS = {
    "ovcae": ("ovcae.schemas", "ovcae.materials"),
    "ovcae.app": ("ovcae", "ovcae.schemas"),
    "ovcae.stream": ("ovcae", "ovcae.app"),
    "ovcae.schemas": (),
    "ovcae.materials": (),
    "ovcae.authoring": ("ovcae.schemas", "ovcae.materials"),
    "ovcae.readers": ("ovcae",),
}
_FORBIDDEN_EXTERNAL_IMPORTS = {
    "ovcae": ("cae_openusd_plugins",),
    "ovcae.app": ("cae_openusd_plugins", "ovrtx", "ovstream"),
    "ovcae.stream": ("cae_openusd_plugins",),
    "ovcae.schemas": ("cae_openusd_plugins",),
    "ovcae.materials": ("cae_openusd_plugins",),
    "ovcae.authoring": ("cae_openusd_plugins",),
    "ovcae.readers": (),
}
# The generic layers must not name a reader's schema vocabulary in code either,
# so a new reader never requires editing them. Prose may point users at
# ovcae.readers; imports and identifiers may not. Generated schema docstrings
# quote schema.usda, which mentions OmniSciDataset, and typed class names strip
# the OmniSci prefix cosmetically; the type name is the dependency.
_FORBIDDEN_TEXT = {
    "ovcae": ("OmniSci",),
    "ovcae.app": ("OmniSci",),
    "ovcae.stream": ("OmniSci",),
    "ovcae.schemas": (),
    "ovcae.materials": ("OmniSci",),
    "ovcae.authoring": ("OmniSciDataset",),
    "ovcae.readers": (),
}


def _module_name(path: Path) -> str:
    parts = path.relative_to(_SRC).with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _layer_of(module: str) -> str:
    for prefix in _SUBPACKAGES:
        if module == prefix or module.startswith(prefix + "."):
            return prefix
    return "ovcae"


def _imports(path: Path) -> Iterator[str]:
    """Yield the absolute module name of every import statement, resolving relative ones."""

    module = _module_name(path)
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                yield node.module or ""
                continue
            base = package.split(".")[: len(package.split(".")) - node.level + 1]
            prefix = ".".join(base)
            yield f"{prefix}.{node.module}" if node.module else prefix
            # ``from . import x`` may name a submodule; count it as one.
            if not node.module:
                yield from (f"{prefix}.{alias.name}" for alias in node.names)


def _sources(layer: str) -> list[Path]:
    return [path for path in sorted((_SRC / "ovcae").rglob("*.py")) if _layer_of(_module_name(path)) == layer]


@pytest.mark.parametrize("layer", sorted(_ALLOWED_INTERNAL_IMPORTS))
def test_layer_imports_only_what_it_may(layer: str) -> None:
    offenders = []
    for path in _sources(layer):
        relative = path.relative_to(_SRC)
        for name in _imports(path):
            top = name.split(".")[0]
            if top == "ovcae":
                target = _layer_of(name)
                if target != layer and target not in _ALLOWED_INTERNAL_IMPORTS[layer]:
                    offenders.append(f"{relative}: imports {name}")
            elif top in _FORBIDDEN_EXTERNAL_IMPORTS[layer]:
                offenders.append(f"{relative}: imports {name}")
        text = path.read_text()
        offenders.extend(f"{relative}: {marker}" for marker in _FORBIDDEN_TEXT[layer] if marker in text)
    assert not offenders, "\n".join(offenders)


def test_every_source_file_belongs_to_a_known_layer() -> None:
    assert {_layer_of(_module_name(p)) for p in (_SRC / "ovcae").rglob("*.py")} == set(_ALLOWED_INTERNAL_IMPORTS)
