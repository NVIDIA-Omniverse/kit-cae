# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Runtime-neutral access to the CAE material library.

CAE scenes bind MDL materials and colormap textures through search-path-relative
asset identifiers such as ``cae/mdl/basic.mdl`` and
``cae/colormaps/gist_rainbow.png`` rather than concrete file paths. The
identifiers stay stable across installations; a renderer resolves them at
runtime against a material library that ships with this distribution. Updating
the library therefore updates every scene without editing the scenes.

Importing this package has no side effects. Applications opt in explicitly with
:func:`register_material_library` before creating their renderer.
"""

from __future__ import annotations

import os
from collections.abc import MutableMapping
from pathlib import Path

MDL_SEARCH_PATH_VARIABLE = "MDL_USER_PATH"
"""Environment variable the MDL SDK (and therefore OVRTX) reads for user search paths."""

MDL_MODULE = "cae/mdl/basic.mdl"
"""Search-path-relative identifier of the bundled MDL module (``::cae::mdl::basic``)."""


def material_library_path() -> Path:
    """Return the directory that MDL search paths must include to resolve CAE materials."""

    path = Path(__file__).parent / "resources"
    required = (
        path / MDL_MODULE,
        path / "cae" / "colormaps" / "gist_rainbow.png",
        path / "cae" / "colormaps" / "gist_gray.png",
    )
    missing = [str(item.relative_to(path)) for item in required if not item.is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete ovcae.materials resource bundle at {path}: missing {', '.join(missing)}")
    return path


def colormap_paths() -> tuple[Path, ...]:
    """Return the bundled colormap lookup textures, sorted by file name."""

    return tuple(sorted((material_library_path() / "cae" / "colormaps").glob("*.png")))


def register_material_library(environ: MutableMapping[str, str] | None = None) -> Path:
    """Prepend the material library to the MDL user search path and return it.

    The MDL SDK reads ``MDL_USER_PATH`` once, when the renderer initializes, so
    call this before constructing ``ovrtx.Renderer()``. Existing entries are
    preserved and the library is not added twice. ``environ`` defaults to
    ``os.environ``; pass another mapping to stage a child-process environment.
    """

    if environ is None:
        environ = os.environ
    library = str(material_library_path())
    existing = [entry for entry in environ.get(MDL_SEARCH_PATH_VARIABLE, "").split(os.pathsep) if entry]
    if library not in existing:
        environ[MDL_SEARCH_PATH_VARIABLE] = os.pathsep.join([library, *existing])
    return Path(library)


__all__ = [
    "MDL_MODULE",
    "MDL_SEARCH_PATH_VARIABLE",
    "colormap_paths",
    "material_library_path",
    "register_material_library",
]
