# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Author CAE visualization recipes into OpenUSD stages with usd-core.

``ovcae.authoring`` constructs the scene prims the runtime later discovers and
evaluates. It depends on ``ovcae.schemas`` and ``ovcae.materials`` only and
never imports the runtime modules, ovstage, or Warp, so it works in a process
without them. It needs the usd-core bindings ovcae depends on; importing
without them raises ImportError.

Importing registers the codeless OmniCaeViz schema plug-in with ``pxr.Plug`` so
API schemas apply with their fallback values. This package knows nothing about
any one dataset representation; the helpers that author native-file payload
holders live with their reader in ``ovcae.readers``.

Builders author through the typed schema classes in :mod:`ovcae.authoring.schemas`
(``FacesAPI.apply(prim).external_only = False``), return the created prim, and
raise :class:`AuthoringError` on invalid input. The same classes let callers
read or adjust authored recipes without spelling ``cae:viz:*`` property names,
and :func:`validate_recipe` reports what would stop ovcae from executing one.
Recipe builders also author Kit-CAE's MDL material scaffolding (materials under
``<prim>/Materials``, rescale-range wiring, opacity role, and binding) unless
``materials=False``.
Every builder accepts an ``edit_target`` (``Usd.EditTarget`` or ``Sdf.Layer``)
to author into a caller-chosen layer.
"""

from __future__ import annotations

from ._errors import AuthoringError
from ._usd import register_schemas, require_usd

require_usd()
register_schemas()

from ovcae.schemas import SchemaError  # noqa: E402

from . import schemas  # noqa: E402
from ._controls import Bounds, create_bounding_box, create_seed_points, create_seed_sphere, sphere_points  # noqa: E402
from ._glyphs import create_glyphs  # noqa: E402
from ._materials import (  # noqa: E402
    bind_material,
    colormap_asset,
    create_mdl_material,
    setup_scalar_materials,
    setup_width_mapping,
    wire_rescale_range,
)
from ._recipes import (  # noqa: E402
    create_faces,
    create_iso_surface,
    create_planar_slice,
    create_points,
    create_streamlines,
)
from ._validate import validate_recipe  # noqa: E402

__all__ = [
    "AuthoringError",
    "Bounds",
    "SchemaError",
    "bind_material",
    "colormap_asset",
    "create_bounding_box",
    "create_faces",
    "create_glyphs",
    "create_iso_surface",
    "create_mdl_material",
    "create_planar_slice",
    "create_points",
    "create_seed_points",
    "create_seed_sphere",
    "create_streamlines",
    "register_schemas",
    "schemas",
    "setup_scalar_materials",
    "setup_width_mapping",
    "sphere_points",
    "validate_recipe",
    "wire_rescale_range",
]
