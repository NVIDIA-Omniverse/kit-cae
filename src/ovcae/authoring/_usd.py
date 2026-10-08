# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Optional usd-core access and codeless schema registration."""

from __future__ import annotations

from ovcae.schemas import register_usd_plugin

_MISSING_USD = (
    "ovcae.authoring requires the usd-core OpenUSD Python bindings (the `pxr` package), "
    "an ovcae dependency; reinstall ovcae or `pip install usd-core`."
)


def require_usd():
    """Import and return the ``pxr`` namespace, raising a clear ImportError when absent."""

    try:
        import pxr
        from pxr import Gf, Plug, Sdf, Usd, UsdGeom, UsdShade, Vt  # noqa: F401
    except ImportError as exc:
        raise ImportError(_MISSING_USD) from exc
    return pxr


def register_schemas() -> None:
    """Register the codeless OmniCaeViz plug-in so API schemas apply with fallbacks.

    Registration is idempotent; Plug ignores an already-registered path.
    """

    require_usd()
    register_usd_plugin()
