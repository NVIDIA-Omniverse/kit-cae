# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Small, schema-backed descriptions of editable operator properties.

Descriptions and UI snapshots are not scene authority: values are read from the
committed OVStage, and only pending user edits are held outside it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Property:
    name: str
    label: str
    type_name: str
    default: object
    choices: tuple[str, ...] = ()
    minimum: float | None = None

    def validate(self, value):
        """Validate before any Stage writes; do not silently clamp user input."""
        if self.type_name == "bool":
            if not isinstance(value, bool):
                raise ValueError(f"{self.label} must be a boolean")
        elif self.type_name in {"float", "int"}:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{self.label} must be finite")
            if self.type_name == "int" and (int(value) != value or not -(2**31) <= value < 2**31):
                raise ValueError(f"{self.label} must be a 32-bit integer")
            if self.type_name == "float" and abs(value) > 3.4028235e38:
                raise ValueError(f"{self.label} exceeds float32 range")
            if self.minimum is not None and value < self.minimum:
                raise ValueError(f"{self.label} must be at least {self.minimum:g}")
        elif value not in self.choices:
            raise ValueError(f"invalid {self.label}: {value}")
        return value


@dataclass(frozen=True)
class Operator:
    """One discovered prim, including unsupported implementations."""

    path: str
    schemas: frozenset[str]
    properties: tuple[Property, ...]

    @property
    def kind(self):
        return " / ".join(label for schema, label in KINDS.items() if schema in self.schemas) or "Operator"


KINDS = {
    "CaeVizFacesAPI": "Surface",
    "CaeVizIsoSurfaceAPI": "Iso surface",
    "CaeVizPointsAPI": "Points",
    "CaeVizGlyphsAPI": "Glyphs",
    "CaeVizStreamlinesAPI": "Streamlines",
    "CaeVizPlanarSliceAPI": "Planar slice",
}
# Deliberately focused controls. Dataset/field selection and transforms remain
# authored in the file. Defaults/types come from the installed schema package.
CONTROLS = {
    "cae:viz:operator:enabled": ("Enabled", (), None),
    "cae:viz:faces:externalOnly": ("External faces only", (), None),
    "cae:viz:isoSurface:isoValue": ("Iso value", (), None),
    "cae:viz:glyphs:maxCount": ("Maximum glyphs (0 = all)", (), 0),
    "cae:viz:glyphs:scale": ("Default glyph scale", (), 0),
    "cae:viz:glyphs:useCellPoints": ("Cell-connected points", (), None),
    "cae:viz:glyphs:orientationsMode": ("Orientation mode", ("eulerAngles", "quaternion"), None),
    "cae:viz:points:maxCount": ("Maximum points (0 = all)", (), 0),
    "cae:viz:points:width": ("Point width", (), 0),
    "cae:viz:points:useCellPoints": ("Cell-connected points", (), None),
    "cae:viz:streamlines:direction": ("Direction", ("forward", "backward", "both"), None),
    "cae:viz:streamlines:maxSteps": ("Maximum steps", (), 1),
    "cae:viz:streamlines:initialStepSize": ("Initial step size", (), 1e-12),
    "cae:viz:streamlines:minStepSize": ("Minimum step size", (), 1e-12),
    "cae:viz:streamlines:maxStepSize": ("Maximum step size", (), 1e-12),
    "cae:viz:streamlines:tolerance": ("Tolerance", (), 1e-12),
    "cae:viz:streamlines:width": ("Curve width", (), 0),
    "cae:viz:planarSlice:mode": ("Planes", ("free", "x", "y", "z", "xy", "xz", "yz", "xyz"), None),
}
VISIBILITY = Property("visibility", "Visible", "token", "inherited", ("inherited", "invisible"))


def describe_operator(path, schemas):
    """Select supported controls for every applied operator schema, at any path."""
    # Importing the typed module registers the bundle with usd-core when needed.
    from ovcae.schemas import typed

    properties = [VISIBILITY]
    for schema in ("CaeVizOperatorAPI", *KINDS):
        if schema not in schemas:
            continue
        for descriptor in getattr(typed, schema.removeprefix("CaeViz")).properties().values():
            name = descriptor.resolve()
            if descriptor.kind == "attribute" and name in CONTROLS:
                label, choices, minimum = CONTROLS[name]
                properties.append(Property(name, label, descriptor.type_name, descriptor.default, choices, minimum))
    return Operator(path, frozenset(schemas), tuple(properties))
