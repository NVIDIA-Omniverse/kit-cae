# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""MDL material setup mirroring Kit-CAE's create commands.

Every recipe builder gives its prim the same material scaffolding Kit-CAE's
``CreateCaeViz*`` commands author: a ``Materials`` scope holding MDL materials
from the bundled ``cae/mdl/basic.mdl`` module, colormap lookup textures named by
search-path-relative identifiers (see ``ovcae.materials``), the
``CaeVizRescaleRangeAPI`` wiring that lets a runtime push field ranges into the
shader inputs, an independently selectable ``opacity`` role, and a
``weakerThanDescendants`` material binding.

The ovcae runtime does not yet process ``CaeVizRescaleRangeAPI``; until it does,
``color_range`` lets callers author a known domain so materials color immediately.
"""

from __future__ import annotations

from collections.abc import Iterable

from . import schemas as S
from ._common import as_path
from ._errors import AuthoringError
from ._usd import require_usd

MDL_MODULE = "cae/mdl/basic.mdl"
DEFAULT_COLORMAP = "gist_rainbow"
OPACITY_COLORMAP = "gist_gray"
UNSET_RANGE = (0.0, -1.0)
"""The ``domain`` value Kit-CAE authors so a runtime rescale replaces it; ``x >= y`` disables coloring."""

Range = tuple[float, float]


def colormap_asset(colormap: str) -> str:
    """Return the search-path-relative asset identifier for a colormap.

    ``colormap`` is either a bundled colormap name such as ``"cividis"`` or an
    explicit asset path (anything containing ``/`` or ending in ``.png``), which is
    recorded as given.
    """

    if not isinstance(colormap, str) or not colormap:
        raise AuthoringError(f"colormap must be a non-empty string; got {colormap!r}")
    if "/" in colormap or colormap.endswith(".png"):
        return colormap
    from ovcae.materials import colormap_paths

    names = sorted(path.stem for path in colormap_paths())
    if colormap not in names:
        raise AuthoringError(f"unknown bundled colormap {colormap!r}; expected one of {', '.join(names)}")
    return f"cae/colormaps/{colormap}.png"


def create_mdl_material(
    stage,
    path,
    sub_identifier: str,
    *,
    colormap: str = DEFAULT_COLORMAP,
    color_range: Range | None = None,
    enable_coloring: bool | None = None,
):
    """Define a ``Material`` at ``path`` whose ``Shader`` child is an export of the CAE MDL module.

    Authors the same inputs Kit-CAE writes: ``lut``, ``domain``, ``opacity_lut``,
    ``opacity_domain``, ``opacity``, and ``enable_opacity``. ``enable_coloring`` is
    authored only when given (Kit-CAE authors ``False`` solely on materials whose
    flag a rescale toggles; the MDL default is ``True``) or when ``color_range`` is
    supplied, which forces it on. Returns the material prim.
    """

    pxr = require_usd()
    path = as_path(path)
    material = pxr.UsdShade.Material.Define(stage, path)
    if not material:
        raise AuthoringError(f"could not define Material at {path}")
    shader = pxr.UsdShade.Shader.Define(stage, path.AppendChild("Shader"))
    shader.SetSourceAsset(pxr.Sdf.AssetPath(MDL_MODULE), "mdl")
    shader.SetSourceAssetSubIdentifier(str(sub_identifier), "mdl")
    out = shader.CreateOutput("out", pxr.Sdf.ValueTypeNames.Token)
    out.SetRenderType("material")
    for output in (
        material.CreateSurfaceOutput("mdl"),
        material.CreateDisplacementOutput("mdl"),
        material.CreateVolumeOutput("mdl"),
    ):
        output.ConnectToSource(out)

    domain = UNSET_RANGE if color_range is None else _range(color_range, "color_range")
    types = pxr.Sdf.ValueTypeNames
    shader.CreateInput("lut", types.Asset).Set(pxr.Sdf.AssetPath(colormap_asset(colormap)))
    shader.CreateInput("domain", types.Float2).Set(pxr.Gf.Vec2f(*domain))
    if color_range is not None:
        enable_coloring = True
    if enable_coloring is not None:
        shader.CreateInput("enable_coloring", types.Bool).Set(bool(enable_coloring))
    shader.CreateInput("opacity_lut", types.Asset).Set(pxr.Sdf.AssetPath(colormap_asset(OPACITY_COLORMAP)))
    shader.CreateInput("opacity_domain", types.Float2).Set(pxr.Gf.Vec2f(*UNSET_RANGE))
    shader.CreateInput("opacity", types.Float).Set(1.0)
    shader.CreateInput("enable_opacity", types.Bool).Set(False)
    material.GetPrim().CreateAttribute("omni:rtx:enableCutoutOpacity", types.Bool, custom=True).Set(True)
    return material.GetPrim()


def shader_input_path(material, name: str):
    """Return the property path of ``inputs:<name>`` on a material's ``Shader`` child."""

    return material.GetPath().AppendChild("Shader").AppendProperty(f"inputs:{name}")


def wire_rescale_range(prim, role: str, *, includes: Iterable = (), enable_includes: Iterable = ()):
    """Apply ``CaeVizRescaleRangeAPI:<role>`` targeting the given property paths.

    ``includes`` are ``float2`` properties a runtime resets to the field's range;
    ``enable_includes`` are ``bool`` properties it toggles on when the field is present.
    Targets may be ``Usd.Property`` objects, ``Sdf.Path`` values, or strings such as
    ``/Prim/Shader.inputs:domain``; prim paths are rejected. Returns the wrapper.
    """

    rescale = S.RescaleRangeAPI.apply(prim, role)
    includes, enable_includes = list(includes), list(enable_includes)
    if includes:
        rescale.includes = includes
    if enable_includes:
        rescale.enable_includes = enable_includes
    return rescale


def bind_material(prim, material) -> None:
    """Bind ``material`` to ``prim`` with ``weakerThanDescendants`` strength, as Kit-CAE does."""

    pxr = require_usd()
    binding = pxr.UsdShade.MaterialBindingAPI.Apply(prim)
    if not binding.Bind(pxr.UsdShade.Material(material), pxr.UsdShade.Tokens.weakerThanDescendants):
        raise AuthoringError(f"could not bind {material.GetPath()} to {prim.GetPath()}")


def setup_scalar_materials(
    prim,
    *,
    sub_identifiers: tuple[str, ...] = ("ScalarColor",),
    bound: str = "ScalarColor",
    colormap: str = DEFAULT_COLORMAP,
    color_range: Range | None = None,
    enable_includes_for: tuple[str, ...] | None = None,
) -> dict[str, object]:
    """Author Kit-CAE's per-recipe materials under ``<prim>/Materials`` and wire them.

    Creates one material per entry of ``sub_identifiers`` (exports of the CAE MDL
    module), targets every material's ``domain`` from ``CaeVizRescaleRangeAPI:colors``,
    targets ``enable_coloring`` for the materials named in ``enable_includes_for``
    (all of them by default), wires the ``opacity`` role into every material, and
    binds the material named ``bound``. Returns the materials keyed by sub-identifier.
    """

    stage = prim.GetStage()
    scope = prim.GetPath().AppendChild("Materials")
    stage.DefinePrim(scope, "Scope")
    enable_for = set(sub_identifiers if enable_includes_for is None else enable_includes_for)
    materials = {
        name: create_mdl_material(
            stage,
            scope.AppendChild(name),
            name,
            colormap=colormap,
            color_range=color_range,
            enable_coloring=False if name in enable_for else None,
        )
        for name in sub_identifiers
    }
    wire_rescale_range(
        prim,
        "colors",
        includes=[shader_input_path(material, "domain") for material in materials.values()],
        enable_includes=[
            shader_input_path(materials[name], "enable_coloring") for name in sub_identifiers if name in enable_for
        ],
    )
    S.FieldSelectionAPI.apply(prim, "opacity")
    wire_rescale_range(
        prim,
        "opacity",
        includes=[shader_input_path(material, "opacity_domain") for material in materials.values()],
        enable_includes=[shader_input_path(material, "enable_opacity") for material in materials.values()],
    )
    if bound not in materials:
        raise AuthoringError(f"bound material {bound!r} is not one of {sub_identifiers}")
    bind_material(prim, materials[bound])
    return materials


def setup_width_mapping(prim, *, width: float | None = None, mapped_range: Range | None = None) -> None:
    """Apply Kit-CAE's ``widths`` role scaffolding: selection, field mapping, and rescale wiring."""

    S.FieldSelectionAPI.apply(prim, "widths")
    mapping = S.FieldMappingAPI.apply(prim, "widths")
    mapping.domain = UNSET_RANGE
    if mapped_range is not None:
        mapping.range = _range(mapped_range, "mapped_range")
    elif width is not None:
        mapping.range = (float(width), float(width) * 2.0)
    wire_rescale_range(prim, "widths", includes=[mapping.attribute("domain")])


def _range(value, what: str) -> Range:
    try:
        low, high = (float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise AuthoringError(f"{what} must be a (min, max) pair of numbers; got {value!r}") from exc
    if not low < high:
        raise AuthoringError(f"{what} must satisfy min < max; got {value!r}")
    return (low, high)
