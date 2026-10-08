# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Metadata-only timelines for OmniSciDataset prims on a composed usd-core stage."""

from __future__ import annotations

from pxr import Usd

from ovcae import DatasetResolutionError


def dataset_time_samples(root) -> tuple[float, ...]:
    """Union composed attribute sample times over descendants and relationships.

    Visit actual attributes, including dynamically declared array values and scalar
    metadata. Also discover lazy values by applied OmniSciArrayAPI instance names.
    Follow prim and property targets outside the initial subtree; a visited
    set terminates back-references. This deliberately includes every reachable field
    and sibling reached through an owning dataset. No attribute values are read.

    Missing targets and unloaded payloads cannot establish a complete timeline and
    raise DatasetResolutionError instead of being mistaken for static data. Querying
    composed times may open value clips, but does not request scientific values.
    """
    if not root:
        raise DatasetResolutionError("cannot discover time samples for an invalid dataset prim")
    stage = root.GetStage()
    pending = [root]
    visited = set()
    attributes = {}
    while pending:
        prim = pending.pop()
        path = prim.GetPath()
        if path in visited:
            continue
        visited.add(path)
        if prim.HasPayload() and not prim.IsLoaded():
            raise DatasetResolutionError(f"dataset dependency {path} has an unloaded payload")
        prim_attributes = {attr.GetName(): attr for attr in prim.GetAttributes()}
        # Native readers can expose lazy value properties through direct lookup
        # without listing them in GetAttributes(). The applied array instance is
        # the common, format-independent discovery contract for these values.
        for schema in prim.GetAppliedSchemas():
            if schema.startswith("OmniSciArrayAPI:"):
                instance = schema.removeprefix("OmniSciArrayAPI:")
                name = f"omni:sci:array:{instance}:value"
                attr = prim.GetAttribute(name)
                if not attr:
                    raise DatasetResolutionError(f"dataset array {path}.{name} has no value attribute")
                prim_attributes[name] = attr
        for attr in prim_attributes.values():
            if attr.HasSpline():
                raise DatasetResolutionError(f"dataset dependency {attr.GetPath()} uses unsupported spline sampling")
            attributes[attr.GetPath()] = attr
        pending.extend(prim.GetFilteredChildren(Usd.TraverseInstanceProxies()))
        for relationship in prim.GetRelationships():
            # Follow relationship properties explicitly too: this validates missing
            # forwarded targets and terminates forwarding cycles through visited.
            for target in relationship.GetTargets():
                target_prim = stage.GetPrimAtPath(target.GetPrimPath())
                if not target_prim or (target.IsPropertyPath() and not stage.GetPropertyAtPath(target)):
                    raise DatasetResolutionError(
                        f"dataset relationship {relationship.GetPath()} targets missing {target}"
                    )
                pending.append(target_prim)
    return tuple(Usd.Attribute.GetUnionedTimeSamples(list(attributes.values())))
