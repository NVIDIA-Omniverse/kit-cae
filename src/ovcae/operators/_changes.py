# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Read-only, execution-scoped comparisons with committed evaluation evidence."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ._errors import InvalidRecipe

if TYPE_CHECKING:
    from .._reuse import Evaluation
    from ..scene import StageView


@dataclass(frozen=True)
class Changes:
    """Compare current inputs with this binding's last verified committed result.

    Session supplies the same private evaluation used by its execution gate. Queries
    only inspect metadata: they never load arrays, publish results, or advance history.
    Missing baselines or unknown input identities count as changed. Native-history
    expiry follows Session's assume-unchanged policy. This view expires with the execution-scoped
    scene when execute returns, including when a caller saves the view separately.
    Obtain it through OperatorContext.changes; Session owns its construction.
    """

    _stage: StageView
    _evaluation: Evaluation

    def recipe_changed(self) -> bool:
        """Report native writes to declared recipe and explicit scene dependencies.

        Session checks OVStage change ranges before publication, relative to the
        rolling recipe checkpoint. A retained identical-value rewrite counts as
        changed. Expired intervals are assumed unchanged; previously observed
        changes remain invalidating. Changed dependencies, unknown schemas, native
        read failures, or no valid baseline return True. This query reads no Stage.
        """
        self._stage._require()
        return self._evaluation.recipe_changed()

    def input_changed(
        self,
        role: str,
        *,
        components: Collection[str] = (),
        fields: Collection[str] = (),
    ) -> bool:
        """Return whether any selected effective component or field differs.

        Components selects topology and/or geometry; fields selects output field roles,
        not backing array names. Compare complete request and source identities,
        including native replacement generations, rather than application time alone.
        At least one available item is required. Invalid roles, selections, and bare
        strings in place of name collections raise InvalidRecipe.
        """
        self._stage._require()
        if not isinstance(role, str) or not role:
            raise InvalidRecipe("input role must be a nonempty string")
        if isinstance(components, str) or isinstance(fields, str):
            raise InvalidRecipe("components and fields must be collections of names, not strings")
        try:
            component_names = set(components)
            field_names = set(fields)
        except TypeError as exc:
            raise InvalidRecipe("components and fields must be collections of names") from exc
        if not component_names <= {"topology", "geometry"}:
            raise InvalidRecipe("components must contain topology and/or geometry")
        if any(not isinstance(name, str) or not name for name in field_names):
            raise InvalidRecipe("fields must contain nonempty output field-role names")
        selected = component_names | {f"field:{name}" for name in field_names}
        try:
            return self._evaluation.input_changed(role, selected)
        except (KeyError, ValueError) as exc:
            raise InvalidRecipe(str(exc)) from exc
