# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Update-scoped operator execution context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ._changes import Changes
from ._errors import InvalidRecipe

if TYPE_CHECKING:
    from .._data import FrameData
    from ..scene import PrimView, StageView


@dataclass(frozen=True)
class OperatorContext:
    """Execution-scoped view of one bound recipe and its scientific inputs.

    Stage and prim are read-only scene handles valid during execute; input_ordinal records
    the minimum requested publication, while output_ordinal is reserved for Session
    writes. Device is the resolved execution device. Private fields connect helpers
    to the Frame Data lease, shared input plan, and read-only change comparisons.
    Legacy unannotated operators have no input plan. Operators
    must not retain this context or its scene handles after execute returns.
    """

    stage: StageView
    prim: PrimView
    input_ordinal: int
    output_ordinal: int
    device: str
    _frame_data: FrameData
    _inputs: object = None
    _changes: Changes | None = None

    @property
    def changes(self) -> Changes:
        """Expose execution-scoped comparisons with the gate's committed baseline.

        Annotated operators receive a read-only Changes view. Unannotated operators
        have no declared comparison evidence and raise InvalidRecipe. Access after
        execution raises the same expired-view error as scene access.
        """
        self.stage._require()
        if self._changes is None:
            raise InvalidRecipe("change queries require an operator with dataset_components")
        return self._changes
