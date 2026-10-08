# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Immutable public Update results."""

from dataclasses import dataclass
from typing import Literal

OperatorStatus = Literal["succeeded", "unchanged", "skipped", "failed", "removed"]
DiagnosticSeverity = Literal["info", "warning", "error"]


@dataclass(frozen=True)
class OperatorResult:
    operator_path: str
    status: OperatorStatus
    reason: str | None = None
    product_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class Diagnostic:
    severity: DiagnosticSeverity
    code: str
    message: str
    operator_path: str | None = None


@dataclass(frozen=True)
class UpdateReport:
    input_ordinal: int
    output_ordinal: int
    operators: tuple[OperatorResult, ...]
    diagnostics: tuple[Diagnostic, ...]
    committable: bool
