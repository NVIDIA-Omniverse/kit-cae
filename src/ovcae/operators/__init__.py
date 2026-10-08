# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Public class-based operator extension Interface."""

from ._changes import Changes
from ._context import OperatorContext
from ._errors import InvalidRecipe
from ._registry import (
    AmbiguousOperatorError,
    OperatorDeclaration,
    OperatorRegistrationError,
    OperatorRegistry,
    operator,
)

__all__ = [
    "AmbiguousOperatorError",
    "Changes",
    "InvalidRecipe",
    "OperatorContext",
    "OperatorDeclaration",
    "OperatorRegistrationError",
    "OperatorRegistry",
    "operator",
]
