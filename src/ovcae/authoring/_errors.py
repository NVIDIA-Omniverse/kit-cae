# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Authoring error types."""

from __future__ import annotations

from ovcae.schemas import SchemaError


class AuthoringError(SchemaError):
    """Raised for arguments or typed schema access that cannot produce a valid recipe.

    Typed schema classes raise it on ``Usd.Prim`` (wrong prim type, disallowed
    token, unapplied schema) because the Usd backend selects it; the
    ``ovcae.schemas.SchemaError`` base covers errors raised before a backend is
    involved, such as an unknown schema name.
    """
