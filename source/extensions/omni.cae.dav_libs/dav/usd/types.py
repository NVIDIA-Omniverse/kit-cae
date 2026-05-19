# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Shared types for :mod:`dav.usd`."""

from dataclasses import dataclass

import dav


@dataclass(frozen=True)
class FieldInfo:
    """Field metadata exposed by a USD adapter.

    Attributes:
        name: Stable field identifier used by :func:`dav.usd.field_from_prim`.
            This is typically the USD instance name for ``OmniSciFieldAPI``.
        label: Human-readable display name for the field, often preserving the
            original solver variable name.
        association: DAV topological association for the field, such as
            vertex- or cell-associated.
    """

    name: str
    label: str
    association: dav.AssociationType
