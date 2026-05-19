# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Base adapter contract for :mod:`dav.usd`.

Concrete adapters translate one or more USD scientific schemas into DAV
datasets and fields. The public :mod:`dav.usd.api` functions never talk to
schema-specific code directly; they always resolve a :class:`PrimAdapter`
through the registry first.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence

from pxr import Usd


class PrimAdapter(ABC):
    """Abstract base class for OpenUSD-to-DAV adapters.

    Adapters are intentionally small and focused:

    - :meth:`can_handle` answers whether the adapter supports a prim
    - :meth:`to_dataset` converts a prim into DAV dataset objects
    - :meth:`list_fields` reports available field metadata
    - :meth:`get_field` loads one field or assembles a vector field from
      multiple components
    """

    @abstractmethod
    def can_handle(self, prim: Usd.Prim) -> bool:
        """Return ``True`` if this adapter supports ``prim``.

        Args:
            prim: USD prim to test.
        """
        raise NotImplementedError

    @abstractmethod
    async def to_dataset(self, prim: Usd.Prim, *, device: str, time_code: Usd.TimeCode):
        """Convert ``prim`` into DAV.

        Args:
            prim: USD prim to convert.
            device: Warp device alias on which DAV arrays should be allocated.
            time_code: USD time code to use when reading time-sampled data.

        Returns:
            dav.Dataset | dav.DatasetCollection: DAV representation of
            ``prim``.
        """
        raise NotImplementedError

    @abstractmethod
    async def list_fields(self, prim: Usd.Prim, *, time_code: Usd.TimeCode):
        """List fields that can be loaded from ``prim``.

        Args:
            prim: USD prim whose fields should be enumerated.
            time_code: USD time code to use when reading field metadata.

        Returns:
            list[dav.usd.FieldInfo]: Metadata describing the available fields.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_field(self, prim: Usd.Prim, *, field_names: str | Sequence[str], device: str, time_code: Usd.TimeCode):
        """Load a field from ``prim``.

        Args:
            prim: USD prim to load from.
            field_names: Field identifier or identifiers. Multiple names may be
                combined into one DAV vector field.
            device: Warp device alias on which DAV arrays should be allocated.
            time_code: USD time code to use when reading time-sampled data.

        Returns:
            dav.Field: Loaded DAV field.
        """
        raise NotImplementedError
