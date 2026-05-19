# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Adapter registry for :mod:`dav.usd`.

The registry is responsible for mapping a USD prim to the first adapter that
claims it can handle that prim. This keeps the public API small while allowing
new schema-specific adapters to be added over time.
"""

from collections.abc import Iterable
from typing import Any

from pxr import Usd

from .adapters.cae import CaeMeshAdapter, CaePointCloudAdapter
from .adapters.cgns import CgnsAdapter
from .adapters.ensight import EnSightPartAdapter
from .adapters.openfoam import OpenFoamBoundaryPatchAdapter, OpenFoamPolyMeshAdapter
from .adapters.vtk import VtkAdapter
from .exceptions import UnsupportedPrimError, USDAdapterError


class AdapterRegistry:
    """Resolve USD prims to DAV adapter implementations.

    Args:
        adapters: Optional iterable of adapter instances. Adapters are checked
            in order, so more specific adapters should typically be registered
            before more generic ones.
    """

    def __init__(self, adapters: Iterable[Any] | None = None):
        self._adapters = list(adapters or [])

    def register(self, adapter: Any):
        """Register an additional adapter instance.

        Args:
            adapter: An object implementing the :class:`dav.usd.adapters.base.PrimAdapter`
                contract.
        """
        self._adapters.append(adapter)

    def find_adapter(self, prim: Usd.Prim):
        """Return the first adapter that can handle ``prim``.

        Args:
            prim: USD prim to test.

        Returns:
            object | None: The matching adapter instance, or ``None`` if no
            adapter matches.

        Raises:
            USDAdapterError: If an adapter's ``can_handle`` check fails because
                a required ``pxr`` schema module is not installed.
        """
        for adapter in self._adapters:
            try:
                matches = adapter.can_handle(prim)
            except ImportError as exc:
                raise USDAdapterError(f"Required pxr schema module is not available: {exc}") from exc
            if matches:
                return adapter
        return None

    def get_adapter(self, prim: Usd.Prim):
        """Return the adapter for ``prim`` or raise if no adapter matches.

        Args:
            prim: USD prim to resolve.

        Returns:
            object: The matching adapter instance.

        Raises:
            UnsupportedPrimError: If no registered adapter can handle
                ``prim``.
        """
        adapter = self.find_adapter(prim)
        if adapter is None:
            raise UnsupportedPrimError(f"No dav.usd adapter is registered for prim {prim.GetPath()}")
        return adapter


#: Default adapter registry used by the high-level :mod:`dav.usd.api` helpers.
registry = AdapterRegistry([CgnsAdapter(), EnSightPartAdapter(), OpenFoamPolyMeshAdapter(), OpenFoamBoundaryPatchAdapter(), VtkAdapter(), CaePointCloudAdapter(), CaeMeshAdapter()])
