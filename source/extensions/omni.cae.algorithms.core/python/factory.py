# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
#  its affiliates is strictly prohibited.

__all__ = ["Factory", "get_factory"]

import logging
from typing import Callable

from omni.cae.data import usd_utils
from pxr import Tf, Usd

from .algorithm import Algorithm

logger = logging.getLogger(__name__)


class Creator:
    primType: str = None
    apiSchemaTypes: list[str] = []
    callback: Callable[[Usd.Prim], Algorithm] = None

    def supports(self, prim: Usd.Prim) -> bool:
        if not prim:
            return False

        ttype = Usd.SchemaRegistry().GetTypeFromName(self.primType)
        if prim and not prim.IsA(ttype) and not prim.GetTypeName() == self.primType:
            return False

        for api in self.apiSchemaTypes:
            if api not in prim.GetAppliedSchemas():
                return False
        return True

    def create(self, prim: Usd.Prim) -> Algorithm:
        if self.callback is not None:
            return self.callback(prim)
        return None


class Factory:

    def __init__(self):
        self._creators: dict[str, list[Creator]] = {}

    def get_creators(self):
        for creators in self._creators.values():
            for h in creators:
                yield h

    def get_convertors(self):
        for convertors in self._convertors.values():
            for h in convertors:
                yield h

    def register_create_callback(
        self, primType: str, appliedApiSchemas: list[str], callback: Callable[[Usd.Prim], Algorithm], moduleName: str
    ):
        creator = Creator()
        creator.primType = primType
        creator.apiSchemaTypes = appliedApiSchemas
        creator.callback = callback

        # add to list
        creators = self._creators.get(moduleName, [])
        creators.append(creator)
        self._creators[moduleName] = creators

    def unregister_all(self, moduleName):
        """Unregister all algorithms for the given module name"""
        if moduleName in self._creators:
            del self._creators[moduleName]

    def get_supported_prim_types(self) -> list[tuple[str, list[str]]]:
        """Returns a list of 2-tuples. Each tuple is the prim type and a
        list of API schemas that should have been applied to those prim types. The list of
        API schemas may be empty."""
        for creator in self.get_creators():
            yield (creator.primType, creator.apiSchemaTypes)

    @usd_utils.quietable
    def create(self, prim: Usd.Prim) -> Algorithm:
        for creator in self.get_creators():
            if creator.supports(prim):
                if algo := creator.create(prim):
                    return algo
        raise usd_utils.QuietableException("Failed to create algorithm for %s" % prim)


_factory = Factory()


def get_factory() -> Factory:
    global _factory
    return _factory
