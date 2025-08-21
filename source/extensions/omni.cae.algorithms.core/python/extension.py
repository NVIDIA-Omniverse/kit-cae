# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
#  its affiliates is strictly prohibited.

from logging import getLogger

from omni.ext import IExt

from ._algorithms import BoundingBox, ExternalFaces, Glyphs, Points, Streamlines
from .factory import Factory, get_factory
from .listener import Listener

logger = getLogger(__name__)


class Extension(IExt):

    def on_startup(self, ext_id):
        self._extId = ext_id
        self._listener = Listener()

        factory: Factory = get_factory()

        # tell the factory all algorithms we can handle.
        factory.register_create_callback("BasisCurves", ["CaeAlgorithmsBoundingBoxAPI"], BoundingBox, self._extId)
        factory.register_create_callback("BasisCurves", ["CaeAlgorithmsStreamlinesAPI"], Streamlines, self._extId)
        factory.register_create_callback("Points", ["CaeAlgorithmsPointsAPI"], Points, self._extId)
        factory.register_create_callback("PointInstancer", ["CaeAlgorithmsGlyphsAPI"], Glyphs, self._extId)
        factory.register_create_callback("Mesh", ["CaeAlgorithmsExternalFacesAPI"], ExternalFaces, self._extId)

        # register commands
        if kit_commands := self.get_commands():
            from . import _commands

            kit_commands.register_all_commands_in_module(_commands)

    def on_shutdown(self):
        if kit_commands := self.get_commands():
            from . import _commands

            kit_commands.unregister_module_commands(_commands)

        factory: Factory = get_factory()
        factory.unregister_all(self._extId)

        del self._listener

    def get_commands(self):
        try:
            from omni.kit import commands

            return commands
        except ImportError:
            return None
