# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
#  its affiliates is strictly prohibited.

import os.path
from logging import getLogger

import omni.client
from omni.ext import IExt
from omni.kit.app import get_app

logger = getLogger(__name__)


class Extension(IExt):

    @staticmethod
    def get_materials_path(extId: str) -> str:
        """
        Return the path to the CAE materials file.

        Returns
        -------
        str
            The path to the CAE materials file.
        """
        # Get the absolute path of the materials directory.
        ext_path = get_app().get_extension_manager().get_extension_path(extId)
        materials_path = os.path.join(ext_path, "materials")
        return materials_path

    def on_startup(self, extId):
        if path := self.get_materials_path(extId):
            # Add the materials path to the omni.client search path.
            omni.client.add_default_search_path(path)
            self._path = path
        else:
            self._path = None

    def on_shutdown(self):
        if self._path:
            # Remove the materials path from the omni.client search path.
            omni.client.remove_default_search_path(self._path)
            self._path = None
