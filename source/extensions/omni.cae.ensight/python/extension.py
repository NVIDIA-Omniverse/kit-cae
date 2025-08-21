# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
#  its affiliates is strictly prohibited.

import omni.ext


class Extension(omni.ext.IExt):
    """Extension class for the Ensight data delegate."""

    def on_startup(self, _ext_id):
        from omni.cae.data import get_data_delegate_registry
        from omni.kit.commands import register_all_commands_in_module

        from . import commands
        from .ensight import EnsightGoldGeoDelegate, EnsightGoldVarDelegate

        self._geo_delegate = EnsightGoldGeoDelegate(_ext_id)
        self._var_delegate = EnsightGoldVarDelegate(_ext_id)

        self._registry = get_data_delegate_registry()
        self._registry.register_data_delegate(self._geo_delegate)
        self._registry.register_data_delegate(self._var_delegate)

        register_all_commands_in_module(commands)

    def on_shutdown(self):
        from omni.kit.commands import unregister_module_commands

        from . import commands

        self._registry.deregister_data_delegate(self._geo_delegate)
        self._registry.deregister_data_delegate(self._var_delegate)
        self._geo_delegate = None
        self._var_delegate = None
        self._registry = None

        unregister_module_commands(commands)
