# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
#  its affiliates is strictly prohibited.

__all__ = ["Extension"]

from omni.ext import IExt

from .context_menu import *


class Extension(IExt):

    def on_startup(self, ext_id):
        self._registered = False

        from omni.kit.widget.context_menu import add_menu

        self._context_menu_create_algos = add_menu(get_algorithms_menu_dict(), "CREATE")
        self._context_menu_create_flow = add_menu(get_flow_menu_dict(), "CREATE")

    def on_shutdown(self):
        del self._context_menu_create_algos
        del self._context_menu_create_flow
