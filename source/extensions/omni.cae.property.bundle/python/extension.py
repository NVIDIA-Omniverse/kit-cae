# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

__all__ = ["Extension"]

from omni.ext import IExt
from omni.kit.property.bundle import GeomPrimSchemeDelegate
from omni.kit.window.property import get_window

from .property_widget import CaeFieldArrayPropertiesWidget, CaeGeomPrimSchemeDelegate, CaePropertiesWidget


class Extension(IExt):

    def _register_widget(self, property_window, scheme, name, *args, **kwargs):
        property_window.register_widget(scheme, name, *args, **kwargs)

    def on_startup(self, ext_id):
        if property_window := get_window():
            property_window.register_widget("prim", "cae", CaePropertiesWidget("CAE"))
            property_window.register_widget("prim", "cae_field_array", CaeFieldArrayPropertiesWidget("CAE Insights"))
            property_window.register_scheme_delegate("prim", "xformable_prim", CaeGeomPrimSchemeDelegate())

    def on_shutdown(self):
        if property_window := get_window():
            property_window.unregister_widget("prim", "cae_field_array")
            property_window.unregister_widget("prim", "cae")
            # restore the default GeomPrimSchemeDelegate
            property_window.register_scheme_delegate("prim", "xformable_prim", GeomPrimSchemeDelegate())
