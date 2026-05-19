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

import omni.kit.context_menu as kit_context_menu
from omni.ext import IExt
from omni.kit.property.usd import PrimPathWidget
from omni.kit.widget.context_menu import add_menu

from . import context_menu, legacy_context_menu


def _is_legacy_ui_enabled() -> bool:
    """Check if legacy UI elements should be shown."""
    try:
        from omni.cae.data.settings import is_legacy_ui_enabled

        return is_legacy_ui_enabled()
    except ImportError:
        # If the function doesn't exist, default to False
        return False


class Extension(IExt):

    def on_startup(self, ext_id):
        self._menu_entries = []
        self._prim_path_widget_entries = []

        self._menu_entries.append(add_menu(context_menu.get_sources_menu_dict(), "CREATE"))
        self._menu_entries.append(add_menu(context_menu.get_operators_menu_dict(), "CREATE"))
        self._menu_entries.append(add_menu(context_menu.get_flow_menu_dict(), "CREATE"))
        self._add_to_prim_path_widget(context_menu.get_add_menu_dict())

        self._colormap_menu_entry = kit_context_menu.add_menu(
            {
                "name": "Copy LUT Texture URL",
                "glyph": "menu_link.svg",
                "show_fn": context_menu.ColormapCopyLutUrl.show,
                "onclick_fn": context_menu.ColormapCopyLutUrl.onclick,
                "appear_after": "Copy Prim Path",
            },
            "MENU",
            "omni.kit.widget.stage",
        )

        # legacy menus - only register if legacy UI is enabled
        if _is_legacy_ui_enabled():
            self._menu_entries.append(add_menu(legacy_context_menu.get_algorithms_menu_dict(), "CREATE"))
            self._menu_entries.append(add_menu(legacy_context_menu.get_flow_menu_dict(), "CREATE"))

    def on_shutdown(self):
        self._colormap_menu_entry = None
        for item in self._prim_path_widget_entries:
            PrimPathWidget.remove_button_menu_entry(item)
        self._prim_path_widget_entries.clear()
        self._menu_entries.clear()

    def _add_to_prim_path_widget(self, menu_dict):
        entires = menu_dict["name"]["CAE"]
        for entry in entires:
            name = entry.get("name")
            if not name:
                continue

            onclick_fn = entry.get("onclick_fn")
            show_fn = entry.get("show_fn", None)

            path = f"CAE/{name}"
            self._prim_path_widget_entries.append(
                PrimPathWidget.add_button_menu_entry(
                    path,
                    onclick_fn=onclick_fn,
                    show_fn=show_fn,
                )
            )
