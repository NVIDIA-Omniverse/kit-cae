# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import List

from omni import ui


class MinimalItem(ui.AbstractItem):
    """
    Helper class for ui.AbstractItem implementations to create items for modal item model.
    """

    def __init__(self, text):
        super().__init__()
        self.model = ui.SimpleStringModel(text)

    @property
    def text(self):
        return self.model.get_value_as_string()


class MinimalModal(ui.AbstractItemModel):
    """
    Helper class for ui.AbstractItemModel implementations to store indexed items for use in Omniverse UI widgets.
    """

    def __init__(self, default_index, item_labels: list[str]):
        super().__init__()
        self._default_index = default_index
        self._current_index = ui.SimpleIntModel(default_index)
        self._current_index.add_value_changed_fn(lambda _: self._item_changed(None))
        self._items = [MinimalItem(label) for label in item_labels]

    @property
    def current_index(self) -> int:
        return self._current_index.as_int

    @property
    def current_text(self) -> str:
        return self._items[self._current_index.get_value_as_int()].text

    def get_item_children(self, _: ui.AbstractItem = None) -> List[ui.AbstractItem]:
        return self._items

    def get_item_value_model(self, item: MinimalItem, _) -> ui.AbstractValueModel:
        if item is None:
            return self._current_index
        return item.model
