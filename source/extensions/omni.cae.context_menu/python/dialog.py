# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Reusable dialog components for context menu operations."""

__all__ = ["TypeSelectionDialog", "ValidatedInputDialog"]

import asyncio
from typing import Callable, Optional

from omni import ui


class TypeSelectionDialog:
    """A modal dialog for selecting a type from a list of options.

    This dialog presents a combobox with options and returns the selected value
    when the user clicks OK, or None if cancelled.

    Example:
        dialog = TypeSelectionDialog(
            title="Choose Volume Type",
            label="Select the volume type:",
            options=["irregular", "nanovdb"],
            default_index=0
        )
        selected = await dialog.exec()
        if selected is not None:
            print(f"Selected: {selected}")
    """

    def __init__(
        self,
        title: str,
        label: str,
        options: list[str],
        default_index: int = 0,
        field_label: str = "Type:",
        field_width: int = 60,
    ):
        """Initialize the dialog.

        Args:
            title: The window title
            label: The description label shown above the combobox
            options: List of string options to display in the combobox
            default_index: The initially selected option index (default: 0)
            field_label: The label shown next to the combobox (default: "Type:")
            field_width: Width of the field label in pixels (default: 60)
        """
        self._title = title
        self._label = label
        self._options = options
        self._default_index = default_index
        self._field_label = field_label
        self._field_width = field_width

        self._window: Optional[ui.Window] = None
        self._dialog_result = asyncio.Event()
        self._accepted = [False]
        self._selected_value = [options[default_index] if options else None]

    async def exec(self) -> Optional[str]:
        """Show the dialog and wait for user response.

        Returns:
            The selected option string if OK was clicked, None if cancelled or no options.
        """
        if not self._options:
            return None

        self._build_ui()
        self._window.visible = True

        # Wait for the user to accept or cancel the dialog
        await self._dialog_result.wait()

        # Get the selected value before closing the window
        result = self._selected_value[0] if self._accepted[0] else None

        # Close and destroy the window
        self._window.visible = False
        self._window.destroy()
        self._window = None

        return result

    def _build_ui(self):
        """Build the dialog UI."""
        self._window = ui.Window(self._title, auto_resize=True, flags=ui.WINDOW_FLAGS_MODAL, visible=False)

        with self._window.frame:
            with ui.VStack(spacing=8):
                ui.Label(self._label)

                # Combobox for type selection
                with ui.HStack(height=0, spacing=4):
                    ui.Label(self._field_label, width=self._field_width)
                    type_combobox = ui.ComboBox(self._default_index, *self._options)

                    def on_type_changed(model, item):
                        selected_index = model.get_item_value_model().as_int
                        self._selected_value[0] = self._options[selected_index]

                    type_combobox.model.add_item_changed_fn(on_type_changed)

                ui.Spacer()
                with ui.HStack(height=0, spacing=4):
                    ui.Spacer()
                    ui.Button("OK", name="ok_button", width=80, clicked_fn=self._on_ok_clicked)
                    ui.Button("Cancel", name="cancel_button", width=80, clicked_fn=self._on_cancel_clicked)

    def _on_ok_clicked(self):
        """Handle OK button click."""
        self._accepted[0] = True
        self._dialog_result.set()

    def _on_cancel_clicked(self):
        """Handle Cancel button click."""
        self._accepted[0] = False
        self._dialog_result.set()


class ValidatedInputDialog:
    """A modal dialog for entering a validated text input.

    This dialog presents a text input field with real-time validation and returns
    the validated value when the user clicks OK, or None if cancelled.

    Example:
        def validate_name(value: str) -> tuple[bool, str]:
            if not value.strip():
                return False, "Name cannot be empty"
            if not value.islower():
                return False, "Name must be lowercase"
            return True, ""

        dialog = ValidatedInputDialog(
            title="Enter Name",
            label="Please enter an instance name:",
            default_value="default",
            validator=validate_name,
            field_label="Name:",
            field_width=100
        )
        result = await dialog.exec()
        if result is not None:
            print(f"Entered: {result}")
    """

    def __init__(
        self,
        title: str,
        label: str,
        validator: Callable[[str], tuple[bool, str]],
        default_value: str = "",
        field_label: str = "Value:",
        field_width: int = 100,
    ):
        """Initialize the dialog.

        Args:
            title: The window title
            label: The description label shown above the input field
            validator: A function that takes the input string and returns (is_valid, error_message)
            default_value: The initial value in the input field (default: "")
            field_label: The label shown next to the input field (default: "Value:")
            field_width: Width of the field label in pixels (default: 100)
        """
        self._title = title
        self._label = label
        self._validator = validator
        self._default_value = default_value
        self._field_label = field_label
        self._field_width = field_width

        self._window: Optional[ui.Window] = None
        self._dialog_result = asyncio.Event()
        self._accepted = [False]
        self._input_value = [default_value]

    async def exec(self) -> Optional[str]:
        """Show the dialog and wait for user response.

        Returns:
            The validated input string if OK was clicked, None if cancelled.
        """
        self._build_ui()
        self._window.visible = True

        # Wait for the user to accept or cancel the dialog
        await self._dialog_result.wait()

        # Get the input value before closing the window
        result = self._input_value[0].strip() if self._accepted[0] else None

        # Close and destroy the window
        self._window.visible = False
        self._window.destroy()
        self._window = None

        return result

    def _build_ui(self):
        """Build the dialog UI."""
        self._window = ui.Window(self._title, auto_resize=True, flags=ui.WINDOW_FLAGS_MODAL, visible=False)

        with self._window.frame:
            with ui.VStack(spacing=8):
                ui.Label(self._label)

                with ui.HStack(height=0, spacing=4):
                    ui.Label(self._field_label, width=self._field_width)
                    input_field = ui.StringField(name="input_field")
                    input_field.model.set_value(self._default_value)

                # Error label (initially hidden)
                error_label = ui.Label("", style={"color": 0xFFFF6B6B, "font_size": 12}, visible=False)

                def on_input_changed(model):
                    value = model.get_value_as_string()
                    self._input_value[0] = value
                    is_valid, error_msg = self._validator(value)
                    ok_button.enabled = is_valid
                    error_label.visible = not is_valid
                    if not is_valid:
                        error_label.text = f"* {error_msg}"

                input_field.model.add_value_changed_fn(on_input_changed)

                ui.Spacer()
                with ui.HStack(height=0, spacing=4):
                    ui.Spacer()
                    ok_button = ui.Button("OK", name="ok_button", width=80, clicked_fn=self._on_ok_clicked)
                    ui.Button("Cancel", name="cancel_button", width=80, clicked_fn=self._on_cancel_clicked)

        # Initial validation
        initial_valid, initial_error = self._validator(self._default_value)
        ok_button.enabled = initial_valid
        error_label.visible = not initial_valid
        if not initial_valid:
            error_label.text = f"* {initial_error}"

    def _on_ok_clicked(self):
        """Handle OK button click."""
        self._accepted[0] = True
        self._dialog_result.set()

    def _on_cancel_clicked(self):
        """Handle Cancel button click."""
        self._accepted[0] = False
        self._dialog_result.set()
