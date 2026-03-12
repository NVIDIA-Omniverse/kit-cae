# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Dialog for adding multi-apply API schemas with smart instance name suggestions."""

__all__ = ["APISchemaDialog"]

import asyncio
import re
from logging import getLogger
from typing import Callable, Optional

from omni import ui
from pxr import Usd

logger = getLogger(__name__)


class APISchemaDialog:
    """A modal dialog for selecting an instance name when applying multi-apply API schemas.

    This dialog shows:
    - Documentation for the API schema
    - An editable text input field for the instance name
    - Clickable suggestion buttons for quick selection
    - Real-time validation

    Example:
        dialog = APISchemaDialog(
            api_schema="CaeVizDatasetSelectionAPI",
            prims=[prim1, prim2],
            suggestions_provider=lambda prims: ["source", "seeds"],
            validator=lambda name: (True, "") if name.islower() else (False, "Must be lowercase")
        )
        instance_name = await dialog.exec()
        if instance_name:
            print(f"Selected instance name: {instance_name}")
    """

    def __init__(
        self,
        api_schema: str,
        prims: list[Usd.Prim],
        suggestions_provider: Optional[Callable[[list[Usd.Prim]], list[str]]] = None,
        validator: Optional[Callable[[str], tuple[bool, str]]] = None,
        default_value: str = "default",
    ):
        """Initialize the dialog.

        Args:
            api_schema: The API schema type name (e.g., "CaeVizDatasetSelectionAPI")
            prims: List of prims to which the API will be applied
            suggestions_provider: Optional function that returns a list of suggested instance names
            validator: Optional function that validates instance names (is_valid, error_message)
            default_value: Default instance name if no suggestions are available
        """
        self._api_schema = api_schema
        self._prims = prims
        self._suggestions_provider = suggestions_provider
        self._validator = validator or self._default_validator
        self._default_value = default_value

        # Get documentation from USD schema
        self._documentation = self._get_schema_documentation()

        # Get suggestions
        self._suggestions = self._get_suggestions()

        self._window: Optional[ui.Window] = None
        self._dialog_result = asyncio.Event()
        self._accepted = [False]
        self._input_value = [self._suggestions[0] if self._suggestions else default_value]
        self._combo_box = None

    def _get_schema_documentation(self) -> str:
        """Extract documentation for the API schema from USD schema registry."""
        try:
            registry = Usd.SchemaRegistry()
            schema_type = registry.GetTypeFromName(self._api_schema)
            if schema_type:
                # Get the prim definition which contains the documentation
                prim_def = registry.FindAppliedAPIPrimDefinition(
                    self._api_schema
                ) or registry.FindConcretePrimDefinition(self._api_schema)
                if prim_def:
                    doc = prim_def.GetDocumentation()
                    if doc:
                        # Clean up whitespace using regex:
                        doc = re.sub(r"\n{3,}", "\v", doc)
                        doc = re.sub(r"[ \t\r\n]+", " ", doc, flags=re.MULTILINE)
                        doc = re.sub(r"\v", "\n", doc)
                        return doc.strip()

            # Fallback message
            return f"Apply {self._api_schema} to the selected prims."
        except Exception as e:
            logger.warning(f"Failed to get documentation for {self._api_schema}: {e}")
            return f"Apply {self._api_schema} to the selected prims."

    def _get_suggestions(self) -> list[str]:
        """Get suggested instance names based on the suggestions provider."""
        if self._suggestions_provider:
            try:
                suggestions = self._suggestions_provider(self._prims)
                # Filter suggestions to only include those that can be applied to all prims
                valid_suggestions = []
                for suggestion in suggestions:
                    can_apply = all(prim.CanApplyAPI(self._api_schema, suggestion) for prim in self._prims)
                    if can_apply:
                        valid_suggestions.append(suggestion)
                return valid_suggestions
            except Exception as e:
                logger.warning(f"Failed to get suggestions: {e}")
        return []

    def _default_validator(self, instance_name: str) -> tuple[bool, str]:
        """Default validator that checks if the instance name can be applied to all prims."""
        if not instance_name or not instance_name.strip():
            return False, "Instance name cannot be empty"

        instance_name = instance_name.strip()

        # Check if instance name contains only lowercase characters
        if not instance_name.islower():
            return False, "Instance name must contain only lowercase characters"

        # Check if all prims can apply this API schema with the instance name
        for prim in self._prims:
            if not prim.CanApplyAPI(self._api_schema, instance_name):
                return False, f"Cannot apply to prim {prim.GetName()}"
            if prim.HasAPI(self._api_schema, instance_name):
                return False, "Instance name already exists"

        return True, ""

    async def exec(self) -> Optional[str]:
        """Show the dialog and wait for user response.

        Returns:
            The validated instance name if OK was clicked, None if cancelled.
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
        self._window = ui.Window(
            f"Add {self._api_schema}",
            width=500,
            height=0,
            flags=ui.WINDOW_FLAGS_MODAL | ui.WINDOW_FLAGS_NO_RESIZE,
            visible=False,
        )

        with self._window.frame:
            with ui.VStack(spacing=8):
                # Documentation section with scroll area
                ui.Label(self._documentation, word_wrap=True)

                ui.Spacer(height=8)

                # Create a container for the input field
                input_field = None
                with ui.HStack(height=0, spacing=4):
                    ui.Label("Instance Name:", width=60)

                    # Use StringField for editable input
                    input_field = ui.StringField(name="instance_name_field")
                    input_field.model.set_value(self._input_value[0])

                # Show suggestions with clickable buttons if available
                if self._suggestions:
                    with ui.HStack(height=0, spacing=4):
                        ui.Label("Suggestions:", width=60, style={"color": 0xFF888888})
                        with ui.VStack(spacing=2):
                            # Display suggestions in a grid with max 6 buttons per row
                            for row_start in range(0, len(self._suggestions), 6):
                                row_suggestions = self._suggestions[row_start : row_start + 6]
                                with ui.HStack(spacing=2, height=0):
                                    for suggestion in row_suggestions:

                                        def create_suggestion_button(sug: str):
                                            """Create a button that applies the suggestion."""

                                            def on_click():
                                                input_field.model.set_value(sug)

                                            return on_click

                                        ui.Button(
                                            suggestion,
                                            clicked_fn=create_suggestion_button(suggestion),
                                            height=20,
                                            width=0,  # Auto width based on content
                                        )
                                    # Add spacer to prevent stretching of last button in incomplete rows
                                    ui.Spacer()

                # Error label (initially hidden)
                error_label = ui.Label("", style={"color": 0xFF3333FF}, visible=True)

                def on_input_changed(model):
                    value = model.get_value_as_string()
                    self._input_value[0] = value
                    is_valid, error_msg = self._validator(value)
                    ok_button.enabled = is_valid
                    if not is_valid:
                        error_label.text = f"* {error_msg}"
                    else:
                        error_label.text = ""

                input_field.model.add_value_changed_fn(on_input_changed)

                ui.Spacer()
                with ui.HStack(height=0, spacing=4):
                    ui.Spacer()
                    ok_button = ui.Button("OK", name="ok_button", width=80, clicked_fn=self._on_ok_clicked)
                    ui.Button("Cancel", name="cancel_button", width=80, clicked_fn=self._on_cancel_clicked)

        # Initial validation
        initial_valid, initial_error = self._validator(self._input_value[0])
        ok_button.enabled = initial_valid
        if not initial_valid:
            error_label.text = f"* {initial_error}"
        else:
            error_label.text = ""

    def _on_ok_clicked(self):
        """Handle OK button click."""
        self._accepted[0] = True
        self._dialog_result.set()

    def _on_cancel_clicked(self):
        """Handle Cancel button click."""
        self._accepted[0] = False
        self._dialog_result.set()
