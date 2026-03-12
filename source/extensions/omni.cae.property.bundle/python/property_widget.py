# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

__all__ = [
    "CaeGeomPrimSchemeDelegate",
    "CaePropertiesWidget",
    "CaeFieldArrayPropertiesWidget",
]

import asyncio
import logging

import omni.timeline
import omni.ui as ui
import omni.usd
from omni.cae.data import array_utils, usd_utils
from omni.cae.schema import cae
from omni.kit.property.bundle import GeomPrimSchemeDelegate
from omni.kit.property.usd.usd_property_widget import MultiSchemaPropertiesWidget, UsdPropertiesWidget
from omni.kit.window.property.style import get_style
from omni.kit.window.property.templates import SimplePropertyWidget
from pxr import Usd

logger = logging.getLogger(__name__)

# Module-level cache for field array metadata
# Key: prim path (str), Value: metadata dict
_FIELD_ARRAY_METADATA_CACHE = {}


class CaePropertiesWidget(UsdPropertiesWidget):
    """A custom properties widget for CAE-related schemas.
    It filters the properties to only show the ones that are part of CAE-related schemas.
    It also groups the properties by the instance name and sorts them by the order of the schema attributes.
    """

    def __init__(self, title: str):
        """Initializes the CAE Properties Widget.

        Args:
            title (str): The title of the widget.
        """
        super().__init__(title, collapsed=False, maintain_property_order=True)
        self._schema_attr_names = set()

        self._multi_schema_properties_widgets = {}

    def on_new_payload(self, payload):
        """Handles a new payload for the widget.

        Args:
            payload (:obj:`Payload`): The new payload to be handled by the widget.
        Returns:
            bool: True if the payload is valid and the widget should be updated, False otherwise.
        """
        if not super().on_new_payload(payload):
            return False

        if not self._payload or len(self._payload) == 0:
            return False

        return self._build_schema_attr_names()

    def _add_multi_schema_properties_widget(self, api_schema):
        """
        Adds a multi-schema properties widget for the given API schema.
        """
        # This track ensures that the widgets we create here are skipped when building default properties widgets
        # like the "Geometry" widget. Otherwise, properties for Cae schemas get duplicated in the "Geometry" widget
        # which can be confusing for the user.
        self._multi_schema_properties_widgets[api_schema] = MultiSchemaPropertiesWidget(
            f"{api_schema} Properties", Usd.Typed, [], api_schemas=[api_schema]
        )

    def _build_schema_attr_names(self):
        """
        Builds the schema attribute names.
        """
        self._schema_attr_names = set()
        self._ordered_schema_attr_names = []
        self._instance_names = {}
        schema_reg = Usd.SchemaRegistry()
        for prim_path in self._payload:
            prim = self._get_prim(prim_path)

            if not prim:
                return False

            for api_schema_full in [prim.GetTypeName()] + list(prim.GetAppliedSchemas()):
                api_schema, api_instance = schema_reg.GetTypeNameAndInstance(api_schema_full)
                if api_schema.startswith("Cae") or api_schema.startswith("Rtwt"):
                    if api_schema not in self._multi_schema_properties_widgets:
                        self._add_multi_schema_properties_widget(api_schema)
                    defn = schema_reg.FindAppliedAPIPrimDefinition(api_schema)
                    defn = defn or schema_reg.FindConcretePrimDefinition(api_schema)
                    if defn:
                        api_prop_names = defn.GetPropertyNames()
                        if api_instance:
                            api_prop_names = [
                                name.replace("__INSTANCE_NAME__", api_instance) for name in api_prop_names
                            ]
                            for name in api_prop_names:
                                self._instance_names[name] = api_instance
                        self._schema_attr_names.update(api_prop_names)
                        self._ordered_schema_attr_names.extend(api_prop_names)

        return len(self._schema_attr_names) > 0

    def _filter_props_to_build(self, props):
        """
        The responsibility of this method is to filter any properties that are not part of CAE-related schemas.
        """

        if len(props) == 0:
            return props

        self._build_schema_attr_names()
        cae_schema_attr_names = self._schema_attr_names

        # print("CaePropertiesWidget: filtering properties for CAE schemas: %s (props: %s)" % (cae_schema_attr_names, [prop.GetName() for prop in props]))
        return [prop for prop in props if prop.GetName() in cae_schema_attr_names]

    def _customize_props_layout(self, props):
        for prop in props:
            if prop.prop_name in self._instance_names:
                instance_name = self._instance_names[prop.prop_name]
                # prop.add_additional_label_kwargs({"instance_name": instance_name})
                # prop.override_display_group(prop.display_group + f" [{instance_name}]")
                prop.override_display_group(f"{instance_name[0].upper()}{instance_name[1:]} [{prop.display_group}]")

        # sort props to be in the order of self._ordered_schema_attr_names
        props.sort(key=lambda p: self._ordered_schema_attr_names.index(p.prop_name))

        # Group properties with the same instance name together when first encountered
        reordered_props = []
        processed = set()

        for prop in props:
            if prop.prop_name in processed:
                continue

            reordered_props.append(prop)
            processed.add(prop.prop_name)

            # If this property has an instance name, add all other properties with the same instance name
            if prop.prop_name in self._instance_names:
                instance_name = self._instance_names[prop.prop_name]
                for other_prop in props:
                    if (
                        other_prop.prop_name not in processed
                        and other_prop.prop_name in self._instance_names
                        and self._instance_names[other_prop.prop_name] == instance_name
                    ):
                        reordered_props.append(other_prop)
                        processed.add(other_prop.prop_name)

        return super()._customize_props_layout(reordered_props)

    def _build_framestack(self, prefix, display_group):
        """Overridden to collapse certain frames by default. This avoid clutter in the property window
        with rarely modified property groups."""
        frame, stack, wid = super()._build_framestack(prefix, display_group)
        suffixes = ["[Rescale Range]", "[Configure XAC Shader]", "[Configure Flow Environment]"]
        if frame is not None and any(wid.endswith(suffix) for suffix in suffixes):
            frame.collapsed = True
        return frame, stack, wid


class CaeFieldArrayPropertiesWidget(SimplePropertyWidget):
    """
    A custom properties widget specifically for CaeFieldArray types and their subtypes.
    This widget will be shown when a prim with CaeFieldArray type is selected.
    Custom UI will be added here in the future.
    """

    def __init__(self, title: str = "Field Array"):
        """Initializes the CaeFieldArray Properties Widget.

        Args:
            title (str): The title of the widget.
        """
        super().__init__(title, collapsed=False)
        self._metadata = {}
        self._info_labels = {}
        self._stats_frame = None
        self._histogram_frame = None
        self._scalar_farray = None  # cached field array for histogram recomputation
        self._hist_min_field = None
        self._hist_max_field = None

    def on_new_payload(self, payload) -> bool:
        """Handles a new payload for the widget.

        Args:
            payload (:obj:`Payload`): The new payload to be handled by the widget.
        Returns:
            bool: True if the payload is valid and the widget should be updated, False otherwise.
        """
        if not super().on_new_payload(payload):
            return False

        if not payload or len(payload) == 0:
            return False

        # Only show widget for single selection
        if len(payload) > 1:
            return False

        # Check if any of the selected prims is a CaeFieldArray or subtype
        stage = omni.usd.get_context().get_stage()

        prim_path = self._payload[0]
        prim = stage.GetPrimAtPath(prim_path)
        if not prim:
            return False

        # Check if prim is a CaeFieldArray or any of its subtypes
        if not prim.IsA(cae.FieldArray):
            return False

        # Try to restore cached metadata for the first selected prim
        self._metadata = {}
        prim_path_str = str(prim.GetPath())
        if prim_path_str in _FIELD_ARRAY_METADATA_CACHE:
            self._metadata = _FIELD_ARRAY_METADATA_CACHE[prim_path_str].copy()
            logger.info(f"Restored cached metadata for {prim_path_str}")

        return True

    async def _fetch_metadata(self, prim_path_str: str):
        """Fetches metadata for the selected field array prim(s) asynchronously.

        Returns:
            dict: Dictionary containing metadata like type, shape, device, etc.
        """

        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path_str)
        if not prim:
            return {}

        timeline = omni.timeline.get_timeline_interface()
        time_code = Usd.TimeCode(round(timeline.get_current_time() * timeline.get_time_codes_per_seconds()))
        farray = await usd_utils.get_array(prim, time_code)
        if not farray:
            return {}

        metadata = {}
        metadata["device"] = str(array_utils.get_device(farray))
        metadata["type"] = farray.dtype.name

        # Format shape with comma-separated numbers and ndim on separate lines
        # e.g., "shape: (1,000,000, 3)\nndim: 2"
        formatted_dims = [str(dim) for dim in farray.shape]
        metadata["shape"] = f"\n\tndim: {farray.ndim}\n\tshape: ({', '.join(formatted_dims)})"

        # Compute range
        ranges = array_utils.get_componentwise_ranges(farray)

        # Determine format string based on dtype (floating vs integer)
        is_float = farray.dtype.kind == "f"

        def format_value(val):
            """Format a single value based on type."""
            return f"{val:,.3f}" if is_float else f"{int(val):,}"

        # Format range based on number of components
        is_scalar = farray.ndim == 1 or farray.shape[-1] == 1
        if is_scalar:
            # Single component: show as (min, max) with comma separators for large numbers
            metadata["range"] = f"({format_value(ranges[0][0])}, {format_value(ranges[0][1])})"
        else:
            # Multi-component: show ranges for each component on separate lines for readability
            # e.g., "Component 0: (min, max)\nComponent 1: (min, max)\n..."
            component_ranges = [
                f"\t{i}:\t({format_value(ranges[i][0])}, {format_value(ranges[i][1])})" for i in range(len(ranges))
            ]
            metadata["range"] = "\n" + "\n".join(component_ranges)

        # Compute statistics and histogram for scalar fields via Warp (avoids full GPU->CPU transfer)
        if is_scalar:
            self._scalar_farray = farray
            scalar_stats = array_utils.get_scalar_stats(farray, num_bins=32)
            metadata["stats"] = {
                "min": scalar_stats["min"],
                "max": scalar_stats["max"],
                "mean": scalar_stats["mean"],
                "median": scalar_stats["median"],
                "q1": scalar_stats["q1"],
                "q2": scalar_stats["q2"],
                "q3": scalar_stats["q3"],
                "q4": scalar_stats["q4"],
            }
            metadata["histogram"] = {
                "counts": scalar_stats["counts"],
                "bin_edges": scalar_stats["bin_edges"],
            }
        else:
            self._scalar_farray = None

        return metadata

    def _on_refresh_clicked(self):
        """Called when the Refresh button is clicked."""
        # Kick off async metadata fetch
        asyncio.ensure_future(self._async_refresh())

    async def _async_refresh(self):
        """Asynchronously refresh metadata."""
        # Update UI to show we're fetching (optional)
        for label in self._info_labels.values():
            label.text = "(Fetching...)"

        prim_path_str = str(self._payload[0])

        # Fetch metadata asynchronously
        self._metadata = await self._fetch_metadata(prim_path_str)

        # Cache the metadata for the selected prim
        _FIELD_ARRAY_METADATA_CACHE[prim_path_str] = self._metadata.copy()
        logger.info(f"Cached metadata for {prim_path_str}")

        # Update UI with fetched data
        self._update_info_labels()

    def _update_info_labels(self):
        """Updates the information labels, stats table, and histogram."""
        metadata = self._metadata
        if "device" in self._info_labels:
            self._info_labels["device"].text = f"Device: {metadata.get('device', '(TBD)')}"
        if "type" in self._info_labels:
            self._info_labels["type"].text = f"Type: {metadata.get('type', '(TBD)')}"
        if "shape" in self._info_labels:
            self._info_labels["shape"].text = f"Size: {metadata.get('shape', '(TBD)')}"
        # Show range on the left only for vector fields (scalar range is in the stats table)
        if "range" in self._info_labels:
            if "stats" in metadata:
                self._info_labels["range"].text = ""
            else:
                self._info_labels["range"].text = f"Range: {metadata.get('range', '(TBD)')}"
        self._build_stats_table()
        self._build_histogram()

    @staticmethod
    def _format_axis_value(val):
        """Format a value for axis labels, keeping it compact."""
        if abs(val) >= 1e6 or (abs(val) < 1e-2 and val != 0):
            return f"{val:.2e}"
        return f"{val:.3g}"

    def _on_hist_range_changed(self, _=None):
        """Recompute histogram when the user changes the range fields."""
        if self._scalar_farray is None or self._hist_min_field is None:
            return
        try:
            hist_min = float(self._hist_min_field.model.get_value_as_string())
            hist_max = float(self._hist_max_field.model.get_value_as_string())
        except ValueError:
            return
        if hist_min >= hist_max:
            return
        result = array_utils.compute_histogram(self._scalar_farray, num_bins=32, range_min=hist_min, range_max=hist_max)
        self._metadata["histogram"] = result
        self._build_histogram_chart()

    def _build_stats_table(self):
        """Builds the statistics table for scalar fields (shown beside info labels)."""
        if self._stats_frame is None:
            return

        self._stats_frame.clear()

        if "stats" not in self._metadata:
            return

        stats = self._metadata["stats"]
        fmt = self._format_axis_value
        label_style = {"font_size": 12, "color": 0xFF999999}
        value_style = {"font_size": 12, "color": 0xFFCCCCCC}

        with self._stats_frame:
            with ui.VStack(spacing=2, height=0):
                # Range, mean, median
                with ui.HStack(height=0, spacing=4):
                    ui.Label("Range:", width=55, style=label_style, height=0)
                    ui.Label(f"[{fmt(stats['min'])}, {fmt(stats['max'])}]", style=value_style, height=0)
                with ui.HStack(height=0, spacing=4):
                    ui.Label("Mean:", width=55, style=label_style, height=0)
                    ui.Label(fmt(stats["mean"]), style=value_style, height=0)
                with ui.HStack(height=0, spacing=4):
                    ui.Label("Median:", width=55, style=label_style, height=0)
                    ui.Label(fmt(stats["median"]), style=value_style, height=0)
                ui.Spacer(height=2)
                # Quartile ranges
                for name, key in [("Q1", "q1"), ("Q2", "q2"), ("Q3", "q3"), ("Q4", "q4")]:
                    lo, hi = stats[key]
                    with ui.HStack(height=0, spacing=4):
                        ui.Label(f"{name}:", width=55, style=label_style, height=0)
                        ui.Label(f"[{fmt(lo)}, {fmt(hi)}]", style=value_style, height=0)

    def _build_histogram(self):
        """Builds the histogram section: range inputs and chart."""
        if self._histogram_frame is None:
            return

        self._histogram_frame.clear()

        if "histogram" not in self._metadata:
            return

        style = get_style()["Label::label"]
        label_style = {"font_size": 12, "color": 0xFF999999}

        with self._histogram_frame:
            with ui.VStack(spacing=4, height=0):
                # Histogram range inputs
                ui.Label("Histogram Range:", style=style)
                with ui.HStack(height=0, spacing=4):
                    ui.Label("Min:", width=30, style=label_style)
                    self._hist_min_field = ui.FloatField(height=20, width=ui.Fraction(1))
                    self._hist_min_field.model.set_value(float(self._metadata["histogram"]["bin_edges"][0]))
                    self._hist_min_field.model.add_end_edit_fn(self._on_hist_range_changed)
                    ui.Label("Max:", width=30, style=label_style)
                    self._hist_max_field = ui.FloatField(height=20, width=ui.Fraction(1))
                    self._hist_max_field.model.set_value(float(self._metadata["histogram"]["bin_edges"][-1]))
                    self._hist_max_field.model.add_end_edit_fn(self._on_hist_range_changed)

                ui.Spacer(height=2)

                # Chart container
                self._histogram_chart_frame = ui.Frame(height=0)
                self._build_histogram_chart()

    def _build_histogram_chart(self):
        """Builds just the histogram bar chart."""
        if self._histogram_chart_frame is None:
            return

        self._histogram_chart_frame.clear()

        hist = self._metadata.get("histogram")
        if not hist:
            return

        counts = hist["counts"]
        bin_edges = hist["bin_edges"]
        stats = self._metadata.get("stats", {})
        median_val = (bin_edges[0] + bin_edges[-1]) / 2
        max_count = max(counts) if counts else 0
        bar_height = 80
        fmt = self._format_axis_value
        axis_style = {"font_size": 11, "color": 0xFF999999}

        with self._histogram_chart_frame:
            with ui.VStack(spacing=2, height=0):
                with ui.ZStack(height=bar_height):
                    ui.Rectangle(style={"background_color": 0xFF1E1E1E, "border_radius": 2})
                    with ui.HStack(spacing=1, content_clipping=True):
                        for i, count in enumerate(counts):
                            h = int((count / max_count) * bar_height) if max_count > 0 else 0
                            lo, hi = bin_edges[i], bin_edges[i + 1]
                            tooltip = f"[{fmt(lo)}, {fmt(hi)})\nCount: {count:,}"
                            with ui.VStack():
                                ui.Spacer()
                                if h > 0:
                                    ui.Rectangle(
                                        height=h,
                                        tooltip=tooltip,
                                        style={"background_color": 0xFF4A90D9, "border_radius": 1},
                                    )
                                else:
                                    ui.Rectangle(
                                        height=1,
                                        tooltip=tooltip,
                                        style={"background_color": 0x00000000},
                                    )
                # X-axis labels: min, median, max
                with ui.HStack(height=0):
                    ui.Label(fmt(bin_edges[0]), style=axis_style, height=0)
                    ui.Spacer()
                    ui.Label(fmt(median_val), style=axis_style, height=0, alignment=ui.Alignment.CENTER)
                    ui.Spacer()
                    ui.Label(fmt(bin_edges[-1]), style=axis_style, height=0, alignment=ui.Alignment.RIGHT_CENTER)

    def build_items(self):
        """Builds the UI items for the widget."""
        style = get_style()["Label::label"]

        with ui.VStack(spacing=5, height=0):

            # Top row: info labels on the left, stats table on the right
            with ui.HStack(height=0, spacing=10):
                with ui.VStack(spacing=3, width=ui.Fraction(1)):
                    self._info_labels["device"] = ui.Label("Device: (TBD)", word_wrap=True, height=0, style=style)
                    self._info_labels["type"] = ui.Label("Type: (TBD)", word_wrap=True, height=0, style=style)
                    self._info_labels["shape"] = ui.Label("Shape: (TBD)", word_wrap=True, height=0, style=style)
                    # Range shown here only for vector fields (scalar range is in the stats table)
                    self._info_labels["range"] = ui.Label("", word_wrap=True, height=0, style=style)
                # Stats table placeholder (populated for scalar fields only)
                self._stats_frame = ui.Frame(width=ui.Fraction(1), height=0)

            # Histogram section: range inputs + chart (populated for scalar fields only)
            self._histogram_frame = ui.Frame(height=0)

            ui.Spacer(height=5)

            # Refresh button
            ui.Button("Refresh", height=20, clicked_fn=self._on_refresh_clicked)

            # If we have cached metadata, update the labels immediately
            self._update_info_labels()


class CaeGeomPrimSchemeDelegate(GeomPrimSchemeDelegate):
    """
    A custom scheme delegate for CAE-related schemas.
    It inserts the "cae" widget before the "geometry" widget, if it exists.
    This can be used to replace the default GeomPrimSchemeDelegate, for example:
    ```
    property_window.register_scheme_delegate("prim", "xformable_prim", CaeGeomPrimSchemeDelegate())
    ```
    """

    def get_widgets(self, payload) -> list[str]:
        """
        Tests the payload and gathers widgets in interest to be drawn in specific order.

        Args:
            payload (PrimSelectionPayload): payload.

        Returns:
            list: list of widgets to build.
        """
        widgets = super().get_widgets(payload)
        # insert "cae" widget before "geometry", if present else insert it after "path"
        # if "path" is not present, insert "cae" at the beginning (this ensures it's before "flow" widgets)
        if "geometry" in widgets:
            widgets.insert(widgets.index("geometry"), "cae")
        elif "path" in widgets:
            # insert "cae" after "path"
            widgets.insert(widgets.index("path") + 1, "cae")
        else:
            widgets = ["cae"] + widgets
        return widgets
