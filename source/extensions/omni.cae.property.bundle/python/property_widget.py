# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
#  its affiliates is strictly prohibited.

__all__ = [
    "CodelessSchemaPropertiesWidget",
    "CaeCodelessSchemaPropertiesWidget",
    "CaeSchemaPropertiesWidget",
    "CAESchemeDelegate",
    "has_algorithms_warp",
    "has_index",
    "has_flow",
]

import logging

from omni import ui
from omni.cae.schema import cae, sids
from omni.kit.app import get_app
from omni.kit.property.usd.usd_property_widget import (
    UsdPropertiesWidget,
    UsdPropertiesWidgetBuilder,
    UsdPropertyUiEntry,
)
from omni.kit.window.property.property_scheme_delegate import PropertySchemeDelegate
from omni.kit.window.property.templates import HORIZONTAL_SPACING
from pxr import Sdf, Usd

logger = logging.getLogger(__name__)


def has_index():
    manager = get_app().get_extension_manager()
    return manager.is_extension_enabled("omni.cae.index")


def has_flow():
    manager = get_app().get_extension_manager()
    return manager.is_extension_enabled("omni.cae.flow")


def has_algorithms_warp():
    manager = get_app().get_extension_manager()
    return manager.is_extension_enabled("omni.cae.algorithms.warp")


class CodelessSchemaPropertiesWidget(UsdPropertiesWidget):
    """A widget that filters and displays properties from a specific USD schema or applied API schema.

    This widget is a modified version of  usd_property_widget.SchemaPropertiesWidget that works with codeless schemas.
    """

    def __init__(self, title: str, schema: str):
        """Initializes the SchemaPropertiesWidget with a specific schema and inclusion settings.

        Args:
            title (str): The title of the widget.
            schema (str): The USD schema type name to filter properties for.
        """
        super().__init__(title, collapsed=False)
        self._schemaTypeName = schema
        self._schemaType = None

    @property
    def schema_type(self):
        """Returns the schema type of the widget."""
        if self._schemaType is None:
            registry = Usd.SchemaRegistry()
            logger.info(
                "GetAPITypeFromSchemaTypeName(%s) = %s",
                self._schemaTypeName,
                registry.GetAPITypeFromSchemaTypeName(self._schemaTypeName),
            )
            logger.info(
                "GetConcreteTypeFromSchemaTypeName(%s) = %s",
                self._schemaTypeName,
                registry.GetConcreteTypeFromSchemaTypeName(self._schemaTypeName),
            )
            self._schemaType = registry.GetAPITypeFromSchemaTypeName(
                self._schemaTypeName
            ) or registry.GetConcreteTypeFromSchemaTypeName(self._schemaTypeName)
            if not self._schemaType:
                raise RuntimeError("Unknown schema '%s' specified. Did you load the USD plugin?" % self._schemaTypeName)

        return self._schemaType

    def on_new_payload(self, payload):
        """Handles a new payload for the widget, filtering properties based on the schema.

        Args:
            payload (:obj:`Payload`): The new payload to be handled by the widget.

        Returns:
            bool: True if the payload is valid and the widget should be updated, False otherwise."""
        if not super().on_new_payload(payload):
            return False

        if not self._payload or len(self._payload) == 0:
            return False

        for prim_path in self._payload:
            prim = self._get_prim(prim_path)

            if not prim:
                return False

            registry = Usd.SchemaRegistry()
            is_api_schema = registry.IsAppliedAPISchema(self.schema_type)
            if not (
                is_api_schema and prim.HasAPI(self.schema_type) or not is_api_schema and prim.IsA(self.schema_type)
            ):
                return False
        return True

    def _filter_props_to_build(self, props):
        """
        See UsdPropertiesWidget._filter_props_to_build
        """
        if len(props) == 0:
            return props

        registry = Usd.SchemaRegistry()
        defn = registry.FindAppliedAPIPrimDefinition(self._schemaTypeName) or registry.FindConcretePrimDefinition(
            self._schemaTypeName
        )
        schema_attr_names = defn.GetPropertyNames()
        return [prop for prop in props if prop.GetName() in schema_attr_names]


class CaeCodelessSchemaPropertiesWidget(CodelessSchemaPropertiesWidget):
    LIMITS = {
        sids.Tokens.caeSidsElementConnectivity: 1,
        sids.Tokens.caeSidsElementStartOffset: 1,
        sids.Tokens.caeSidsGridCoordinates: 3,
    }

    def get_additional_kwargs(self, ui_prop: UsdPropertyUiEntry):
        """Gets additional keyword arguments for building the label or UI widget.

        Args:
            ui_prop (:obj:`UsdPropertyUiEntry`): The UsdPropertyUiEntry to get additional kwargs for.

        Returns:
            tuple: A tuple containing additional_label_kwargs and additional_widget_kwargs."""

        additional_label_kwargs, additional_widget_kwargs = super().get_additional_kwargs(ui_prop)
        if ui_prop.property_type == Usd.Relationship:
            targets_limit = 0
            if ui_prop.prop_name.endswith("dataset") and ui_prop.prop_name.startswith("omni:cae:"):
                targets_limit = 1
            else:
                targets_limit = self.LIMITS.get(ui_prop.prop_name, 0)
            # see UsdPropertiesWidgetBuilder.relationship_builder(...)
            widget_kwargs = {"targets_limit": targets_limit}
            if additional_widget_kwargs:
                widget_kwargs.update(additional_widget_kwargs)
            return additional_label_kwargs, widget_kwargs
        else:
            return additional_label_kwargs, additional_widget_kwargs

    def build_property_item(self, stage, ui_prop: UsdPropertyUiEntry, prim_paths: list[Sdf.Path]):
        # if ui_prop.prop_name.endswith(":primVars"):  # FIXME
        #     ui_prop.build_fn = partial(CaePropertyWidgetBuilder.field_builder, dataset_prop_name="omni:cae:hull:dataset")
        if ui_prop.property_type == Usd.Relationship:
            ui_prop.build_fn = CaePropertyWidgetBuilder.relationship_builder
        return super().build_property_item(stage, ui_prop, prim_paths)


class CaeSchemaPropertiesWidget(CaeCodelessSchemaPropertiesWidget):
    def __init__(self, title: str, schema):
        registry = Usd.SchemaRegistry()
        if registry.IsConcrete(schema):
            schemaTypeName = registry.GetConcreteSchemaTypeName(schema)
        else:
            schemaTypeName = registry.GetAPISchemaTypeName(schema)
        super().__init__(title, schemaTypeName)


class CaePropertyWidgetBuilder:

    @classmethod
    def relationship_builder(
        cls,
        stage,
        attr_name,
        metadata,
        type_name,
        prim_paths: list[Sdf.Path],
        additional_label_kwargs=None,
        additional_widget_kwargs=None,
        dataset_prop_name=None,
    ):
        if len(prim_paths) == 1:
            return UsdPropertiesWidgetBuilder.relationship_builder(
                stage, attr_name, metadata, prim_paths, additional_label_kwargs, additional_widget_kwargs
            )
        else:
            # if more than 1 prim is selected, we don't show (nor allow editing of) the relationship
            return cls.mixed_builder(attr_name, metadata, additional_label_kwargs)

    @classmethod
    def mixed_builder(cls, attr_name, metadata, additional_label_kwargs):
        # if more than 1 prim is selected, we don't show (nor allow editing of) the relationship
        with ui.HStack(spacing=HORIZONTAL_SPACING):
            UsdPropertiesWidgetBuilder.create_label(attr_name, metadata, additional_label_kwargs)
            with ui.ZStack(alignment=ui.Alignment.CENTER):
                ui.Rectangle(alignment=ui.Alignment.CENTER, name="mixed_overlay_text")
                ui.Label("Mixed", name="mixed_overlay", alignment=ui.Alignment.CENTER)
            ui.Spacer(width=12)


class CAESchemeDelegate(PropertySchemeDelegate):

    def _is_schema(self, payload, schema):
        anchor_prim = None
        stage = payload.get_stage()
        if stage:
            for prim_path in payload:
                prim = stage.GetPrimAtPath(prim_path)
                if prim:
                    if not prim.IsA(schema):
                        return None
                    anchor_prim = prim
        return anchor_prim is not None

    def _has_schema(self, payload, schema):
        anchor_prim = None
        stage = payload.get_stage()
        if stage:
            for prim_path in payload:
                prim = stage.GetPrimAtPath(prim_path)
                if prim:
                    if not prim.HasAPI(schema):
                        return None
                    anchor_prim = prim
        return anchor_prim is not None

    def get_widgets(self, payload) -> list[str]:
        """
        Tests the payload and gathers widgets in interest to be drawn in specific order.

        Args:
            payload (PrimSelectionPayload): payload.

        Returns:
            list: list of widgets to build.
        """
        if self._is_schema(payload, cae.CgnsFieldArray):
            return ["path", "cae_cgns_field_array"]
        if self._is_schema(payload, cae.NumPyFieldArray):
            return ["path", "cae_np_field_array"]
        if self._is_schema(payload, cae.Hdf5FieldArray):
            return ["path", "cae_hdf5_field_array"]
        if self._has_schema(payload, sids.UnstructuredAPI):
            return ["path", "cae_sids_unstructured"]
        if self._has_schema(payload, "OmniCaeAlgorithmsBoundingBoxAPI"):
            return ["path", "cae_algorithms_bounding_box"]
        if self._has_schema(payload, "OmniCaeAlgorithmsPointsAPI"):
            return ["path", "cae_algorithms_points"]
        if self._has_schema(payload, "OmniCaeAlgorithmsGlyphsAPI"):
            return ["path", "cae_algorithms_glyphs"]
        if self._has_schema(payload, "OmniCaeAlgorithmsExternalFacesAPI"):
            return ["path", "cae_algorithms_external_faces"]
        if self._has_schema(payload, "OmniCaeAlgorithmsStreamlinesAPI"):
            return ["path", "cae_algorithms_streamlines"]

        if has_algorithms_warp():
            if self._has_schema(payload, "OmniCaeAlgorithmsWarpStreamlinesAPI"):
                return ["path", "cae_algorithms_warp_streamlines"]

        if has_index():
            if self._has_schema(payload, "OmniCaeIndeXSliceAPI"):
                return ["path", "cae_index_slice"]
            if self._has_schema(payload, "OmniCaeIndeXNanoVdbSliceAPI"):
                return ["path", "cae_index_nanovdbslice"]
            if self._has_schema(payload, "OmniCaeIndeXVolumeAPI"):
                return ["path", "cae_index_volume"]
            if self._has_schema(payload, "OmniCaeIndeXNanoVdbVolumeAPI"):
                return ["path", "cae_index_nanovdbvolume"]
        if has_flow():
            if self._has_schema(payload, "OmniCaeFlowDataSetEmitterAPI"):
                return ["path", "cae_flow_dataset_emitter"]
        return []

    def get_unwanted_widgets(self, payload) -> list[str]:
        unwanted_widgets = []
        if not self._is_schema(payload, cae.CgnsFieldArray):
            unwanted_widgets.append("cae_cgns_field_array")
        if not self._is_schema(payload, cae.NumPyFieldArray):
            unwanted_widgets.append("cae_np_field_array")
        if not self._has_schema(payload, sids.UnstructuredAPI):
            unwanted_widgets.append("cae_sids_unstructured")
        return unwanted_widgets
