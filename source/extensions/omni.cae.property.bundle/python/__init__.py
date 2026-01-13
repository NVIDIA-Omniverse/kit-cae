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

import omni.kit.window.property as property_window_ext
from omni.cae.schema import cae, ensight, sids, vtk
from omni.ext import IExt

# from omni.kit.property.usd.usd_property_widget import SchemaPropertiesWidget
from .property_widget import *

# from omni.kit.window.content_browser import get_content_window


class Extension(IExt):

    def _register_widget(self, property_window, scheme, name, *args, **kwargs):
        property_window.register_widget(scheme, name, *args, **kwargs)
        self._registered_widgets.append((scheme, name))

    def on_startup(self, ext_id):
        self._registered_widgets: list[tuple[str, str]] = []
        # get_content_window().show_window(None, False)

        property_window = property_window_ext.get_window()
        if property_window:
            self._register_widget(
                property_window,
                "prim",
                "cae_cgns_field_array",
                CaeSchemaPropertiesWidget("CGNS Field Array", cae.CgnsFieldArray),
            )
            self._register_widget(
                property_window,
                "prim",
                "cae_np_field_array",
                CaeSchemaPropertiesWidget("NumPy Field Array", cae.NumPyFieldArray),
            )
            self._register_widget(
                property_window,
                "prim",
                "cae_hdf5_field_array",
                CaeSchemaPropertiesWidget("HDF5 Field Array", cae.Hdf5FieldArray),
            )
            self._register_widget(
                property_window,
                "prim",
                "cae_point_cloud",
                CaeSchemaPropertiesWidget("Point Cloud", cae.PointCloudAPI),
            )
            self._register_widget(
                property_window,
                "prim",
                "cae_dense_volume",
                CaeSchemaPropertiesWidget("Dense Volume", cae.DenseVolumeAPI),
            )
            self._register_widget(
                property_window,
                "prim",
                "cae_mesh",
                CaeSchemaPropertiesWidget("Mesh", cae.MeshAPI),
            )
            self._register_widget(
                property_window,
                "prim",
                "cae_sids_unstructured",
                CaeSchemaPropertiesWidget("SIDS Unstructured", sids.UnstructuredAPI),
            )

            self._register_widget(
                property_window,
                "prim",
                "cae_vtk_field_array",
                CaeSchemaPropertiesWidget("VTK Field Array", vtk.FieldArray),
            )

            self._register_widget(
                property_window,
                "prim",
                "cae_vtk_unstructured_grid",
                CaeSchemaPropertiesWidget("VTK Unstructured Grid", vtk.UnstructuredGridAPI),
            )

            self._register_widget(
                property_window,
                "prim",
                "cae_ensight_gold_geo_field_array",
                CaeSchemaPropertiesWidget("EnSight Gold Geometry Field Array", ensight.GoldGeoFieldArray),
            )

            self._register_widget(
                property_window,
                "prim",
                "cae_ensight_gold_var_field_array",
                CaeSchemaPropertiesWidget("EnSight Gold Variable Field Array", ensight.GoldVarFieldArray),
            )

            self._register_widget(
                property_window,
                "prim",
                "cae_ensight_unstructured_piece",
                CaeSchemaPropertiesWidget("EnSight Unstructured", ensight.UnstructuredPieceAPI),
            )

            self._register_widget(
                property_window,
                "prim",
                "cae_algorithms_bounding_box",
                CaeCodelessSchemaPropertiesWidget("CAE Bounding Box", "CaeAlgorithmsBoundingBoxAPI"),
            )
            self._register_widget(
                property_window,
                "prim",
                "cae_algorithms_glyphs",
                CaeCodelessSchemaPropertiesWidget("CAE Glyphs", "CaeAlgorithmsGlyphsAPI"),
            )
            self._register_widget(
                property_window,
                "prim",
                "cae_algorithms_points",
                CaeCodelessSchemaPropertiesWidget("CAE Points", "CaeAlgorithmsPointsAPI"),
            )
            self._register_widget(
                property_window,
                "prim",
                "cae_algorithms_external_faces",
                CaeCodelessSchemaPropertiesWidget("CAE External Faces", "CaeAlgorithmsExternalFacesAPI"),
            )
            self._register_widget(
                property_window,
                "prim",
                "cae_algorithms_streamlines",
                CaeCodelessSchemaPropertiesWidget("CAE Streamlines", "CaeAlgorithmsStreamlinesAPI"),
            )

            if has_algorithms_warp():
                self._register_widget(
                    property_window,
                    "prim",
                    "cae_algorithms_warp_streamlines",
                    CaeCodelessSchemaPropertiesWidget("CAE NanoVDB Streamlines", "CaeAlgorithmsWarpStreamlinesAPI"),
                )

            if has_index():
                self._register_widget(
                    property_window,
                    "prim",
                    "cae_index_slice",
                    CaeCodelessSchemaPropertiesWidget("Slice (Unstructured)", "CaeIndeXSliceAPI"),
                )
                self._register_widget(
                    property_window,
                    "prim",
                    "cae_index_nanovdbslice",
                    CaeCodelessSchemaPropertiesWidget("Slice (NanoVDB)", "CaeIndeXNanoVdbSliceAPI"),
                )
                self._register_widget(
                    property_window,
                    "prim",
                    "cae_index_volume",
                    CaeCodelessSchemaPropertiesWidget("Volume (Unstructured)", "CaeIndeXVolumeAPI"),
                )
                self._register_widget(
                    property_window,
                    "prim",
                    "cae_index_nanovdbvolume",
                    CaeCodelessSchemaPropertiesWidget("Volume (NanoVDB)", "CaeIndeXNanoVdbVolumeAPI"),
                )

            if has_flow():
                self._register_widget(
                    property_window,
                    "prim",
                    "cae_flow_dataset_emitter",
                    CaeCodelessSchemaPropertiesWidget("DataSet Emitter", "CaeFlowDataSetEmitterAPI"),
                )

            property_window.register_scheme_delegate("prim", "cae_delegate", CAESchemeDelegate())
            # HACK: this seems necessary otherwise the delegate is never called, consequently the widget is never removed.
            property_window._scheme_delegates_layout["prim"].insert(0, "cae_delegate")
            self._registered = True

    def on_shutdown(self):
        if self._registered_widgets:
            property_window = property_window_ext.get_window()
            if property_window:
                for scheme, name in self._registered_widgets:
                    property_window.unregister_widget(scheme, name)
                self._registered_widgets = []

                property_window.unregister_scheme_delegate("prim", "cae_delegate")
                property_window._scheme_delegates_layout["prim"].remove("cae_delegate")
