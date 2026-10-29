# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import asyncio
import logging
import warnings
from unittest import mock

import numpy as np
import omni.cae.simdata as cae_simdata
import omni.kit.test
import omni.usd
import warp as wp
import warp_simdata as simdata
from omni.cae.core import settings, usd_utils
from omni.cae.schema import viz as cae_viz
from omni.cae.testing import get_test_data_path, new_stage
from omni.cae.usd_plugins_importers import import_to_stage
from omni.cae.viz import utils
from omni.cae.viz.utils import RtSubPrimGuard
from omni.kit.app import get_app
from omni.usd import get_context
from pxr import Gf, OmniSci, OmniSciFlash, Sdf, Usd, UsdGeom, Vt
from usdrt import Sdf as SdfRT
from usdrt import Usd as UsdRT
from warp_simdata.data_models.custom import voronoi_point_cloud as simdata_data_models_voronoi_point_cloud
from warp_simdata.data_models.flash import axisymmetric_dual as simdata_axisymmetric_dual


def _apply_array(prim: Usd.Prim, name: str, values, value_type) -> None:
    OmniSci.ArrayAPI.Apply(prim, name)
    prim.CreateAttribute(f"omni:sci:array:{name}:value", value_type).Set(values)


def _make_flash_prim(stage: Usd.Stage) -> Usd.Prim:
    prim = OmniSci.Dataset.Define(stage, "/World/Flash").GetPrim()
    api = OmniSciFlash.AmrAPI.Apply(prim)
    api.CreateSpatialDimensionAttr(2)
    api.CreateBoundingBoxShapeAttr(Vt.IntArray([3, 2]))
    api.CreateFieldShapeAttr(Vt.IntArray([1, 1, 2]))
    _apply_array(prim, "nodeType", Vt.IntArray([1, 2, 1]), Sdf.ValueTypeNames.IntArray)
    _apply_array(
        prim,
        "gid",
        Vt.IntArray(
            [
                -2,
                -2,
                -2,
                3,
                -1,
                -1,
                -1,
                -1,
                -1,
                -2,
                -2,
                -2,
                -2,
                -1,
                -1,
                -1,
                -1,
                -1,
                -2,
                -2,
                1,
                -2,
                -1,
                -1,
                -1,
                -1,
                -1,
            ]
        ),
        Sdf.ValueTypeNames.IntArray,
    )
    _apply_array(
        prim,
        "boundingBox",
        Vt.FloatArray(
            [
                0.0,
                1.0,
                -1.0,
                0.0,
                0.0,
                0.1,
                1.0,
                2.0,
                -1.0,
                0.0,
                0.0,
                0.1,
                0.0,
                1.0,
                0.0,
                1.0,
                0.0,
                0.1,
            ]
        ),
        Sdf.ValueTypeNames.FloatArray,
    )
    field_api = OmniSci.FieldAPI.Apply(prim, "dens")
    field_api.CreateNameAttr("dens")
    field_api.CreateAssociationAttr("element")
    _apply_array(
        prim,
        "dens",
        Vt.FloatArray([10.0, 11.0, 20.0, 21.0, 30.0, 31.0]),
        Sdf.ValueTypeNames.FloatArray,
    )
    return prim


class TestUtils(omni.kit.test.AsyncTestCase):
    def test_log_runtime_warnings_routes_through_warning_logger(self):
        test_logger = logging.getLogger("omni.cae.viz.tests.runtime_warnings")
        with self.assertLogs(test_logger, level=logging.WARNING) as captured:
            with utils.log_runtime_warnings(test_logger, "SimData slice"):
                warnings.warn("output may contain holes", RuntimeWarning)

        self.assertEqual(
            captured.output,
            ["WARNING:omni.cae.viz.tests.runtime_warnings:SimData slice: output may contain holes"],
        )

    async def setUp(self):
        self.datasets = {}
        usd_context = get_context()
        await usd_context.new_stage_async()
        self.stage = usd_context.get_stage()

        # load the test data
        await import_to_stage(get_test_data_path("disk_out_ref.npz"), "/World/disk_out_ref_npz", schema="CGNS")
        await import_to_stage(get_test_data_path("disk_out_ref_tet.npz"), "/World/disk_out_ref_tet_npz", schema="CGNS")

        self.datasets["disk_out_ref_npz"] = {
            "dataset": "/World/disk_out_ref_npz/Base/Zone/Section",
            "nb_cells": 7_472,
            "nb_points": 8_499,
            "bds": (wp.vec3f(-5.75, -5.75, -10.0), wp.vec3f(5.75, 5.75, 10.16)),
            "field_ranges": {
                "Temp": (293.15, 913.15),
                "V_2": (-21.1259765625, 6.7119011878967285),
                "V_Mag": [0, 22.418820167105352],
            },
        }

        self.datasets["disk_out_ref_tet_npz"] = {
            "dataset": "/World/disk_out_ref_tet_npz/Base/Zone/Section",
            "nb_cells": 44_832,
            "nb_points": 8_499,
            "bds": (wp.vec3f(-5.75, -5.75, -10.0), wp.vec3f(5.75, 5.75, 10.16)),
            "field_ranges": {
                "Temp": (293.15, 913.15),
                "V_2": (-21.1259765625, 6.7119011878967285),
                "V_Mag": [0, 22.418820167105352],
            },
        }

    async def tearDown(self):
        del self.stage
        usd_context = get_context()
        await usd_context.close_stage_async()

    async def test_utils_get_input_dataset_with_dual_representation(self):
        dataset_prim = _make_flash_prim(self.stage)
        operator_prim = self.stage.DefinePrim("/World/DualContour")
        cae_viz.DatasetSelectionAPI.Apply(operator_prim, "source").CreateTargetRel().SetTargets(
            {dataset_prim.GetPath()}
        )
        cae_viz.DatasetDualAPI.Apply(operator_prim, "source")
        cae_viz.FieldSelectionAPI.Apply(operator_prim, "contour").CreateFieldNamesAttr().Set(["dens"])

        dataset = await utils.get_input_dataset(
            operator_prim,
            "source",
            timeCode=Usd.TimeCode.Default(),
            device="cpu",
            required_fields={"contour"},
            field_associations={"contour": simdata.AssociationType.NODE},
        )

        self.assertIs(dataset.data_model, simdata_axisymmetric_dual.DataModel)
        self.assertEqual(dataset.get_field("contour").association, simdata.AssociationType.NODE)

    async def test_utils_get_input_dataset_with_authored_flash_representation(self):
        dataset_prim = _make_flash_prim(self.stage)
        operator_prim = self.stage.DefinePrim("/World/PartialDualContour")
        cae_viz.DatasetSelectionAPI.Apply(operator_prim, "source").CreateTargetRel().SetTargets(
            {dataset_prim.GetPath()}
        )
        cae_viz.DatasetDualAPI.Apply(operator_prim, "source")
        representation_api = cae_viz.DatasetAxisymmetricRepresentationAPI.Apply(operator_prim, "source")
        representation_api.CreateAngularCellsAttr().Set(8)
        representation_api.CreateMinimumAngleAttr().Set(45.0)
        representation_api.CreateMaximumAngleAttr().Set(135.0)

        dataset = await utils.get_input_dataset(
            operator_prim,
            "source",
            timeCode=Usd.TimeCode.Default(),
            device="cpu",
            needs_fields=False,
        )

        self.assertIs(dataset.data_model, simdata_axisymmetric_dual.DataModel)
        self.assertEqual(int(dataset.handle.flash.angular_cells), 8)
        self.assertAlmostEqual(float(dataset.handle.flash.angle_min), np.pi / 4.0)
        self.assertAlmostEqual(float(dataset.handle.flash.angle_step), np.pi / 16.0)

        representation_api.GetAngularCellsAttr().Set(10)
        await get_app().next_update_async()
        updated_dataset = await utils.get_input_dataset(
            operator_prim,
            "source",
            timeCode=Usd.TimeCode.Default(),
            device="cpu",
            needs_fields=False,
        )
        self.assertIsNot(updated_dataset, dataset)
        self.assertEqual(int(updated_dataset.handle.flash.angular_cells), 10)

    async def test_utils_get_input_dataset_rejects_invalid_flash_representation(self):
        dataset_prim = _make_flash_prim(self.stage)
        operator_prim = self.stage.DefinePrim("/World/InvalidPartialContour")
        cae_viz.DatasetSelectionAPI.Apply(operator_prim, "source").CreateTargetRel().SetTargets(
            {dataset_prim.GetPath()}
        )
        representation_api = cae_viz.DatasetAxisymmetricRepresentationAPI.Apply(operator_prim, "source")
        representation_api.CreateAngularCellsAttr().Set(2)

        with self.assertRaisesRegex(usd_utils.QuietableException, "Angular Cells must be at least 3"):
            await utils.get_input_dataset(
                operator_prim,
                "source",
                timeCode=Usd.TimeCode.Default(),
                device="cpu",
                needs_fields=False,
            )

    async def test_utils_get_input_dataset_rejects_unsupported_dual_representation(self):
        dataset_prim = OmniSci.Dataset.Define(self.stage, "/World/UnsupportedDual").GetPrim()
        operator_prim = self.stage.DefinePrim("/World/UnsupportedDualContour")
        cae_viz.DatasetSelectionAPI.Apply(operator_prim, "source").CreateTargetRel().SetTargets(
            {dataset_prim.GetPath()}
        )
        cae_viz.DatasetDualAPI.Apply(operator_prim, "source")

        with self.assertRaisesRegex(usd_utils.QuietableException, "does not support a dual representation"):
            await utils.get_input_dataset(
                operator_prim,
                "source",
                timeCode=Usd.TimeCode.Default(),
                device="cpu",
                needs_fields=False,
            )

    async def test_utils_get_input_dataset(self):
        operator_prim = self.stage.DefinePrim("/World/OperatorTestInputDataset")
        # cae_viz.OperatorAPI.Apply(operator_prim)
        cae_viz.DatasetSelectionAPI.Apply(operator_prim, "source")
        dataset_selection_api_source = cae_viz.DatasetSelectionAPI(operator_prim, "source")
        dataset_selection_api_source.CreateTargetRel().SetTargets({self.datasets["disk_out_ref_npz"]["dataset"]})

        cae_viz.DatasetSelectionAPI.Apply(operator_prim, "other_dataset")
        dataset_selection_api_other_dataset = cae_viz.DatasetSelectionAPI(operator_prim, "other_dataset")
        dataset_selection_api_other_dataset.CreateTargetRel().SetTargets(
            {self.datasets["disk_out_ref_tet_npz"]["dataset"]}
        )

        cae_viz.DatasetSelectionAPI.Apply(operator_prim, "empty_dataset")

        dataset = await utils.get_input_dataset(
            operator_prim, "source", timeCode=Usd.TimeCode.EarliestTime(), device="cpu"
        )
        self.assertIsNotNone(dataset, "Dataset should not be None")
        self.assertEqual(
            dataset.get_num_nodes(), self.datasets["disk_out_ref_npz"]["nb_points"], "Mismatch in number of points"
        )
        self.assertEqual(
            dataset.get_num_elems(), self.datasets["disk_out_ref_npz"]["nb_cells"], "Mismatch in number of cells"
        )

        # we don't expect any fields to be present
        self.assertEqual(
            len(dataset.get_field_names()), 0, "Expected 0 fields in the dataset, got {len(dataset.get_field_names())}"
        )

        # we expect the bounds to be correct
        bds = dataset.get_bounds()
        self.assertEqual(bds[0], self.datasets["disk_out_ref_npz"]["bds"][0], "Mismatch in bounds min")
        self.assertEqual(bds[1], self.datasets["disk_out_ref_npz"]["bds"][1], "Mismatch in bounds max")

        other_dataset = await utils.get_input_dataset(
            operator_prim, "other_dataset", timeCode=Usd.TimeCode.EarliestTime(), device="cpu"
        )
        self.assertIsNotNone(other_dataset, "Other dataset should not be None")
        self.assertEqual(
            other_dataset.get_num_nodes(),
            self.datasets["disk_out_ref_tet_npz"]["nb_points"],
            "Mismatch in number of points",
        )
        self.assertEqual(
            other_dataset.get_num_elems(),
            self.datasets["disk_out_ref_tet_npz"]["nb_cells"],
            "Mismatch in number of cells",
        )

        # we don't expect any fields to be present
        self.assertEqual(
            len(other_dataset.get_field_names()),
            0,
            "Expected 0 fields in the dataset, got {len(other_dataset.get_field_names())}",
        )

        # we expect the bounds to be correct
        bds = other_dataset.get_bounds()
        self.assertEqual(bds[0], self.datasets["disk_out_ref_tet_npz"]["bds"][0], "Mismatch in bounds min")
        self.assertEqual(bds[1], self.datasets["disk_out_ref_tet_npz"]["bds"][1], "Mismatch in bounds max")

        with self.assertRaises(
            ValueError, msg="Expected ValueError when getting input dataset for non-existent instance"
        ):
            await utils.get_input_dataset(
                operator_prim, "bogus_dataset", timeCode=Usd.TimeCode.EarliestTime(), device="cpu"
            )

        with self.assertRaises(
            usd_utils.QuietableException, msg="Expected QuietableException when getting input dataset for empty dataset"
        ):
            await utils.get_input_dataset(
                operator_prim, "empty_dataset", timeCode=Usd.TimeCode.EarliestTime(), device="cpu"
            )

    # disabled till we start support suc collections.
    # async def test_utils_get_input_collection_dataset(self):

    #     operator_prim = self.stage.DefinePrim("/World/OperatorTestCollectionDataset")
    #     # cae_viz.OperatorAPI.Apply(operator_prim)
    #     cae_viz.DatasetSelectionAPI.Apply(operator_prim, "collection_dataset")
    #     dataset_selection_api_collection_dataset = cae_viz.DatasetSelectionAPI(operator_prim, "collection_dataset")
    #     dataset_selection_api_collection_dataset.CreateTargetRel().SetTargets(
    #         {self.datasets["disk_out_ref_npz"]["dataset"], self.datasets["disk_out_ref_tet_npz"]["dataset"]}
    #     )

    #     collection_dataset = await utils.get_input_dataset(
    #         operator_prim, "collection_dataset", timeCode=Usd.TimeCode.EarliestTime(), device="cpu"
    #     )
    #     self.assertIsNotNone(collection_dataset, "Collection dataset should not be None")
    #     self.assertEqual(
    #         collection_dataset.get_num_nodes(),
    #         self.datasets["disk_out_ref_npz"]["nb_points"] + self.datasets["disk_out_ref_tet_npz"]["nb_points"],
    #         "Mismatch in number of points",
    #     )
    #     self.assertEqual(
    #         collection_dataset.get_num_elems(),
    #         self.datasets["disk_out_ref_npz"]["nb_cells"] + self.datasets["disk_out_ref_tet_npz"]["nb_cells"],
    #         "Mismatch in number of cells",
    #     )

    #     # we don't expect any fields to be present
    #     self.assertEqual(
    #         len(collection_dataset.get_field_names()),
    #         0,
    #         "Expected 0 fields in the dataset, got {len(collection_dataset.get_field_names())}",
    #     )

    #     # we expect the bounds to be correct
    #     bds = collection_dataset.get_bounds()
    #     self.assertEqual(bds[0], self.datasets["disk_out_ref_npz"]["bds"][0], "Mismatch in bounds min")
    #     self.assertEqual(bds[1], self.datasets["disk_out_ref_npz"]["bds"][1], "Mismatch in bounds max")

    async def test_utils_get_input_dataset_with_fields(self):
        with settings.override_setting(settings.SettingsKeys.DOWN_CONVERT_64BIT, False):
            operator_prim = self.stage.DefinePrim("/World/OperatorTestInputDatasetWithFields")
            # cae_viz.OperatorAPI.Apply(operator_prim)
            cae_viz.DatasetSelectionAPI.Apply(operator_prim, "source")
            dataset_selection_api_source = cae_viz.DatasetSelectionAPI(operator_prim, "source")
            dataset_selection_api_source.CreateTargetRel().SetTargets({self.datasets["disk_out_ref_npz"]["dataset"]})

            cae_viz.FieldSelectionAPI.Apply(operator_prim, "Temp")
            field_selection_api_temp = cae_viz.FieldSelectionAPI(operator_prim, "Temp")
            field_selection_api_temp.CreateFieldNamesAttr().Set(["Temp"])

            cae_viz.FieldSelectionAPI.Apply(operator_prim, "V")
            field_selection_api_v = cae_viz.FieldSelectionAPI(operator_prim, "V")
            field_selection_api_v.CreateFieldNamesAttr().Set(["V"])

            # V_2 is the second component of the V field
            cae_viz.FieldSelectionAPI.Apply(operator_prim, "V_2")
            field_selection_api_v_2 = cae_viz.FieldSelectionAPI(operator_prim, "V_2")
            field_selection_api_v_2.CreateFieldNamesAttr().Set(["V"])
            field_selection_api_v_2.CreateModeAttr().Set(cae_viz.Tokens.selected_component)
            field_selection_api_v_2.CreateComponentIndexAttr().Set(2)

            # V_Mag is the vector magnitude of the V field
            cae_viz.FieldSelectionAPI.Apply(operator_prim, "V_Mag")
            field_selection_api_v_mag = cae_viz.FieldSelectionAPI(operator_prim, "V_Mag")
            field_selection_api_v_mag.CreateFieldNamesAttr().Set(["V"])
            field_selection_api_v_mag.CreateModeAttr().Set(cae_viz.Tokens.vector_magnitude)

            dataset_prim = self.stage.GetPrimAtPath(self.datasets["disk_out_ref_npz"]["dataset"])
            temp_direct = await cae_simdata.GetField.invoke(
                dataset_prim, "Temp", timeCode=Usd.TimeCode.EarliestTime(), device="cpu"
            )
            self.assertEqual(temp_direct.dtype, wp.float64, "Mismatch in direct scalar field dtype")
            self.assertAlmostEqual(
                temp_direct.get_range()[0],
                self.datasets["disk_out_ref_npz"]["field_ranges"]["Temp"][0],
                3,
                "Mismatch in direct scalar field range min",
            )
            self.assertAlmostEqual(
                temp_direct.get_range()[1],
                self.datasets["disk_out_ref_npz"]["field_ranges"]["Temp"][1],
                3,
                "Mismatch in direct scalar field range max",
            )

            v_direct = await cae_simdata.GetField.invoke(
                dataset_prim, "V", timeCode=Usd.TimeCode.EarliestTime(), device="cpu"
            )
            self.assertEqual(v_direct.dtype, wp.vec3d, "Mismatch in direct vector field dtype")

            with self.assertRaises(
                usd_utils.QuietableException,
                msg="Expected QuietableException when getting input dataset for non-existent field",
            ):
                await utils.get_input_dataset(
                    operator_prim,
                    "source",
                    timeCode=Usd.TimeCode.EarliestTime(),
                    device="cpu",
                    required_fields={"Bogus"},
                )

            dataset = await utils.get_input_dataset(
                operator_prim,
                "source",
                timeCode=Usd.TimeCode.EarliestTime(),
                device="cpu",
                required_fields={"Temp", "V", "V_2", "V_Mag"},
            )
            self.assertIsNotNone(dataset, "Dataset should not be None")
            self.assertEqual(
                len(dataset.get_field_names()),
                4,
                "Expected 4 fields in the dataset, got {len(dataset.get_field_names())}",
            )

            self.assertEqual(dataset.get_field_names()[0], "Temp", "Mismatch in field name")
            self.assertEqual(dataset.get_field_names()[1], "V", "Mismatch in field name")
            self.assertEqual(dataset.get_field_names()[2], "V_2", "Mismatch in field name")
            self.assertEqual(dataset.get_field_names()[3], "V_Mag", "Mismatch in field name")

            temp_field = dataset.get_field("Temp")
            self.assertAlmostEqual(
                temp_field.get_range()[0],
                self.datasets["disk_out_ref_npz"]["field_ranges"]["Temp"][0],
                3,
                "Mismatch in field range min",
            )
            self.assertAlmostEqual(
                temp_field.get_range()[1],
                self.datasets["disk_out_ref_npz"]["field_ranges"]["Temp"][1],
                3,
                "Mismatch in field range max",
            )
            self.assertEqual(temp_field.dtype, wp.float64, "Mismatch in field dtype")

            v_field = dataset.get_field("V")
            self.assertEqual(v_field.dtype, wp.vec3d, "Mismatch in field dtype")

            v_2_field = dataset.get_field("V_2")
            self.assertEqual(v_2_field.dtype, wp.float64, "Mismatch in field dtype")
            self.assertAlmostEqual(
                v_2_field.get_range()[0],
                self.datasets["disk_out_ref_npz"]["field_ranges"]["V_2"][0],
                3,
                "Mismatch in field range min",
            )
            self.assertAlmostEqual(
                v_2_field.get_range()[1],
                self.datasets["disk_out_ref_npz"]["field_ranges"]["V_2"][1],
                3,
                "Mismatch in field range max",
            )

            v_mag_field = dataset.get_field("V_Mag")
            self.assertEqual(v_mag_field.dtype, wp.float64, "Mismatch in field dtype")
            self.assertAlmostEqual(
                v_mag_field.get_range()[0],
                self.datasets["disk_out_ref_npz"]["field_ranges"]["V_Mag"][0],
                3,
                "Mismatch in field range min",
            )
            self.assertAlmostEqual(
                v_mag_field.get_range()[1],
                self.datasets["disk_out_ref_npz"]["field_ranges"]["V_Mag"][1],
                3,
                "Mismatch in field range max",
            )

    async def test_utils_get_input_dataset_converts_and_caches_field_associations(self):
        await import_to_stage(get_test_data_path("hex_timesteps.cgns"), "/World/hex_timesteps")
        dataset_path = "/World/hex_timesteps/Base/Zone/ElementsUniform"
        operator_prim = self.stage.DefinePrim("/World/OperatorAssociationConversion")

        cae_viz.DatasetSelectionAPI.Apply(operator_prim, "source")
        cae_viz.DatasetSelectionAPI(operator_prim, "source").CreateTargetRel().SetTargets({dataset_path})

        cae_viz.FieldSelectionAPI.Apply(operator_prim, "node_values")
        cae_viz.FieldSelectionAPI(operator_prim, "node_values").CreateFieldNamesAttr().Set(["CellSinusoid"])
        cae_viz.FieldSelectionAPI.Apply(operator_prim, "element_values")
        cae_viz.FieldSelectionAPI(operator_prim, "element_values").CreateFieldNamesAttr().Set(["PointSinusoid"])

        associations = {
            "node_values": simdata.AssociationType.NODE,
            "element_values": simdata.AssociationType.ELEMENT,
        }
        with (
            mock.patch.object(
                utils.simdata_node_field, "compute", wraps=utils.simdata_node_field.compute
            ) as node_compute,
            mock.patch.object(
                utils.simdata_element_field, "compute", wraps=utils.simdata_element_field.compute
            ) as element_compute,
        ):
            dataset = await utils.get_input_dataset(
                operator_prim,
                "source",
                timeCode=Usd.TimeCode.EarliestTime(),
                device="cpu",
                required_fields=set(associations),
                field_associations=associations,
            )
            cached_dataset = await utils.get_input_dataset(
                operator_prim,
                "source",
                timeCode=Usd.TimeCode.EarliestTime(),
                device="cpu",
                required_fields=set(associations),
                field_associations=associations,
            )

        self.assertIs(dataset, cached_dataset)
        self.assertEqual(dataset.get_field("node_values").association, simdata.AssociationType.NODE)
        self.assertEqual(dataset.get_field("node_values").size, dataset.get_num_nodes())
        self.assertEqual(dataset.get_field("element_values").association, simdata.AssociationType.ELEMENT)
        self.assertEqual(dataset.get_field("element_values").size, dataset.get_num_elems())
        node_compute.assert_called_once()
        element_compute.assert_called_once()

    async def test_utils_get_input_dataset_with_voxelization(self):
        operator_prim = self.stage.DefinePrim("/World/OperatorTestInputDatasetWithVoxelization")
        # cae_viz.OperatorAPI.Apply(operator_prim)
        cae_viz.DatasetSelectionAPI.Apply(operator_prim, "source")
        dataset_selection_api_source = cae_viz.DatasetSelectionAPI(operator_prim, "source")
        dataset_selection_api_source.CreateTargetRel().SetTargets({self.datasets["disk_out_ref_npz"]["dataset"]})

        cae_viz.DatasetVoxelizationAPI.Apply(operator_prim, "source")

        cae_viz.FieldSelectionAPI.Apply(operator_prim, "Temp")
        field_selection_api_temp = cae_viz.FieldSelectionAPI(operator_prim, "Temp")
        field_selection_api_temp.CreateFieldNamesAttr().Set(["Temp"])

        cae_viz.FieldSelectionAPI.Apply(operator_prim, "V")
        field_selection_api_v = cae_viz.FieldSelectionAPI(operator_prim, "V")
        field_selection_api_v.CreateFieldNamesAttr().Set(["V"])

        # Older Kit runtimes only allocate NanoVDB on CUDA; newer Warp builds
        # support CPU volumes too. Validate both supported device paths.
        for device in ("cpu", "cuda:0"):
            try:
                dataset = await utils.get_input_dataset(
                    operator_prim,
                    "source",
                    timeCode=Usd.TimeCode.EarliestTime(),
                    device=device,
                    required_fields={"Temp", "V"},
                )
            except RuntimeError as exc:
                if device != "cpu" or str(exc) != "Only CUDA devices are supported for allocate_by_tiles":
                    raise
                continue
            self.assertEqual(dataset.device, device)
            self.assertIsNotNone(dataset, "Dataset should not be None")
            self.assertEqual(
                len(dataset.get_field_names()),
                3,
                "Expected 3 fields in the dataset, got {len(dataset.get_field_names())}",
            )
            self.assertEqual(dataset.get_field_names()[0], "Temp", "Mismatch in field name")
            self.assertEqual(dataset.get_field_names()[1], "cae_mask", "Mismatch in field name")
            self.assertEqual(dataset.get_field_names()[2], "V", "Mismatch in field name")

            temp_field = dataset.get_field("Temp")
            self.assertEqual(temp_field.dtype, wp.float32, "Mismatch in field dtype")
            self.assertAlmostEqual(
                temp_field.get_range()[0],
                self.datasets["disk_out_ref_npz"]["field_ranges"]["Temp"][0],
                3,
                msg="Mismatch in field range min",
            )
            self.assertAlmostEqual(
                temp_field.get_range()[1],
                self.datasets["disk_out_ref_npz"]["field_ranges"]["Temp"][1],
                3,
                msg="Mismatch in field range max",
            )
            self.assertTrue(isinstance(temp_field.get_data(), wp.Volume), "Mismatch in field data type")

            v_field = dataset.get_field("V")
            self.assertEqual(v_field.dtype, wp.vec3f, "Mismatch in field dtype")
            self.assertTrue(isinstance(v_field.get_data(), wp.Volume), "Mismatch in field data type")

    async def test_utils_get_input_dataset_with_vox_voxel_size_mode(self):
        operator_prim = self.stage.DefinePrim("/World/OperatorTestInputDatasetWithVoxelization")
        # cae_viz.OperatorAPI.Apply(operator_prim)
        cae_viz.DatasetSelectionAPI.Apply(operator_prim, "source")
        dataset_selection_api_source = cae_viz.DatasetSelectionAPI(operator_prim, "source")
        dataset_selection_api_source.CreateTargetRel().SetTargets({self.datasets["disk_out_ref_npz"]["dataset"]})

        cae_viz.DatasetVoxelizationAPI.Apply(operator_prim, "source")
        dataset_voxelization_api_source = cae_viz.DatasetVoxelizationAPI(operator_prim, "source")
        dataset_voxelization_api_source.CreateVoxelSizeModeAttr().Set(cae_viz.Tokens.voxelSize)
        dataset_voxelization_api_source.CreateVoxelSizeAttr().Set((0.1, 0.1, 0.1))

        cae_viz.FieldSelectionAPI.Apply(operator_prim, "Temp")
        field_selection_api_temp = cae_viz.FieldSelectionAPI(operator_prim, "Temp")
        field_selection_api_temp.CreateFieldNamesAttr().Set(["Temp"])

        dataset = await utils.get_input_dataset(
            operator_prim, "source", timeCode=Usd.TimeCode.EarliestTime(), device="cuda:0", required_fields={"Temp"}
        )
        self.assertIsNotNone(dataset, "Dataset should not be None")
        self.assertEqual(
            len(dataset.get_field_names()), 2, "Expected 2 fields in the dataset, got {len(dataset.get_field_names())}"
        )
        self.assertEqual(dataset.get_field_names()[0], "Temp", "Mismatch in field name")
        self.assertEqual(dataset.get_field_names()[1], "cae_mask", "Mismatch in field name")

        temp_field = dataset.get_field("Temp")
        self.assertEqual(temp_field.dtype, wp.float32, "Mismatch in field dtype")
        self.assertTrue(isinstance(temp_field.get_data(), wp.Volume), "Mismatch in field data type")
        self.assertEqual(temp_field.get_data().get_voxel_size(), wp.vec3f(0.1), "Mismatch in voxel size")
        self.assertAlmostEqual(
            temp_field.get_range()[0],
            self.datasets["disk_out_ref_npz"]["field_ranges"]["Temp"][0],
            3,
            msg="Mismatch in field range min",
        )
        self.assertAlmostEqual(
            temp_field.get_range()[1],
            self.datasets["disk_out_ref_npz"]["field_ranges"]["Temp"][1],
            3,
            msg="Mismatch in field range max",
        )

    async def test_utils_voxelization_parameters_field_centering(self):
        data_bounds = Gf.Range3d(Gf.Vec3d(-1.0, -1.0, -1.0), Gf.Vec3d(1.0, 1.0, 1.0))

        default_prim = self.stage.DefinePrim("/World/OperatorVoxelizationDefaultCentering")
        cae_viz.DatasetVoxelizationAPI.Apply(default_prim, "source")
        default_api = cae_viz.DatasetVoxelizationAPI(default_prim, "source")

        default_params = utils._get_voxelization_parameters(
            default_prim, "source", data_bounds, Usd.TimeCode.EarliestTime()
        )
        self.assertEqual(default_api.GetFieldCenteringAttr().Get(), cae_viz.Tokens.point)
        self.assertEqual(default_params.field_centering, "point")

        cell_prim = self.stage.DefinePrim("/World/OperatorVoxelizationCellCentering")
        cae_viz.DatasetVoxelizationAPI.Apply(cell_prim, "source")
        cell_api = cae_viz.DatasetVoxelizationAPI(cell_prim, "source")
        cell_api.CreateFieldCenteringAttr().Set(cae_viz.Tokens.cell)

        cell_params = utils._get_voxelization_parameters(cell_prim, "source", data_bounds, Usd.TimeCode.EarliestTime())
        self.assertEqual(cell_params.field_centering, "cell")

        invalid_prim = self.stage.DefinePrim("/World/OperatorVoxelizationInvalidCentering")
        cae_viz.DatasetVoxelizationAPI.Apply(invalid_prim, "source")
        invalid_api = cae_viz.DatasetVoxelizationAPI(invalid_prim, "source")
        invalid_api.CreateFieldCenteringAttr().Set("node")

        with self.assertRaisesRegex(ValueError, "Unsupported voxelization field centering"):
            utils._get_voxelization_parameters(invalid_prim, "source", data_bounds, Usd.TimeCode.EarliestTime())

    async def test_utils_point_cloud_data_model_api_order(self):
        voronoi_first_prim = self.stage.DefinePrim("/World/OperatorPointCloudVoronoiFirst")
        cae_viz.DatasetSelectionAPI.Apply(voronoi_first_prim, "source")
        cae_viz.DatasetVoronoiPointCloudAPI.Apply(voronoi_first_prim, "source")
        cae_viz.DatasetGaussianSplattingAPI.Apply(voronoi_first_prim, "source")

        with mock.patch.object(utils.logger, "warning") as warning:
            self.assertEqual(utils._get_dataset_point_cloud_model(voronoi_first_prim, "source"), "voronoi")
        warning.assert_called_once()

        gaussian_first_prim = self.stage.DefinePrim("/World/OperatorPointCloudGaussianFirst")
        cae_viz.DatasetSelectionAPI.Apply(gaussian_first_prim, "source")
        cae_viz.DatasetGaussianSplattingAPI.Apply(gaussian_first_prim, "source")
        cae_viz.DatasetVoronoiPointCloudAPI.Apply(gaussian_first_prim, "source")

        with mock.patch.object(utils.logger, "warning") as warning:
            self.assertEqual(utils._get_dataset_point_cloud_model(gaussian_first_prim, "source"), "gaussian")
        warning.assert_called_once()

    async def test_utils_process_dataset_voronoi_point_cloud_data_model(self):
        operator_prim = self.stage.DefinePrim("/World/OperatorVoronoiPointCloud")
        cae_viz.DatasetSelectionAPI.Apply(operator_prim, "source")
        cae_viz.DatasetSelectionAPI(operator_prim, "source").CreateTargetRel().SetTargets(
            {self.datasets["disk_out_ref_npz"]["dataset"]}
        )
        cae_viz.FieldSelectionAPI.Apply(operator_prim, "Temp")
        cae_viz.FieldSelectionAPI(operator_prim, "Temp").CreateFieldNamesAttr().Set(["Temp"])

        source_dataset = await utils.get_input_dataset(
            operator_prim,
            "source",
            timeCode=Usd.TimeCode.EarliestTime(),
            device="cpu",
            required_fields={"Temp"},
        )

        cae_viz.DatasetVoronoiPointCloudAPI.Apply(operator_prim, "source")
        voronoi_dataset = await utils._process_dataset_voronoi_point_cloud(
            source_dataset, operator_prim, "source", Usd.TimeCode.EarliestTime()
        )

        self.assertEqual(voronoi_dataset.data_model, simdata_data_models_voronoi_point_cloud.DataModel)
        self.assertEqual(voronoi_dataset.get_num_nodes(), source_dataset.get_num_nodes())
        self.assertEqual(voronoi_dataset.get_num_elems(), source_dataset.get_num_nodes())
        self.assertTrue(voronoi_dataset.has_field("Temp"))

    async def test_utils_get_input_dataset_with_voronoi_point_cloud_without_voxelization(self):
        import os
        import tempfile

        n_points = 16
        coords = np.linspace(-1.0, 1.0, n_points * 3, dtype=np.float32).reshape(n_points, 3)
        temperature = np.linspace(0.0, 1.0, n_points, dtype=np.float32)

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            npz_path = f.name
            np.savez(f, coordinates=coords, temperature=temperature)

        try:
            dataset_prim = await import_to_stage(npz_path, "/World/VoronoiPointCloud", schema="Point Cloud")

            operator_prim = self.stage.DefinePrim("/World/OperatorVoronoiPointCloudNoVoxelization")
            cae_viz.DatasetSelectionAPI.Apply(operator_prim, "source")
            cae_viz.DatasetSelectionAPI(operator_prim, "source").CreateTargetRel().SetTargets({dataset_prim.GetPath()})
            cae_viz.DatasetVoronoiPointCloudAPI.Apply(operator_prim, "source")
            cae_viz.FieldSelectionAPI.Apply(operator_prim, "colors")
            cae_viz.FieldSelectionAPI(operator_prim, "colors").CreateFieldNamesAttr().Set(["temperature"])

            dataset = await utils.get_input_dataset(
                operator_prim,
                "source",
                timeCode=Usd.TimeCode.EarliestTime(),
                device="cpu",
                required_fields={"colors"},
            )

            self.assertEqual(dataset.data_model, simdata_data_models_voronoi_point_cloud.DataModel)
            self.assertEqual(dataset.get_num_nodes(), n_points)
            self.assertEqual(dataset.get_num_elems(), n_points)
            self.assertTrue(dataset.has_field("colors"))
        finally:
            os.unlink(npz_path)

    async def test_utils_get_input_dataset_with_subset(self):
        """Verify CaeVizDatasetSubsetAPI restricts the input dataset to cells inside the ROI bounds."""
        dataset_info = self.datasets["disk_out_ref_npz"]
        total_cells = dataset_info["nb_cells"]

        # Axis-aligned ROI cube centered at origin with extents (-2, 2) on each axis —
        # a strict subset of the dataset bounds (~ (-5.75, -5.75, -10) to (5.75, 5.75, 10.16)).
        roi_path = "/World/SubsetROI"
        roi_cube = UsdGeom.Cube.Define(self.stage, roi_path)
        roi_cube.CreateSizeAttr(4.0)

        counts = {}
        for mode in ("all", "any", "centroid"):
            operator_prim = self.stage.DefinePrim(f"/World/OperatorSubset_{mode}")
            cae_viz.DatasetSelectionAPI.Apply(operator_prim, "source")
            cae_viz.DatasetSelectionAPI(operator_prim, "source").CreateTargetRel().SetTargets({dataset_info["dataset"]})

            cae_viz.DatasetSubsetAPI.Apply(operator_prim, "source")
            subset_api = cae_viz.DatasetSubsetAPI(operator_prim, "source")
            subset_api.CreateRoiRel().SetTargets({roi_path})
            subset_api.CreateModeAttr().Set(mode)

            cae_viz.FieldSelectionAPI.Apply(operator_prim, "Temp")
            cae_viz.FieldSelectionAPI(operator_prim, "Temp").CreateFieldNamesAttr().Set(["Temp"])

            dataset = await utils.get_input_dataset(
                operator_prim,
                "source",
                timeCode=Usd.TimeCode.EarliestTime(),
                device="cuda:0",
                required_fields={"Temp"},
            )
            self.assertIsNotNone(dataset, f"mode={mode}: dataset should not be None")
            count = dataset.get_num_elems()
            counts[mode] = count

            self.assertGreater(count, 0, f"mode={mode}: subset should contain at least one cell")
            self.assertLess(count, total_cells, f"mode={mode}: subset should exclude at least one cell")
            self.assertTrue(dataset.has_field("Temp"), f"mode={mode}: Temp should pass through")
            # element_idx is an internal indirection used by pass_fields and must not leak to callers.
            self.assertFalse(dataset.has_field("element_idx"), f"mode={mode}: element_idx should not be exposed")

        # For convex cells inside a convex (AABB) ROI, every cell selected by "all" has its
        # centroid inside the box (centroid is a convex combination of vertices) and at least
        # one vertex inside (trivially). So "all" is a subset of both "centroid" and "any".
        self.assertLessEqual(counts["all"], counts["centroid"], '"all" should be a subset of "centroid"')
        self.assertLessEqual(counts["all"], counts["any"], '"all" should be a subset of "any"')

    async def test_utils_get_input_dataset_with_subset_inflate_bounds(self):
        """Verify inflateBounds expands the effective ROI before cell selection."""
        dataset_info = self.datasets["disk_out_ref_npz"]

        # Tight ROI — extents (-0.5, 0.5) on each axis.
        roi_path = "/World/SubsetROITight"
        roi_cube = UsdGeom.Cube.Define(self.stage, roi_path)
        roi_cube.CreateSizeAttr(1.0)

        def build_operator(name: str, inflate: int) -> Usd.Prim:
            operator_prim = self.stage.DefinePrim(f"/World/{name}")
            cae_viz.DatasetSelectionAPI.Apply(operator_prim, "source")
            cae_viz.DatasetSelectionAPI(operator_prim, "source").CreateTargetRel().SetTargets({dataset_info["dataset"]})
            cae_viz.DatasetSubsetAPI.Apply(operator_prim, "source")
            subset_api = cae_viz.DatasetSubsetAPI(operator_prim, "source")
            subset_api.CreateRoiRel().SetTargets({roi_path})
            subset_api.CreateModeAttr().Set("any")
            subset_api.CreateInflateBoundsAttr().Set(inflate)
            return operator_prim

        tight_prim = build_operator("OperatorSubsetTight", 0)
        inflated_prim = build_operator("OperatorSubsetInflated", 400)

        tight_ds = await utils.get_input_dataset(
            tight_prim, "source", timeCode=Usd.TimeCode.EarliestTime(), device="cuda:0"
        )
        inflated_ds = await utils.get_input_dataset(
            inflated_prim, "source", timeCode=Usd.TimeCode.EarliestTime(), device="cuda:0"
        )

        self.assertGreater(
            inflated_ds.get_num_elems(),
            tight_ds.get_num_elems(),
            "Inflating ROI bounds should select more cells",
        )
        self.assertLess(
            inflated_ds.get_num_elems(),
            dataset_info["nb_cells"],
            "Inflated ROI should still exclude some cells from the full dataset",
        )

    async def test_utils_get_input_dataset_caching(self):
        """Test that get_input_dataset properly caches and invalidates cache."""
        operator_prim = self.stage.DefinePrim("/World/OperatorTestCaching")

        # Setup dataset selection
        cae_viz.DatasetSelectionAPI.Apply(operator_prim, "source")
        dataset_selection_api = cae_viz.DatasetSelectionAPI(operator_prim, "source")
        dataset_selection_api.CreateTargetRel().SetTargets({self.datasets["disk_out_ref_npz"]["dataset"]})

        # Setup field selection
        cae_viz.FieldSelectionAPI.Apply(operator_prim, "Temp")
        field_selection_api = cae_viz.FieldSelectionAPI(operator_prim, "Temp")
        field_selection_api.CreateFieldNamesAttr().Set(["Temp"])

        # First call - should NOT be cached
        dataset1 = await utils.get_input_dataset(
            operator_prim,
            "source",
            timeCode=Usd.TimeCode.EarliestTime(),
            device="cpu",
            required_fields={"Temp"},
        )
        self.assertIsNotNone(dataset1, "First dataset should not be None")
        self.assertEqual(len(dataset1.get_field_names()), 1, "Should have 1 field")

        # Second call with same parameters - should be cached
        dataset2 = await utils.get_input_dataset(
            operator_prim,
            "source",
            timeCode=Usd.TimeCode.EarliestTime(),
            device="cpu",
            required_fields={"Temp"},
        )
        self.assertIsNotNone(dataset2, "Second dataset should not be None")
        # Verify it's the same cached object
        self.assertIs(dataset1, dataset2, "Second call should return cached dataset")

        # Modify a property on the prim - this should invalidate the cache
        field_selection_api.GetModeAttr().Set(cae_viz.Tokens.unchanged)
        await get_app().next_update_async()

        # Third call - should NOT be cached anymore due to prim modification
        dataset3 = await utils.get_input_dataset(
            operator_prim,
            "source",
            timeCode=Usd.TimeCode.EarliestTime(),
            device="cpu",
            required_fields={"Temp"},
        )
        self.assertIsNotNone(dataset3, "Third dataset should not be None")
        # This should be a new object since cache was invalidated
        self.assertIsNot(dataset1, dataset3, "Third call should return new dataset after cache invalidation")

        # Fourth call - should be cached again
        dataset4 = await utils.get_input_dataset(
            operator_prim,
            "source",
            timeCode=Usd.TimeCode.EarliestTime(),
            device="cpu",
            required_fields={"Temp"},
        )
        self.assertIs(dataset3, dataset4, "Fourth call should return cached dataset")

        # Test that different instance names use different cache entries
        # Note: Adding a new API to the prim will invalidate ALL caches for that prim
        # (as per the current implementation which uses prim-level invalidation)
        cae_viz.DatasetSelectionAPI.Apply(operator_prim, "other")
        dataset_selection_api_other = cae_viz.DatasetSelectionAPI(operator_prim, "other")
        dataset_selection_api_other.CreateTargetRel().SetTargets({self.datasets["disk_out_ref_tet_npz"]["dataset"]})
        await get_app().next_update_async()

        dataset_other = await utils.get_input_dataset(
            operator_prim,
            "other",
            timeCode=Usd.TimeCode.EarliestTime(),
            device="cpu",
            needs_fields=False,
        )
        self.assertIsNotNone(dataset_other, "Other instance dataset should not be None")
        self.assertIsNot(dataset3, dataset_other, "Different instance should have different cache entry")

        # Because we modified the prim by adding a new API, the cache for "source" was also invalidated
        # This is expected behavior with prim-level invalidation
        dataset5 = await utils.get_input_dataset(
            operator_prim,
            "source",
            timeCode=Usd.TimeCode.EarliestTime(),
            device="cpu",
            required_fields={"Temp"},
        )
        self.assertIsNot(dataset3, dataset5, "Adding new API invalidated all caches for this prim")

        # But now both should be cached
        dataset6 = await utils.get_input_dataset(
            operator_prim,
            "source",
            timeCode=Usd.TimeCode.EarliestTime(),
            device="cpu",
            required_fields={"Temp"},
        )
        self.assertIs(dataset5, dataset6, "Should be cached after re-fetch")

    async def test_utils_get_input_dataset_cache_invalidation_on_dataset_change(self):
        """Test that cache is invalidated when the underlying dataset prim changes."""
        operator_prim = self.stage.DefinePrim("/World/OperatorTestCacheDatasetChange")

        # Setup dataset selection
        cae_viz.DatasetSelectionAPI.Apply(operator_prim, "source")
        dataset_selection_api = cae_viz.DatasetSelectionAPI(operator_prim, "source")
        dataset_selection_api.CreateTargetRel().SetTargets({self.datasets["disk_out_ref_npz"]["dataset"]})

        # First call
        dataset1 = await utils.get_input_dataset(
            operator_prim,
            "source",
            timeCode=Usd.TimeCode.EarliestTime(),
            device="cpu",
            needs_fields=False,
        )
        self.assertIsNotNone(dataset1, "First dataset should not be None")
        self.assertEqual(
            dataset1.get_num_elems(),
            self.datasets["disk_out_ref_npz"]["nb_cells"],
            "Should have correct number of cells",
        )

        # Second call - should be cached
        dataset2 = await utils.get_input_dataset(
            operator_prim,
            "source",
            timeCode=Usd.TimeCode.EarliestTime(),
            device="cpu",
            needs_fields=False,
        )
        self.assertIs(dataset1, dataset2, "Should return cached dataset")

        # Modify the dataset prim (change the target to a different dataset)
        dataset_selection_api.GetTargetRel().SetTargets({self.datasets["disk_out_ref_tet_npz"]["dataset"]})
        await get_app().next_update_async()

        # Third call - should NOT be cached due to dataset change
        dataset3 = await utils.get_input_dataset(
            operator_prim,
            "source",
            timeCode=Usd.TimeCode.EarliestTime(),
            device="cpu",
            needs_fields=False,
        )
        self.assertIsNotNone(dataset3, "Third dataset should not be None")
        self.assertIsNot(dataset1, dataset3, "Should return new dataset after target change")
        self.assertEqual(
            dataset3.get_num_elems(),
            self.datasets["disk_out_ref_tet_npz"]["nb_cells"],
            "Should have correct number of cells from new dataset",
        )

    async def test_utils_get_input_dataset_cache_invalidation_on_roi_xform(self):
        """Cache must be invalidated when the ROI prim's xform changes.

        DatasetSubsetAPI/DatasetVoxelizationAPI derive their box from the ROI prim's world
        transform. Without a watch on the ROI prim itself, moving the ROI leaves the cached
        pre-move result in place and downstream operators run against stale data.
        """
        dataset_info = self.datasets["disk_out_ref_npz"]

        # Cube size 4.0 => extents (-2, 2) on each axis; strict subset of the dataset.
        roi_path = "/World/SubsetROIXform"
        roi_cube = UsdGeom.Cube.Define(self.stage, roi_path)
        roi_cube.CreateSizeAttr(4.0)
        translate_op = roi_cube.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(0.0, 0.0, 0.0))

        operator_prim = self.stage.DefinePrim("/World/OperatorSubsetROIXform")
        cae_viz.DatasetSelectionAPI.Apply(operator_prim, "source")
        cae_viz.DatasetSelectionAPI(operator_prim, "source").CreateTargetRel().SetTargets({dataset_info["dataset"]})
        cae_viz.DatasetSubsetAPI.Apply(operator_prim, "source")
        subset_api = cae_viz.DatasetSubsetAPI(operator_prim, "source")
        subset_api.CreateRoiRel().SetTargets({roi_path})
        subset_api.CreateModeAttr().Set("any")

        dataset1 = await utils.get_input_dataset(
            operator_prim, "source", timeCode=Usd.TimeCode.EarliestTime(), device="cuda:0", needs_fields=False
        )
        count1 = dataset1.get_num_elems()
        self.assertGreater(count1, 0, "Initial ROI should select at least one cell")

        # Sanity: repeated call hits the cache.
        dataset2 = await utils.get_input_dataset(
            operator_prim, "source", timeCode=Usd.TimeCode.EarliestTime(), device="cuda:0", needs_fields=False
        )
        self.assertIs(dataset1, dataset2, "Repeated call should return the cached dataset")

        # Shift the ROI along Z (dataset Z-extent is ~[-10, 10.16]) so the effective box moves to
        # [6, 10] in Z and selects a different set of cells.
        translate_op.Set(Gf.Vec3d(0.0, 0.0, 8.0))
        await get_app().next_update_async()

        dataset3 = await utils.get_input_dataset(
            operator_prim, "source", timeCode=Usd.TimeCode.EarliestTime(), device="cuda:0", needs_fields=False
        )
        self.assertIsNot(dataset1, dataset3, "Cache must be invalidated by ROI xform change")
        self.assertNotEqual(count1, dataset3.get_num_elems(), "Selected cell count must change when the ROI moves")


class TestRtSubPrimGuard(omni.kit.test.AsyncTestCase):
    """Tests for RtSubPrimGuard stage-transition correctness."""

    async def test_registry_cleared_on_stage_detach(self):
        """Registry is emptied when a stage is detached.

        Verifies that ``clear_all()`` is called via the stage-update subscription
        on stage close, so stale guards from the previous stage do not linger.
        """
        SOURCE_PATH = "/World/Source"
        RT_PATH = SdfRT.Path("/RT/SubPrim")

        async with new_stage() as stage:
            stage_id = omni.usd.get_context().get_stage_id()
            rt_stage = UsdRT.Stage.Attach(stage_id)

            source_prim = stage.DefinePrim(SOURCE_PATH, "Xform")
            rt_stage.DefinePrim(RT_PATH)
            RtSubPrimGuard.register(source_prim, rt_stage, [RT_PATH])

            self.assertIn(SOURCE_PATH, RtSubPrimGuard._registry, "Guard should be registered")

            del rt_stage  # release Fabric reference before stage closes

        # on_detach → clear_all() must have fired by the time close_stage_async() returns.
        self.assertEqual(len(RtSubPrimGuard._registry), 0, "Registry must be empty after stage detach")
