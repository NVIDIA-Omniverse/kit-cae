# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from logging import getLogger

import numpy as np
import omni.kit.app
import omni.kit.test
import omni.usd
import warp as wp
from omni.cae.sids import sids_unstructured
from omni.cae.testing import get_test_data_path
from pxr import Sdf, Usd

logger = getLogger(__name__)


class TestOmniCaeSIDSUnstructured(omni.kit.test.AsyncTestCase):

    def get_data_path(self, relative_path: str) -> str:
        "compute the absolute path of the stage"
        return get_test_data_path(relative_path)

    async def load(self, dataset: str):
        source_layer: Sdf.Layer = Sdf.Layer.FindOrOpen(self.get_data_path(dataset), {})
        stage = Usd.Stage.Open(source_layer)
        usd_context = omni.usd.get_context()
        await usd_context.attach_stage_async(stage)
        return stage

    async def test_uniform(self):
        stage = await self.load("hex_uniform.cgns")

        with wp.ScopedDevice("cpu"):
            prim = stage.GetPrimAtPath("/World/hex_uniform_cgns/Base/Zone/ElementsUniform")
            self.assertIsNotNone(prim)

            # force split into 2 chunks for testing
            elems = await sids_unstructured.get_sections(prim, Usd.TimeCode.EarliestTime(), 25)
            print(elems)
            self.assertEqual(len(elems), 2)
            self.assertEqual(elems[0].elementRange, wp.vec2i(1, 3))
            self.assertEqual(elems[0].elementStartOffsetShift, 0)
            self.assertTrue(
                np.array_equal(
                    elems[0].elementConnectivity.numpy(),
                    np.array([1, 2, 4, 3, 5, 6, 8, 7, 5, 6, 8, 7, 9, 10, 12, 11, 9, 10, 12, 11, 13, 14, 16, 15]),
                )
            )

            self.assertEqual(elems[1].elementRange, wp.vec2i(4, 5))
            self.assertEqual(elems[1].elementStartOffsetShift, 24)
            self.assertTrue(
                np.array_equal(
                    elems[1].elementConnectivity.numpy(),
                    np.array([13, 14, 16, 15, 17, 18, 20, 19, 17, 18, 20, 19, 21, 22, 24, 23]),
                )
            )

            # get single chunk and confirm the 2 chunks have all the data.
            elem = (await sids_unstructured.get_sections(prim, Usd.TimeCode.EarliestTime()))[0]
            self.assertEqual(elem.elementRange[0], elems[0].elementRange[0])
            self.assertEqual(elem.elementRange[1], elems[1].elementRange[1])
            self.assertTrue(
                np.array_equal(
                    elem.elementConnectivity.numpy(),
                    np.concatenate(
                        (elems[0].elementConnectivity.numpy(), elems[1].elementConnectivity.numpy()), axis=None
                    ),
                )
            )

            vtx_ids = await sids_unstructured.get_grid_coordinates_ids(prim, Usd.TimeCode.EarliestTime())
            self.assertTrue(np.array_equal(vtx_ids.numpy(), np.arange(1, 25)))

            coords = await sids_unstructured.get_grid_coordinates(prim, Usd.TimeCode.EarliestTime())
            self.assertEqual(coords.shape, (24,))
            self.assertEqual(coords.numpy().shape, (24, 3))

            centers = await sids_unstructured.get_cell_centers(prim, Usd.TimeCode.EarliestTime())
            self.assertTrue(
                np.array_equal(
                    centers.numpy(),
                    np.array(
                        [
                            [0.5, 0.5, 0.5],
                            [0.5, 0.5, 1.5],
                            [0.5, 0.5, 2.5],
                            [0.5, 0.5, 3.5],
                            [0.5, 0.5, 4.5],
                        ]
                    ),
                )
            )

            field_prim = stage.GetPrimAtPath(
                "/World/hex_uniform_cgns/Base/Zone/SolutionCellCenter/CellDistanceToCenter"
            )
            self.assertIsNotNone(field_prim)
            c2p = (await sids_unstructured.get_cell_field_as_point(prim, [field_prim]))[0]
            self.assertAlmostEqual(np.amin(c2p.numpy(), axis=0), 1.064, 2)
            self.assertAlmostEqual(np.amax(c2p.numpy(), axis=0), 2.1281943, 2)

    async def test_mixed(self):
        stage = await self.load("hex_mixed.cgns")

        with wp.ScopedDevice("cpu"):
            prim = stage.GetPrimAtPath("/World/hex_mixed_cgns/Base/Zone/ElementsMixed")
            self.assertIsNotNone(prim)

            # force split into 2 chunks for testing
            elems = await sids_unstructured.get_sections(prim, Usd.TimeCode.EarliestTime(), 25)
            print(elems)
            self.assertEqual(len(elems), 2)
            self.assertEqual(elems[0].elementRange, wp.vec2i(1, 3))
            self.assertEqual(elems[0].elementStartOffsetShift, 0)
            self.assertEqual(
                elems[0].elementStartOffset.numpy()[-1] - elems[0].elementStartOffsetShift,
                elems[0].elementConnectivity.shape[0],
            )

            self.assertEqual(elems[1].elementRange, wp.vec2i(4, 5))
            self.assertEqual(elems[1].elementStartOffsetShift, 27)
            self.assertEqual(
                elems[1].elementStartOffset.numpy()[-1] - elems[1].elementStartOffsetShift,
                elems[1].elementConnectivity.shape[0],
            )
            # print(elems)

            # get single chunk and confirm the 2 chunks have all the data.
            elem = (await sids_unstructured.get_sections(prim, Usd.TimeCode.EarliestTime()))[0]
            print(elem)
            self.assertEqual(elem.elementRange[0], elems[0].elementRange[0])
            self.assertEqual(elem.elementRange[1], elems[1].elementRange[1])
            self.assertTrue(
                np.array_equal(
                    elem.elementConnectivity.numpy(),
                    np.concatenate(
                        (elems[0].elementConnectivity.numpy(), elems[1].elementConnectivity.numpy()), axis=None
                    ),
                )
            )
            self.assertTrue(
                np.array_equal(
                    elems[0].elementConnectivity.numpy(),
                    elem.elementConnectivity.numpy()[
                        elems[0].elementStartOffset.numpy()[0] : elems[0].elementStartOffset.numpy()[-1]
                    ],
                )
            )
            self.assertTrue(
                np.array_equal(
                    elems[1].elementConnectivity.numpy(),
                    elem.elementConnectivity.numpy()[
                        elems[1].elementStartOffset.numpy()[0] : elems[1].elementStartOffset.numpy()[-1]
                    ],
                )
            )

            vtx_ids = await sids_unstructured.get_grid_coordinates_ids(prim, Usd.TimeCode.EarliestTime())
            self.assertTrue(np.array_equal(vtx_ids.numpy(), np.arange(1, 25)))

            coords = await sids_unstructured.get_grid_coordinates(prim, Usd.TimeCode.EarliestTime())
            self.assertEqual(coords.shape, (24,))
            self.assertEqual(coords.numpy().shape, (24, 3))

            centers = await sids_unstructured.get_cell_centers(prim, Usd.TimeCode.EarliestTime())
            self.assertTrue(
                np.array_equal(
                    centers.numpy(),
                    np.array(
                        [
                            [0.5, 0.5, 0.5],
                            [0.5, 0.5, 1.5],
                            [0.5, 0.5, 2.5],
                            [0.5, 0.5, 3.5],
                            [0.5, 0.5, 4.5],
                        ]
                    ),
                )
            )

            field_prim = stage.GetPrimAtPath("/World/hex_mixed_cgns/Base/Zone/SolutionCellCenter/CellDistanceToCenter")
            self.assertIsNotNone(field_prim)
            c2p = (await sids_unstructured.get_cell_field_as_point(prim, [field_prim]))[0]
            self.assertAlmostEqual(np.amin(c2p.numpy(), axis=0), 1.064, 2)
            self.assertAlmostEqual(np.amax(c2p.numpy(), axis=0), 2.1281943, 2)

    async def test_mixed_static_mixer(self):
        stage = await self.load("StaticMixer.cgns")

        with wp.ScopedDevice("cpu"):
            prim = stage.GetPrimAtPath("/World/StaticMixer_cgns/Base/StaticMixer/B1_P3")
            self.assertIsNotNone(prim)
            elems = await sids_unstructured.get_sections(prim, Usd.TimeCode.EarliestTime())
            self.assertEqual(len(elems), 1)

            centers = await sids_unstructured.get_cell_centers(prim, Usd.TimeCode.EarliestTime())
            self.assertEqual(centers.shape, (13761,))
            self.assertEqual(centers.numpy().shape, (13761, 3))

    async def test_wedge_polyhedra(self):
        stage = await self.load("wedge_polyhedra.cgns")
        tc = Usd.TimeCode.EarliestTime()

        with wp.ScopedDevice("cpu"):
            prim = stage.GetPrimAtPath("/World/wedge_polyhedra_cgns/Base/Zone/ElementsNfaces")
            self.assertIsNotNone(prim)
            elems = await sids_unstructured.get_sections(prim, tc)
            self.assertEqual(len(elems), 1)

            centers = await sids_unstructured.get_cell_centers(prim, tc)
            self.assertEqual(centers.shape, (65536,))
            self.assertEqual(centers.numpy().shape, (65536, 3))
            self.assertTrue(np.allclose(np.amin(centers.numpy(), axis=0), np.array([0.33333334, 0.33333334, 0.5])))
            self.assertTrue(np.allclose(np.amax(centers.numpy(), axis=0), np.array([31.66666, 31.66666, 31.5])))

            field_prim = stage.GetPrimAtPath(
                "/World/wedge_polyhedra_cgns/Base/Zone/SolutionCellCenter/CellDistanceToCenter"
            )
            self.assertIsNotNone(field_prim)
            c2p = (await sids_unstructured.get_cell_field_as_point(prim, [field_prim]))[0]
            self.assertAlmostEqual(np.amin(c2p.numpy(), axis=0), 1.052, 2)
            self.assertAlmostEqual(np.amax(c2p.numpy(), axis=0), 27.049, 2)
