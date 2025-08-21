# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
#  its affiliates is strictly prohibited.

import logging
import threading
from time import sleep

import numpy as np
import omni.kit.app
import omni.kit.test
import omni.usd
from omni.cae.data import get_data_delegate_registry
from omni.cae.data.delegates import DataDelegateBase
from omni.cae.schema import cae
from pxr import Sdf, Usd, UsdGeom

logger = logging.getLogger(__name__)


class TestDataDelegate(DataDelegateBase):
    def __init__(self, extensionId: str, array: np.ndarray):
        super().__init__(extensionId)
        self._array = array

    def __del__(self):
        logger.info("TestDataDelegate.__del__")

    def get_field_array(self, prim: Usd.Prim, time: Usd.TimeCode) -> np.ndarray:
        logger.info("TestDataDelegate.get_field_array(%s,m %f)", str(prim.GetPath()), time.GetValue())
        return self._array

    def can_provide(self, prim: Usd.Prim) -> bool:
        logger.info("TestDataDelegate.can_provide(%s)", str(prim.GetPath()))
        return True


class TestLargeDataDelegate(DataDelegateBase):
    def __init__(self, extensionId: str):
        super().__init__(extensionId)

    def __del__(self):
        logger.info("TestDataDelegate.__del__")

    def get_field_array(self, prim: Usd.Prim, time: Usd.TimeCode) -> np.ndarray:
        assert threading.current_thread() != threading.main_thread(), "should not be called from main thread"

        delay = 2
        logger.info(f"sleeping for {delay} seconds")
        sleep(delay)
        logger.info("TestDataDelegate.get_field_array(%s,m %f)", str(prim.GetPath()), time.GetValue())
        return np.ones((1024, 3), dtype=np.float32)

    def can_provide(self, prim: Usd.Prim) -> bool:
        logger.info("TestDataDelegate.can_provide(%s)", str(prim.GetPath()))
        return True


class TestOmniCaeDataDelegate(omni.kit.test.AsyncTestCase):
    async def setUp(self):
        self.fail_n_log_error = True
        self._registry = get_data_delegate_registry()
        self._ext_id = "omni.cae.data.tests.test_data_delegate"

    async def tearDown(self):
        self._registry.deregister_all_data_delegates_for_extension(self._ext_id)
        del self._registry

    async def test_data_delegate_callbacks(self):
        stage = Usd.Stage.CreateInMemory()
        xform = UsdGeom.Xform.Define(stage, "/Root")
        fieldArray0 = cae.FieldArray.Define(stage, "/Root/Array0")
        fieldArray1 = cae.FieldArray.Define(stage, "/Root/Array1")

        await omni.usd.get_context().attach_stage_async(stage)

        data0 = np.random.default_rng(1212).normal(loc=10, scale=2, size=(1024, 3))
        logger.info(f"data0: {data0}")
        delegate = TestDataDelegate(self._ext_id, data0)
        self._registry.register_data_delegate(delegate)
        result = self._registry.get_field_array(fieldArray0.GetPrim(), Usd.TimeCode.EarliestTime())

        logger.info(f"result: {result}")
        self.assertEqual(np.array_equal(data0, result), True)
        self.assertEqual(np.array_equal(data0.shape, result.shape), True)
        self.assertEqual(np.array_equal(data0.dtype, result.dtype), True)
        self.assertIsNotNone(np.asarray(result).base)
        self.assertIsNotNone(result.numpy().base)

        # try get_field on non-field-array prim
        self.assertIsNone(self._registry.get_field_array(xform.GetPrim(), Usd.TimeCode.EarliestTime()))

        # remove delegate
        self._registry.deregister_data_delegate(delegate)

        # HACK: need to handle preserving object in Python wrapping code.
        # For now, we expect the delegate to be kept alive in Python until
        # `deregister_data_delegate`.
        del delegate

        # try array1, it should fail.
        self.assertIsNone(self._registry.get_field_array(fieldArray1.GetPrim(), Usd.TimeCode.EarliestTime()))

        # now try getting the array0 again, it should use cached value
        self.assertTrue(self._registry.is_field_array_cached(fieldArray0.GetPrim(), Usd.TimeCode.EarliestTime()))
        self.assertIsNotNone(self._registry.get_field_array(fieldArray0.GetPrim(), Usd.TimeCode.EarliestTime()))

        # Do some unrelated change, should not invalidate fieldArray0
        xform.AddTranslateOp().Set((0, 0, 0))
        self.assertTrue(self._registry.is_field_array_cached(fieldArray0.GetPrim(), Usd.TimeCode.EarliestTime()))

        # now do some change to fieldArray0 so it is evicted from the cache
        fieldArray0.CreateFileNamesAttr().Set(["/somewhere"])
        self.assertFalse(self._registry.is_field_array_cached(fieldArray0.GetPrim(), Usd.TimeCode.EarliestTime()))

        omni.usd.get_context().close_stage()
        del stage

    async def test_async_data_delegate(self):
        delegate = TestLargeDataDelegate(self._ext_id)
        self._registry.register_data_delegate(delegate)

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.Xform.Define(stage, "/Root")
        fieldArray0 = cae.FieldArray.Define(stage, "/Root/Array0")
        await omni.usd.get_context().attach_stage_async(stage)

        array = await self._registry.get_field_array_async(fieldArray0.GetPrim(), Usd.TimeCode.EarliestTime())

        self.assertIsNotNone(array)
        self.assertEqual(array.shape, [1024, 3])

        self._registry.deregister_data_delegate(delegate)
        omni.usd.get_context().close_stage()
        del stage
