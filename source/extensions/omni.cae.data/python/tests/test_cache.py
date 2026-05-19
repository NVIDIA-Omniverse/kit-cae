# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import logging

import carb.settings
import omni.kit.app
import omni.kit.test
import omni.usd
from omni.cae.data.impl import cache
from omni.cae.schema import cae
from omni.cae.schema import viz as cae_viz
from pxr import Sdf, Usd, UsdGeom

logger = logging.getLogger(__name__)


class TestCache(omni.kit.test.AsyncTestCase):
    async def setUp(self):
        """Set up test fixtures"""
        cache._initialize()
        settings = carb.settings.get_settings()
        self._cache_mode_key = "/persistent/exts/omni.cae.data/cacheMode"
        self._original_cache_mode = settings.get_as_string(self._cache_mode_key)
        settings.set_string(self._cache_mode_key, "always")

    async def tearDown(self):
        """Clean up after tests"""
        settings = carb.settings.get_settings()
        settings.set_string(self._cache_mode_key, self._original_cache_mode)
        cache.clear()
        cache._finalize()
        ctx = omni.usd.get_context()
        if ctx.get_stage():
            ctx.close_stage()

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    async def _attach_stage(self, stage):
        await omni.usd.get_context().attach_stage_async(stage)

    async def _next_frame(self):
        await omni.kit.app.get_app().next_update_async()

    # -----------------------------------------------------------------------
    # Legacy put() API — existing tests kept intact
    # -----------------------------------------------------------------------

    async def test_source_prim_field_array_modification_drops_cache(self):
        """
        Test that when a sourcePrim is a Dataset prim with a field relationship
        pointing to FieldArray, and the FieldArray is modified, the cache should
        be automatically dropped.
        """
        stage = Usd.Stage.CreateInMemory()
        dataset = cae.DataSet.Define(stage, "/Root/DataSet")
        field_array = cae.FieldArray.Define(stage, "/Root/DataSet/Field1")

        rel = dataset.GetPrim().CreateRelationship("field:Field1")
        rel.AddTarget(field_array.GetPrim().GetPath())

        await self._attach_stage(stage)

        cache_key = "test_key_1"
        test_data = {"value": 42}
        cache.put(
            cache_key,
            test_data,
            sourcePrims=[dataset.GetPrim()],
            consumerPrims=[],
            force=True,
        )

        self.assertIsNotNone(cache.get(cache_key))
        self.assertEqual(cache.get(cache_key), test_data)

        field_array.CreateFileNamesAttr().Set(["/some/path/to/file"])

        await self._next_frame()

        self.assertIsNone(cache.get(cache_key))

    async def test_source_prim_field_array_property_change_drops_cache(self):
        """
        Test that property changes on a FieldArray that is a target of a
        field relationship from a source Dataset prim causes cache invalidation.
        """
        stage = Usd.Stage.CreateInMemory()
        dataset = cae.DataSet.Define(stage, "/Root/DataSet")
        field_array = cae.FieldArray.Define(stage, "/Root/DataSet/Field1")

        rel = dataset.GetPrim().CreateRelationship("field:Field1")
        rel.AddTarget(field_array.GetPrim().GetPath())

        await self._attach_stage(stage)

        cache_key = "test_key_2"
        test_data = {"value": 100}
        cache.put(
            cache_key,
            test_data,
            sourcePrims=[dataset.GetPrim()],
            consumerPrims=[],
            force=True,
        )

        self.assertIsNotNone(cache.get(cache_key))

        attr = field_array.GetPrim().CreateAttribute("testAttr", Sdf.ValueTypeNames.Float)
        attr.Set(123.0)

        await self._next_frame()

        self.assertIsNone(cache.get(cache_key))

    async def test_consumer_prim_property_change_does_not_drop_cache(self):
        """
        Test that when a consumer prim is modified (property change only),
        the cache should NOT be dropped.
        """
        stage = Usd.Stage.CreateInMemory()
        consumer_prim = UsdGeom.Xform.Define(stage, "/Root/Consumer")

        await self._attach_stage(stage)

        cache_key = "test_key_3"
        test_data = {"value": 200}
        cache.put(
            cache_key,
            test_data,
            sourcePrims=[],
            consumerPrims=[consumer_prim.GetPrim()],
            force=True,
        )

        self.assertIsNotNone(cache.get(cache_key))

        consumer_prim.AddTranslateOp().Set((1.0, 2.0, 3.0))

        await self._next_frame()

        self.assertIsNotNone(cache.get(cache_key))
        self.assertEqual(cache.get(cache_key), test_data)

    async def test_consumer_prim_resync_drops_cache(self):
        """
        Test that when a consumer prim is resynced (e.g., API schema added),
        the cache should be dropped.
        """
        stage = Usd.Stage.CreateInMemory()
        consumer_prim = UsdGeom.Mesh.Define(stage, "/Root/Consumer")

        await self._attach_stage(stage)

        cache_key = "test_key_4"
        test_data = {"value": 300}
        cache.put(
            cache_key,
            test_data,
            sourcePrims=[],
            consumerPrims=[consumer_prim.GetPrim()],
            force=True,
        )

        self.assertIsNotNone(cache.get(cache_key))

        cae_viz.FacesAPI.Apply(consumer_prim.GetPrim())

        await self._next_frame()

        self.assertIsNone(cache.get(cache_key))

    async def test_consumer_prim_deletion_drops_cache(self):
        """
        Test that when a consumer prim is deleted, the cache should be dropped.
        """
        stage = Usd.Stage.CreateInMemory()
        consumer_prim = UsdGeom.Xform.Define(stage, "/Root/Consumer")

        await self._attach_stage(stage)

        cache_key = "test_key_5"
        test_data = {"value": 400}
        cache.put(
            cache_key,
            test_data,
            sourcePrims=[],
            consumerPrims=[consumer_prim.GetPrim()],
            force=True,
        )

        self.assertIsNotNone(cache.get(cache_key))

        stage.RemovePrim(consumer_prim.GetPrim().GetPath())

        await self._next_frame()

        self.assertIsNone(cache.get(cache_key))

    async def test_source_prim_resync_drops_cache(self):
        """
        Test that when a source prim is resynced, the cache should be dropped.
        """
        stage = Usd.Stage.CreateInMemory()
        dataset = cae.DataSet.Define(stage, "/Root/DataSet")

        await self._attach_stage(stage)

        cache_key = "test_key_6"
        test_data = {"value": 500}
        cache.put(
            cache_key,
            test_data,
            sourcePrims=[dataset.GetPrim()],
            consumerPrims=[],
            force=True,
        )

        self.assertIsNotNone(cache.get(cache_key))

        cae_viz.DatasetSelectionAPI.Apply(dataset.GetPrim(), "foo")

        await self._next_frame()

        self.assertIsNone(cache.get(cache_key))

    async def test_source_prim_property_change_drops_cache(self):
        """
        Test that when a source prim's property is changed, the cache should be dropped.
        """
        stage = Usd.Stage.CreateInMemory()
        dataset = cae.DataSet.Define(stage, "/Root/DataSet")

        await self._attach_stage(stage)

        cache_key = "test_key_7"
        test_data = {"value": 600}
        cache.put(
            cache_key,
            test_data,
            sourcePrims=[dataset.GetPrim()],
            consumerPrims=[],
            force=True,
        )

        self.assertIsNotNone(cache.get(cache_key))

        attr = dataset.GetPrim().CreateAttribute("testAttr", Sdf.ValueTypeNames.String)
        attr.Set("test_value")

        await self._next_frame()

        self.assertIsNone(cache.get(cache_key))

    async def test_multiple_source_prims_one_change_drops_cache(self):
        """
        Test that when multiple source prims are tracked, changing one drops the cache.
        """
        stage = Usd.Stage.CreateInMemory()
        dataset1 = cae.DataSet.Define(stage, "/Root/DataSet1")
        dataset2 = cae.DataSet.Define(stage, "/Root/DataSet2")

        await self._attach_stage(stage)

        cache_key = "test_key_8"
        test_data = {"value": 700}
        cache.put(
            cache_key,
            test_data,
            sourcePrims=[dataset1.GetPrim(), dataset2.GetPrim()],
            consumerPrims=[],
            force=True,
        )

        self.assertIsNotNone(cache.get(cache_key))

        attr = dataset1.GetPrim().CreateAttribute("testAttr", Sdf.ValueTypeNames.Float)
        attr.Set(123.0)

        await self._next_frame()

        self.assertIsNone(cache.get(cache_key))

    async def test_multiple_consumer_prims_one_resync_drops_cache(self):
        """
        Test that when multiple consumer prims are tracked, resyncing one drops the cache.
        """
        stage = Usd.Stage.CreateInMemory()
        consumer1 = UsdGeom.Mesh.Define(stage, "/Root/Consumer1")
        consumer2 = UsdGeom.Mesh.Define(stage, "/Root/Consumer2")

        await self._attach_stage(stage)

        cache_key = "test_key_9"
        test_data = {"value": 800}
        cache.put(
            cache_key,
            test_data,
            sourcePrims=[],
            consumerPrims=[consumer1.GetPrim(), consumer2.GetPrim()],
            force=True,
        )

        self.assertIsNotNone(cache.get(cache_key))

        cae_viz.FacesAPI.Apply(consumer1.GetPrim())

        await self._next_frame()

        self.assertIsNone(cache.get(cache_key))

    async def test_field_array_relationship_target_modification_drops_cache(self):
        """
        Test that modifying a FieldArray that is a target of a field relationship
        from a Dataset source prim causes cache invalidation.
        """
        stage = Usd.Stage.CreateInMemory()
        dataset = cae.DataSet.Define(stage, "/Root/DataSet")
        field_array = cae.FieldArray.Define(stage, "/Root/DataSet/Field1")

        rel = dataset.GetPrim().CreateRelationship("field:Field1")
        rel.AddTarget(field_array.GetPrim().GetPath())

        await self._attach_stage(stage)

        cache_key = "test_key_10"
        test_data = {"value": 900}
        cache.put(
            cache_key,
            test_data,
            sourcePrims=[dataset.GetPrim()],
            consumerPrims=[],
            force=True,
        )

        self.assertIsNotNone(cache.get(cache_key))

        field_array.CreateFileNamesAttr().Set(["/modified/path"])

        await self._next_frame()

        self.assertIsNone(
            cache.get(cache_key),
            "Cache should be dropped when a FieldArray target of a field relationship is modified",
        )

    # -----------------------------------------------------------------------
    # put_ex() — on="any" mode (mirrors old sourcePrims behaviour)
    # -----------------------------------------------------------------------

    async def test_put_ex_any_mode_property_update_drops_cache(self):
        """on="any": a property change on the watched prim drops the cache."""
        stage = Usd.Stage.CreateInMemory()
        field_array = cae.FieldArray.Define(stage, "/Root/Field")

        await self._attach_stage(stage)

        key = "ex_any_update"
        cache.put_ex(key, {"v": 1}, prims=[cache.PrimWatch(field_array.GetPrim(), on="any")], force=True)
        self.assertIsNotNone(cache.get(key))

        field_array.CreateFileNamesAttr().Set(["/a/path"])

        await self._next_frame()

        self.assertIsNone(cache.get(key))

    async def test_put_ex_any_mode_structural_resync_drops_cache(self):
        """on="any": a structural resync (API schema applied) drops the cache."""
        stage = Usd.Stage.CreateInMemory()
        mesh = UsdGeom.Mesh.Define(stage, "/Root/Mesh")

        await self._attach_stage(stage)

        key = "ex_any_resync"
        cache.put_ex(key, {"v": 1}, prims=[cache.PrimWatch(mesh.GetPrim(), on="any")], force=True)
        self.assertIsNotNone(cache.get(key))

        cae_viz.FacesAPI.Apply(mesh.GetPrim())

        await self._next_frame()

        self.assertIsNone(cache.get(key))

    # -----------------------------------------------------------------------
    # put_ex() — on="update" mode
    # -----------------------------------------------------------------------

    async def test_put_ex_update_mode_property_change_drops_cache(self):
        """on="update": a property value change drops the cache."""
        stage = Usd.Stage.CreateInMemory()
        field_array = cae.FieldArray.Define(stage, "/Root/Field")

        await self._attach_stage(stage)

        key = "ex_update_prop"
        cache.put_ex(key, {"v": 1}, prims=[cache.PrimWatch(field_array.GetPrim(), on="update")], force=True)
        self.assertIsNotNone(cache.get(key))

        field_array.CreateFileNamesAttr().Set(["/a/path"])

        await self._next_frame()

        self.assertIsNone(cache.get(key))

    async def test_put_ex_update_mode_structural_resync_does_not_drop_cache(self):
        """on="update": a structural resync alone does NOT drop the cache."""
        stage = Usd.Stage.CreateInMemory()
        mesh = UsdGeom.Mesh.Define(stage, "/Root/Mesh")

        await self._attach_stage(stage)

        key = "ex_update_resync"
        cache.put_ex(key, {"v": 1}, prims=[cache.PrimWatch(mesh.GetPrim(), on="update")], force=True)
        self.assertIsNotNone(cache.get(key))

        # Apply an API schema — causes a structural resync but no property update.
        cae_viz.FacesAPI.Apply(mesh.GetPrim())

        await self._next_frame()

        self.assertIsNotNone(cache.get(key), "on='update' should not drop cache on structural resync")

    # -----------------------------------------------------------------------
    # put_ex() — on="resync" mode (mirrors old consumerPrims behaviour)
    # -----------------------------------------------------------------------

    async def test_put_ex_resync_mode_property_change_does_not_drop_cache(self):
        """on="resync": a property-only change does NOT drop the cache."""
        stage = Usd.Stage.CreateInMemory()
        xform = UsdGeom.Xform.Define(stage, "/Root/Xform")

        await self._attach_stage(stage)

        key = "ex_resync_prop"
        cache.put_ex(key, {"v": 1}, prims=[cache.PrimWatch(xform.GetPrim(), on="resync")], force=True)
        self.assertIsNotNone(cache.get(key))

        xform.AddTranslateOp().Set((1.0, 2.0, 3.0))

        await self._next_frame()

        self.assertIsNotNone(cache.get(key), "on='resync' should not drop cache on property-only change")

    async def test_put_ex_resync_mode_structural_resync_drops_cache(self):
        """on="resync": a structural resync drops the cache."""
        stage = Usd.Stage.CreateInMemory()
        mesh = UsdGeom.Mesh.Define(stage, "/Root/Mesh")

        await self._attach_stage(stage)

        key = "ex_resync_resync"
        cache.put_ex(key, {"v": 1}, prims=[cache.PrimWatch(mesh.GetPrim(), on="resync")], force=True)
        self.assertIsNotNone(cache.get(key))

        cae_viz.FacesAPI.Apply(mesh.GetPrim())

        await self._next_frame()

        self.assertIsNone(cache.get(key))

    # -----------------------------------------------------------------------
    # put_ex() — on="delete" mode
    # -----------------------------------------------------------------------

    async def test_put_ex_delete_mode_property_change_does_not_drop_cache(self):
        """on="delete": a property change does NOT drop the cache."""
        stage = Usd.Stage.CreateInMemory()
        xform = UsdGeom.Xform.Define(stage, "/Root/Xform")

        await self._attach_stage(stage)

        key = "ex_delete_prop"
        cache.put_ex(key, {"v": 1}, prims=[cache.PrimWatch(xform.GetPrim(), on="delete")], force=True)
        self.assertIsNotNone(cache.get(key))

        xform.AddTranslateOp().Set((1.0, 2.0, 3.0))

        await self._next_frame()

        self.assertIsNotNone(cache.get(key), "on='delete' should not drop cache on property change")

    async def test_put_ex_delete_mode_resync_without_deletion_does_not_drop_cache(self):
        """on="delete": a structural resync where the prim still exists does NOT drop the cache."""
        stage = Usd.Stage.CreateInMemory()
        mesh = UsdGeom.Mesh.Define(stage, "/Root/Mesh")

        await self._attach_stage(stage)

        key = "ex_delete_resync_no_del"
        cache.put_ex(key, {"v": 1}, prims=[cache.PrimWatch(mesh.GetPrim(), on="delete")], force=True)
        self.assertIsNotNone(cache.get(key))

        # Structural resync but the prim is NOT deleted.
        cae_viz.FacesAPI.Apply(mesh.GetPrim())

        await self._next_frame()

        self.assertIsNotNone(cache.get(key), "on='delete' should not drop cache on resync without deletion")

    async def test_put_ex_delete_mode_prim_deletion_drops_cache(self):
        """on="delete": deleting the prim drops the cache."""
        stage = Usd.Stage.CreateInMemory()
        xform = UsdGeom.Xform.Define(stage, "/Root/Xform")

        await self._attach_stage(stage)

        key = "ex_delete_deleted"
        cache.put_ex(key, {"v": 1}, prims=[cache.PrimWatch(xform.GetPrim(), on="delete")], force=True)
        self.assertIsNotNone(cache.get(key))

        stage.RemovePrim(xform.GetPrim().GetPath())

        await self._next_frame()

        self.assertIsNone(cache.get(key))

    # -----------------------------------------------------------------------
    # put_ex() — schema filtering
    # -----------------------------------------------------------------------

    async def test_put_ex_schema_filter_matching_property_drops_cache(self):
        """schemas filter: a change to a schema-declared property drops the cache."""
        stage = Usd.Stage.CreateInMemory()
        field_array = cae.FieldArray.Define(stage, "/Root/Field")

        await self._attach_stage(stage)

        key = "ex_schema_match"
        cache.put_ex(
            key,
            {"v": 1},
            prims=[cache.PrimWatch(field_array.GetPrim(), on="update", schemas=[cae.FieldArray])],
            force=True,
        )
        self.assertIsNotNone(cache.get(key))

        # fileNames is declared by CaeFieldArray — should trigger invalidation.
        field_array.CreateFileNamesAttr().Set(["/changed/path"])

        await self._next_frame()

        self.assertIsNone(cache.get(key))

    async def test_put_ex_schema_filter_non_matching_property_does_not_drop_cache(self):
        """schemas filter: a change to a property NOT in the schema does NOT drop the cache."""
        stage = Usd.Stage.CreateInMemory()
        field_array = cae.FieldArray.Define(stage, "/Root/Field")

        await self._attach_stage(stage)

        key = "ex_schema_no_match"
        cache.put_ex(
            key,
            {"v": 1},
            prims=[cache.PrimWatch(field_array.GetPrim(), on="update", schemas=[cae.FieldArray])],
            force=True,
        )
        self.assertIsNotNone(cache.get(key))

        # customAttr is NOT declared by CaeFieldArray — should NOT trigger invalidation.
        field_array.GetPrim().CreateAttribute("customAttr", Sdf.ValueTypeNames.Float).Set(99.0)

        await self._next_frame()

        self.assertIsNotNone(cache.get(key), "Non-schema property change should not drop cache")

    async def test_put_ex_schema_filter_any_mode_resync_still_drops_cache(self):
        """schemas filter with on="any": a structural resync drops the cache even with a schema filter."""
        stage = Usd.Stage.CreateInMemory()
        field_array = cae.FieldArray.Define(stage, "/Root/Field")

        await self._attach_stage(stage)

        key = "ex_schema_any_resync"
        cache.put_ex(
            key,
            {"v": 1},
            prims=[cache.PrimWatch(field_array.GetPrim(), on="any", schemas=[cae.FieldArray])],
            force=True,
        )
        self.assertIsNotNone(cache.get(key))

        # Structural resync — schema filter does not block this for on="any".
        cae_viz.DatasetSelectionAPI.Apply(field_array.GetPrim(), "bar")

        await self._next_frame()

        self.assertIsNone(cache.get(key), "Structural resync should drop cache even with schema filter")

    async def test_put_ex_schema_filter_string_name(self):
        """schemas filter: schema type names can be provided as strings."""
        stage = Usd.Stage.CreateInMemory()
        field_array = cae.FieldArray.Define(stage, "/Root/Field")

        await self._attach_stage(stage)

        key = "ex_schema_str"
        cache.put_ex(
            key,
            {"v": 1},
            prims=[cache.PrimWatch(field_array.GetPrim(), on="update", schemas=["CaeFieldArray"])],
            force=True,
        )
        self.assertIsNotNone(cache.get(key))

        # Schema-declared property — should trigger.
        field_array.CreateFileNamesAttr().Set(["/str/path"])

        await self._next_frame()

        self.assertIsNone(cache.get(key))

    # -----------------------------------------------------------------------
    # put_ex() — multi-apply schema filtering (tuple and qualified-string forms)
    # -----------------------------------------------------------------------

    async def test_put_ex_multi_apply_tuple_matching_instance_drops_cache(self):
        """Multi-apply tuple: a property change on the watched instance drops the cache."""
        stage = Usd.Stage.CreateInMemory()
        mesh = UsdGeom.Mesh.Define(stage, "/Root/Mesh")
        cae_viz.DatasetTransformingAPI.Apply(mesh.GetPrim(), "source")

        await self._attach_stage(stage)

        key = "ex_multi_tuple_match"
        cache.put_ex(
            key,
            {"v": 1},
            prims=[cache.PrimWatch(mesh.GetPrim(), on="update", schemas=[(cae_viz.DatasetTransformingAPI, "source")])],
            force=True,
        )
        self.assertIsNotNone(cache.get(key))

        # Change a property on the "source" instance — should trigger invalidation.
        mesh.GetPrim().GetAttribute("cae:viz:dataset_transforming:source:useGlobalTransform").Set(True)

        await self._next_frame()

        self.assertIsNone(cache.get(key))

    async def test_put_ex_multi_apply_tuple_other_instance_does_not_drop_cache(self):
        """Multi-apply tuple: a property change on a different instance does NOT drop the cache."""
        stage = Usd.Stage.CreateInMemory()
        mesh = UsdGeom.Mesh.Define(stage, "/Root/Mesh")
        cae_viz.DatasetTransformingAPI.Apply(mesh.GetPrim(), "source")
        cae_viz.DatasetTransformingAPI.Apply(mesh.GetPrim(), "other")

        await self._attach_stage(stage)

        key = "ex_multi_tuple_other"
        cache.put_ex(
            key,
            {"v": 1},
            prims=[cache.PrimWatch(mesh.GetPrim(), on="update", schemas=[(cae_viz.DatasetTransformingAPI, "source")])],
            force=True,
        )
        self.assertIsNotNone(cache.get(key))

        # Change a property on the "other" instance — should NOT trigger invalidation.
        mesh.GetPrim().GetAttribute("cae:viz:dataset_transforming:other:useGlobalTransform").Set(True)

        await self._next_frame()

        self.assertIsNotNone(cache.get(key), "Property change on a different instance should not drop cache")

    async def test_put_ex_multi_apply_string_qualified_drops_cache(self):
        """Multi-apply string form: 'SchemaName:instance' drops cache on matching property change."""
        stage = Usd.Stage.CreateInMemory()
        mesh = UsdGeom.Mesh.Define(stage, "/Root/Mesh")
        cae_viz.DatasetTransformingAPI.Apply(mesh.GetPrim(), "source")

        await self._attach_stage(stage)

        key = "ex_multi_str_qual"
        cache.put_ex(
            key,
            {"v": 1},
            prims=[cache.PrimWatch(mesh.GetPrim(), on="update", schemas=["CaeVizDatasetTransformingAPI:source"])],
            force=True,
        )
        self.assertIsNotNone(cache.get(key))

        # Change a property on the "source" instance — should trigger invalidation.
        mesh.GetPrim().GetAttribute("cae:viz:dataset_transforming:source:useGlobalTransform").Set(True)

        await self._next_frame()

        self.assertIsNone(cache.get(key))


class TestPutExExpansion(omni.kit.test.AsyncTestCase):
    """Tests for the auto-expansion of PrimWatch prims in put_ex().

    put_ex() must automatically expand each watched prim to include all
    transitively related CaeDataSet and CaeFieldArray prims so that changes
    to FieldArray children correctly invalidate the cache.
    """

    async def setUp(self):
        cache._initialize()
        settings = carb.settings.get_settings()
        self._cache_mode_key = "/persistent/exts/omni.cae.data/cacheMode"
        self._original_cache_mode = settings.get_as_string(self._cache_mode_key)
        settings.set_string(self._cache_mode_key, "always")

    async def tearDown(self):
        settings = carb.settings.get_settings()
        settings.set_string(self._cache_mode_key, self._original_cache_mode)
        cache.clear()
        cache._finalize()
        ctx = omni.usd.get_context()
        if ctx.get_stage():
            ctx.close_stage()

    async def _attach_stage(self, stage):
        await omni.usd.get_context().attach_stage_async(stage)

    async def _next_frame(self):
        await omni.kit.app.get_app().next_update_async()

    async def test_field_array_property_change_invalidates_dataset_watch(self):
        """Core regression: PrimWatch(dataset) invalidates when a child FieldArray property changes."""
        stage = Usd.Stage.CreateInMemory()
        dataset = cae.DataSet.Define(stage, "/Root/DataSet")
        field = cae.FieldArray.Define(stage, "/Root/DataSet/Field1")
        dataset.GetPrim().CreateRelationship("field:Field1").AddTarget(field.GetPrim().GetPath())

        await self._attach_stage(stage)

        key = "expansion_field_change"
        cache.put_ex(key, {"v": 1}, prims=[cache.PrimWatch(dataset.GetPrim())], force=True)
        self.assertIsNotNone(cache.get(key))

        # Change a FieldArray property — must invalidate the cache (was broken before this fix)
        field.CreateFileNamesAttr().Set(["/new/file.dat"])

        await self._next_frame()

        self.assertIsNone(cache.get(key))

    async def test_expansion_respects_on_mode_resync(self):
        """Expanded watches inherit the on= mode of the original.

        With on="resync", a property change on the child FieldArray must NOT invalidate;
        a structural resync of the FieldArray MUST invalidate.
        """
        stage = Usd.Stage.CreateInMemory()
        dataset = cae.DataSet.Define(stage, "/Root/DataSet")
        field = cae.FieldArray.Define(stage, "/Root/DataSet/Field1")
        dataset.GetPrim().CreateRelationship("field:Field1").AddTarget(field.GetPrim().GetPath())

        await self._attach_stage(stage)

        key = "expansion_resync_mode"
        cache.put_ex(key, {"v": 1}, prims=[cache.PrimWatch(dataset.GetPrim(), on="resync")], force=True)
        self.assertIsNotNone(cache.get(key))

        # Property change on FieldArray — should NOT invalidate a resync-only watch
        field.CreateFileNamesAttr().Set(["/new/file.dat"])
        await self._next_frame()
        self.assertIsNotNone(cache.get(key))

        # Structural resync on FieldArray — SHOULD invalidate
        cae_viz.FacesAPI.Apply(field.GetPrim())
        await self._next_frame()
        self.assertIsNone(cache.get(key))

    async def test_schema_filtered_watch_on_operator_prim_is_noop_expansion(self):
        """Schema-filtered watch on a non-data prim (no field relationships) → expansion is a no-op.

        The cache must still be invalidated when the schema property on the operator prim changes.
        """
        stage = Usd.Stage.CreateInMemory()
        op_prim = UsdGeom.Xform.Define(stage, "/Root/OpPrim").GetPrim()
        cae_viz.DatasetSelectionAPI.Apply(op_prim, "inst0")

        await self._attach_stage(stage)

        key = "expansion_noop_op_prim"
        cache.put_ex(
            key,
            {"v": 1},
            prims=[cache.PrimWatch(op_prim, on="any", schemas=[(cae_viz.DatasetSelectionAPI, "inst0")])],
            force=True,
        )
        self.assertIsNotNone(cache.get(key))

        # Change a DatasetSelectionAPI property on the operator prim → must invalidate
        ds_api = cae_viz.DatasetSelectionAPI(op_prim, "inst0")
        ds_api.GetTargetRel().ClearTargets(True)

        await self._next_frame()

        self.assertIsNone(cache.get(key))
