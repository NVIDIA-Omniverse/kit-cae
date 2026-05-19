# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import uuid

import numpy as np
import omni.kit.test
from omni.cae.testing import new_stage, wait_for_update
from omni.cae.viz.colormap_texture_manager import (
    ColormapTextureManager,
    build_colormap_lut,
    get_dynamic_url_for_identifier,
)
from pxr import Sdf, Usd


def _define_colormap(stage: Usd.Stage, path: str) -> Usd.Prim:
    """Define a Colormap prim with CaeVizColormapTextureAPI applied and two control points."""
    from omni.cae.schema import viz as cae_viz

    prim = stage.DefinePrim(path, "Colormap")
    prim.CreateAttribute("rgbaPoints", Sdf.ValueTypeNames.Float4Array).Set([(0.0, 0.0, 1.0, 0.0), (1.0, 0.0, 0.0, 1.0)])
    prim.CreateAttribute("xPoints", Sdf.ValueTypeNames.FloatArray).Set([0.0, 1.0])
    cae_viz.ColormapTextureAPI.Apply(prim).GetIdentifierAttr().Set(uuid.uuid4().hex[:12])
    return prim


def _define_colormap_no_api(stage: Usd.Stage, path: str) -> Usd.Prim:
    """Define a bare Colormap prim without CaeVizColormapTextureAPI."""
    prim = stage.DefinePrim(path, "Colormap")
    prim.CreateAttribute("rgbaPoints", Sdf.ValueTypeNames.Float4Array).Set([(0.0, 0.0, 1.0, 0.0), (1.0, 0.0, 0.0, 1.0)])
    prim.CreateAttribute("xPoints", Sdf.ValueTypeNames.FloatArray).Set([0.0, 1.0])
    return prim


class TestColormapUtilityFunctions(omni.kit.test.AsyncTestCase):
    """Unit tests for pure utility functions — no stage required."""

    async def test_get_dynamic_url_for_identifier(self):
        identifier = "abc123def456"
        self.assertEqual(get_dynamic_url_for_identifier(identifier), "dynamic://cae_colormap_abc123def456")

    async def test_build_colormap_lut_interpolates_points(self):
        lut = build_colormap_lut([(0.0, 0.0, 1.0, 0.0), (1.0, 0.0, 0.0, 1.0)], [0.0, 1.0], resolution=5)
        np.testing.assert_allclose(lut[0], [0.0, 0.0, 1.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(lut[-1], [1.0, 0.0, 0.0, 1.0], atol=1e-6)
        np.testing.assert_allclose(lut[2], [0.5, 0.0, 0.5, 0.5], atol=1e-6)


class TestColormapTextureManager(omni.kit.test.AsyncTestCase):
    """Integration tests for ColormapTextureManager — uses the singleton owned by the extension."""

    def _get_manager(self) -> ColormapTextureManager:
        manager = ColormapTextureManager.get_instance()
        self.assertIsNotNone(manager, "ColormapTextureManager singleton not available — is the extension loaded?")
        return manager

    async def test_refresh_discovers_colormap_prims(self):
        """Manager should register a texture entry after a Colormap prim with the API is added."""
        async with new_stage() as stage:
            manager = self._get_manager()
            prim = _define_colormap(stage, "/World/Test/Material/Colormap")
            await wait_for_update(0)

            self.assertTrue(manager.has_colormap(prim))

            entry = manager.get_entry(prim)
            self.assertIsNotNone(entry)

            from omni.cae.schema import viz as cae_viz

            identifier = cae_viz.ColormapTextureAPI(prim).GetIdentifierAttr().Get()
            expected_url = get_dynamic_url_for_identifier(identifier)
            self.assertEqual(manager.get_dynamic_url(prim), expected_url)
            self.assertEqual(entry.texture_name, f"cae_colormap_{identifier}")

    async def test_colormap_without_api_not_tracked(self):
        """Manager must not create a texture entry for a Colormap prim without CaeVizColormapTextureAPI."""
        async with new_stage() as stage:
            manager = self._get_manager()
            prim = _define_colormap_no_api(stage, "/World/Test/Material/ColormapNoAPI")
            await wait_for_update(0)

            self.assertFalse(manager.has_colormap(prim))

    async def test_refresh_updates_when_colormap_changes(self):
        """Fingerprint should change after rgbaPoints is modified."""
        async with new_stage() as stage:
            manager = self._get_manager()
            prim = _define_colormap(stage, "/World/Test/Material/Colormap")
            await wait_for_update(0)

            initial_fingerprint = manager.get_entry(prim).fingerprint
            self.assertIsNotNone(initial_fingerprint)

            prim.GetAttribute("rgbaPoints").Set([(0.0, 1.0, 0.0, 1.0), (1.0, 1.0, 1.0, 1.0)])
            await wait_for_update(0)

            self.assertNotEqual(initial_fingerprint, manager.get_entry(prim).fingerprint)

    async def test_refresh_removes_deleted_colormap(self):
        """Entry should be removed after the Colormap prim is deleted from the stage."""
        async with new_stage() as stage:
            manager = self._get_manager()
            prim = _define_colormap(stage, "/World/Test/Material/Colormap")
            await wait_for_update(0)
            self.assertTrue(manager.has_colormap(prim))

            stage.RemovePrim(prim.GetPath())
            await wait_for_update(0)

            self.assertFalse(manager.has_colormap("/World/Test/Material/Colormap"))

    async def test_multiple_colormaps_tracked_independently(self):
        """Each Colormap prim on the stage should get its own texture entry with a distinct URL."""
        async with new_stage() as stage:
            manager = self._get_manager()
            prim_a = _define_colormap(stage, "/World/MatA/Colormap")
            prim_b = _define_colormap(stage, "/World/MatB/Colormap")
            await wait_for_update(0)

            self.assertTrue(manager.has_colormap(prim_a))
            self.assertTrue(manager.has_colormap(prim_b))
            self.assertNotEqual(
                manager.get_entry(prim_a).texture_name,
                manager.get_entry(prim_b).texture_name,
            )

    async def test_unchanged_colormap_preserves_fingerprint(self):
        """Fingerprint should remain stable across update cycles when data is unchanged."""
        async with new_stage() as stage:
            manager = self._get_manager()
            prim = _define_colormap(stage, "/World/Test/Material/Colormap")
            await wait_for_update(0)

            fingerprint_before = manager.get_entry(prim).fingerprint
            await wait_for_update(0)

            self.assertEqual(fingerprint_before, manager.get_entry(prim).fingerprint)
