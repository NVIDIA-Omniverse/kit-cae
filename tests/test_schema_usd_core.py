# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

import pytest

from ovcae.schemas import register_usd_plugin


def test_codeless_schema_registers_with_classic_usd() -> None:
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom

    register_usd_plugin()
    stage = Usd.Stage.CreateInMemory()
    prim = UsdGeom.Mesh.Define(stage, "/Faces").GetPrim()

    assert prim.ApplyAPI("CaeVizFacesAPI")
    assert prim.HasAPI("CaeVizFacesAPI")
    assert prim.GetAttribute("cae:viz:faces:externalOnly").Get() is True
