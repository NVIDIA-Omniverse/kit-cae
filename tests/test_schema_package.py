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

import json
from pathlib import Path

import ovcae.schemas


def _load_plug_info(path: Path) -> dict:
    text = "\n".join(line for line in path.read_text().splitlines() if not line.lstrip().startswith("#"))
    return json.loads(text)


def test_schema_bundle_is_complete() -> None:
    root = ovcae.schemas.schema_resource_path()
    assert ovcae.schemas.schema_source_path() == root / "schema.usda"
    plugin = _load_plug_info(root / "plugInfo.json")["Plugins"][0]
    assert plugin["Type"] == "resource"
    assert "LibraryPath" not in plugin
    types = plugin["Info"]["Types"]
    assert types["OmniCaeVizFacesAPI"]["schemaIdentifier"] == "CaeVizFacesAPI"
    assert types["OmniCaeVizIndeXVolumeAPI"]["schemaIdentifier"] == "CaeVizIndeXVolumeAPI"
