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

from unittest.mock import Mock

import ovcae
import ovcae.schemas


def test_ovcae_schema_helpers_register_with_ovstage(monkeypatch) -> None:
    import ovstage

    assert ovcae.codeless_schema_path() == ovcae.schemas.schema_resource_path()
    register = Mock()
    monkeypatch.setattr(ovstage.population, "register_usd_schemas", register)

    identifiers = ovcae.register_schema_paths()

    register.assert_called_once_with(str(ovcae.schemas.schema_resource_path()))
    assert "CaeVizFacesAPI" in identifiers and "CaeVizFieldSelectionAPI" in identifiers
    assert set(identifiers) <= set(ovcae.registered_schemas())
    assert ovcae.codeless_schema_path().resolve() == ovcae.schemas.schema_resource_path().resolve()


def test_ovstage_population_desc_includes_domains_and_cae_properties() -> None:
    import ovstage

    desc = ovcae.ovstage_population_desc(domains=ovstage.PopulationDomain.RENDERING)

    assert desc.domains == ovstage.PopulationDomain.RENDERING
    assert len(desc.selectors) == 1
