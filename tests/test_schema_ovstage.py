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

import ovcae
import ovstage


def test_codeless_schema_application_reaches_ovstage_population() -> None:
    scene = """#usda 1.0
def Mesh "Faces" (
    prepend apiSchemas = ["CaeVizFacesAPI"]
)
{
}
"""
    ovcae.register_schema_paths()
    stage = ovstage.Stage("ovcae.tests.codeless-schema")
    try:
        ovstage.population.open_usd_from_string_with_desc(
            stage,
            scene,
            ordinal=1,
            time_code=0.0,
            descs=ovcae.ovstage_population_desc(domains=ovstage.PopulationDomain.ALL),
        )
        stage.advance_write_floor(1).wait()

        with stage.query(
            filter=ovstage.Filter(
                [
                    ovstage.Predicate("usd-schemas", ovstage.FilterOp.CONTAINS, ["CaeVizFacesAPI"]),
                ]
            )
        ) as query:
            query.wait()
            assert query.result().total_prim_count == 1
    finally:
        stage.destroy()
