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

from contextlib import contextmanager
from pathlib import Path

import ovstage

import ovcae
import ovcae.readers

SCENES = Path(__file__).parent / "data" / "scenes"


@contextmanager
def open_scenario(name: str, *, stage_name: str):
    """Open one authored recipe through both sides of the ovcae boundary."""

    path = (SCENES / f"{name}.usda").resolve()
    ovcae.register_schema_paths()
    stage = ovstage.Stage(stage_name)
    source = ovcae.readers.UsdDataSource.open(path)
    try:
        # OVStage owns the lightweight recipes and the payload holder; the
        # Data Source composes the same layers with the CAE file-format plug-ins.
        ovstage.population.open_usd_with_desc(
            stage,
            str(path),
            ordinal=1,
            time_code=0.0,
            descs=ovcae.ovstage_population_desc(domains=ovstage.PopulationDomain.ALL),
        )
        stage.advance_write_floor(1).wait()
        yield stage, source
    finally:
        source.close()
        stage.destroy()
