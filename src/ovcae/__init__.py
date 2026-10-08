# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Standalone CAE visualization for OVStage applications.

The top level is the runtime: it evaluates recipes and publishes products into
an OVStage. Importing it loads only the standard library and registers nothing.
The subpackages have their own import footprints: ``ovcae.schemas`` and
``ovcae.materials`` are standard-library only, ``ovcae.authoring`` loads
usd-core and registers the schemas with it on import, and ``ovcae.readers``
needs the CAE OpenUSD Plugins and registers the file formats.
"""

from . import data, materials, operators, scene, schemas
from ._data import CompositeDataSource, DataSource, FrameData
from ._errors import (
    ActiveFrameDataError,
    DatasetResolutionError,
    DataSourceClosedError,
    DataSourceOpenError,
    FrameDataError,
    OrdinalError,
    OvcaeError,
    SessionClosedError,
    StageSessionConflictError,
    UpdateError,
    UpdateInFlightError,
)
from ._results import Diagnostic, OperatorResult, UpdateReport
from ._session import Session, Update
from ._setup import (
    codeless_schema_path,
    load_readers,
    ovstage_population_desc,
    register_schema_paths,
    registered_schemas,
    setup,
)
from ._sources import MemoryDataSource
from ._version import __version__
from .materials import register_material_library

__all__ = [
    "ActiveFrameDataError",
    "CompositeDataSource",
    "DataSource",
    "DataSourceClosedError",
    "DataSourceOpenError",
    "DatasetResolutionError",
    "Diagnostic",
    "FrameData",
    "FrameDataError",
    "MemoryDataSource",
    "OperatorResult",
    "OrdinalError",
    "OvcaeError",
    "Session",
    "SessionClosedError",
    "StageSessionConflictError",
    "Update",
    "UpdateError",
    "UpdateInFlightError",
    "UpdateReport",
    "__version__",
    "codeless_schema_path",
    "data",
    "load_readers",
    "materials",
    "operators",
    "ovstage_population_desc",
    "register_material_library",
    "register_schema_paths",
    "registered_schemas",
    "scene",
    "schemas",
    "setup",
]
