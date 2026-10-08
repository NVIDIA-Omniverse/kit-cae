# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Session-wide test setup.

usd-core fixes its file-format registry when the first stage is created, so the
reader must be imported before any test module builds a stage. usd-core likewise
builds its schema registry on the first schema read, so the custom-schema tests'
plug-in is registered here as well. Doing both here gives every module the same
starting point regardless of selection or order; modules that need the reader
still ``importorskip`` it for the skip behavior. Fresh-process tests that
exercise the ordering rules spawn their own interpreter.
"""

from contextlib import suppress
from pathlib import Path

with suppress(ImportError):
    import ovcae.readers  # noqa: F401

with suppress(ImportError):
    from ovcae.schemas import register_usd_plugin

    register_usd_plugin(Path(__file__).parent / "data" / "scaled_points_schema")
