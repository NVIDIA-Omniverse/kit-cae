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

from ovcae.schemas import schema_source_path


def test_schema_source_is_kept_with_project() -> None:
    text = schema_source_path().read_text()
    assert 'class "CaeVizFacesAPI"' in text
    assert 'class "CaeVizStreamlinesAPI"' in text
