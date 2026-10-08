# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Public ovcae exception hierarchy."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._results import UpdateReport


class OvcaeError(Exception):
    """Base class for ovcae errors."""


class DataSourceOpenError(OvcaeError):
    """Raised when an accepted Data Source cannot be opened."""


class DatasetResolutionError(OvcaeError):
    """Raised when Frame Data cannot resolve a requested dataset."""


class DataSourceClosedError(OvcaeError):
    """Raised when an operation needs an open Data Source."""


class ActiveFrameDataError(OvcaeError):
    """Raised when closing a Data Source with an application lease active."""


class FrameDataError(OvcaeError):
    """Raised when Frame Data is invalid for a new Update."""


class SessionClosedError(OvcaeError):
    """Raised when an operation needs an open Session."""


class UpdateInFlightError(OvcaeError):
    """Raised when a Session already has an Update awaiting completion."""


class OrdinalError(OvcaeError, ValueError):
    """Raised when an Update violates the ordinal contract."""


class StageSessionConflictError(OvcaeError):
    """Raised when a live Session already publishes to an Execution Stage."""


class UpdateError(OvcaeError):
    """Raised when process-wide failure prevents a reliable Update outcome."""

    def __init__(self, message: str, report: UpdateReport | None = None):
        super().__init__(message)
        self.report = report
