# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
#  its affiliates is strictly prohibited.

import pathlib
from logging import getLogger

logger = getLogger(__name__)

_test_data_root = pathlib.Path(__file__).parent.parent.parent.parent / "shared" / "data"
_test_stage_root = pathlib.Path(__file__).parent.parent.parent.parent / "shared" / "stages"


def get_test_data_root() -> str:
    return str(_test_data_root)


def get_test_stage_root() -> str:
    return str(_test_stage_root)


def get_test_data_path(relative_path: str) -> str:
    if relative_path is None:
        return _test_data_root
    elif pathlib.Path(relative_path).is_absolute():
        #  check if path is absolute
        return pathlib.Path(relative_path)
    else:
        path = str(_test_data_root / relative_path)
        logger.info("Using test data %s", path)
        return path


def get_test_stage_path(relative_path: str) -> str:
    if relative_path is None:
        return _test_stage_root
    elif pathlib.Path(relative_path).is_absolute():
        #  check if path is absolute
        return pathlib.Path(relative_path)
    else:
        path = str(_test_stage_root / relative_path)
        logger.info("Using test stage %s", path)
        return path
