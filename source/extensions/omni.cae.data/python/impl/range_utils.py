# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
#  its affiliates is strictly prohibited.

r"""
There are too many cases where we are dealing with computing
and specifying ranges of arrays. This module provides
utilities to make this easier.
"""

from logging import getLogger
from typing import Union

import numpy as np
from pxr import Usd
from usdrt import Usd as UsdRt

from . import array_utils, usd_utils
from .typing import FieldArrayLike

logger = getLogger(__name__)


async def get_range(
    dataset: Usd.Prim, field_name_or_names: Union[str, list[str]], timeCode=Usd.TimeCode.EarliestTime()
):
    """Get the range (min, max) of values in the specified field(s).

    If multiple fields are specified, they are combined into a single multi-component array.
    For multi-component arrays, the range is computed using the magnitude of each element.

    Args:
        dataset (Usd.Prim): The dataset prim.
        field_name_or_names (Union[str, list[str]]): The field name or list of field names.
        timeCode (Usd.TimeCode, optional): The time code to use. Defaults to Usd.TimeCode.EarliestTime().

    Returns:
        tuple[float, float]: The (min, max) range of values.
    """
    fields = field_name_or_names if isinstance(field_name_or_names, list) else [field_name_or_names]
    arrays = []
    for field in fields:
        array = await usd_utils.get_array_from_relationship(dataset, f"field:{field}", timeCode)
        arrays.append(array)

    if not arrays:
        raise usd_utils.QuietableException(f"No arrays found for fields: {field_name_or_names}")

    min_val, max_val = await get_range_from_arrays(arrays)
    logger.info(f"Range for fields {field_name_or_names}: ({min_val}, {max_val})")
    return (min_val, max_val)


async def get_range_from_arrays(arrays: list[FieldArrayLike]) -> tuple[float, float]:
    combined_array = array_utils.as_numpy_array(array_utils.column_stack(arrays))
    if combined_array.ndim == 1 or combined_array.shape[1] == 1:
        min_val = float(np.min(combined_array))
        max_val = float(np.max(combined_array))
        logger.debug(f"Computed range for single-component array: ({min_val}, {max_val})")
    else:
        magnitudes = np.linalg.norm(combined_array, axis=1)
        min_val = float(np.min(magnitudes))
        max_val = float(np.max(magnitudes))
        logger.debug(f"Computed range for multi-component array: ({min_val}, {max_val})")
    return (min_val, max_val)


async def compute_and_set_range(
    attr: Union[Usd.Attribute, UsdRt.Attribute],
    array_or_arrays: Union[FieldArrayLike, list[FieldArrayLike]],
    precomputed_range: tuple[float, float] = None,
    force: bool = False,
) -> Union[tuple[float, float], None]:
    """Set the range (min, max) of values in the specified attribute based on the provided arrays.

    If multiple arrays are specified, they are combined into a single multi-component array.
    For multi-component arrays, the range is computed using the magnitude of each element.

    Args:
        attr (Union[Usd.Attribute, UsdRt.Attribute]): The attribute to set the range on.
        array_or_arrays (Union[FieldArrayLike, list[FieldArrayLike]]): The array or list of arrays to compute the range from.
        precomputed_range (tuple[float, float], optional): If provided, this range is used instead of computing it. Defaults to None.
        force (bool, optional): If True, forces recomputation of the range even if the attribute already has a valid range. Defaults to False.

    Raises:
        QuietableException: If no arrays are provided.
    """
    cur_val = attr.Get()
    if force or cur_val is None or cur_val[0] > cur_val[1]:
        if precomputed_range is not None:
            min_val, max_val = precomputed_range
            logger.info(f"Using precomputed range for attribute {attr}: ({min_val}, {max_val})")
        else:
            if array_or_arrays is None:
                return None

            if not isinstance(array_or_arrays, list) and not isinstance(array_or_arrays, tuple):
                arrays = [array_or_arrays]
            else:
                arrays = array_or_arrays

            if not arrays:
                return None

            min_val, max_val = await get_range_from_arrays(arrays)

        if min_val > max_val:
            logger.warning(f"Invalid range for attribute {attr}: ({min_val}, {max_val})")
            return None

        logger.info(f"Setting range for attribute {attr}: ({min_val}, {max_val})")
        attr.Set((min_val, max_val))
        return (min_val, max_val)
    else:
        return None
