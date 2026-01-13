# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import Any, Protocol, Union

from numpy.typing import ArrayLike


class CudaArrayLike(Protocol):
    """Protocol for objects that support CUDA Array Interface"""

    @property
    def __cuda_array_interface__(self) -> dict[str, Any]:
        """CUDA Array Interface dictionary"""


FieldArrayLike = Union[CudaArrayLike, ArrayLike]
"""
Protocol for objects that support the Numpy Array Interface or CUDA Array Interface
"""
