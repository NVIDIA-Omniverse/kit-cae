# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
#  its affiliates is strictly prohibited.

"""
Omni CAE DataDelegate
----------------------
"""

from .impl.bindings import *
from .impl.extension import Extension
from .impl.types import IJKExtents, Range3i


# Cached interface pointer
def get_data_delegate_interface() -> IDataDelegateInterface:
    """Returns IDataDelegateInterface interface"""
    if not hasattr(get_data_delegate_interface, "iface"):
        get_data_delegate_interface.iface = acquire_data_delegate_interface()
    return get_data_delegate_interface.iface


def get_data_delegate_registry() -> IDataDelegateRegistry:
    """Returns data delegate registry"""
    # this is necessary to ensure the IDataDelegateRegistry definition
    # has been decorated with custom Python methods.
    from .impl import delegates

    return get_data_delegate_interface().get_data_delegate_registry()
