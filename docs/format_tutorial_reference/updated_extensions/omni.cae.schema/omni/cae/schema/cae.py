# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from OmniCae import *
from OmniCaeCgns import CgnsFieldArray  # noqa: F401
from OmniCaeHdf5 import Hdf5FieldArray  # noqa: F401

# Re-export format-specific field array types for backward compatibility.
# These types are now defined in their own schema libraries but are re-exported
# here so that existing code using cae.NumPyFieldArray etc. continues to work.
# Import only the class types — do NOT use `import *` as it would overwrite
# shared symbols like Tokens with format-specific versions.
from OmniCaeNumPy import NumPyFieldArray  # noqa: F401
from OmniCaeScae import ScaeFieldArray  # noqa: F401
