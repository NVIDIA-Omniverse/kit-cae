# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Application glue for interactive OVCAE: the update loop and typed stage writes.

Every interactive OVCAE application repeats the same steps: write a batch of
recipe edits at a new ordinal, advance the write floor, run an update, commit
its products, and render at the committed ordinal; presentation edits such as
a camera move skip the update. :class:`UpdateLoop` owns those steps and the
Session behind them for any renderer; :func:`write_attribute` is the typed
write it uses. Importing this
package loads only the standard library and the runtime package.
"""

from ._loop import EditKey, UpdateLoop
from ._writes import committed_type, element_layout, write_attribute

__all__ = ["EditKey", "UpdateLoop", "committed_type", "element_layout", "write_attribute"]
