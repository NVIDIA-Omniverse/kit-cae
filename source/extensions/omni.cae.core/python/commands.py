# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import inspect
from logging import getLogger

from omni.kit import commands
from pxr import Tf, Usd

from .usd_utils import QuietableException

logger = getLogger(__name__)


def _get_type_names(prim: Usd.Prim):
    stype = prim.GetPrimTypeInfo().GetSchemaType()
    root_type = Tf.Type.GetRoot()
    return [t.typeName for t in stype.GetAllAncestorTypes() if t != root_type]


async def execute(suffix, prim: Usd.Prim, **kwargs):
    attempted_commands = []
    for apiName in prim.GetAppliedSchemas():
        className = f"{apiName[:-3]}{suffix}"
        attempted_commands.append(className)
        logger.info("looking for command with name '%s'", className)
        if commands.get_command_class(className) is not None:
            status, result = commands.execute(className, **kwargs)
            if not status:
                raise QuietableException("Failed to execute '%s'" % className)
            return (await result) if inspect.isawaitable(result) else result

    # traverse type hierarchy.
    for typeName in _get_type_names(prim):
        className = f"{typeName}{suffix}"
        attempted_commands.append(className)
        logger.info("looking for command with name '%s'", className)
        if commands.get_command_class(className) is not None:
            status, result = commands.execute(className, **kwargs)
            if not status:
                raise QuietableException("Failed to execute '%s'" % className)
            return (await result) if inspect.isawaitable(result) else result

    message = f"Failed to execute command '{suffix}' on {prim}."
    if attempted_commands:
        message += f" No registered command matched; tried: {', '.join(attempted_commands)}."
    if prim.GetTypeName() == "CaeDataSet":
        message += " Legacy CaeDataSet commands are provided by 'omni.cae.legacy.bundle'; ensure it is enabled."
    raise NotImplementedError(message)


async def execute_command(command_name: str, **kwargs):
    status, result = commands.execute(command_name, **kwargs)
    if not status:
        raise QuietableException(f"Failed to execute command '{command_name}'")
    if inspect.isawaitable(result):
        return status, await result
    return status, result
