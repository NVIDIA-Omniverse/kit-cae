# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""
Ahead-of-Time (AOT) Kernel Compilation for DAV

This module compiles all Warp GPU kernels in the DAV operators ahead of time, so they
are ready to load at runtime without any JIT compilation overhead.

Overview
--------
DAV operators contain Warp kernels that are normally JIT-compiled on first use. AOT
compilation runs all those compilations up front — typically as part of a build or
installation step — and caches the results on disk so that subsequent runs load the
pre-built binaries directly.

Usage
-----
AOT compilation is controlled by the ``DAV_COMPILE_KERNELS_AOT`` environment variable.
It must be set to ``1`` before invoking this module, otherwise the compile step is
skipped with an error message.

Run from the command line::

    DAV_COMPILE_KERNELS_AOT=1 python -m dav.aot_compile

Optionally supply a recorded configuration JSON (produced by ``dav.recorder``) to
compile only the specializations that were actually exercised during a representative
session, rather than the full default set.  Recorder-produced configs omit the
``"devices"`` key, so you must supply ``--devices`` explicitly::

    DAV_COMPILE_KERNELS_AOT=1 python -m dav.aot_compile --config aot_config.json --devices cpu cuda:0

Use ``--devices`` alone to override the built-in default device list::

    DAV_COMPILE_KERNELS_AOT=1 python -m dav.aot_compile --devices cuda:0

Or call :func:`compile` programmatically::

    import os
    os.environ["DAV_COMPILE_KERNELS_AOT"] = "1"

    import dav.aot_compile
    dav.aot_compile.compile(devices=["cpu"])                             # default config, explicit device
    dav.aot_compile.compile(config_path="aot.json", devices=["cuda:0"]) # recorded config

What it compiles
----------------
The :func:`compile` function iterates over every operator listed by
:func:`dav.core.aot.get_operators` and imports it.  Each operator module has a
top-level ``if dav.config.compile_kernels_aot:`` block that triggers kernel
specialization and calls ``wp.compile_aot_module`` for every combination of data
model, field model, and other parameters relevant to that operator.

Target devices are determined by :func:`dav.core.aot.get_devices`, which respects
the ``DAV_AOT_DEVICES`` environment variable (defaults to all available CUDA devices).

Logging
-------
Progress is reported via the standard :mod:`logging` module at ``INFO`` level.
Each operator and each kernel variant logs a message before compilation begins so
that long-running compilations can be monitored.
"""

import importlib
import json
import logging

import dav
from dav.core import aot

logging.basicConfig(level=logging.INFO, format="[%(levelname)-5s] %(name)s — %(message)s")

logger = logging.getLogger(__name__)


def compile(config_path: str = None, devices: list[str] = None):
    """Compile all DAV operator kernels ahead of time.

    Iterates over every operator returned by :func:`dav.core.aot.get_operators`,
    imports it, and relies on each operator's module-level AOT block to call
    ``wp.compile_aot_module`` for every kernel variant.

    Requires the ``DAV_COMPILE_KERNELS_AOT`` environment variable to be set to
    ``1``; if it is not set the function logs an error and returns immediately.

    Args:
        config_path: Optional path to a JSON file produced by
            :func:`dav.recorder.build_config_from_cache` or
            :meth:`dav.recorder.Recorder.save`.  When provided, the file is
            loaded and used to replace the default :data:`dav.core.aot.configuration`
            before compilation begins, so that only the recorded specializations
            are compiled rather than the full default set.
        devices: Optional list of device strings to compile for (e.g.
            ``["cpu", "cuda:0"]``).  When provided, overrides the ``"devices"``
            key in the active configuration (or in the loaded *config_path* file).
            When omitted, the value already present in
            :data:`dav.core.aot.configuration` is used.

    Raises:
        ImportError: If an operator module cannot be imported.
        FileNotFoundError: If *config_path* is given but does not exist.
        json.JSONDecodeError: If *config_path* cannot be parsed as JSON.
    """
    if not dav.config.compile_kernels_aot:
        logger.error("AOT compilation is disabled in the configuration. Skipping AOT compilation. Use environment variable DAV_COMPILE_KERNELS_AOT=1 to enable.")
        return

    if config_path is not None:
        logger.info(f"Loading AOT configuration from: {config_path}")
        with open(config_path) as f:
            loaded_config = json.load(f)
        # Replace the active configuration in-place so all aot.get_*() helpers
        # pick up the new values without needing a reference update.
        aot.configuration.clear()
        aot.configuration.update(loaded_config)
        logger.info(f"Configuration loaded: {list(loaded_config.get('operators', {}).keys())} operators")

    if devices is not None:
        aot.configuration["devices"] = devices

    if not aot.get_devices():
        logger.error("No target devices specified. Pass --devices (CLI) or the devices= argument, or include a 'devices' key in the config.")
        return

    logger.info(f"Starting AOT compilation on {', '.join(aot.get_devices())} (this can take a while!)")

    # import all operators to trigger AOT compilation for their kernels
    # and their dependencies (data model APIs, field APIs, etc.)
    for operator_name in aot.get_operators():
        logger.info(f"Compiling operator: {operator_name}")
        importlib.import_module(f"dav.operators.{operator_name}")
    logger.info("AOT compilation completed successfully!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compile DAV operator kernels ahead of time.", epilog="Requires DAV_COMPILE_KERNELS_AOT=1 to be set in the environment.")
    parser.add_argument(
        "--config",
        metavar="PATH",
        default=None,
        help=(
            "Path to a recorded AOT configuration JSON file "
            "(produced by dav.recorder.Recorder.save() or dav.recorder.build_config_from_cache()). "
            "When supplied, replaces the default configuration so that only the "
            "recorded kernel specializations are compiled."
        ),
    )
    parser.add_argument(
        "--devices",
        nargs="+",
        metavar="DEVICE",
        default=None,
        help=(
            "One or more devices to compile for, e.g. --devices cpu cuda:0. "
            "Overrides the 'devices' key in the config (or the built-in default). "
            "Required when the config file was produced by dav.recorder (which omits devices)."
        ),
    )
    args = parser.parse_args()
    compile(config_path=args.config, devices=args.devices)
