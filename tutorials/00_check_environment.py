# %%
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# %% [markdown]
# # 0. Check your environment
#
# Run this first. It reports the kernel's environment and selected Warp
# device, and checks that OVCAE and its runtime dependencies import.
# Every other tutorial assumes this setup is ready.
#
# If you opened this as a notebook, use **Run All**. If you are reading the
# `.py` source in an editor with cell support, run it cell by cell or as a
# plain script.

# %%
import sys
from pathlib import Path

print("Python  :", sys.executable)
print("Version :", sys.version.split()[0])

# %% [markdown]
# The executable above should live inside this checkout's `.venv`. If it does
# not, the notebook is running on a different kernel; see the README for how
# to select the right one.

# %%
# As a script, locate the checkout from this file; as a notebook, __file__ is
# undefined and the working directory is the tutorials folder.
tutorials_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
expected_venv = (tutorials_dir.parent / ".venv").resolve()
if not Path(sys.executable).is_relative_to(expected_venv):
    print(f"Warning: expected an interpreter under {expected_venv}")
else:
    print("Interpreter is inside the project .venv")

# %% [markdown]
# ## Imports
#
# OVCAE sits on OVStage for the scene database, on Warp and Warp SimData for
# compute, and on usd-core plus the CAE OpenUSD plug-ins for authoring and
# reading recipe scenes. OVRTX is the renderer. All of these come from the
# sync command in the README, and the tutorials treat every one of them as
# required so that any tutorial can render what it computes.

# %%
from importlib.metadata import version

import cae_openusd_plugins  # noqa: F401  (import check; version comes from package metadata)
import ovrtx  # noqa: F401
import ovstage  # noqa: F401
import pxr  # noqa: F401
import warp as wp
import warp_simdata  # noqa: F401

import ovcae

# %% [markdown]
# ## Initialize OVCAE
#
# Importing `ovcae` registers nothing. Initialization is explicit through
# `ovcae.setup()`, which performs, in order, the registrations that have to
# happen at a particular moment in the process:
#
# 1. Import every installed reader (discovered through the `ovcae.readers`
#    entry-point group). The bundled reader registers the CAE OpenUSD
#    plug-ins, so usd-core learns the CGNS and other native file formats.
#    This has to happen before usd-core creates its first stage, because the
#    file-format registry is fixed at that point.
# 2. Register the codeless CAE schemas with usd-core, so recipe prims resolve
#    their API schemas on authored stages.
# 3. Register the CAE schema paths with OVStage, so recipe properties populate
#    and resolve their fallbacks when a scene is opened into an OVStage.
# 4. Register the bundled MDL material library, which the MDL SDK reads once
#    when an OVRTX renderer is constructed.
#
# The rule every tutorial follows: call `ovcae.setup()` right after importing
# and before creating any usd-core stage, OVStage, or renderer. When a
# tutorial renders with OVRTX, `ovrtx.register_schema_paths()` comes first,
# then `ovcae.setup()`. Repeated calls are harmless. A reader whose optional
# dependencies are missing is skipped with a warning; pass `strict=True` to
# make that an error instead, which is what this check wants.

# %%
ovcae.setup(strict=True)

# %% [markdown]
# ## Versions
#
# Versions come from package metadata so that they match what `uv sync`
# installed, not what a module happens to expose.


# %%
print("ovcae               :", ovcae.__version__)
print("ovstage             :", version("ovstage"))
print("warp-lang           :", wp.config.version)
print("warp-simdata        :", version("warp-simdata"))
print("usd-core            :", version("usd-core"))
print("cae-openusd-plugins :", version("cae-openusd-plugins"))
print("ovrtx               :", version("ovrtx"))

# %% [markdown]
# The CAE OpenUSD plug-ins must be registered with the usd-core build that is
# actually loaded, or reading CGNS and other native files fails later.
# `ovcae.setup()` did that above; confirm a CAE file format is visible.

# %%
from pxr import Sdf

if Sdf.FileFormat.FindByExtension("cgns") is None:
    raise RuntimeError("CGNS file format not registered; check that cae-openusd-plugins matches usd-core")
print("CAE OpenUSD plug-ins registered with usd-core")

# %% [markdown]
# ## Compute device
#
# OVCAE workflows require an RTX-capable NVIDIA GPU, including for headless
# rendering. Tutorials use the first GPU by default. Set the
# `OVCAE_TUTORIAL_DEVICE` environment variable to `cuda:N` to choose
# explicitly; `tutorials/run.py --device` does that for you, and in a live
# notebook you set it in your shell before launching JupyterLab. The shared
# `support.tutorial_device()` helper applies that rule, and every tutorial
# calls it.

# %%
from support import tutorial_device

device_name = tutorial_device()
device = wp.get_device(device_name)
print("Requested device :", device_name)
print("Resolved device  :", device)
print("CUDA available   :", wp.is_cuda_available())
if device.is_cuda:
    print("GPU              :", device.name)

# %% [markdown]
# Check that the interpreter is inside `.venv`, GPU availability is `True`,
# and the selected GPU is listed above. Tutorial 1 checks rendering next.
