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
# # A1. Native data without OVCAE
#
# Use **CAE OpenUSD Plugins** and **Warp SimData** directly to read a CGNS
# file, extract its external faces and an iso-surface, and validate the
# results. The plug-ins expose native data through an OpenUSD stage; Warp
# SimData loads the selected dataset and field and runs the operators.
#
# Warp SimData processing runs on CPU or GPU. This example uses no OVCAE
# recipes, OVStage, or renderer.
# It uses `tests/data/StaticMixer.cgns`, included in the checkout through
# Git LFS. Run `git lfs pull` before starting.

# %% [markdown]
# ## Register the plug-ins
#
# usd-core discovers file-format plug-ins once, when the first stage in the
# process is created, and ignores anything registered later. So the plug-in
# package must be registered before any stage exists. `check_runtime`
# confirms the installed plug-in build matches the active usd-core version;
# `register_usd_plugins` adds the plug-in directory to usd-core's search
# path. This is the first thing `ovcae.setup()` does on your behalf in the
# numbered tutorials.

# %%
import cae_openusd_plugins

print(cae_openusd_plugins.check_runtime(raise_on_error=True).message)
plugin_root = cae_openusd_plugins.register_usd_plugins()
print("Plug-ins registered from", plugin_root)

# %% [markdown]
# ## Compose the file
#
# A CGNS file becomes part of a stage through a **payload** on an empty
# prim. Composing the stage makes the plug-in read the file's structure and
# express it as prims: a `CGNSZone` per zone, an `OmniSciDataset` per element
# section or boundary in it, a `CGNSFlowSolution` holding the fields, and a
# `GridCoordinates` prim for the points. The arrays themselves stay in the
# file; the prims carry attributes that resolve to them on demand.

# %%
from pxr import Usd
from support import REPO_ROOT

cgns_path = REPO_ROOT / "tests" / "data" / "StaticMixer.cgns"

stage = Usd.Stage.CreateInMemory()
holder = stage.DefinePrim("/Data")
holder.GetPayloads().AddPayload(str(cgns_path))

for prim in stage.Traverse():
    print(f"{prim.GetPath()!s:<44} {prim.GetTypeName()}")

# %% [markdown]
# Warp SimData knows which of these prims it can turn into a dataset, and
# which fields a dataset can carry. `find_supported_prims` and `list_fields`
# are the two questions to ask before loading anything.
#
# One detail matters here and will matter again below: the plug-in authors
# the arrays as **time samples**, at time code 0 for a steady-state result.
# Asking for the default time code finds no value. Every call that touches
# array data takes an explicit `time_code`.

# %%
import warp_simdata.usd as simdata_usd

TIME = Usd.TimeCode(0.0)

for prim in simdata_usd.find_supported_prims(stage):
    print(f"{prim.GetPath()!s:<44} {prim.GetTypeName()}")

volume = stage.GetPrimAtPath("/Data/Base/StaticMixer/B1_P3")
print("\nFields on", volume.GetPath())
for info in simdata_usd.list_fields(volume, time_code=TIME):
    print(f"  {info.name:<28} per {info.association.name.lower()}")

# %% [markdown]
# ## Load a dataset onto a device
#
# `dataset_from_prim` reads the topology and points for one dataset prim
# and builds a Warp SimData `Dataset` on the device you name. `B1_P3` is
# the mixer's fluid volume; the other datasets are its inlet, outlet, and
# wall boundaries. Fields are loaded separately with `field_from_prim` and
# attached with `add_field`, so a dataset carries only the fields an
# operation needs.

# %%
import warp as wp
from support import tutorial_device

wp.config.log_level = wp.LOG_WARNING
device = tutorial_device()

dataset = simdata_usd.dataset_from_prim(volume, device=device, time_code=TIME)
temperature = simdata_usd.field_from_prim(volume, "Temperature", device=device, time_code=TIME)
dataset.add_field("Temperature", temperature)

low, high = dataset.get_bounds()
print("Device       :", device)
print("Nodes        :", dataset.get_num_nodes())
print("Elements     :", dataset.get_num_elems())
print("Bounds       :", tuple(round(v, 2) for v in low), "to", tuple(round(v, 2) for v in high))
print(
    "Temperature  :",
    temperature.association.name.lower(),
    "field, range",
    tuple(round(v, 2) for v in temperature.get_range()),
)

# %% [markdown]
# ## Run two operators
#
# These Warp SimData operators take a dataset and return a new one. OVCAE's
# Faces and Iso Surface operators use the same computations.
#
# `element_faces` extracts the faces of the volume elements; with
# `external_only` it keeps only the faces that belong to one element, which
# is the boundary of the mesh. The result is a surface dataset whose
# `node_idx` and `element_idx` fields say where each output point and face
# came from, so any field on the volume can be carried across.
#
# `iso_surface` contours a node field at a value and interpolates the fields
# you name onto the new surface. Both operators run on the dataset's device.

# %%
import time

from warp_simdata.operators import element_faces, iso_surface

started = time.perf_counter()
boundary = element_faces.compute(dataset, external_only=True)
elapsed = time.perf_counter() - started
print(f"External faces: {boundary.get_num_nodes()} nodes, {boundary.get_num_elems()} faces, {elapsed:.3f} s")
print("  fields:", boundary.get_field_names())

iso_value = 0.5 * (temperature.get_range()[0] + temperature.get_range()[1])
started = time.perf_counter()
contour = iso_surface.compute(dataset, "Temperature", iso_value, field_names=["Temperature"])
elapsed = time.perf_counter() - started
print(f"Iso-surface at {iso_value:.1f}: {contour.get_num_nodes()} nodes, {contour.get_num_elems()} triangles")
print(f"  computed in {elapsed:.3f} s")
print("  fields:", contour.get_field_names())

# %% [markdown]
# ## Validate the results
#
# Without a renderer, correctness is checked with numbers. Three checks
# that catch most mistakes:
#
# - Every point of the iso-surface carries the contour field, and its value
#   there must equal the iso value to interpolation precision.
# - The iso-surface must lie inside the volume's bounds.
# - The external boundary must have far fewer faces than the volume has
#   elements, and it must not be empty.
#
# The arrays come back as Warp arrays on the device; `numpy()` copies them
# to the host for inspection.

# %%
import numpy as np

contour_temperature = contour.get_field("Temperature").to_array().numpy()
deviation = float(np.abs(contour_temperature - iso_value).max())
iso_low, iso_high = contour.get_bounds()
inside = all(np.asarray(iso_low) >= np.asarray(low) - 1e-5) and all(np.asarray(iso_high) <= np.asarray(high) + 1e-5)

print(f"Contour field deviation from iso value : {deviation:.2e}")
print(f"Iso-surface within volume bounds       : {inside}")
print(f"Boundary faces / volume elements        : {boundary.get_num_elems()} / {dataset.get_num_elems()}")
assert deviation < 1e-3 and inside and 0 < boundary.get_num_elems() < dataset.get_num_elems()
print("All checks passed")

# %% [markdown]
# If a GPU is available and this ran on it, the same operations on
# CPU should agree exactly on counts. Run the cell to confirm; it is skipped
# when CPU was already the device.

# %%
if device != "cpu":
    cpu_dataset = simdata_usd.dataset_from_prim(volume, device="cpu", time_code=TIME)
    cpu_dataset.add_field(
        "Temperature", simdata_usd.field_from_prim(volume, "Temperature", device="cpu", time_code=TIME)
    )
    cpu_boundary = element_faces.compute(cpu_dataset, external_only=True)
    cpu_contour = iso_surface.compute(cpu_dataset, "Temperature", iso_value, field_names=["Temperature"])
    print("CPU boundary faces  :", cpu_boundary.get_num_elems(), "vs", boundary.get_num_elems())
    print("CPU contour triangles:", cpu_contour.get_num_elems(), "vs", contour.get_num_elems())
    assert cpu_boundary.get_num_elems() == boundary.get_num_elems()
    assert cpu_contour.get_num_elems() == contour.get_num_elems()
    print("CPU and", device, "agree")
else:
    print("Already on CPU; nothing to compare")

# %% [markdown]
# ## Where OVCAE fits
#
# Everything above is what OVCAE does inside for you, in the same order.
# `ovcae.setup()` registers the plug-ins. The USD Data Source composes the
# scene's layers, calls `dataset_from_prim` and `field_from_prim` for the
# datasets and fields the recipes select, always with the scene's time
# code, and caches the results across updates. The Faces and Iso Surface
# operators call `element_faces.compute` and `iso_surface.compute` on those
# datasets and publish the result to an OVStage as a mesh, which is the
# step this appendix does not need.
#
# What OVCAE adds is the coordination: recipes that record which dataset,
# field, and value were used so the result is reproducible; a session that
# reruns only what changed; a report that says what happened; and the
# stage that a renderer or an application reads. Tutorial 3 is the same
# static-mixer-style workflow with those pieces in place, on a larger file.
