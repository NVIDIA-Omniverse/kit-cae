# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Reader for datasets the CAE OpenUSD Plugins compose from native engineering files.

The runtime package defines the Data Source contract in ``ovcae.data`` and knows
nothing about any one dataset representation. This package is the first-party
reader for the representation the CAE OpenUSD Plugins produce: a native file
payloaded onto a holder prim composes into typed ``OmniSciDataset`` prims, and
:class:`UsdDataSource` resolves recipe selections against them on a usd-core
stage. :func:`import_dataset` authors the holder prim that payloads a native
file. The subpackage depends on the runtime for the Data Source contract and on
nothing else in the wheel. A reader for another representation follows the
same shape, as its own package, without touching the runtime or authoring;
appendix A2 of the tutorials writes one over VTK.

Importing this package requires usd-core and the CAE OpenUSD Plugins and
registers the plug-ins' file formats with usd-core. usd-core fixes its
file-format registry on first stage creation, so import this package before
the process creates any stage, or call ``ovcae.setup()``, which imports every
installed reader first. Importing it later raises RuntimeError rather than
leaving native files silently uncomposed.
"""

from __future__ import annotations

_MISSING = (
    "ovcae.readers requires usd-core (the `pxr` package) and the CAE OpenUSD Plugins. "
    "Install the optional dependency with `pip install 'ovcae[usd]'`."
)

try:
    import cae_openusd_plugins
    import pxr  # noqa: F401
except ImportError as exc:
    raise ImportError(_MISSING) from exc

try:
    cae_openusd_plugins.check_runtime(raise_on_error=True)
    cae_openusd_plugins.register_usd_plugins()
except Exception as exc:
    # A plug-in build that does not match the active pxr is an unusable optional
    # dependency, the same as an absent one, so ovcae.setup() can skip it.
    raise ImportError(f"ovcae.readers cannot initialize the CAE OpenUSD Plugins: {exc}") from exc


def _check_file_formats_registered() -> None:
    """Fail loudly when usd-core fixed its file-format registry before this import.

    usd-core builds the registry on first use, typically when the process creates
    its first stage, and ignores plug-ins registered afterwards. Nothing in this
    package can work in that state: holders never compose and the reader claims
    nothing, both silently. Detect it here, where the fix is one import earlier.
    """

    from pxr import Plug, Sdf, Tf

    base = Tf.Type.FindByName("SdfFileFormat")
    formats = [t.typeName for t in Plug.Registry().GetAllDerivedTypes(base) if t.typeName.startswith("OmniSci")]
    if formats and not any(Sdf.FileFormat.FindById(name) for name in formats):
        raise RuntimeError(
            "usd-core initialized its file-format registry before ovcae.readers was imported, so the CAE file "
            "formats are unavailable in this process. Import ovcae.readers before creating any pxr.Usd.Stage."
        )


_check_file_formats_registered()

from ._authoring import dataset_prims, import_dataset  # noqa: E402
from ._source import UsdDataSource  # noqa: E402

__all__ = [
    "UsdDataSource",
    "dataset_prims",
    "import_dataset",
]
