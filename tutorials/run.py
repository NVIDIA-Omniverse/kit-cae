# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Execute tutorials in fresh kernels and save the rendered notebooks under build/.

Each tutorial is a percent-format ``.py`` file in this directory. This script
turns it into a notebook with jupytext, runs every cell top to bottom with
nbclient, and writes the executed notebook (with outputs) to ``--output``.
A failing cell fails the run; the partially executed notebook is still saved
so the failure can be inspected in Jupyter.

Examples::

    python tutorials/run.py                       # every tutorial; first GPU if present, else CPU
    python tutorials/run.py --device cpu          # force CPU; tutorials that need a GPU are skipped
    python tutorials/run.py --assets ~/cae-data   # scenes and datasets copied somewhere else
    python tutorials/run.py 00_check_environment  # one tutorial by name
"""

import argparse
import os
import sys
from pathlib import Path

import jupytext
import nbformat
from nbclient import NotebookClient
from support import ASSETS_VARIABLE, DEVICE_VARIABLE, assets_dir, tutorial_device

TUTORIALS_DIR = Path(__file__).resolve().parent
TUTORIAL_GLOBS = ("[0-9][0-9]_*.py", "A[0-9]_*.py")  # numbered tutorials, then appendices
KERNEL_NAME = "python3"
# A tutorial that cannot run without a GPU says so with this line in its first cell;
# run.py skips it when the device is cpu instead of failing part way through.
REQUIRES_GPU_MARKER = "# tutorial-requires: gpu"


def discover(names: list[str]) -> list[Path]:
    """Resolve tutorial names (or every tutorial, numbered ones first) to source paths."""
    if not names:
        return [path for pattern in TUTORIAL_GLOBS for path in sorted(TUTORIALS_DIR.glob(pattern))]
    paths = []
    for name in names:
        candidate = TUTORIALS_DIR / (name if name.endswith(".py") else f"{name}.py")
        if not candidate.is_file():
            sys.exit(f"No tutorial named {name!r} in {TUTORIALS_DIR}")
        paths.append(candidate)
    return paths


def requires_gpu(source: Path) -> bool:
    return REQUIRES_GPU_MARKER in source.read_text(encoding="utf-8")


def execute(source: Path, output_dir: Path, timeout: int) -> None:
    notebook = jupytext.read(source, fmt="py:percent")
    notebook.metadata.setdefault("kernelspec", {"name": KERNEL_NAME, "language": "python", "display_name": "Python 3"})
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name=KERNEL_NAME,
        resources={"metadata": {"path": str(TUTORIALS_DIR)}},
    )
    try:
        client.execute()
    finally:
        nbformat.write(notebook, output_dir / source.with_suffix(".ipynb").name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "names", nargs="*", help="Tutorial file names to run; default is every tutorial, numbered ones then appendices"
    )
    parser.add_argument(
        "--device",
        help="Warp device the tutorials evaluate on: cpu or cuda:N. Default: the first CUDA device, else cpu",
    )
    parser.add_argument(
        "--assets",
        type=Path,
        help=f"Directory holding the tutorial scenes and datasets. Default: ${ASSETS_VARIABLE} or tutorials/data",
    )
    parser.add_argument("--output", type=Path, default=Path("build/tutorials"), help="Where executed notebooks go")
    parser.add_argument("--timeout", type=int, default=600, help="Per-cell timeout in seconds")
    args = parser.parse_args()

    device = args.device or tutorial_device()
    os.environ[DEVICE_VARIABLE] = device
    if args.assets:
        os.environ[ASSETS_VARIABLE] = str(args.assets.expanduser().resolve())
    print(f"Assets directory: {assets_dir()}")
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = discover(args.names)
    if not sources:
        print(f"No tutorials found in {TUTORIALS_DIR}")
        return
    skipped = []
    for source in sources:
        if device == "cpu" and requires_gpu(source):
            print(f"Skipped {source.name}: needs a GPU", flush=True)
            skipped.append(source.name)
            continue
        print(f"Running {source.name} on {device}", flush=True)
        execute(source, output_dir, args.timeout)
        print(f"Passed {source.name}", flush=True)
    if skipped:
        print(f"Skipped on cpu: {', '.join(skipped)}")
    print(f"Executed notebooks are in {output_dir}")


if __name__ == "__main__":
    main()
