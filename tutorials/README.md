# OVCAE tutorials

Hands-on, executable walkthroughs of OVCAE. Each tutorial is a notebook you
run cell by cell in JupyterLab or your editor. The committed source is a
plain Python file in [percent format](https://jupytext.org/formats/scripts/#the-percent-format);
[Jupytext](https://jupytext.org/) turns it into a notebook when you
open it, so the repository never stores notebook JSON or outputs.

OVCAE workflows require an RTX-capable NVIDIA GPU, including for headless
and offscreen rendering.

For the learning sequence and additional tutorials, see
[Examples and tutorials](../docs/examples-and-tutorials.md#tutorials).

## Tutorial data

Scenes and datasets are checked in under [`data/`](data/README.md), except
A1's static mixer in [`../tests/data/`](../tests/data/README.md). Large files
use Git LFS; run `git lfs pull` after cloning. Those READMEs record provenance
and licenses. Generated scenes and images go under `build/tutorials/`.

Tutorials 1 to 3 can read scenes and datasets from another directory: set
`OVCAE_TUTORIAL_ASSETS` or pass `run.py --assets`. A1, A2, and A3 use their
repository data paths directly. Colormap textures ship with OVCAE's material
library.

## Streaming to the browser

Tutorial 2 and appendices A2 and A3 stream the live render into the notebook page
through `ovcae.stream`, which uses
[OVStream](https://github.com/NVIDIA-Omniverse/ovstream); the `tutorials`
group installs it, as the `stream` extra does for an application. The browser
side is ovstream's stock WebRTC client, loaded from the ovstream repository at
the pinned release through jsDelivr, so the notebook page needs internet
access for that one script. WebRTC
carries video over UDP, which means:

- On the GPU machine itself, or on a network where the browser can reach it
  directly, nothing needs configuring. The client connects to the host the
  notebook page was loaded from, on signaling port 49100.
- An SSH port forward is not enough for a remote JupyterLab: it carries the
  signaling port but not the UDP media. The browser must reach the GPU
  host's signaling port and UDP media port (47998 by default) directly, or
  through a TURN server. Set `OVCAE_TUTORIAL_STREAM_HOST` to the name or
  address the browser should use when it differs from the notebook page's
  host. ovstream's `docs/WEBRTC_CONNECTIVITY.md` covers the failure modes.

Without a connected browser the tutorial still runs to completion: the
render thread drops frames until a client connects. The tutorials also save
the latest frame as a PNG. Wait for the scene to appear in the live video
before running the snapshot cell; a non-interactive run can capture a blank
initial frame because the snapshot does not wait for rendering.

## Prerequisites

- Linux x86-64, Python 3.12, and [`uv`](https://docs.astral.sh/uv/) on your PATH.
- An RTX-capable NVIDIA GPU with a working driver (`nvidia-smi` should list it).
- A checkout of Kit-CAE's `experimental-ovcae` branch; see the
  [installation guide](../docs/installation.md). Run the commands below from
  the repository root.
- The native system libraries listed in the
  [installation prerequisites](../docs/installation.md#prerequisites), including
  for headless use.
- A browser, or an editor with Jupyter support (VS Code, PyCharm).

Tutorials use the first GPU by default. Set `OVCAE_TUTORIAL_DEVICE` to
`cuda:N` to choose a different GPU. OVRTX, the renderer, is installed with
the tutorials. Appendix A1 uses Warp SimData directly and can run its
processing on CPU or GPU without a renderer.

## Set up

```console
git lfs pull
uv sync --frozen --group tutorials --extra usd
```

This creates or updates `.venv` with this checkout installed in editable mode
plus JupyterLab, Jupytext, the IPython kernel, and OVRTX. `uv sync` may
remove packages outside the selected groups; add `--group dev` if you also
run the test suite from the same environment.

## Run the tutorials

Start each tutorial in a fresh kernel. For tutorial 2 and appendices A2 and
A3, run through the cell that displays the video and sliders, then pause to
try them. Run the remaining cells, including **Shut down**, when finished.
**Run All** executes the whole tutorial and closes the live view. Close the
streaming notebook tab before opening another in JupyterLab, even after
running **Shut down**.

### In JupyterLab

```console
uv run --no-sync jupyter lab tutorials
```

In the file browser, right-click a tutorial such as `00_check_environment.py`
and choose **Open With > Notebook**. Jupytext pairs it with a generated
`.ipynb` next to the source (gitignored) and keeps the two in sync on save.
Pick the **Python 3 (ipykernel)** kernel, which is the one inside `.venv`
because JupyterLab was launched from it.

To pick a GPU explicitly, or to
read the data from somewhere other than `tutorials/data`, set the variables
before launching:

```console
OVCAE_TUTORIAL_DEVICE=cuda:0 OVCAE_TUTORIAL_ASSETS=~/cae-data uv run --no-sync jupyter lab tutorials
```

On a remote machine, start JupyterLab with `--no-browser --ip=127.0.0.1`,
forward its port over SSH, and open the token-bearing URL it prints.

### In VS Code or PyCharm

Open the `.py` file directly. Both editors recognize `# %%` cell markers and
offer **Run Cell** and **Run All** with an interactive window. Select
`.venv/bin/python` as the interpreter. If you prefer the notebook UI, convert
first:

```console
uv run --no-sync jupytext --to ipynb tutorials/00_check_environment.py
```

### Non-interactively

`run.py` converts every tutorial to a notebook, executes it top to bottom in
a fresh kernel, and writes the executed notebooks with outputs to
`build/tutorials/`. A failing cell fails the run and still saves the partial
notebook for inspection.

```console
uv run --no-sync python tutorials/run.py --device cuda:0                       # all tutorials
uv run --no-sync python tutorials/run.py --device cuda:0 --assets ~/cae-data   # data for tutorials 1 to 3
uv run --no-sync python tutorials/run.py --device cuda:0 00_check_environment  # one tutorial
```

Each tutorial also runs as an ordinary script:

```console
.venv/bin/python tutorials/00_check_environment.py
```

## Writing a tutorial

- Name it `NN_topic.py` with a two-digit prefix, or `AN_topic.py` for an
  appendix that stands alone; `run.py` runs the numbered tutorials in order
  and then the appendices. Add a row to the matching table when you add a
  file.
- Start with a `# %%` cell holding only the SPDX header (Jupytext needs the
  marker so the header round-trips as its own cell), then a `# %% [markdown]`
  cell holding the title as a level-one heading. Alternate `# %% [markdown]` prose cells and `# %%`
  code cells. Keep one idea per cell so readers can run and inspect step by
  step.
- Import a module in the cell where the prose introduces it. Ruff's
  "import not at top of file" rule is disabled for this folder; its other
  rules and the formatter still apply, so run `uv run --no-sync ruff format tutorials`
  and `uv run --no-sync ruff check tutorials` before committing.
- Take the device from `support.tutorial_device()` so the
  `OVCAE_TUTORIAL_DEVICE` setting is honored.
- Resolve scenes and datasets with `support.asset("name")`, never with a path
  relative to the tutorial, so the data directory setting is honored. Add
  new files under `data/` with a provenance entry in its README, large ones
  through Git LFS. Write
  generated files under `support.OUTPUT_DIR` (`build/tutorials/`) or a
  temporary directory, never next to the source.
- Shared helpers go in `support.py`, which tutorials import. Keep evaluation,
  session, and renderer calls visible in the tutorial itself; helpers are for
  device and asset resolution, image output, the OVStage write boilerplate,
  and the browser stream client. Data a later tutorial reuses without
  re-explaining it can live in its own module, as `ripple.py` does for the
  plate tutorials 4 and 5 share, and `exodus_source.py` does for the
  appendices. Code that every interactive application needs belongs in
  `ovcae` itself, as the render thread of tutorial 2 now does in
  `ovcae.stream`.
- A tutorial that cannot run without a GPU says so in its prose and carries
  the line `# tutorial-requires: gpu` in its first cell, right after the SPDX
  header. Rendering uses the scene's own camera and `RenderProduct`.
- Do not commit `.ipynb` files. If you edited the paired notebook in
  JupyterLab, the `.py` is already updated on save. If in doubt run
  `uv run --no-sync jupytext --sync tutorials/NN_topic.ipynb`.
- Before opening a merge request run `uv run --no-sync python tutorials/run.py` so
  every tutorial still executes end to end.
