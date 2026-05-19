# Kit-CAE Agent Skill Evals

Evaluation suite for measuring how well an AI agent can use the Kit-CAE skill set
to accomplish CAE post-processing tasks.

## Structure

```
evals/
├── README.md              ← this file
├── prompts/               ← one YAML file per eval
│   ├── 01_import_inspect.yaml
│   ├── 02_volume_render.yaml
│   ├── 03_faces_colormap.yaml
│   ├── 04_multi_viz.yaml
│   ├── 05_streamlines.yaml
│   ├── 06_time_varying_video.yaml
│   ├── 07_field_statistics.yaml
│   └── 08_creativity.yaml
├── validators/            ← one Kit --exec script per graded eval
│   ├── validate_01.py
│   ├── validate_02.py
│   ├── validate_03.py
│   ├── validate_04.py
│   ├── validate_05.py
│   ├── validate_06.py
│   └── validate_07.py
├── run_eval.py            ← CLI runner (provides prompt, runs validator, records results)
└── results/
    ├── SUMMARY.md                      ← auto-generated matrix across harness/model combos
    └── <harness>/<model>/
        ├── summary.json                ← per-combo aggregate
        ├── <eval_key>.json             ← latest-wins per eval
        └── renders/                    ← PNG/MP4 outputs (gitignored)
```

JSON summaries and `SUMMARY.md` are tracked in git so pass/fail across harness/model
combinations is visible at a glance. Rendered images and videos under `renders/` are
gitignored — run the eval locally to (re)generate them.

## How It Works

Each eval simulates a realistic user request. The agent receives a natural-language
prompt and the Kit-CAE skill set, then produces a script. The validator checks whether
the script accomplished the task correctly.

### Workflow

1. **Select an eval** — pick a prompt YAML from `prompts/`
2. **Give the prompt to the agent** — the agent uses Kit-CAE skills to write and run a script
3. **Run the validator** — checks the agent's output for correctness
4. **Record results** — JSON with pass/fail, timing, checks

### Running Evals

> **All scripts run through Kit Python** via `./repo.sh launch ... --exec`.
> The eval runner (`run_eval.py`) is the sole exception — it is a pure-stdlib
> orchestration script that spawns validators through `repo.sh launch` internally.
> If you write any new scripts (data generators, custom validators, analysis),
> they must run via `--exec`, never system `python3`.

```bash
cd <kit-cae-dir>

# List available evals (runner is pure-stdlib, calls repo.sh for validators)
python skills/evals/run_eval.py --list

# Show an eval prompt (give this to the agent)
python skills/evals/run_eval.py --eval 01_import_inspect --prompt

# Show a prompt with concrete render paths for a specific harness/model
python skills/evals/run_eval.py --eval 01_import_inspect --prompt \
    --harness claude-code --model opus-4.7

# Run a validator (validators run inside Kit via --exec)
# --harness and --model are required; they determine where renders and JSON land.
python skills/evals/run_eval.py --eval 02_volume_render --validate \
    --harness claude-code --model opus-4.7

# Run all graded validators for a combination
python skills/evals/run_eval.py --all --validate --harness claude-code --model opus-4.7

# View the cross-combination results matrix
python skills/evals/run_eval.py --results
```

### Harness / model tagging

`--harness` identifies the driver running the agent (e.g. `claude-code`, `openclaw`,
`codex`). `--model` identifies the LLM (e.g. `opus-4.7`, `codex-5.4`). Pick stable
short labels — they become directory names under `results/` and rows in the summary
matrix. Both are required for `--validate` so nothing accidentally overwrites an
unrelated combination's results.

### Manual Workflow (without runner)

You can also use evals manually:

1. Read the prompt YAML and give the `prompt` text to the agent. Prompts contain
   a `{render_dir}` placeholder; either substitute your chosen path by hand, or
   run `run_eval.py --prompt --harness ... --model ...` to have the runner fill
   it in.
2. The agent writes and runs a Kit-CAE script that saves renders to `{render_dir}`.
3. Run the validator inside Kit, setting `KIT_CAE_EVAL_RENDER_DIR` so the validator
   reads outputs from the same place the agent wrote them:
   ```bash
   KIT_CAE_EVAL_RENDER_DIR=skills/evals/results/claude-code/opus-4.7/renders \
   ./repo.sh launch -n omni.cae.kit -- \
       --exec skills/evals/validators/validate_02.py --no-window \
       --/app/asyncRendering=false \
       --/rtx/materialDb/syncLoads=true \
       --/omni.kit.plugin/syncUsdLoads=true \
       --/rtx/hydra/materialSyncLoads=true \
       --/rtx-transient/resourcemanager/texturestreaming/async=false \
       --/rtx-transient/resourcemanager/enableTextureStreaming=false \
       --/exts/omni.kit.window.viewport/blockingGetViewportDrawable=true \
       --/rtx-transient/dlssg/enabled=false \
       --/persistent/app/viewport/defaults/fillViewport=false
   ```
   For validators that create volumes (02, 04, 06), also add:
   ```bash
       --/renderer/enabled="rtx" \
       --/rtx/rendermode="RaytracedLighting" \
       --/rtx/directLighting/sampledLighting/enabled=true
   ```
   If `KIT_CAE_EVAL_RENDER_DIR` is unset the validator falls back to
   `<repo>/data/renders`, matching pre-harness behavior.
4. Look for `EVAL_RESULT` JSON in output

## Eval Descriptions

### Graded Evals (1–7)

| # | Name | Tests | Data File | Kit |
|---|------|-------|-----------|-----|
| 01 | Import & Inspect | File import, stage discovery, field enumeration | `StaticMixer.cgns` | `omni.cae.kit` |
| 02 | Volume Render | Volume creation, field binding, screenshot | `StaticMixer.cgns` | `omni.cae.kit` |
| 03 | Faces + Colormap | Surface viz, field coloring, screenshot | `multicomb_0_polyhedra.vtu` | `omni.cae_vtk.kit` |
| 04 | Multi-Viz Composition | Multiple viz types, field wiring, NPZ import | `disk_out_ref.npz` | `omni.cae.kit` |
| 05 | Streamlines | Flow viz, seed geometry, velocity binding | `disk_out_ref.npz` | `omni.cae.kit` |
| 06 | Time-Varying Video | Time-varying data, camera animation, H.264 MP4 | `hex_timesteps.cgns` | `omni.cae.kit` |
| 07 | Field Statistics | Programmatic stats query, numeric validation | `StaticMixer.cgns` | `omni.cae.kit` |

### Ungraded Eval (8)

| # | Name | Tests |
|---|------|-------|
| 08 | Creativity | Invent a custom data format, generate data, produce a visually striking video |

Eval 08 is **not graded**. It measures the model's ability to synthesize format
onboarding, data generation, visualization, and capture into something coherent
and compelling. Evaluated by human review.

## Grading

Each graded validator outputs structured JSON:

```json
{
  "eval": "02_volume_render",
  "pass": true,
  "score": 100,
  "checks": [
    {"name": "script_ran", "pass": true, "detail": "No errors during execution"},
    {"name": "volume_prim_exists", "pass": true, "detail": "/World/CAE/Volume"},
    {"name": "field_bound", "pass": true, "detail": "Eddy_Viscosity"},
    {"name": "screenshot_exists", "pass": true, "detail": "output.png (1280x720)"},
    {"name": "screenshot_not_black", "pass": true, "detail": "mean pixel > 5"}
  ]
}
```

A check passes or fails. The score is `(passed_checks / total_checks) * 100`.

## Results Tracking

Per-run JSON lands at `results/<harness>/<model>/<eval_key>.json` (latest-wins — a
re-run overwrites the previous entry for that combination):

```json
{
  "harness": "claude-code",
  "model": "opus-4.7",
  "date": "2026-04-21T08:30:00Z",
  "eval": "02_volume_render",
  "pass": true,
  "score": 100,
  "duration_seconds": 45.2,
  "checks": [...]
}
```

After each run the runner regenerates:
- `results/<harness>/<model>/summary.json` — aggregate for the combination.
- `results/SUMMARY.md` — matrix across all harness/model combinations plus per-combo
  detail. **Auto-generated; do not edit by hand.**

### Eval 08 Summary Requirement

After completing eval 08, add an `eval_08_creativity` object to `summary.json` with:
- `format`: name, structure, functional advantage, schema, onboarding steps
- `data`: phenomenon, physics/math basis, grid size, timestep count, field choice
- `visualization`: viz type, colormap strategy, renderer
- `video`: resolution, fps, duration, codec, camera narrative description

Also append a one-line summary to `SUMMARY.md` under the model's table.
This allows cross-model comparison of creative choices without watching every video.

This enables cross-harness / cross-model comparison at a glance.

## Artifact Hygiene

All eval outputs — rendered images, videos, and any generated data files (e.g.,
custom-format timestep files for eval 08) — must be written under
`skills/evals/results/<harness>/<model>/`. Do **not** write generated data to
`data/`, scripts to `scripts/`, or artifacts anywhere else in the repo.

The `results/` `.gitignore` excludes `**/renders/` so binary outputs won't be
committed.

## Test Data

All evals use data files already in `<kit-cae-dir>/data/`. No additional data
generation is needed for graded evals.

| File | Format | Size | Used By |
|------|--------|------|---------|
| `StaticMixer.cgns` | CGNS | 728K | Evals 01, 02, 07 |
| `multicomb_0_polyhedra.vtu` | VTK Unstructured | 2.2M | Eval 03 |
| `disk_out_ref.npz` | NumPy | 312K | Evals 04, 05 |
| `hex_timesteps.cgns` | CGNS (time-varying) | 42M | Eval 06 |

## Adding New Evals

1. Create a YAML file in `prompts/` following the schema
2. Create a validator in `validators/` if graded
3. Update this README
