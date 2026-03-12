# CAE Expression Variables (`omni.cae.exVars`)

Injects [USD expression variables](https://openusd.org/release/api/usd_page_front.html#usd_expressionVariables)
into the session layer of every stage that is opened, sourcing the values from
Carbonite settings supplied on the command line.

---

## How it works

On every stage-attach event the extension reads all settings nested under:

```
/exts/omni.cae.exVars/variables/<NAME>
```

and writes them into `stage.GetSessionLayer().expressionVariables`.
Because the session layer sits at the top of the layer stack these values
override any same-named variables that may already be present in the stage's
own layers.

---

## Setting variables from the command line

Pass one `--/` flag per variable:

```bash
./omni.cae.kit.sh \
  --/exts/omni.cae.exVars/variables/DATASET_ROOT="/mnt/data/sim_results" \
  --/exts/omni.cae.exVars/variables/TIMESTEP="42"
```

These flags are forwarded to Carbonite's settings registry before any
extension starts, so the values are available the moment the first stage opens.

### Multiple variables

```bash
./omni.cae.kit.sh \
  --/exts/omni.cae.exVars/variables/DATASET_ROOT="/mnt/data/sim_results" \
  --/exts/omni.cae.exVars/variables/VARIANT="high_res" \
  --/exts/omni.cae.exVars/variables/TIMESTEP="100"
```

---

## Using variables in a USD stage

Reference a variable anywhere an asset path or string is accepted using the
`${NAME}` syntax:

```usda
#usda 1.0
(
    defaultPrim = "World"
)

def Xform "World"
{
    def CaeCgnsFieldArray "Simulation"
    {
        asset[] fileNames = [@${DATASET_ROOT}/results.cgns@]
    }
}
```

With the flag `--/exts/omni.cae.exVars/variables/DATASET_ROOT="/mnt/data/sim_results"`,
the asset path resolves to `/mnt/data/sim_results/results.cgns` at runtime.

---

## Notes

- Variables are **non-persistent** — they are not saved to disk and must be
  supplied on every launch.
- If no variables are configured the extension does nothing on stage attach.
- Variables set via this extension are merged with any expression variables
  already present in the session layer; existing entries are not removed.
