# Format Tutorial Reference Files

This directory contains all the files needed for the
[Format Onboarding Tutorial](../FormatOnboarding.md).

## Quick Setup

```bash
python docs/format_tutorial_reference/autocomplete_tutorial.py
./repo.sh build -xr
./repo.sh launch -n omni.cae.kit -- --exec scripts/generate_scae_data.py
```

## Contents

| Path | Purpose |
|------|---------|
| `autocomplete_tutorial.py` | Auto-setup script (stdlib only, idempotent) |
| `generate_scae_data.py` | Kit launch script — generates data and imports it |
| `repo_schemas.toml` | Updated schema config with `omniCaeScae` added |
| `premake5.lua` | Updated schemas build script |
| `updated_extensions/` | Extension files to overlay onto `source/extensions/` |
| `new_schemas/formats/omniCaeScae/` | New USD schema definition |

See the tutorial for detailed step-by-step instructions.
