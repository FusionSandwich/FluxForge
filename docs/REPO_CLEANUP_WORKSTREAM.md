# Repo Cleanup Workstream

This document tracks the current folder-by-folder cleanup queue for the active branch.

- Total reviewed files: `474`
- CSV inventory: `docs/REPO_CLEANUP_WORKSTREAM.csv`

## Folder Buckets
- `.github`: `2` files
- `README.md`: `1` files
- `docs`: `28` files
- `environment.yml`: `1` files
- `examples`: `175` files
- `pyproject.toml`: `1` files
- `setup.cfg`: `1` files
- `src/fluxforge`: `141` files
- `src/fluxforge_gui`: `14` files
- `testing_validation`: `1` files
- `tests`: `99` files
- `tools`: `10` files

## Suggested Actions
- `delete`: `3` files
- `keep`: `391` files
- `refactor`: `70` files
- `split`: `10` files

## Highest-Priority Hotspots

| Path | Lines | Action | Accuracy Risk | Notes |
| --- | ---: | --- | --- | --- |
| `src/fluxforge/examples/rafm_workflow.py` | 4212 | `split` | `low` | hotspot-size |
| `src/fluxforge/cli/app.py` | 3501 | `split` | `medium` | 3.5k-line CLI monolith that should be grouped by subcommand family |
| `src/fluxforge/examples/flux_wire/flux_wire_spectrum_analysis.py` | 3021 | `split` | `low` | hotspot-size |
| `src/fluxforge/analysis/flux_wire_analysis.py` | 2792 | `split` | `high` | hotspot-size |
| `src/fluxforge_gui/ui_builder.py` | 2440 | `split` | `medium` | large mixed-responsibility widget builder with copied imports |
| `src/fluxforge_gui/app.py` | 2133 | `split` | `medium` | mechanical split left a duplicated import header and a 2k+ line shell module |
| `src/fluxforge/analysis/peakfit.py` | 2095 | `split` | `high` | hotspot-size |
| `src/fluxforge/data/irdff.py` | 1710 | `split` | `high` | hotspot-size |
| `src/fluxforge/examples/flux_wire/batch_compare_spectra.py` | 1551 | `split` | `low` | hotspot-size |
| `src/fluxforge_gui/commands.py` | 1078 | `split` | `medium` | large command dispatch module with copied imports and handler density |
| `tests/test_cli_app.py` | 1465 | `refactor` | `low` | fixture/setup consolidation candidate |
| `src/fluxforge/analysis/k0_workflow.py` | 1395 | `refactor` | `high` | none flagged yet |
| `src/fluxforge/analysis/naa_ann.py` | 1330 | `refactor` | `high` | none flagged yet |
| `examples/RAFM_irradiation/compare_peak_count_methods.py` | 1300 | `refactor` | `low` | none flagged yet |
| `docs/FLUXFORGE_CONSOLIDATED_MASTER.md` | 1226 | `refactor` | `low` | none flagged yet |

## Review Policy

- `baseline-captured` means the file is in scope for the cleanup campaign and has an initial suggested action.
- Accuracy-sensitive folders (`analysis`, `data`, `physics`, `solvers`, `validation`) require characterization tests before logic changes.
- GUI and CLI hotspots should be split in staged passes rather than one large formatting diff.
