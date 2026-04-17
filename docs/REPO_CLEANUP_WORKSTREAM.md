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

## Post-Phase-6 Cleanup Plan (2026-04-17)

This plan is the next execution track after the Phase 6 landing.

### Current Measured Baseline

- Formatter check (Ruff): `143 files would be reformatted` under `src/fluxforge` + `tests`.
- Lint check (Ruff E/F): `1642` findings total.
  - `E501 line-too-long`: `1193`
  - `F401 unused-import`: `317`
  - `F841 unused-variable`: `55`
- Large-file hotspots (line count):
  - `src/fluxforge/cli/app.py`: `6998`
  - `src/fluxforge/gui/panels/modern_shell.py`: `3409`
  - `src/fluxforge/analysis/flux_wire_analysis.py`: `2876`
  - `src/fluxforge/examples/rafm_workflow.py`: `4412`
  - `src/fluxforge_gui/app.py`: `2185`

### Formatter + PEP8 Execution Policy

- Scope: enforce PEP8 for code under `src/fluxforge` and `tests`, excluding data files.
- Formatter tool: use `ruff format` as the active formatter in this environment.
  - Note: `black` currently blocks in this workspace due Python `3.12.5` safety guard.
- Lint tool: use `ruff check --select E,F` as the baseline gate.
- Keep formatting commits separate from behavior-changing commits.

### Phase A: Safety-Preserving Auto-Fixes (No Behavior Change)

1. Run `python -m ruff format src/fluxforge tests` in focused slices (module-by-module).
2. Run `python -m ruff check src/fluxforge tests --select E,F --fix` for safe fixes.
3. Re-run targeted tests for touched areas after each slice.
4. Run full suite after each major slice.

Exit gate:
- zero formatter drift for touched slices
- no regression in full pytest suite

### Phase B: Modularization of Oversized Files

1. Continue modern-shell decomposition:
	- split `modern_shell.py` into `modern_shell_bottom.py`, `modern_shell_peak.py`, `modern_shell_activity.py`, and `modern_shell_batch.py` while preserving public imports from `fluxforge.gui.panels`.
2. Split CLI monolith `src/fluxforge/cli/app.py` into subcommand modules:
	- `cli/commands/activity.py`
	- `cli/commands/optimization.py`
	- `cli/commands/library.py`
	- `cli/commands/parity.py`
3. Split analysis hotspots:
	- `analysis/flux_wire_analysis.py` into parser/model/solver/report modules.
4. Keep shim imports so existing call sites remain stable during migration.

Exit gate:
- each split has parity tests + CLI/GUI route tests
- no public command regressions

### Phase C: Legacy and Unused Surface Reduction

1. Inventory legacy Tk surfaces under `src/fluxforge_gui/` and mark each as:
	- active fallback
	- deprecated bridge
	- removable
2. Remove unused scripts and dead helpers in `tools/` and stale examples after traceability review.
3. Add a CI guard to fail on new unused imports/variables in `src/fluxforge`.

Exit gate:
- documented deprecation map
- no orphaned workflow entrypoints

### Phase D: Continuous Quality Gates

1. Add CI jobs:
	- `ruff format --check src/fluxforge tests`
	- `ruff check src/fluxforge tests --select E,F`
2. Keep full-suite pytest as release gate.
3. Require new large modules to stay under an agreed soft cap (for example `<=1200` lines) unless explicitly waived.
