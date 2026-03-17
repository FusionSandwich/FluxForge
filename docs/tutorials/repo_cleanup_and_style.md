# Repository Cleanup and Style Migration

## Scope
This document tracks active cleanup steps for FluxForge repository deduplication and Python style enforcement.

## Historical Cleanup Work Completed Earlier On 2026-03-17
- Created safety branch for cleanup work.
- Archived recovered GUI copies from active source tree:
  - `src/fluxforge_gui/app_recovered.py`
  - `src/fluxforge_gui/app_recovered_39.py`
  - `src/fluxforge_gui/app_recovered_311.py`
  - archived under `docs/archive/recovery/2026-03-17/`.
- Removed generated temporary output directories:
  - `examples/output`
  - `examples_output`
  - `examples/RAFM_irradiation/results_tmp_debug`
  - `examples/RAFM_irradiation/results_tmp_single`
  - `examples/RAFM_irradiation/results_qg_cli_check`
  - `examples/RAFM_irradiation/results_qg_fluxwire_check`
  - `examples/RAFM_irradiation/results_qg_single_check`
- Removed merge artifact files (`*.orig`, `*.rej`) and ad hoc `out.txt` in RAFM examples.
- Added ignore rules for generated outputs and artifact files in `.gitignore`.
- Added style baseline in `pyproject.toml`:
  - Black formatter config
  - Flake8 baseline settings (PEP8-focused)

## Current Audit Snapshot (2026-03-17)
- Branch under active cleanup review: `chore/folder-audit-native-gui-review-20260317`
- Starting branch baseline:
  - `fc2b4c4` `Refresh cleanup audit baseline`
- Current broad lint snapshot:
  - `475` flake8 findings across the current review buckets (`src/fluxforge_gui`, `src/fluxforge/cli`, `src/fluxforge/analysis`, `src/fluxforge/data`, `tests`, and `tools`)
- Highest-payoff folder buckets from the current audit:
  - `src/fluxforge_gui`: `90` findings after the first GUI-folder cleanup pass; the dead-import and unused-local debt in `app.py`, `ui_builder.py`, and `commands.py` is now removed, and the remaining debt is primarily line length in `app.py`, `ui_builder.py`, and `reporting.py`
  - `src/fluxforge/cli`: `30` findings, all in the `3501`-line `app.py` monolith
  - `src/fluxforge/analysis`: `116` findings, including several real code issues (`F821`, `E731`, unused state)
  - `src/fluxforge/data` + `tests` + `tools`: `239` findings, concentrated in `kayzero_k0.py`, `nuclear_data_sources.py`, `irdff.py`, and older test bootstrap modules
- Largest active source files:
  - `src/fluxforge_gui/ui_builder.py`: `2540` lines
  - `src/fluxforge_gui/app.py`: `2176` lines
  - `src/fluxforge_gui/commands.py`: `1190` lines
  - `src/fluxforge/cli/app.py`: `3501` lines
  - `src/fluxforge/analysis/flux_wire_analysis.py`: `2792` lines
- Additional cleanup hotspots outside the main packages:
  - `docs/REPO_CLEANUP_WORKSTREAM.csv` now records the folder-by-folder keep/refactor/split/archive/delete queue.
  - `src/fluxforge_gui/split_app.py`, `_sync_probe.txt`, and `xyz.txt` have been removed from the shipped GUI package.
  - `tools/github_issues/*.py` still use hard-coded local filesystem paths.
  - `examples/` and parts of `docs/` still contain workstation-specific absolute paths that should be converted to repo-relative usage or clearly marked as local-only examples.
  - Native screenshot evidence now has a committed Linux baseline under `tests/data/gui_review_baselines/linux/` plus a review-gallery manifest.
  - The latest native Linux evidence bundle is reproducible under `artifacts/gui_review/current_linux/` and currently matches all six baseline checkpoints.

## Verified Earlier Cleanup Work
- Completed: Archived recovered GUI files and removed them from active source tree.
- Completed: Removed generated/temp outputs and merge artifacts; expanded `.gitignore` guardrails.
- Completed: Consolidated duplicated k0 physics helpers into canonical shared modules with compatibility wrappers.
- Completed: Consolidated RAFM comparison script bootstrap/pairing duplication into shared helper(s).
- Completed: Added and validated formatting/lint/tooling baseline (`black`, `flake8`, CI checks).
- Completed: Added regression coverage for dedup targets (k0 helpers, isotope parsing, N42 parsing, metadata/CSV QC).
- Completed: Updated important project documents/configs, including `setup.cfg`, `environment.yml`, and `README.md`.
- Completed: Re-ran targeted and full test suites after each phase; current suite status is green aside from expected environment-data skips.
- Completed: First GUI-folder cleanup pass removed duplicated split-module import headers, preserved `fluxforge_gui.app` compatibility re-exports, and kept native desktop acceptance green.

## Phase 2 Completed (2026-03-17)
- Added canonical shared module: `src/fluxforge/k0_physics.py`.
- Rewired duplicate implementations to shared helpers:
  - `src/fluxforge/analysis/k0_naa.py`
  - `src/fluxforge/triga/cd_ratio.py`
  - `src/fluxforge/triga/k0.py`
- Added compatibility wrapper: `src/fluxforge/analysis/k0_physics.py`.
- Added targeted regression tests: `tests/test_k0_physics.py`.
- Verified focused test suite passes:
  - `tests/test_k0_physics.py`
  - `tests/test_k0_naa.py`
  - `tests/test_k0_workflow.py`
  - `tests/test_kayzero_k0.py`

### Duplicate clusters audited beyond scripts
- Repeated `Q0(alpha)` implementations in `analysis/k0_naa.py` and `triga/cd_ratio.py`.
- Repeated S/D/C timing factor implementations in `analysis/k0_naa.py` and `triga/k0.py`.
- Additional repeated timing helpers were identified in `physics/sigphi.py`, but these include stability shortcuts and will be handled in a separate, test-first pass.

## Phase 3 Active Roadmap
1. `src/fluxforge_gui`
   - Reduce copied import blocks and dead imports in `app.py`, `ui_builder.py`, and `commands.py`.
   - Continue splitting widget construction, dispatch, and state helpers into smaller focused modules.
   - Keep native desktop GUI tests green after each extraction.
2. `src/fluxforge/cli`
   - Break `cli/app.py` into command-group builders or subcommand modules.
   - Preserve CLI names and help text while shrinking the monolith.
3. `src/fluxforge/analysis` and `src/fluxforge/data`
   - Fix true lint/code-quality issues first (`F821`, bare `except`, lambda assignments, ambiguous names).
   - Then reduce duplicated parsing, validation, and plotting-input preparation helpers.
4. `tests`
   - Consolidate repeated GUI/bootstrap setup into shared fixtures.
   - Separate helper tests, native desktop tests, and long-running validation flows more clearly.
5. `tools`, `examples`, and `docs`
   - Remove or parameterize hard-coded local paths.
   - Mark developer-only scripts explicitly.
   - Keep shipped docs and examples runnable from a clean checkout.

## New Supporting Tooling On This Branch
- `tools/qa/build_cleanup_inventory.py`
  - regenerates `docs/REPO_CLEANUP_WORKSTREAM.csv` and `docs/REPO_CLEANUP_WORKSTREAM.md`
- `tools/qa/run_native_gui_evidence.py`
  - runs the real desktop GUI driver, saves screenshots/artifacts, and writes `run.json`
- `tools/qa/build_gui_review_gallery.py`
  - builds `review_gallery/index.html` from a captured evidence bundle plus platform baselines

## Formatting and Audit Commands
Run from repository root.

```bash
black src tests examples
flake8 src tests
python -m pytest -q
```

## Current Success Criteria For Cleanup
- No workstation-specific absolute paths in shipped runtime code.
- Smaller GUI and CLI module boundaries with less copied import boilerplate.
- PEP8/flake8 debt reduced in staged, reviewable slices instead of one large reformat-only diff.
- Regression coverage stays green after each folder-level cleanup pass.
- Native GUI evidence remains reproducible, with reviewable screenshots and gallery output after GUI-affecting changes.
