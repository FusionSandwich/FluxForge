# Repository Cleanup and Style Migration

## Scope
This document tracks active cleanup steps for FluxForge repository deduplication and Python style enforcement.

## Phase 1 Completed (2026-03-17)
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

## Formatting Commands
Run from repository root.

```bash
black src tests examples
flake8 src tests
```

## Next Implementation Steps
All original cleanup-plan goals are now completed.

## Plan Completion Status (2026-03-17)
- Completed: Archived recovered GUI files and removed them from active source tree.
- Completed: Removed generated/temp outputs and merge artifacts; expanded `.gitignore` guardrails.
- Completed: Consolidated duplicated k0 physics helpers into canonical shared modules with compatibility wrappers.
- Completed: Consolidated RAFM comparison script bootstrap/pairing duplication into shared helper(s).
- Completed: Added and validated formatting/lint/tooling baseline (`black`, `flake8`, CI checks).
- Completed: Added regression coverage for dedup targets (k0 helpers, isotope parsing, N42 parsing, metadata/CSV QC).
- Completed: Updated important project documents/configs, including `setup.cfg`, `environment.yml`, and `README.md`.
- Completed: Re-ran targeted and full test suites after each phase; current suite status is green aside from expected environment-data skips.

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

## Current Verification Snapshot
- Full suite: `963 passed, 6 skipped` (OpenMC statepoint file unavailable in this workspace).
- MCNP workflow tests: passing (`tests/test_mcnp_io.py`, `tests/test_transport_io.py -k MCNP`).
