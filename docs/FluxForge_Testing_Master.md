# FluxForge Testing Master

**Status:** active testing and validation source of truth  
**Last Updated:** 2026-04-19  
**Purpose:** consolidated testing contract for unit, integration, GUI, parity, and
artifact-backed validation work.

## 1. Testing Source Order

Use the testing docs set in this order:

1. `docs/ROADMAP_EXECUTION_STATUS.md` for live run status and latest completion notes
2. `docs/GUI_TEST_COVERAGE_LEDGER.md` for explicit GUI surface coverage tracking
3. `docs/FluxForge_Testing_Master.md` for testing policy and acceptance requirements
4. `docs/FLUXFORGE_CONSOLIDATED_MASTER.md` for feature scope
5. `docs/GUI_PLAN.md` for GUI-specific interaction acceptance

## 2. Test Contract

Every adopted capability should land with the following where applicable:

- reusable core or data-layer coverage
- CLI or workflow coverage
- GUI workflow coverage for user-facing desktop surfaces
- saved artifact, report, or machine-readable output that can be reviewed later

Additional rules:

- Standards-locked workflows require explicit Standards-mode tests.
- Parity work requires source-linked fixtures and declared tolerances.
- Interaction-heavy GUI surfaces require native probes or artifact-gallery review, not only headless unit checks.
- Export and provenance formats require schema/round-trip checks.

## 3. Current Coverage Baseline

| Coverage Area | Representative Tests / Artifacts | Focus |
|---|---|---|
| File and schema IO | `tests/test_n42.py`, `tests/test_cnf_io.py`, `tests/test_csv_readers.py`, `tests/test_spectrum_io_parity.py` | spectrum import/export, calibration extraction, parity baselines |
| Calibration and efficiency | `tests/test_calibration_workspace_qt.py`, `tests/test_pipeline_validation.py`, `tests/test_genie_overrides.py` | energy calibration, efficiency workflows, Qt calibration behavior |
| Analysis workspace and peak workflows | `tests/test_analysis_workspace_qt.py`, `tests/test_roi_analysis_core.py`, `tests/test_peak_finder_methods.py` | peak search, ROI/background workflows, GUI peak review, library behavior |
| Phase 5 crosswalk and parity review | `tests/test_phase5_crosswalk.py`, `tests/test_reference_parity_runner.py`, `tests/test_parity_fixture_manifests.py`, `tests/test_cli_app.py`, `tests/test_modern_gui_shell.py`, `tests/gui_phase5_parity_probe.py`, `artifacts/gui_review/phase5_parity/` | machine-readable writeup crosswalk validation, executable parity-manifest workflows (including spectrum IO/background/fit plus overlay-role and ROI-statistics workflow bundles), CLI report + parity contract, modern-shell state persistence and workflow fixture execution, native GUI evidence, and browser-lane gallery audit |
| Phase 6 optimization and saved workflows | `tests/test_cli_app.py`, `tests/test_analysis_workspace_qt.py`, `tests/test_modern_gui_shell.py`, `tests/gui_phase6_optimization_probe.py`, `tests/_phase6_real_data.py` | real RAFM/LDRD activity-review inputs, masking, optimization, second irradiation, `.ffexp`, GUI workflow/session persistence, and mouse-driven probe actions for Playwright artifact review |
| Unfolding and inverse analysis | `tests/test_unfolding_registry.py`, `tests/test_unfolding_workflows.py`, `tests/test_unfolding_reference_parity.py`, validation scripts under `examples/validation/` | registry-backed unfolding, external parity, diagnostic outputs |
| Standards and QA | `tests/test_astm_e261.py`, `tests/test_astm_e262.py`, `tests/test_module3_backends.py`, `tests/test_module3_workflows_qt.py`, `tests/test_gui_dialogs_qt.py` | governed calculations, QA monitor, standards review, reporting, and direct dialog-level GUI checks |
| Batch, reporting, and predictive GUI | `tests/test_cli_app.py`, `tests/test_predictive_features.py`, `tests/test_predictive_dashboard_qt.py` | CLI contract, offline predictive features, dashboard/status behavior |
| Native GUI probes | `tests/gui_unfolding_workspace_probe.py`, `tests/gui_module3_workflows_probe.py`, `tests/gui_predictive_dashboard_probe.py`, `artifacts/gui_review/` | screenshot-backed inspection of interaction-heavy Qt surfaces |
| Browser-lane gallery audit | `tests/gui_gallery_playwright_audit.js`, `artifacts/gui_review/playwright_audit/audit_report.json`, `artifacts/gui_review/playwright_audit/audit_report.md` | Playwright validation of gallery page load, section/card visibility, image health, viewport metadata, and responsive styling |

## 4. Verification Baseline

Documented verification state:

- Full-suite baseline in this workspace is `1286 passed, 2 skipped`.
- Targeted parity/fixture/CLI verification:
  `pytest -q tests/test_reference_parity_runner.py tests/test_parity_fixture_manifests.py tests/test_parity_phase3_scaffolding.py tests/test_cli_app.py -k "parity or fixture or manifest"`
  with `10 passed, 67 deselected`.
- Targeted Qt workflow verification:
  `pytest -q tests/test_analysis_workspace_qt.py tests/test_module3_workflows_qt.py tests/test_modern_gui_shell.py`
  with `49 passed`.
- Saved-workflow and shell-modularization regression:
  `PYTHONPATH=src pytest -q tests/test_modern_gui_shell.py`
  with `14 passed`.
- Phase 6 CLI/export regression on the real RAFM/LDRD corpus:
  `PYTHONPATH=src pytest -q tests/test_cli_app.py -k "second_irradiation_plan_writes_json_and_csv_outputs or optimization_sweep_builds_candidates_from_activity_review or ffexp_export_packages_phase6_products"`
  with `3 passed`.
- Phase 6 Qt workflow regression after the modern-shell split:
  `PYTHONPATH=src pytest -q tests/test_analysis_workspace_qt.py -k "masking_review_panel_runs_and_exports_tables or optimization_workspace_panel_runs_and_exports_phase6_bundle or optimization_workspace_panel_advanced_guard_and_second_irradiation_panel"`
  with `3 passed, 29 deselected`.
- Phase 6 worked-example CLI command + existing CLI/export checks:
  `PYTHONPATH=src pytest -q tests/test_cli_app.py -k "phase6_ldrd_worked_example or second_irradiation_plan_writes_json_and_csv_outputs or optimization_sweep_builds_candidates_from_activity_review or ffexp_export_packages_phase6_products"`
  with `5 passed, 71 deselected`.
- Phase 6 Qt workflow regression including the new worked-example panel action:
  `PYTHONPATH=src pytest -q tests/test_analysis_workspace_qt.py -k "optimization_workspace_panel_runs_ldrd_worked_example_action or masking_review_panel_runs_and_exports_tables or optimization_workspace_panel_runs_and_exports_phase6_bundle or optimization_workspace_panel_advanced_guard_and_second_irradiation_panel"`
  with `4 passed, 29 deselected`.
- Direct CLI command execution on the real RAFM/LDRD corpus:
  `PYTHONPATH=src python -m fluxforge.cli.app phase6-ldrd-worked-example --sample-id RAFM4-C_15dEOI --output-root /tmp/phase6_ldrd_cli_validation`
  generated a complete artifact bundle and summary.
- Broad Phase 3 regression slice:
  `pytest -q tests/test_unfolding_registry.py tests/test_unfolding_workflows.py tests/test_unfolding_workspace_qt.py tests/test_module3_backends.py tests/test_module3_workflows_qt.py tests/test_analysis_workspace_qt.py tests/test_calibration_workspace_qt.py tests/test_modern_gui_shell.py tests/test_cli_app.py`
  with `204 passed`.
- Native probe galleries were refreshed at:
  `artifacts/gui_review/phase326_probe/`,
  `artifacts/gui_review/phase317_calibration_probe/`,
  `artifacts/gui_review/phase31x_unfolding_probe/`, and
  `artifacts/gui_review/phase327_probe/`, and
  `artifacts/gui_review/phase6_optimization_probe/`.
- The Phase 6 probe now drives masking/optimization/second-irradiation actions with
  `QTest.mouseClick(...)` before screenshot capture so browser artifact inspection
  can validate mouse-path behavior instead of method-only programmatic calls.
- The Phase 6 probe also drives the new `Run LDRD Worked Example` GUI action and now
  captures six screenshots, including the worked-example state.
- Optimization-helper dedup regression verification:
  `PYTHONPATH=src pytest -q tests/test_optimization_common.py tests/test_optimization_difom.py tests/test_optimization_fim.py tests/test_optimization_mwdcs.py tests/test_optimization_bassd.py tests/test_optimization_stbdmr.py`
  with `22 passed`.
- Modern-shell dedup guard verification:
  `PYTHONPATH=src pytest -q tests/test_modern_gui_shell.py -k "modern_shell_reuses_shared_demo_and_selection_helpers"`
  with `1 passed, 14 deselected`.
- Full-suite revalidation after cleanup/quality tooling updates:
  `PYTHONPATH=src pytest -q`
  with `1286 passed, 2 skipped`.
- Phase 5.1 crosswalk targeted verification:
  `PYTHONPATH=src pytest -q tests/test_phase5_crosswalk.py tests/test_cli_app.py -k "phase5 or crosswalk"`
  with `4 passed, 78 deselected`.
- Phase 5.1 shell-state regression verification:
  `PYTHONPATH=src pytest -q tests/test_modern_gui_shell.py -k "main_window_restores_saved_workflow_state_across_sessions"`
  with `1 passed, 14 deselected`.
- Phase 5.1 native probe generation:
  `PYTHONPATH=src /usr/bin/python tests/gui_phase5_parity_probe.py artifacts/gui_review/phase5_parity`
  producing `artifacts/gui_review/phase5_parity/index.html` and
  `artifacts/gui_review/phase5_parity/phase5_probe_report.json`.
- Phase 5.1 browser-lane audit:
  `node tests/gui_gallery_playwright_audit.js artifacts/gui_review/phase5_parity artifacts/gui_review/phase5_parity/playwright_audit`
  with `1 audited, 0 failing` and reports at
  `artifacts/gui_review/phase5_parity/playwright_audit/audit_report.json` plus
  `artifacts/gui_review/phase5_parity/playwright_audit/audit_report.md`.
- Phase 5.2 parity-workflow and fixture-contract verification:
  `PYTHONPATH=src pytest -q tests/test_reference_parity_runner.py tests/test_parity_fixture_manifests.py tests/test_parity_phase3_scaffolding.py`
  with `8 passed`.
- Phase 5.2 focused backend/CLI regression:
  `PYTHONPATH=src pytest -q tests/test_phase5_crosswalk.py tests/test_cli_app.py -k "phase5 or crosswalk or parity"`
  with `6 passed, 76 deselected`.
- Phase 5.2 CLI execution check:
  `PYTHONPATH=src python -m fluxforge.cli.app parity-check --scope algorithm --fixture-id spectrum_io_normalization_algorithm_case --output /tmp/phase5_2_parity_check.json`
  with `1 passed, 0 failed`.
- Phase 5.2 GUI scope/fixture-control verification:
  `PYTHONPATH=src pytest -q tests/test_modern_gui_shell.py -k "main_window_restores_saved_workflow_state_across_sessions"`
  with `1 passed, 14 deselected`, plus refreshed
  `tests/gui_phase5_parity_probe.py` and Playwright audit rerun for
  `artifacts/gui_review/phase5_parity/` (`1 audited, 0 failing`).
- Phase 5.3 workflow-bundle and GUI execution verification:
  `PYTHONPATH=src pytest -q tests/test_reference_parity_runner.py tests/test_parity_fixture_manifests.py tests/test_parity_phase3_scaffolding.py tests/test_phase5_crosswalk.py tests/test_modern_gui_shell.py -k "phase5 or parity or manifest or main_window_restores_saved_workflow_state_across_sessions or phase5_parity_panel_runs_workflow_fixture_bundle"`
  with `15 passed, 14 deselected`.
- Phase 5.3 CLI workflow execution check:
  `PYTHONPATH=src python -m fluxforge.cli.app parity-check --scope workflow --fixture-id roi_statistics_workflow_case --output /tmp/phase5_3_parity_check.json`
  with `1 passed, 0 failed`.
- Phase 5.3 probe and browser-lane refresh:
  `PYTHONPATH=src /usr/bin/python tests/gui_phase5_parity_probe.py artifacts/gui_review/phase5_parity`
  plus
  `node tests/gui_gallery_playwright_audit.js artifacts/gui_review/phase5_parity artifacts/gui_review/phase5_parity/playwright_audit`
  with `1 audited, 0 failing`.
- Playwright gallery audit verification:
  `NODE_PATH=/tmp/pw-audit/node_modules node tests/gui_gallery_playwright_audit.js artifacts/gui_review /tmp/gui_playwright_audit`
  with `21 audited, 0 failing`; evidence copied to
  `artifacts/gui_review/playwright_audit/audit_report.json` and
  `artifacts/gui_review/playwright_audit/audit_report.md`.
- Expanded GUI regression verification (dialogs + widgets + workspaces + shell):
  `PYTHONPATH=src pytest -q tests/test_gui_dialogs_qt.py tests/test_gui_widgets_qt.py tests/test_analysis_workspace_qt.py tests/test_calibration_workspace_qt.py tests/test_module3_workflows_qt.py tests/test_predictive_dashboard_qt.py tests/test_unfolding_workspace_qt.py tests/test_modern_gui_shell.py`
  with `91 passed, 9 warnings`.
- Changed-file quality-gate validation (non-data Python files):
  `ruff check --select I <changed files under src/fluxforge and tests>`
  with local dry-run passing after import normalization.
- Post-gate regression slice:
  `PYTHONPATH=src pytest -q tests/test_optimization_common.py tests/test_optimization_difom.py tests/test_optimization_fim.py tests/test_optimization_mwdcs.py tests/test_optimization_bassd.py tests/test_optimization_stbdmr.py tests/test_modern_gui_shell.py -k "modern_shell_reuses_shared_demo_and_selection_helpers or test_"`
  with `37 passed`.

Use `docs/ROADMAP_EXECUTION_STATUS.md` for the live verification snapshot after this date.

## 5. Planned Testing Roadmap

| Step | Testing Deliverable | Status |
|---|---|---|
| 3.21 | Curate repo-backed parity fixtures and manifests under `tests/spectra/reference_parity/` with source repo, source case, workflow, expected outputs, tolerances, and provenance notes | Planned |
| 3.22 | Add algorithm-level parity tests for parsing, calibration, peak search, fit, activity, detector-response, detection-limit, dose/shielding, and k0 workflows | Planned |
| 3.23 | Add end-to-end workflow parity suites covering "same input -> same workflow -> nearly same result" behavior for each source family and tutorial/example dataset | Planned |
| 3.27 | Expand Qt workflow tests, native probes, artifact galleries, and release-blocking acceptance checklists for every new parity workspace | Complete (repo) |

Additive testing overlay from the activation / FISPACT-style supplement:

- Add `tests/activation_inventory/` for pure decay, Bateman-chain, EOI reconstruction, observable regression, CSV export, and plot smoke tests.
- Extend `3.21` through `3.23` with masking-analysis fixtures, optimization-grid fixtures, second-irradiation planner cases, long-term dose endpoint cases, and `.ffexp` benchmark export fixtures.
- Add library-governance regression coverage for opt-in library download selection, GUI/CLI custom-library registration by path, reserved bundled-ID enforcement, collision-safe aliasing, and provenance capture for user libraries.
- Add configuration-audit tests that push removable hardcoded defaults behind registries or explicit config while preserving the intentional carve-outs for QuantumGold, PeakEasy, standards workflows, and RAFM irradiation-analysis cases.

## 6. Additional Parity and Validation Requirements

- Add InterSpec-style tutorial/example regression packs for nuclide ID,
  energy calibration, detector-response creation, buried-source/shielding,
  relative efficiency, uranium enrichment, and batch analysis.
- Add machine-readable handoff-bundle schema tests covering readiness flags,
  activity payloads, corrected activation rates, and optional adjusted spectra.
- Add comparator, k0, activation-only, and arbitrary-time activity tests once
  those workflows land in `3.18` through `3.20`.
- Add dedicated regression coverage for spectrum-level activation review:
  isotope-at-EOI CSV export, uncertainty propagation from counts/efficiency/emission
  inputs, and saved half-life/Bateman review plots in both CLI and Qt workflows.
- Keep active regression coverage around the governed library registry: bundled
  GSA/NASA source loading, capability-bucket filtering so prompt/reaction-gamma
  tables do not leak into activity-review choices, collision-safe user-library
  registration/removal in CLI and Qt, and GUI activity-unit selectors for
  activity/inventory views.
- Add covariance-aware GLS/STAYSL experimental-workflow tests when the governed
  inverse-analysis surfaces land.
- Add benchmark experimental bundle tests for `.ffexp`, `activities_at_irradiation.csv`,
  `inventory_timeseries.csv`, `dominant_contributors.csv`, `dose_endpoints.csv`,
  `optimization_grid.csv`, and plot-manifest sidecars.
- Add GUI and CLI tests for user-supplied library locations, opt-in downloadable
  library families, duplicate-name rejection or alias reassignment, and protection
  against accidental overwrite of bundled libraries.
- Add masking, optimization, second-irradiation, and shutdown-to-100-year dose
  regression coverage once those `3N` additive workflows land.
- Keep regression coverage around workflow-preset persistence, active-workflow
  restore, and the built-in `quantumgold-workflow` / `astm-ldrd-irradiation`
  starting points as the modern shell continues to be split into smaller modules.

## 7. Artifact and Probe Requirements

- Every new interaction-heavy GUI workspace needs a native probe path and a saved review state.
- Every new parity fixture family needs golden outputs and declared tolerances.
- Every new export/report contract needs a stable JSON/CSV/PDF validation path.
- Every new standards or governed workflow needs explicit compliance-oriented assertions.

## 8. Archived Testing Inputs

Testing-related source material retained for later reference includes:

- `docs/archive/planning_snapshot_2026-04-06/FLUXFORGE_CONSOLIDATED_MASTER.md`
- `docs/archive/planning_snapshot_2026-04-06/FluxForge_Additions_v3_Final.md`
- `docs/archive/planning_snapshot_2026-04-06/FluxForge_Final_Additions.md`
- `docs/archive/planning_snapshot_2026-04-06/FluxForge_NAA_InterSpec_STAYSL_Addition.md`
- `docs/archive/legacy_plans/testing_gui_survey.md`
- `docs/archive/legacy_plans/testing_repo_capability_matrix.md`
