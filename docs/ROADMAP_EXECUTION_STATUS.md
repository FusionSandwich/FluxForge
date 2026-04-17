# FluxForge Roadmap Execution Status

**Date:** 2026-04-17  
**Controlling roadmap document:** `docs/FLUXFORGE_CONSOLIDATED_MASTER.md`  
**Companion GUI document:** `docs/GUI_PLAN.md`  
**Companion testing document:** `docs/FluxForge_Testing_Master.md`  
**Machine-readable tracker:** `.github/project-management/implementation_steps.json`

## Stepwise Execution Rule

The roadmap is now tracked in two dimensions:

- `repo_status`: what already exists in the repository
- `sequence_status`: what counts as the formal next step in the roadmap order

This matters because some additive scaffolding already exists for early Phase 1, but the
original documents require Stage 0 to be fully closed before Phase 1 can count as
officially in progress.

## Current Sequence Gate

| Step | Sequence Status | Repo Status | Meaning |
|---|---|---|---|
| 3.18 | `next` | `in-progress` | Identification / activity / reference parity is now in progress. The repo now includes bundled GSA-v4 edited plus natural libraries; NASA-gamma common-lab, natural, CapGam, IAEA capture, delayed-activation, Baghdad inelastic, and TALYS 14 MeV reference libraries; ENDF/B-VIII supplement; ICRP-107 plus Kayzero registrations; GUI/CLI user-library registration with reserved bundled names; and activity-review plus inventory views with uncertainty-bearing decay/Bateman outputs and GUI activity-unit selectors. |
| 4.1 | `pending` | `not-started` | HAL driver work remains on the roadmap, but it is formally deferred until the new Phase 3B offline-parity module closes. |

## Additive Roadmap Overlays

The numbered sequence gate above remains the controlling roadmap map. The
following overlays were adopted additively and do not replace that map.

### Activation / FISPACT-Style `3N` Overlay

- `3N.1` through `3N.4` extend the active `3.18` through planned `3.25` work with
  explicit irradiation schedules, inventory states, arbitrary-time solvers,
  Bateman/half-life plots, and time-evolution workspaces.
- `3N.5` through `3N.14` extend `3.19`, `3.20`, and `3.25` with FISPACT-style
  observables, dominant-contributor plots, masking analysis, NAA mass and
  concentration inference, optimization sweeps, second-irradiation planners,
  and shutdown-through-100-year dose studies.
- `3N.15` extends the reporting/export track with a benchmark experimental bundle
  contract centered on `.ffexp`.
- `3N.16` extends `3.21` through `3.23` with activation-inventory fixtures,
  plot smoke tests, and workflow-level validation.
- The additive issue-seed set `3N.1` through `3N.16` is adopted for future
  tracker synchronization and should be kept distinct from the controlling
  `3.18` through `3.27` sequence.

### Library Governance and De-Hardcoding Overlay

- Library-management planning now explicitly includes opt-in downloadable
  libraries, GUI and CLI registration of external library locations, reserved
  bundled IDs, and collision-safe aliasing so user libraries cannot silently
  overwrite built-in sources.
- InterSpec compatibility planning now explicitly includes a SandiaDecay
  bridge track (decay XML + reaction-gamma XML + reference-line overlays) as
  a governed adapter layer under Step `3.18`, with validation gates before any
  imported value is surfaced as a recommended display value.
- A de-hardcoding cleanup pass is now planned across the parity work: move
  removable hardcoded workflows and values behind registries or explicit config,
  while keeping the intentional carve-outs for QuantumGold parity, PeakEasy
  parity, governed standards workflows such as ASTM and k0-NAA, and RAFM
  irradiation-analysis paths.

## Completed In Sequence

| Step | Repo Status | Evidence |
|---|---|---|
| S0.1 | `complete` | Live GitHub labels, milestones, issue templates, board config source, and successful `Sync Project Planning` run on 2026-03-30 |
| S0.2 | `complete` | The original Stage 0 epic seed set from `.github/project-management/issues.json` is live on GitHub; newer Phase 3B epic additions are present in-repo and await sync |
| S0.3 | `complete` | The original Stage 0 seed issue set from `.github/project-management/issues.json` is live on GitHub; newer Phase 3B issue additions are present in-repo and await sync |
| S0.4 | `complete` | `docs/adr/ADR-001` through `ADR-007` |
| S0.5 | `complete` | `docs/adr/`, `tests/spectra/`, `.github/ISSUE_TEMPLATE/` |

| Step | Repo Status | Evidence |
|---|---|---|
| 1.1 | `complete` | `src/fluxforge/gui/`, `plugins/`, `hal/`, `reporting/`, `standards/`, `unfolding/` |
| 1.2 | `complete` | `src/fluxforge/plugins/registry.py` |
| 1.3 | `complete` | `src/fluxforge/gui/main_window.py` + `src/fluxforge/gui/panels/modern_shell.py` now implement the Qt shell with dock zones A-F, QDockWidget layout, and layout persistence |
| 1.4 | `complete` | `src/fluxforge/gui/mode_manager.py` + `src/fluxforge/gui/widgets/mode_switcher.py` now persist and surface Simple / Expert / Standards |
| 1.5 | `complete` | `src/fluxforge/gui/selection_bus.py` is now wired across the shell surfaces |
| 1.6 | `complete` | `src/fluxforge/gui/spectrum_canvas.py` + `src/fluxforge/gui/backends/pyqtgraph_backend.py` provide the abstraction and default backend |
| 1.7 | `complete` | `src/fluxforge/gui/backends/vispy_backend.py` provides the required additive stub |
| 1.8 | `complete` | `src/fluxforge/gui/spectrum_canvas.py` implements `HierarchicalSpectrumBuffer` |
| 1.9 | `complete` | `src/fluxforge/hal/base.py` + `src/fluxforge/io/session.py` + `src/fluxforge/io/spe.py` now cover the device registry, mock-device snapshotting, and spectrum source metadata wiring |
| 1.10 | `complete` | `src/fluxforge/io/spe.py` + `src/fluxforge/io/session.py` now provide the `.ffs` session container, round-trip persistence, and GPS/source fields |
| 1.11 | `complete` | `src/fluxforge/io/reader_factory.py` + `src/fluxforge/io/spc.py` + `src/fluxforge/io/spectrum_csv.py` + `src/fluxforge/gui/file_workflow.py` + `src/fluxforge/gui/main_window.py` now cover factory dispatch, SPC/CSV support, recent files, and drag/drop opening |
| 1.12 | `complete` | `src/fluxforge/io/n42.py` + `src/fluxforge/resources/schemas/n42_2012.xsd` now validate 2012 exports with lxml |
| 1.13 | `complete` | `src/fluxforge/data/nuclide_library.py` now builds the SQLite schema with `decay_chains` and FTS5-backed search when available |
| 1.14 | `complete` | `src/fluxforge/gui/panels/modern_shell.py` + `src/fluxforge/gui/nuclide_search.py` + `src/fluxforge/gui/selection_bus.py` now provide live nuclide search and instant overlay wiring |
| 1.15 | `complete` | `src/fluxforge/gui/panels/modern_shell.py` now reserves the QA & Standards sidebar section |
| 1.16 | `complete` | `src/fluxforge/gui/panels/modern_shell.py` reserves the Dashboard tab in Zone C |
| 1.17 | `complete` | `src/fluxforge/gui/widgets/hardware_led.py` + `src/fluxforge/gui/main_window.py` add the status-bar hardware LED |
| 1.18 | `complete` | `src/fluxforge/gui/theme_manager.py` + `src/fluxforge/gui/themes/` provide dark/light/system theme handling |
| 1.19 | `complete` | `tests/spectra/`, `tests/test_n42.py`, `tests/test_cnf_io.py`, `tests/test_csv_readers.py`, and CI provide the current Phase 1 test harness baseline |
| 2.1 | `complete` | `src/fluxforge/core/calibration.py` + `src/fluxforge/gui/dialogs/calibration_dialog.py` + `src/fluxforge/gui/main_window.py` now provide the live Qt calibration workspace with embedded spectrum review, energy/FWHM fits, ASTM E181 order locking, and residual-first diagnostics |
| 2.2 | `complete` | `src/fluxforge/core/calibration.py` + `src/fluxforge/gui/dialogs/calibration_dialog.py` + `src/fluxforge/gui/main_window.py` + `src/fluxforge/gui/panels/modern_shell.py` now provide the additive quick-slider calibration mode and launch points |
| 2.3 | `complete` | `src/fluxforge/core/calibration.py` + `src/fluxforge/io/spe.py` + `src/fluxforge/gui/dialogs/calibration_dialog.py` now provide InterSpec-style deviation-pair fine tuning and persistence |
| 2.4 | `complete` | `src/fluxforge/core/peak_fitting.py` + `src/fluxforge/gui/dialogs/calibration_dialog.py` now provide draggable ROI fitting with live Gaussian diagnostics and table application actions |
| 2.5 | `complete` | `src/fluxforge/core/peak_fitting.py` + `src/fluxforge/plugins/registry.py` now register the additive skewed-Gaussian fitter beside the default Gaussian fitter |
| 2.6 | `complete` | `src/fluxforge/gui/widgets/method_selector.py` + `src/fluxforge/gui/dialogs/calibration_dialog.py` now drive fitter choice from the shared registry with Standards-mode locking |
| 2.7 | `complete` | `src/fluxforge/gui/main_window.py` + `src/fluxforge/gui/phase2_workspace.py` + `src/fluxforge/gui/panels/modern_shell.py` now route peak-table mutations through a shared undo/redo command stack |
| 2.8 | `complete` | `src/fluxforge/gui/panels/modern_shell.py` now provides the live Peak Table with SelectionBus synchronization, candidate columns, and status dots |
| 2.9 | `complete` | `src/fluxforge/core/phase2_analysis.py` + `src/fluxforge/gui/dialogs/auto_peak_review_dialog.py` + `src/fluxforge/gui/main_window.py` now provide auto peak search with review dialog and `Ctrl+A` wiring |
| 2.10 | `complete` | `src/fluxforge/core/phase2_analysis.py` now registers and applies a Bayesian line-matching engine through the modern peak workflow |
| 2.11 | `complete` | `src/fluxforge/core/phase2_analysis.py` + `src/fluxforge/gui/dialogs/efficiency_dialog.py` + `src/fluxforge/gui/panels/modern_shell.py` now provide the efficiency calibration workflow with registered models |
| 2.12 | `complete` | `src/fluxforge/core/phase2_analysis.py` + `src/fluxforge/gui/panels/modern_shell.py` now compute activities with propagated counting uncertainty in the Activity Results panel |
| 2.13 | `complete` | `src/fluxforge/core/phase2_analysis.py` + `src/fluxforge/physics/decay_chain.py` now expose source-age correction and Bateman-based summaries in the activity workflow |
| 2.14 | `complete` | `src/fluxforge/core/phase2_analysis.py` + `src/fluxforge/gui/panels/modern_shell.py` now provide simple, scaled, and statistical background subtraction modes |
| 2.15 | `complete` | `src/fluxforge/gui/backends/pyqtgraph_backend.py` now renders mini residual subplots for active peaks in Expert and Standards modes |
| 2.16 | `complete` | `src/fluxforge/core/phase2_analysis.py` + `src/fluxforge/io/n42.py` + `src/fluxforge/gui/panels/modern_shell.py` now surface GPS extraction through the Survey Map panel |
| 2.17 | `complete` | `src/fluxforge/gui/panels/modern_shell.py` + `src/fluxforge/gui/main_window.py` now provide foreground/background/secondary overlay spectrum tabs above the canvas |
| 2.18 | `complete` | `src/fluxforge/gui/panels/modern_shell.py` + `src/fluxforge/gui/phase2_workspace.py` now provide pinned nuclides and peak tagging in the modern shell |
| 2.19 | `complete` | `src/fluxforge/core/phase2_analysis.py` + `src/fluxforge/gui/backends/pyqtgraph_backend.py` now render dotted cascade-sum overlays for pinned nuclides |
| 2.20 | `complete` | `src/fluxforge/core/peak_fitting.py` + `src/fluxforge/gui/dialogs/calibration_dialog.py` now provide the Bayesian Gaussian ROI fitter registered beside the Gaussian and skew options |
| 2.21 | `complete` | `src/fluxforge/gui/panels/modern_shell.py` + `src/fluxforge/gui/phase2_workspace.py` now provide manual isotope assignment, replacement, and clear actions for selected peaks |
| 2.22 | `complete` | `src/fluxforge/gui/panels/modern_shell.py` + `src/fluxforge/gui/nuclide_search.py` + `src/fluxforge/data/nuclide_library.py` now provide a centroid-driven isotope browser with a default ±2 keV window and active-library filtering |
| 2.23 | `complete` | `src/fluxforge/core/analysis_workspace.py` + `src/fluxforge/gui/panels/modern_shell.py` + `src/fluxforge/gui/backends/pyqtgraph_backend.py` + `src/fluxforge/plots/spectrum_inspection.py` + `src/fluxforge/cli/app.py` now render gamma-phenomena guidance for Compton edge/backscatter/annihilation/escape with estimated feature heights, auto continuum-driver selection, and saved spectrum-plot overlays/reports |
| 2.24 | `complete` | `src/fluxforge/gui/panels/modern_shell.py` + `src/fluxforge/gui/selection_bus.py` + `src/fluxforge/gui/backends/pyqtgraph_backend.py` now keep peak IDs editable after automated workflows with SelectionBus overlays and undo/redo |
| 3.1 | `complete` | `src/fluxforge/core/unfolding_inputs.py` + `src/fluxforge/unfolding/base.py` + `src/fluxforge/unfolding/gravel.py` + `src/fluxforge/unfolding/__init__.py` now provide the registry-backed GRAVEL API, shared pyunfold-style nonnegative input validation, the built-in unfolder registration path, and local reference-parity coverage in `tests/test_unfolding_reference_parity.py` |
| 3.2 | `complete` | `src/fluxforge/unfolding/maxed.py` + `src/fluxforge/unfolding/gravel.py` + `src/fluxforge/unfolding/__init__.py` now provide the registry-backed MAXED method with uncertainty estimates and shared pytest coverage in `tests/test_unfolding_registry.py` and `tests/test_unfolding_workflows.py` |
| 3.3 | `complete` | `src/fluxforge/unfolding/rmle.py` + `src/fluxforge/unfolding/gravel.py` + `src/fluxforge/unfolding/__init__.py` now provide the registry-backed RMLE default, while `src/fluxforge/gui/dialogs/unfolding_dialog.py`, `src/fluxforge/cli/app.py`, and `src/fluxforge/workflows/spectrum_unfolding.py` expose it through the modern Qt workspace, CLI, and public workflow |
| 3.4 | `complete` | `src/fluxforge/unfolding/ml_seed.py` + `src/fluxforge/unfolding/rmle.py` + `src/fluxforge/workflows/spectrum_unfolding.py` now provide ML Seed as both a standalone method and an RMLE/GRAVEL initializer |
| 3.5 | `complete` | `src/fluxforge/gui/dialogs/unfolding_dialog.py` + `tests/test_unfolding_workspace_qt.py` + `tests/gui_unfolding_workspace_probe.py` now provide the comparison-mode unfolding dialog with response heatmap, uncertainty bands, and response-source loading |
| 3.6 | `complete` | `src/fluxforge/ml/peak_analysis.py` + `src/fluxforge/core/phase2_analysis.py` + `src/fluxforge/gui/panels/modern_shell.py` now provide registry-backed ML peak proposals in the modern peak table |
| 3.7 | `complete` | `src/fluxforge/standards/e181.py` + `src/fluxforge/standards/__init__.py` now provide ASTM E181 compliance checks and registry registration |
| 3.8 | `complete` | `src/fluxforge/standards/e1297.py` now provides Currie-method MDA evaluation through the standards registry |
| 3.9 | `complete` | `src/fluxforge/standards/e1218.py` + `src/fluxforge/standards/c1232.py` now provide calibration-bracketing and lab-QA checks |
| 3.10 | `complete` | `src/fluxforge/standards/c1030.py` + `src/fluxforge/gui/dialogs/pu_isotopics_dialog.py` + `src/fluxforge/gui/main_window.py` now provide the ASTM C1030 Pu isotopics backend and Expert/Standards wizard |
| 3.11 | `complete` | `src/fluxforge/standards/qa_monitor.py` + `src/fluxforge/gui/dialogs/qa_history_dialog.py` + `src/fluxforge/gui/dialogs/standards_review_dialog.py` + `src/fluxforge/gui/main_window.py` now provide SQLite QA history, drift status, `Tools → QA History`, and a direct `Run ASTM Check` review surface from the modern sidebar |
| 3.12 | `complete` | `src/fluxforge/standards/__init__.py` + `src/fluxforge/gui/widgets/method_selector.py` + `src/fluxforge/gui/panels/modern_shell.py` now surface registry-driven standards locks and padlock summaries in the modern GUI |
| 3.13 | `complete` | `src/fluxforge/reporting/engine.py` + `src/fluxforge/reporting/templates/` + `src/fluxforge/gui/dialogs/report_export_dialog.py` now provide the Jinja2 report engine with the three bundled templates and HTML/PDF export |
| 3.14 | `complete` | `src/fluxforge/core/batch_analysis.py` + `src/fluxforge/gui/panels/modern_shell.py` now provide the ProcessPoolExecutor-backed batch queue, visible progress tracking, and JSON/CSV outputs |
| 3.15 | `complete` | `src/fluxforge/unfolding/gpu_backend.py` + `src/fluxforge/ml/peak_analysis.py` + `src/fluxforge/core/batch_analysis.py` now provide optional CuPy backend selection with clean CPU fallback |
| 3.16 | `complete` | `src/fluxforge/core/analysis_workspace.py` + `src/fluxforge/gui/analysis_workspace.py` + `src/fluxforge/gui/panels/modern_shell.py` + `src/fluxforge/cli/app.py` now provide registry-backed peak-search selection, explicit ROI/background workflows, overlap decomposition, ROI statistics, the Qt ROI Tools panel, and matching CLI commands covered by `tests/test_roi_analysis_core.py`, `tests/test_cli_app.py`, and `tests/test_analysis_workspace_qt.py` |
| 3.17 | `complete` | `src/fluxforge/gui/dialogs/calibration_dialog.py` + `src/fluxforge/gui/dialogs/efficiency_dialog.py` + `src/fluxforge/gui/library_manager.py` + `src/fluxforge/gui/panels/modern_shell.py` + `src/fluxforge/gui/main_window.py` now provide detector-slot recall, preserved/fine-tuned calibration flows, NASA smart seeding, all four registered efficiency models, and standards-aware calibration/identification source locking covered by `tests/test_calibration_workspace_qt.py`, `tests/test_analysis_workspace_qt.py`, and `tests/test_cli_app.py` |
| 4P.1 | `complete` | `src/fluxforge/core/predictive.py` + `src/fluxforge/gui/panels/modern_shell.py` now forecast ROI time-to-target counts from offline spectra |
| 4P.2 | `complete` | `src/fluxforge/core/predictive.py` + `src/fluxforge/gui/panels/modern_shell.py` now project dead-time trend and saturation warnings without live MCA transport |
| 4P.3 | `complete` | `src/fluxforge/core/predictive.py` + `src/fluxforge/gui/main_window.py` + `src/fluxforge/gui/panels/modern_shell.py` now forecast recalibration timing from `QAMonitor` history and surface it in the dashboard, sidebar, and status bar |
| 4P.4 | `complete` | `src/fluxforge/gui/panels/modern_shell.py` + `src/fluxforge/gui/nuclide_search.py` now provide a saved-list nuclide workbench in the modern sidebar |
| 4P.5 | `complete` | `src/fluxforge/gui/nuclide_search.py` + `src/fluxforge/gui/panels/modern_shell.py` now surface parent/daughter relationships, age-aware line tables, specific activity, and dose context from the active library |
| 4P.6 | `complete` | `src/fluxforge/gui/panels/modern_shell.py` now provides an editable nuclide-mixture builder with normalization and combined overlay publishing |
| 4P.7 | `complete` | `src/fluxforge/gui/main_window.py` + `src/fluxforge/gui/backends/pyqtgraph_backend.py` + `src/fluxforge/gui/panels/modern_shell.py` now expose direct `Log Scale` and `Peak Labels` toggles on the primary canvas |

## Scheduled / Not Started

| Step | Sequence Status | Repo Status | Meaning |
|---|---|---|---|
| 3.18 | `next` | `in-progress` | Implement the remaining identification/activity/reference parity work on top of the now-bundled GSA-v4 edited/natural, NASA-gamma common-lab/natural/capture/delayed-activation/inelastic families, ENDF/B-VIII supplement, ICRP-107 plus Kayzero source registrations, collision-safe GUI/CLI user-library registration, and GUI activity-unit selectors, then finish relative activity, source-age overlays, deeper decay-dataset consumption, and the remaining parity surfaces. |
| 3.19 | `pending` | `in-progress` | Detection-limit/dose/shielding workspaces remain pending, but the Phase 6 irradiation-optimization slice is now active: CLI `masking-review`, `optimization-sweep`, `second-irradiation-plan`, `activity-review`, and `inventory-review` flows plus the shared support-artifact builder in `src/fluxforge/workflows/irradiation_optimization.py` now produce masking, inventory, optimization-grid, recommended-schedule, and long-term dose-endpoint outputs. |
| 3.20 | `pending` | `in-progress` | CLI now includes explicit `file-query` and `batch-compare` commands in `src/fluxforge/cli/app.py`, and the Phase 6 benchmark experimental-bundle track is now active via `ffexp-export` plus GUI `.ffexp` export in `src/fluxforge/gui/panels/phase6.py`; GUI workflow/workspace persistence is now also active through `src/fluxforge/gui/workflow_presets.py` + `src/fluxforge/gui/main_window.py` with built-in `quantumgold-workflow` and `astm-ldrd-irradiation` presets. ROI-statistics and k0 workflows remain active, and archive/workbench depth still needs broader GUI parity follow-through. |
| 3.21 | `pending` | `in-progress` | Fixture-manifest scaffolding now includes expanded source-linked case placeholders under `tests/spectra/reference_parity/cases/` and `tests/activation_inventory/fixtures/` (including second-irradiation planning), with contract checks in `tests/test_parity_fixture_manifests.py` and `tests/test_parity_phase3_scaffolding.py`. |
| 3.22 | `pending` | `in-progress` | Initial algorithm-level parity scaffolding is now present via `parity_scope=algorithm` manifests plus discovery checks in `tests/test_parity_phase3_scaffolding.py`; full parser/calibration/fit/activity/dose/k0 golden comparisons remain pending. |
| 3.23 | `pending` | `in-progress` | Initial workflow-level parity scaffolding is now present via `parity_scope=workflow` manifests (including activity/inventory and second-irradiation placeholders) plus scaffold verification tests; source-linked end-to-end golden-result suites remain pending. |
| 3.24 | `pending` | `not-started` | Add direct-manipulation canvas parity: peak add/delete/move, ROI dragging, background handles, and overlay-role actions. |
| 3.25 | `pending` | `in-progress` | Add dedicated Qt workspaces for ROI statistics, detection limit, dose/shielding, relative activity, file query/batch compare, reference libraries, and k0 reporting. The first Phase 6 parity surfaces are now live in the modern shell via `Line Interference / Masking`, `Irradiation Optimizer`, and `Second Irradiation` tabs backed by `src/fluxforge/gui/panels/phase6.py`, and their state now round-trips through the saved-workflow system. |
| 3.26 | `pending` | `complete` | GUI polish parity is now implemented via saved theme profiles (`src/fluxforge/gui/mode_manager.py` + `src/fluxforge/gui/widgets/mode_switcher.py`), stronger graph-table synchronization (`src/fluxforge/gui/backends/pyqtgraph_backend.py` + `src/fluxforge/gui/panels/modern_shell.py`), clearer launch/discovery actions (`src/fluxforge/gui/main_window.py`), and the maintainability split of the oversized shell into `modern_shell_center.py`, `modern_shell_sidebar.py`, `modern_shell_context.py`, and `modern_shell_shared.py`, with Qt coverage in `tests/test_analysis_workspace_qt.py`, `tests/test_module3_workflows_qt.py`, and `tests/test_modern_gui_shell.py`. |
| 3.27 | `pending` | `complete` | GUI verification/release acceptance is now implemented with expanded Qt/CLI coverage (`tests/test_modern_gui_shell.py`, `tests/test_module3_workflows_qt.py`, `tests/test_cli_app.py`), native probe evidence (`tests/gui_phase327_release_probe.py` and `artifacts/gui_review/phase327_probe/index.html`), and a release-blocking checklist at `docs/PHASE3_27_RELEASE_CHECKLIST.md` plus CLI validation command `gui-acceptance-check`. |
| 4.1 | `pending` | `not-started` | Implement the first real HAL drivers after the offline-parity module closes. |
| 4.2 | `pending` | `not-started` | Implement device discovery and thumbnail/device-list surfaces once HAL transport exists. |
| 4.3 | `pending` | `not-started` | Implement the live Digital Twin dashboard after HAL transport and device telemetry exist. |
| 4.4 | `pending` | `not-started` | Implement the live spectrogram panel after time-energy acquisition lands. |

## GUI Direction

- The Qt redesign under `src/fluxforge/gui/` is now the primary GUI implementation path.
- The prior Tk application under `src/fluxforge_gui/` is intentionally retained as a
  legacy/archive fallback while the redesign reaches feature parity.
- Older planning references are archived under
  `docs/archive/planning_snapshot_2026-04-06/`.
- CI keeps lightweight modern-shell checks in the regular push path and limits the
  old desktop automation flow to manual dispatch.

## Verification

- New scaffolding modules compile under the workspace Python.
- Direct Python verification passed for the tracker assets, ordered step tracker, plugin
  registries, GUI scaffolding, and mock HAL device.
- `pytest` was upgraded in the user environment to `8.4.2`.
- The TensorFlow-specific tests were run explicitly in this round, and the only remaining skips are the CUDA library checks that are correct for this CPU-only workspace.
- The full suite now passes in this workspace: `1279 passed, 2 skipped`.
- Latest continuation verification (2026-04-17) also passed:
  `10 passed, 67 deselected` for the parity/fixture/CLI slice,
  `49 passed` for targeted Qt workflow tests,
  and `204 passed` for the broad Phase 3 regression slice across unfolding, module3, analysis, calibration, and CLI.
- Native GUI probe galleries were refreshed in this continuation at
  `artifacts/gui_review/phase326_probe/`,
  `artifacts/gui_review/phase317_calibration_probe/`,
  `artifacts/gui_review/phase31x_unfolding_probe/`, and
  `artifacts/gui_review/phase327_probe/`, and the `gui-acceptance-check` CLI command wrote
  `artifacts/gui_review/phase327_probe/gui_acceptance_check.json`.
- Phase 6 irradiation-optimization verification was refreshed on 2026-04-17
  against the real RAFM example corpus rather than synthetic activity-review
  payloads:
  `PYTHONPATH=src pytest -q tests/test_cli_app.py -k "second_irradiation_plan_writes_json_and_csv_outputs or optimization_sweep_builds_candidates_from_activity_review or ffexp_export_packages_phase6_products"`
  with `3 passed`,
  `PYTHONPATH=src pytest -q tests/test_analysis_workspace_qt.py -k "masking_review_panel_runs_and_exports_tables or optimization_workspace_panel_runs_and_exports_phase6_bundle or optimization_workspace_panel_advanced_guard_and_second_irradiation_panel"`
  with `3 passed, 29 deselected`,
  and `PYTHONPATH=src /usr/bin/python tests/gui_phase6_optimization_probe.py artifacts/gui_review/phase6_optimization_probe`
  which generated `artifacts/gui_review/phase6_optimization_probe/index.html`
  plus `phase6_probe.ffexp` from sample `RAFM4-C_15dEOI`.
- The refreshed Phase 6 gallery was reviewed in the Playwright/browser lane via
  the generated `index.html`; all five screenshots loaded successfully and the
  gallery summary matched the native probe counts (`inventory_rows=150`,
  `masking_rows=6`, `optimization_rows=10`, `second_irradiation_rows=5`).
- Saved-workflow persistence and the split modern shell were also regression-checked
  on 2026-04-17 via `PYTHONPATH=src pytest -q tests/test_modern_gui_shell.py`
  with `14 passed`, covering built-in/user preset persistence plus active-workflow
  restore across GUI sessions.
- The redesigned Qt GUI was verified beyond unit tests in this round:
  mouse-driven peak picking in the calibration canvas, governed data-library
  selectors in the sidebar, library-assisted calibration-line assignment, and
  both Manual and Standards calibration workflows were exercised successfully.
- Additional Phase 2 completion coverage now exists for the redesigned Qt shell:
  auto peak review, undo/redo, Bayesian matching, efficiency fitting, activity
  calculation, background subtraction, pinned nuclides, cascade lines, survey
  map rendering, and multi-spectrum tab switching all ran successfully in the
  modern GUI tests.
- Additional Qt GUI identification coverage now exists for the reopened Phase 2
  items as well: centroid-driven isotope browsing, adjustable energy windows,
  manual isotope reassignment after Bayesian matching, and gamma-phenomena guide
  overlays are now exercised in the modern GUI tests.
- Phase 3.1 now has dedicated core coverage as well: the new unfolding contract,
  GRAVEL adapter, and built-in unfolder registration path are exercised in
  `tests/test_unfolding_registry.py`, including parity checks against the
  existing iterative GRAVEL solver and a no-GUI-import boundary check.
- Public unfolding workflow coverage is now broader as well: `tests/test_unfolding_workflows.py`
  exercises the public neutron and gamma unfolding entrypoints, CLI methods,
  registry-backed GRAVEL adapter, standards-style `SpectrumUnfolder`, `quick_unfold`,
  and the optional IBU/RMLE wrappers on synthetic spectra, including uncertainty
  payload checks and pyunfold-style negative-input rejection checks.
- External-reference unfolding parity coverage now exists as well:
  `tests/test_unfolding_reference_parity.py` compares FluxForge GRAVEL and MLEM
  against the local Neutron-Unfolding repo on its bundled neutron dataset using
  the same tolerance settings, and compares `NeutronUnfolderIBU` against the local
  pyunfold repo on five published/example cases while checking unfolded values,
  uncertainties, and iteration statistics.
- Phase 3 GUI coverage now exists in the redesigned Qt shell as well:
  `tests/test_unfolding_workspace_qt.py` exercises the unfolding workspace,
  mouse-driven algorithm comparison, RMLE default selection, uncertainty-visible
  results table, and main-window launch wiring for the native unfolding dialog.
- A native unfolding review probe now exists at
  `tests/gui_unfolding_workspace_probe.py` so the modern dialog can be captured
  and inspected visually in the same artifact-gallery flow used for the Phase 2 shell.
- Additional Module 3 backend coverage now exists in `tests/test_module3_backends.py`:
  the standards registry, Currie MDA implementation, C1030 isotopics backend,
  QA monitor, reporting engine, response-matrix loader, optional GPU selector,
  ML peak engine, and batch-analysis writer are all exercised directly.
- Additional Module 3 GUI coverage now exists in `tests/test_module3_workflows_qt.py`:
  the QA History dialog, direct `Run ASTM Check` sidebar action, clickable
  hardware LED dashboard shortcut, report export dialog, C1030 wizard, batch
  queue tab, ML peak table action, and sidebar QA/lock summary are now exercised
  in the redesigned Qt shell.
- A native Module 3 review probe now exists at `tests/gui_module3_workflows_probe.py`
  so the QA history, ASTM standards review, report export, Pu isotopics, and
  batch queue surfaces can be inspected visually in the artifact-gallery flow as
  well.
- The refreshed Module 3 artifact gallery now lives at
  `artifacts/gui_review/module3_workflows_review/`, and the `03-standards-review`
  state was re-checked in Playwright after the sidebar `Run ASTM Check` path was
  added to the modern shell.
- Additional predictive backend coverage now exists in `tests/test_predictive_features.py`:
  ROI count-target ETA, dead-time saturation forecasting, and QA-history
  recalibration prediction are exercised directly from deterministic spectra and
  QA records.
- Additional predictive Qt coverage now exists in `tests/test_predictive_dashboard_qt.py`:
  mouse navigation into the Dashboard tab, predictive summary rendering, QA sidebar
  predictive lines, count/dead-time trend plots, and the status-bar forecast label
  are now exercised in the modern shell.
- A native predictive review probe now exists at `tests/gui_predictive_dashboard_probe.py`
  so the offline predictive dashboard and QA sidebar can be inspected visually in
  the artifact-gallery flow as well.
- Additional PeakEasy-parity GUI coverage now exists too:
  `tests/test_foundation_integration.py` now validates nuclide detail snapshots
  and decay relatives, while `tests/test_analysis_workspace_qt.py` now exercises
  saved user lists, mixtures, and the primary-canvas display toggles with native
  Qt interaction.
- Native Qt review artifacts for the peak-ID browser were generated at
  `artifacts/gui_review/mouse_peak_id_review/`, and the resulting browser
  gallery was inspected with Playwright against the `06-peak-id-browser` state
  after a mouse-driven probe path through the peak table, `Use Selected Peak`,
  and the candidate list to confirm centroid-driven isotope browsing, manual
  reassignment readiness, and guide overlays in the rendered GUI.
- Native review artifacts for the Phase 2 calibration tools were generated at
  `artifacts/gui_review/phase2_tools_review/`, including quick-slider,
  deviation-pair, Gaussian ROI, skew ROI, and standards-workflow states in a
  browser review gallery inspected with Playwright.
- Native Phase 2 completion artifacts were also generated at
  `artifacts/gui_review/phase2_complete/`, and the resulting browser gallery was
  inspected with Playwright against the pinned/tagged and survey-map review states.
- GitHub verification on 2026-03-30 confirmed the pre-Phase-3B tracker assets are live: the original required labels, 7 milestones, 10 epics, and the initial seed issue set from the manifest are present on `FusionSandwich/FluxForge`.
- The newly added Phase 3B milestone/epic/issue seeds are now present in the in-repo planning manifests and still require the next planning-sync run before they can be considered live on GitHub.

## Review Result

- Stage 0 is complete both in-repo and on the live GitHub repository.
- Phase 1 is complete in the repository and formally complete in sequence.
- Phase 2.1 is complete in the repository and formally complete in sequence.
- Phase 2.2 through 2.20 are complete in the repository and formally complete in sequence.
- Phase 2 as a whole is now complete in the repository and formally complete in sequence.
- Phase 2.21 through 2.24 are now complete in the repository and formally complete in sequence.
- Phase 3.1 is now complete in the repository and formally complete in sequence.
- Phase 3.2 is now complete in the repository and formally complete in sequence.
- Phase 3.3 is now complete in the repository and formally complete in sequence.
- Phase 3.4 is now complete in the repository and formally complete in sequence.
- Phase 3.5 is now complete in the repository and formally complete in sequence.
- Phase 3.6 through 3.15 are now complete in the repository and formally complete in sequence.
- Module 3 as a whole is now complete in the repository and formally complete in sequence.
- A new formal follow-on module now sits between Module 3 and Phase 4:
  Phase 3B — Offline Spectroscopy Parity.
- Phase 3.16 is now complete in the repository and formally complete in sequence.
- Phase 3.17 is now complete in the repository and formally complete in sequence.
- Phase 3.18 remains the active sequence gate and is not yet complete.
- Phase 3.19 through 3.24 remain open in sequence (in-progress/planned).
- Phase 3.26 and 3.27 are implemented in-repo but remain sequence-pending until the earlier open Phase 3 gates are closed.
- All final-plan GUI features except the deferred live MCA / HAL transport work in
  Phase 4.1 through 4.4 are now implemented in the modern Qt shell.
- The roadmap's currently active offline-parity tranche is Phase 3.18 through
  3.23, with 3.20 command surfaces and 3.21 through 3.23 scaffold fixtures/tests
  now in progress and pending golden-source expansion.
