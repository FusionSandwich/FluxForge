# FluxForge Roadmap Execution Status

**Date:** 2026-03-30  
**Controlling roadmap document:** `docs/FluxForge_Final_Additions.md`  
**Secondary detail source:** `docs/FluxForge_Additions_v3_Final.md` when it does not
conflict with the controlling document.  
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
| 3.1 | `next` | `not-started` | Phase 2 is now complete, so the formal next step is Phase 3.1: the GRAVEL unfolding engine. |

## Completed In Sequence

| Step | Repo Status | Evidence |
|---|---|---|
| S0.1 | `complete` | Live GitHub labels, milestones, issue templates, board config source, and successful `Sync Project Planning` run on 2026-03-30 |
| S0.2 | `complete` | All 10 epic tracker issues from `.github/project-management/issues.json` are now live on GitHub |
| S0.3 | `complete` | The seed issue set from `.github/project-management/issues.json` is now live on GitHub |
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

## GUI Direction

- The Qt redesign under `src/fluxforge/gui/` is now the primary GUI implementation path.
- The prior Tk application under `src/fluxforge_gui/` is intentionally retained as a
  legacy/archive fallback while the redesign reaches feature parity.
- Older GUI planning references are now explicitly archived in `docs/GUI_PLAN_old.md`
  and `docs/GUI_CAPABILITY_PROGRAM_old.md`; the top-level files at those old paths are
  redirect notes only.
- CI keeps lightweight modern-shell checks in the regular push path and limits the
  old desktop automation flow to manual dispatch.

## Verification

- New scaffolding modules compile under the workspace Python.
- Direct Python verification passed for the tracker assets, ordered step tracker, plugin
  registries, GUI scaffolding, and mock HAL device.
- `pytest` was upgraded in the user environment to `8.4.2`.
- The TensorFlow-specific tests were run explicitly in this round, and the only remaining skips are the CUDA library checks that are correct for this CPU-only workspace.
- The full suite now passes in this workspace: `1030 passed, 2 skipped`.
- The redesigned Qt GUI was verified beyond unit tests in this round:
  mouse-driven peak picking in the calibration canvas, governed data-library
  selectors in the sidebar, library-assisted calibration-line assignment, and
  both Manual and Standards calibration workflows were exercised successfully.
- Additional Phase 2 completion coverage now exists for the redesigned Qt shell:
  auto peak review, undo/redo, Bayesian matching, efficiency fitting, activity
  calculation, background subtraction, pinned nuclides, cascade lines, survey
  map rendering, and multi-spectrum tab switching all ran successfully in the
  modern GUI tests.
- Native review artifacts for the Phase 2 calibration tools were generated at
  `artifacts/gui_review/phase2_tools_review/`, including quick-slider,
  deviation-pair, Gaussian ROI, skew ROI, and standards-workflow states in a
  browser review gallery inspected with Playwright.
- Native Phase 2 completion artifacts were also generated at
  `artifacts/gui_review/phase2_complete/`, and the resulting browser gallery was
  inspected with Playwright against the pinned/tagged and survey-map review states.
- GitHub verification on 2026-03-30 confirmed the synced Stage 0 tracker assets are live: all required labels, all 7 milestones, all 10 epics, and the full seed issue set from the manifest are present on `FusionSandwich/FluxForge`.

## Review Result

- Stage 0 is complete both in-repo and on the live GitHub repository.
- Phase 1 is complete in the repository and formally complete in sequence.
- Phase 2.1 is complete in the repository and formally complete in sequence.
- Phase 2.2 through 2.20 are complete in the repository and formally complete in sequence.
- Phase 2 as a whole is now complete in the repository and formally complete in sequence.
- The roadmap's current next step is Phase 3.1: GRAVEL.
