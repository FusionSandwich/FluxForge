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
| S0.1 | `next` | `repo-complete-remote-pending` | The repo contains milestones, labels, board config, templates, and sync automation, but the live GitHub tracker has not been applied yet. |
| S0.2 | `blocked` | `repo-complete-remote-pending` | Epic definitions exist in `.github/project-management/issues.json`, but they are not yet live on GitHub. |
| S0.3 | `blocked` | `repo-complete-remote-pending` | Initial issue seeds exist in `.github/project-management/issues.json`, but they are not yet live on GitHub. |

## Completed In Sequence

| Step | Repo Status | Evidence |
|---|---|---|
| S0.4 | `complete` | `docs/adr/ADR-001` through `ADR-007` |
| S0.5 | `complete` | `docs/adr/`, `tests/spectra/`, `.github/ISSUE_TEMPLATE/` |

## Implemented Ahead of the Formal Gate

These items exist additively in the repository, but the roadmap does not credit them as the
active next step until Stage 0 is fully satisfied.

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
- The full suite now passes in this workspace: `1011 passed, 2 skipped`.

## Review Result

- Stage 0 is complete only in-repo. The live GitHub tracker application steps remain
  remote-pending until the sync workflow has been pushed and verified on GitHub.
- Phase 1 is now repo-complete, but it remains sequence-blocked in the tracker until
  Stage 0 is closed on the live GitHub project side.
