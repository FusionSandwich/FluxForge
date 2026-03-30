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
| 1.9 | `partial` | `src/fluxforge/hal/base.py` without session/device-registry integration |
| 1.10 | `partial` | `src/fluxforge/io/spe.py` and `src/fluxforge/io/artifacts.py` provide a spectrum model and JSON artifacts, but the roadmap `.ffs` session format is not implemented yet |
| 1.11 | `partial` | `src/fluxforge/io/hpge.py` provides an auto-detect reader path and multiple file readers exist, but SPC support plus GUI drag/drop and recent-files are still missing |
| 1.12 | `partial` | `src/fluxforge/io/n42.py` writes N42 XML, but the 2012 XSD validation path is still missing |
| 1.13 | `partial` | Bundled nuclide data exists in `src/fluxforge/data/`, but the Phase 1 SQLite schema with `decay_chains` and FTS5 search is not implemented |
| 1.14 | `partial` | The Qt shell reserves a nuclide-search surface, but live search and instant overlay wiring are not complete |
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
- `pytest` was upgraded in the user environment to `8.4.2` and the full suite now passes:
  `997 passed, 8 skipped` in this workspace.

## Review Result

- Stage 0 is complete only in-repo. The live GitHub tracker application steps remain
  remote-pending because this workspace has no `gh` CLI or authenticated GitHub API path.
- Phase 1 is not fully complete yet. The remaining repo-side gaps are `1.9` through `1.14`
  as documented above.
