# Branch 02: Direct-Manipulation Evidence

Branch: `gui/02-bgamma-direct-manipulation`

Integration base: `optimization-workflows` at `761ffdf8cb2437f5a0f86b027ca1d31c6eec409f`

Validated: 2026-07-22

## Implemented behavior

- Shift-drag creates a canonical ROI with signal and left/right background ranges.
- Signal and sideband handles update the selected `AnalysisROI` once when a drag finishes.
- Peak markers select by stable peak ID, including peaks with identical centroid energy.
- The selected centroid is draggable; one focused undo command is committed per drag.
- Context actions cover peak add/select/move/delete, nuclide assignment and clearing,
  pin/unpin, tags, component add/split/merge, ROI create/select/delete, spectrum roles,
  reset, crosshair, and export.
- Peak-table selection highlights and zooms to the exact canonical ROI; plot selection
  selects the exact table row.
- Scientific edits invalidate stale ROI/activity results and show an explicit
  re-analysis-required state.
- ROI, peak, assignment, role, viewport, and workflow-state changes persist in
  `WorkspaceDocument v2` sessions and participate in focused undo/redo.

## Automated evidence

Windows 11, Python 3.12.10, PySide6 6.11.1:

- Offscreen direct-manipulation, workspace-state, and session slice: 68 passed.
- Native `QT_QPA_PLATFORM=windows`: 15 passed (9 renderer interactions plus
  6 shell integration workflows).
- `tests/test_production_gui_mode.py`: 9 passed.
- `tests/test_modern_gui_shell.py`: 23 passed.

The native pytest runs returned exit code 0, but PySide6 emitted a post-summary
Windows COM teardown diagnostic. A standalone native `QApplication` and the
installed GUI launcher both exited cleanly. The diagnostic is recorded as a test
harness limitation and remains part of final clean-install qualification.

Linux WSLg/X11, Python 3.13.5:

- Native `QT_QPA_PLATFORM=xcb` direct/integration suite on the final source:
  15 passed (9 renderer interactions plus 6 shell integration workflows).
- Offscreen direct/state/session slice: 68 passed.

An independent source audit reran 42 focused tests and found no remaining
branch-level blockers. Direct `QT_QPA_PLATFORM=wayland` initialization was not
available in this WSLg environment, so Wayland is not claimed as passed.

The real Qt tests use `QTest` mouse press, move, release, and click events against
the rendered PyQtGraph viewport. They assert emitted intent payloads, stable IDs,
workspace mutations, undo/redo, persistence, table/plot synchronization, and
debounced viewport commits rather than only counting widgets or rows.

## Native desktop observation

The installed Windows launcher (`fluxforge-gui.exe`) was exercised at 1866 x 1079.
Default startup was empty; `File > Open Example` loaded foreground, background,
and overlay spectra. The crosshair produced a live coordinate readout. The spectrum
context menu exposed the full action set, and adding a manual peak produced a visible
selected centroid, peak-table row, and residual row. The automatic peak-review dialog
displayed three candidates; automated acceptance through the native accessibility
layer was inconclusive, so acceptance of that modal is not claimed as branch evidence.

## Remaining release-level gates

This branch closes roadmap step 3.24 and bGamma carryovers BG.4-BG.6. BG.7 remains
prototype until consistent navigation and keyboard behavior is applied to every
scientific plot. Wayland, high-DPI matrices, accessibility, and visual-golden checks
remain part of the final GUI release gate.
