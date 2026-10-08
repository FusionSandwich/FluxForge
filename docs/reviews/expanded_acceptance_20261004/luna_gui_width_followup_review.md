# Qt laptop-width follow-up review

Read-only re-review of the uncommitted fix in `main_window.py`, `pyqtgraph_backend.py`, and `test_gui_calibration_target.py`. Scope remained limited to these GUI/test changes; the separately fetched unsaved-session changes were not reviewed. Parent reports the updated offscreen scope passed 37 tests and measured 1204 px, down from prior failures at 1297 px on Linux CI and 1986 px on Windows offscreen. I did not rerun those tests.

No concrete regression found in the current diff. Canvas metadata and crosshair readout can shrink and wrap, the six canvas actions are split into two three-button rows, and status labels use ignored horizontal size with full text retained in their tooltip. Status-bar labels have stretch allocations so long library identifiers no longer force desktop-width minimums. The strengthened test checks all six buttons remain visible, inside the window, and unobscured at 1280 px, then verifies a long library string remains available in both label text and tooltip without widening the window.

The change trades some vertical canvas-header space for width; metadata remains wrapped and accessible. The previous calibration action/plot layout and action-row behavior are outside this diff and remain unchanged. Button object names and callbacks are preserved. `git diff --check` reported no whitespace errors.

The working tree also contains an untracked `CRASH` file outside the reviewed scope; I left it untouched. The review does not include the separate unsaved-session branch changes or a fresh GUI run.
