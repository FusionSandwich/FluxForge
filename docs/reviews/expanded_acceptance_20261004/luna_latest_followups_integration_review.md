# GUI follow-ups integration review (read-only)

Reviewed the current uncommitted follow-up diff at base `b59664e20af199a82c9c145d86e838817ba72ce9`. No functional blocker found in the GUI integration. I did not run tests or edit source.

- `ElidingLabel` keeps the complete string in `text()`, tooltip, and accessible name, while painting a middle-elided one-line label. Its capped size hint and small minimum width fit the existing status-bar stretch layout. The main window replaces the prior tooltip-only status label with this widget without changing status-bar insertion order or stretch factors.
- `PredictiveDashboardPanel` now owns the scroll area directly, so the public `central_tabs.predictive_dashboard` reference still points to the panel. The forecast controls and summary remain inside the scroll content; the updated compact test checks the count target and bottom summary are reachable and that the plots keep their 160-pixel floor.
- The context dock's 220-pixel minimum width and as-needed horizontal scrollbar make unusually long diagnostic fields accessible while preserving the previous vertical scroll behavior.
- The autouse fixture's lazy import path only imports `qt_compat` for `test_qt_` cases that have not already imported it. It remains import-safe when Qt is unavailable; `wait_for_report_export` continues using `pytest.importorskip`.

## Dependency cap qualification

The new `PySide6>=6.6,<6.12` cap is effective in the GUI CI install because that job installs the `native-gui` extra. It is a broad user-facing upper bound based on a CI shutdown crash observed after tests passed. Treat it as a temporary compatibility cap, not proof that PySide6 6.12 itself is defective; the crash should be isolated in a minimal clean process before making a stronger upstream attribution or leaving the cap indefinitely.

## Validation limits

Source/diff review only; the parent asked for no test run. `git diff --check` reported no whitespace errors. I did not inspect the unrelated untracked `CRASH` file.
