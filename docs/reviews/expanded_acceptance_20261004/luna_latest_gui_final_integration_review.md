# Latest GUI integration review (read-only)

Reviewed `e3801d1bda4887a9cd6b51b3173ebe5452a02e9d` in the integration worktree. Scope: async report export, report snapshot/instrument override binding, unsaved-session replacement guards, and compact context scrolling. No source edits or tests were run; focused tests were already running in the primary checkout.

## Findings

No blocking correctness issue found in this diff.

- Report export captures the template, resolved path, PDF option, and a deep-copied context on the GUI thread before starting `_ReportExportWorker` (`src/fluxforge/gui/dialogs/report_export_dialog.py:249`, `:408`). The worker only receives these values, and the tests exercise source/path mutation after start, single capture, GUI responsiveness, retry after failure, and close/reject protection (`tests/test_gui_report_worker.py`; `tests/test_gui_report_bundle.py`).
- Dialog close/reject is refused while its QThread runs (`report_export_dialog.py:365`, `:375`); the main window independently refuses close while either the report or spectrum-conversion worker is active (`src/fluxforge/gui/main_window.py:2221`). This closes the parent-destruction path that could otherwise destroy a live thread. On completion the UI restores controls, updates the result, schedules `deleteLater`, and emits a success flag (`report_export_dialog.py:457`).
- The export input is a detached snapshot. Instrument overrides are keyed to spectrum IDs and cleared when the document ID changes or a spectrum's counts object is replaced; calibration-only edits preserve them (`report_export_dialog.py:336`). This prevents the reviewed stale-dialog case while allowing harmless calibration updates. Bundle/instructional validation is still delegated to the provenance builder and remains marked `scientific_admission: false` (`src/fluxforge/reporting/instrument_provenance.py`).
- Dirty workspace confirmation is applied before close, session replacement, bundled-example replacement, and reset (`src/fluxforge/gui/main_window.py:1024`, `:1045`, `:1295`, `:1944`, `:2221`). Save failure/cancel and discard behavior are covered in `tests/test_gui_session_guard.py`.
- The context panel is a resizable vertical scroll area with wrapped summaries (`src/fluxforge/gui/panels/modern_shell_context.py:18`); the compact-window test checks scrolling, visibility of the lower analysis summary, and retained central-canvas height (`tests/test_gui_compact_context.py`).

## Non-blocking persistence limitation

Instrument settings manually entered in the report dialog are per-dialog report provenance, not workspace-session state. They persist across spectrum switches within that dialog and are captured into an exported report, but closing the dialog before exporting discards them. This appears consistent with the current report-only UI; if users are expected to reuse those entries across app restarts, persist them separately in a future change. No evidence here that they contaminate an export or are silently attached to a different acquisition.

## Validation limits

This was source/test review only. I did not run tests concurrently with the parent checkout's focused test run and did not inspect the unrelated untracked `CRASH` file.
