# Branch 03 HPGe calibration and efficiency evidence

Validation date: 2026-09-25. Integration target: `optimization-workflows`.

## Implemented and checked

- Measured calibration points are entered or imported through an exact-header CSV. Each row is validated before import or fitting. Certificate and PDF ingestion are unavailable.
- Log polynomial, Gray, and semi empirical fits return point diagnostics, residuals, covariance, and model comparison. A fit receives `review_required` if any point exceeds ±3%, its chi square p value is below 0.05, or it has no residual degrees of freedom.
- Conditional sparse semi empirical fits name the fixed length assumptions. Invalid physical inputs, nonphysical fitted efficiencies, and negative covariance are rejected.
- The active spectrum's detector profile stores the fit, points, uncertainty model, geometry metadata, and provenance. Migration removes legacy global fit state; changing spectra clears stale activity results. A fit to a shared profile creates a spectrum-specific copy and supports undo.
- A fitted covariance contributes to downstream activity uncertainty after session save/load. Geometry metadata does not perform a transfer correction.

## Windows evidence

Existing QA interpreter: Python 3.12.10 at `D:\FluxForgeQA\envs\fluxforge-py312\Scripts\python.exe`.

- HPGe model, persistence, Qt, and action-catalog files: **48 passed** in one verbose run. Each HPGe test file also passed separately with `-q` (17, 4, 8, and 2 tests respectively). Separate per-file invocations are used in CI because combined quiet Qt runs sometimes terminated after printing all passing dots without a pytest summary; the cause is unverified.
- Related non-Qt workspace/model/document/undo/migration suite: **87 passed, 9 skipped**.
- Calibration, activity, undo, document, session, and atomic-save suite: **92 passed, 10 skipped**.
- `tests/test_analysis_workspace_qt.py`: **32 passed, 1 deselected** in the broad run. The separately rerun `test_main_window_exposes_log_scale_and_peak_label_toggles` fails on zero saved-overlay annotation labels in both offscreen and native Windows Qt. This is outside the HPGe editing path, but the broader GUI suite is not wholly green.
- Native `QT_QPA_PLATFORM=windows` HPGe Qt dialog and integration tests: **6 passed**. The native [HPGe dialog screenshot](../artifacts/hpge_branch03_native_windows/hpge-efficiency-fit.png) and [probe receipt](../artifacts/hpge_branch03_native_windows/summary.json) show five fitted points, measured/fitted curves, residual band, and diagnostics. Visual inspection found the review readable at 1200 pixels wide.
- The production action catalog, parity ledger JSON, and CI workflow YAML parse successfully; `git diff --check` found no whitespace errors.

An independent Sol review reproduced the original cross-spectrum migration leak, negative covariance, shared-profile alias, malformed point, and false-pass chi square counterexamples before fixes, then reran them after fixes. It verified that canonical malformed points and nested negative covariance are rejected at profile persistence, while legacy generic efficiency point shapes still validate.

## Open acceptance limits

- Calibration point errors are weighted as independent. A shared source activity uncertainty creates cross-line correlation that the nine-column CSV and fitted covariance do not represent. The p value and covariance are conditional diagnostics for such inputs.
- Detector geometry and certificate fields are stored as metadata. No validated geometry transfer, window/dead-layer inference, or certificate/PDF parser is claimed.
- Standards golden calibration fixtures, release installer checks, high-DPI/accessibility review, and full Windows/Linux GUI parity are still open.
- Debian 13 WSL has Python 3.13.5 and display variables, but no existing `pytest` or PySide6. Linux Qt tests were not run, and Wayland was not verified. No dependencies were installed for this branch.
