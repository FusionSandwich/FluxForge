# Branch 03 HPGe calibration and efficiency evidence

Validation date: 2026-09-25. Integration target: `optimization-workflows`.

## Implemented and checked

- Measured calibration points are entered or imported through an exact-header CSV. Each row is validated before import or fitting. Certificate and PDF ingestion are unavailable.
- Log polynomial, Gray, and semi empirical fits return point diagnostics, residuals, covariance, and model comparison. Same-source activity errors enter a grouped covariance; chi square uses each model's fitted residual space. A fit receives `review_required` if any point exceeds ±3%, its chi square p value is below 0.05, it has no residual degrees of freedom, or a nonzero activity uncertainty has no source ID.
- Conditional sparse semi empirical fits name the fixed length assumptions. Invalid physical inputs, nonphysical fitted efficiencies, and negative covariance are rejected.
- The active spectrum's detector profile stores the fit, points, uncertainty model, geometry metadata, and provenance. Migration removes legacy global fit state; changing spectra clears stale activity results. A fit to a shared profile creates a spectrum-specific copy and supports undo.
- A fitted covariance contributes to downstream activity uncertainty after session save/load. Geometry metadata does not perform a transfer correction.

## Windows evidence

Existing QA interpreter: Python 3.12.10 at `D:\FluxForgeQA\envs\fluxforge-py312\Scripts\python.exe`.

- All four HPGe files passed together on native Windows Qt: **35 passed**. The model and persistence files also passed together (**29**); the dialog file passed separately (**4**); and the profile integration file passed twice in separate processes (**2** each). Before explicit undo-stack and panel teardown, the integration test passed its assertions but triggered a Windows heap-corruption exit during pytest's final garbage collection. Separate per-file invocations remain in CI because combined quiet Qt runs had sometimes terminated after passing dots without a pytest summary; broader teardown behavior has not been proven.
- Related non-Qt workspace/model/document/undo/migration suite: **87 passed, 9 skipped**.
- Calibration, activity, undo, document, session, and atomic-save suite: **92 passed, 10 skipped**.
- `tests/test_analysis_workspace_qt.py`: **33 passed** in the full native Windows run after restoring saved-overlay annotations on empty-trace refresh. Related pyqtgraph viewport and direct-interaction tests: **11 passed**.
- Native `QT_QPA_PLATFORM=windows` HPGe dialog and profile integration tests: **6 passed** across their separate files. The native [HPGe dialog screenshot](../artifacts/hpge_branch03_native_windows/hpge-efficiency-fit.png) and [probe receipt](../artifacts/hpge_branch03_native_windows/summary.json) were refreshed after the UI changes. They show five fitted points, measured/fitted curves, residual band, and diagnostics. Visual inspection found the review readable at 1200 pixels wide.
- The production action catalog, parity ledger JSON, and CI workflow YAML parse successfully; `git diff --check` found no whitespace errors.

Independent Sol reviews reproduced the original cross-spectrum migration leak, negative covariance, shared-profile alias, malformed point, and false-pass chi square counterexamples before fixes, then reran them after fixes. A later review found that log-model chi square used linear residuals and could flip review at p = 0.05; the fix and regression test now use log residuals with the fitted covariance. A Luna audit independently checked the two-point shared-source uncertainty floor. Canonical malformed points and nested negative covariance are rejected at profile persistence, while legacy generic efficiency point shapes still validate.

## Open acceptance limits

- Lines with the same manually entered Activity Source ID use fully correlated activity errors. The nine-column CSV does not carry source IDs, and correlations across distinct certificates or gamma emission probabilities are not modeled. Blank IDs with nonzero activity uncertainty require review; the p value and covariance remain conditional diagnostics.
- Detector geometry and certificate fields are stored as metadata. No validated geometry transfer, window/dead-layer inference, or certificate/PDF parser is claimed.
- Standards golden calibration fixtures, release installer checks, high-DPI/accessibility review, and full Windows/Linux GUI parity are still open.
- Debian 13 WSL has Python 3.13.5 and display variables, but no existing `pytest` or PySide6. Linux Qt tests were not run, and Wayland was not verified. No dependencies were installed for this branch.
