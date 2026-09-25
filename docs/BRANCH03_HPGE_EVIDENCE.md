# Branch 03 HPGe calibration and efficiency evidence

Validation date: 2026-09-25. Integration target: `optimization-workflows`.

## Implemented and checked

- Measured calibration points are entered or imported through a CSV with nine required numeric headers and an optional activity source ID. Each row is validated before import or fitting. Certificate and PDF ingestion are unavailable.
- Log polynomial, Gray, and semi empirical fits return point diagnostics, residuals, covariance, and model comparison. Same-source activity errors enter a grouped covariance; chi square uses each model's fitted residual space. A fit receives `review_required` if any point exceeds ±3%, its chi square p value is below 0.05, it has no residual degrees of freedom, or a nonzero activity uncertainty has no source ID.
- Conditional sparse semi empirical fits name the fixed length assumptions. Invalid physical inputs, nonphysical fitted efficiencies, and negative covariance are rejected.
- The active spectrum's detector profile stores the fit, points, uncertainty model, geometry metadata, and provenance. Migration removes legacy global fit state; changing spectra clears stale activity results. A fit to a shared profile creates a spectrum-specific copy and supports undo.
- A fitted covariance contributes to downstream activity uncertainty after session save/load. Geometry metadata does not perform a transfer correction.

## Windows evidence

Existing QA interpreter: Python 3.12.10 at `D:\FluxForgeQA\envs\fluxforge-py312\Scripts\python.exe`.

- All four HPGe files passed together on native Windows Qt: **36 passed**. The dialog file also passed in offscreen Qt (**5 passed**). The model and persistence files passed together (**29**), and the profile integration file passed twice in separate processes (**2** each). Before explicit undo-stack and panel teardown, the integration test passed its assertions but triggered a Windows heap-corruption exit during pytest's final garbage collection. Separate per-file invocations remain in CI because combined quiet Qt runs had sometimes terminated after passing dots without a pytest summary; broader teardown behavior has not been proven.
- Related non-Qt workspace/model/document/undo/migration suite: **87 passed, 9 skipped**.
- Calibration, activity, undo, document, session, and atomic-save suite: **92 passed, 10 skipped**.
- `tests/test_analysis_workspace_qt.py`: **33 passed** in the full native Windows run after restoring saved-overlay annotations on empty-trace refresh. Related pyqtgraph viewport and direct-interaction tests: **11 passed**.
- Local CI-selected core groups passed: tracker/scaffold **42**, k0 **16**, RAFM/wire/pipeline **70**, and CLI/standards **94**. The focused calibration, activity, undo, document, session, and canvas suite passed **94**, with **10 skipped**. GUI files passed separately: calibration workspace **20**, unfolding **11**, Module 3 **10**, predictive dashboard **1**, modern scaffold **29**, legacy GUI app **27**, and native Windows desktop acceptance **1**. The latter generated and validated its screenshot and artifact bundle in a temporary directory.
- The [native HPGe screenshot](../artifacts/hpge_branch03_native_windows/hpge-efficiency-fit.png) and [probe receipt](../artifacts/hpge_branch03_native_windows/summary.json) show five fitted points, measured/fitted curves, residual band, and diagnostics. A [200% scale screenshot](../artifacts/hpge_branch03_high_dpi_windows/hpge-efficiency-fit.png) and [receipt](../artifacts/hpge_branch03_high_dpi_windows/summary.json) were captured with `QT_SCALE_FACTOR=2`; all five dialog tests also passed at that scale. Both views were readable, and the point and diagnostic tables scroll to the fifth line.
- The HPGe files plus saved-overlay, calibration, and predictive regressions passed together in offscreen Qt (**39 passed**). Black and Flake8 CI scopes passed locally after correcting one formatting-only test line. The production action catalog, parity ledger JSON, and CI workflow YAML parse successfully; `git diff --check` found no whitespace errors.

Independent Sol reviews reproduced the original cross-spectrum migration leak, negative covariance, shared-profile alias, malformed point, and false-pass chi square counterexamples before fixes, then reran them after fixes. A later review found that log-model chi square used linear residuals and could flip review at p = 0.05; the fix and regression test now use log residuals with the fitted covariance. Another review challenged CSV source-ID preservation; an optional CSV column, an atomic second-row regression, and an eight-case Luna probe now cover that path. A proposed dialog-constructor failure was retracted after checking signal connection order and running native Qt tests. A Luna audit independently checked the two-point shared-source uncertainty floor. Canonical malformed points and nested negative covariance are rejected at profile persistence, while legacy generic efficiency point shapes still validate.

## Open acceptance limits

- Lines with the same Activity Source ID use fully correlated activity errors. The optional tenth CSV column preserves source IDs; nine-column files can be annotated in the table after import. Correlations across distinct certificates or gamma emission probabilities are not modeled. Blank IDs with nonzero activity uncertainty require review; the p value and covariance remain conditional diagnostics.
- Detector geometry and certificate fields are stored as metadata. No validated geometry transfer, window/dead-layer inference, or certificate/PDF parser is claimed.
- Standards golden calibration fixtures, release installer checks, keyboard/screen-reader accessibility, and full Windows/Linux GUI parity are still open. A 200% Windows visual check passed; it does not establish screen-reader or keyboard accessibility.
- Debian 13 WSL has Python 3.13.5 and display variables, but no existing `pytest` or PySide6. Linux Qt tests were not run, and Wayland was not verified. The local QA interpreter has no PyInstaller, and no release installer pipeline is defined by the CI workflow. No dependencies were installed for this branch. GitHub Actions were not dispatched at the user's direction.
