# GUI Test Coverage Ledger

Status: active
Last Updated: 2026-04-19
Purpose: explicit inventory of FluxForge GUI parts tested so far, with evidence links and remaining coverage targets.

## 1. Latest Validation Snapshot

- Broad Qt GUI regression:
  - Command:
    - `PYTHONPATH=src pytest -q tests/test_gui_dialogs_qt.py tests/test_gui_widgets_qt.py tests/test_analysis_workspace_qt.py tests/test_calibration_workspace_qt.py tests/test_module3_workflows_qt.py tests/test_predictive_dashboard_qt.py tests/test_unfolding_workspace_qt.py tests/test_modern_gui_shell.py`
  - Result: `91 passed, 9 warnings`
- Full browser-lane gallery audit:
  - Command:
    - `NODE_PATH=/tmp/pw-audit/node_modules node tests/gui_gallery_playwright_audit.js artifacts/gui_review /tmp/gui_playwright_audit`
  - Result: `21 audited, 0 failing`
  - Persisted evidence:
    - `artifacts/gui_review/playwright_audit/audit_report.json`
    - `artifacts/gui_review/playwright_audit/audit_report.md`
- Phase 5 parity panel targeted validation:
  - Commands:
    - `PYTHONPATH=src pytest -q tests/test_modern_gui_shell.py -k "main_window_restores_saved_workflow_state_across_sessions"`
    - `PYTHONPATH=src /usr/bin/python tests/gui_phase5_parity_probe.py artifacts/gui_review/phase5_parity`
    - `node tests/gui_gallery_playwright_audit.js artifacts/gui_review/phase5_parity artifacts/gui_review/phase5_parity/playwright_audit`
  - Results: `1 passed, 14 deselected`; native probe gallery generated with scope+fixture controls; Playwright `1 audited, 0 failing`
  - Persisted evidence:
    - `artifacts/gui_review/phase5_parity/index.html`
    - `artifacts/gui_review/phase5_parity/phase5_probe_report.json`
    - `artifacts/gui_review/phase5_parity/playwright_audit/audit_report.json`
    - `artifacts/gui_review/phase5_parity/playwright_audit/audit_report.md`

## 2. Tested GUI Areas (Evidence Matrix)

### 2.1 Main Window and Shell Scaffolding

- Main shell startup, docks, and workflow framing
  - Tests: `tests/test_analysis_workspace_qt.py`, `tests/test_module3_workflows_qt.py`, `tests/test_modern_gui_shell.py`
  - Probe galleries:
    - `artifacts/gui_review/analysis_workspace_review/index.html`
    - `artifacts/gui_review/phase327_probe/index.html`

### 2.2 Analysis Workspace (Phase 2 surfaces)

- Peak review, pinned/tagged nuclides, activity results, survey map, background selector
  - Tests: `tests/test_analysis_workspace_qt.py`
  - Probe galleries:
    - `artifacts/gui_review/analysis_workspace_review/index.html`
    - `artifacts/gui_review/phase2_complete/index.html`
    - `artifacts/gui_review/peak_id_browser_review/index.html`
    - `artifacts/gui_review/background_selector_review/index.html`

### 2.3 Calibration Workspace

- Manual/standards calibration, mouse peak pick, line assignment, ROI fitting, advanced tools
  - Tests: `tests/test_calibration_workspace_qt.py`
  - Probe galleries:
    - `artifacts/gui_review/phase317_calibration_probe/index.html`
    - `artifacts/gui_review/phase2_calibration_workspace/index.html`

### 2.4 Unfolding Workspace

- Unfolding methods, comparison workflows, uncertainty/result table behavior
  - Tests: `tests/test_unfolding_workspace_qt.py`
  - Probe galleries:
    - `artifacts/gui_review/unfolding_workspace_review/index.html`
    - `artifacts/gui_review/phase31x_unfolding_probe/index.html`

### 2.5 Predictive Dashboard

- Predictive controls and QA/sidebar integration path
  - Tests: `tests/test_predictive_dashboard_qt.py`
  - Probe gallery:
    - `artifacts/gui_review/predictive_dashboard_review/index.html`

### 2.6 Module 3 Dialog Workflows

- QA history, ASTM review launch path, report export, batch queue actions
  - Tests: `tests/test_module3_workflows_qt.py`, `tests/test_gui_dialogs_qt.py`
  - Probe gallery:
    - `artifacts/gui_review/module3_workflows_review/index.html`

### 2.7 Phase 6 Optimization Workspace

- Masking review, optimization workspace, second-irradiation planner, worked-example action
  - Tests: `tests/test_analysis_workspace_qt.py`, `tests/test_cli_app.py`
  - Probe gallery:
    - `artifacts/gui_review/phase6_optimization_probe/index.html`

### 2.8 GUI Widgets and Mode Controls

- Method selector, mode switcher, hardware LED status/click behavior
  - Tests: `tests/test_gui_widgets_qt.py`, `tests/test_calibration_workspace_qt.py`, `tests/test_module3_workflows_qt.py`

### 2.9 Direct Dialog Coverage Added This Round

- `QAHistoryDialog`
  - Test: `tests/test_gui_dialogs_qt.py::test_qa_history_dialog_refresh_and_csv_export`
- `ReportExportDialog`
  - Test: `tests/test_gui_dialogs_qt.py::test_report_export_dialog_uses_engine_for_preview_and_html_export`
- `StandardsReviewDialog`
  - Test: `tests/test_gui_dialogs_qt.py::test_standards_review_dialog_renders_table_and_detail`
- `PuIsotopicsDialog`
  - Test: `tests/test_gui_dialogs_qt.py::test_pu_isotopics_dialog_populates_ratios_and_report`

### 2.10 Phase 5 Crosswalk and Parity Panel

- Crosswalk review table, replay-state filter persistence, parity scope/fixture targeting, and parity-launch visibility
  - Tests: `tests/test_modern_gui_shell.py`, `tests/test_cli_app.py`, `tests/test_phase5_crosswalk.py`
  - Probe gallery:
    - `artifacts/gui_review/phase5_parity/index.html`

## 3. Browser-Lane Coverage Status

- All generated GUI review galleries under `artifacts/gui_review/**/index.html` pass current Playwright checks:
  - page loads
  - heading/section visibility
  - card visibility
  - image health and alt text checks
  - viewport metadata and responsive media-query presence

## 4. Remaining Expansion Targets (Continue Testing)

The items below are tracked as direct-expansion targets to approach full GUI-part coverage:

- Add focused panel-construction/behavior tests for:
  - `src/fluxforge/gui/panels/phase6.py` panel classes
  - `src/fluxforge/gui/panels/modern_shell.py` panel classes where only integrated coverage exists today
- Add explicit backend-canvas coverage for:
  - `src/fluxforge/gui/backends/pyqtgraph_backend.py` (`PyQtGraphSpectrumCanvas`)

## 5. Operating Rule for Future Runs

After any GUI-affecting change:

1. Run the broad Qt GUI regression slice.
2. Refresh probe galleries with `tests/gui_*_probe.py`.
3. Run the Playwright gallery audit script.
4. Update this ledger and roadmap/handoff docs with findings and fixes.
