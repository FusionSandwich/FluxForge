# Validation continuation — 18 September 2026

The handoff remains incomplete. No goal is promoted by this bounded validation run.

## Fresh results

| Scope | Result | Interpretation |
| --- | --- | --- |
| Native Windows Qt shell, sessions and unfolding workspace | 39 passed, 1 warning; 227.30 seconds | Automated source GUI checks, including mouse-driven method comparison, session replacement and calibration undo |
| Numerical, activity, reader, background and session contracts | 173 passed; 13.96 seconds | Software correctness checks; not physical experimental qualification |
| INL 10-bin unfolding regression | 1 failed; 5.02 seconds | Stops before unfolding at incompatible sample/background energy grids |

The GUI warning is an ill-conditioned Hessian inversion in `solvers/rmle.py:846` during the bundled UWNR RAFM rate CSV replay (reported reciprocal condition estimate approximately 9.20e-32). Successful UI execution does not validate those uncertainty estimates. The replay uses a simplified response. The test named `loads_real_rate_artifact` writes synthetic rates; its name is not evidence of measured INL provenance.

The failed regression raises: “Background subtraction requires identical energy grids; conservative rebinning with covariance propagation is not supported.” The existing guard was retained. The rebin primitive passes its checks, but end-to-end background integration and covariance storage/consumers remain incomplete; see [background follow-up](BACKGROUND_FOLLOWUP_20260917.md).

## Reproduction and evidence

Source: `C:/Users/Josh/projects/FluxForge-validation`, branch `recovery-validation`, starting HEAD `78aee3ee9da3d4c44dd77dec05cb674321971c79`. The working tree was clean before the initial run. The continuation changed documentation and added the targeted interface regression tests described below; no production code was changed.

Interpreter: `C:/Users/Josh/projects/FluxForge-recovery-20260916/windows-env/Scripts/python.exe` (Python 3.12.10, pytest 9.1.1, Qt 6.11.2). Runs explicitly set `PYTHONPATH=C:/Users/Josh/projects/FluxForge-validation/src`; the GUI run explicitly set `QT_QPA_PLATFORM=windows`. These are source tests using the recovery interpreter, not fresh installed-wheel qualification or manual desktop inspection.

Run `python -m pytest` with the following selections using that interpreter and source environment:

- GUI: `-v tests/test_modern_gui_shell.py tests/test_workspace_session_qt.py tests/test_unfolding_workspace_qt.py -o faulthandler_timeout=90`
- Core: `-q tests/test_unfolding_workflows.py tests/test_gls_scientific_contract.py tests/test_activity_scientific_correctness.py tests/test_histogram_rebin.py tests/test_reader_integrity.py tests/test_background_normalization_contract.py tests/test_signed_workspace_session.py`
- INL: `-v tests/test_flux_unfolding_10bin.py`

Each invocation also wrote JUnit XML and captured console output in `D:/FluxForge-validation-resume-20260918`: `gui.xml`/`gui.log`, `core.xml`/`core.log`, and `inl.xml`/`inl.log`. All processes finished; the first two exited 0 and INL exited 1.

## Remaining acceptance gates

### Targeted CLI/GUI investigation after the initial run

The initial 173-test backend selection did not exercise a real CLI subprocess, and the 39-test GUI selection did not load this failing sample/background pair. It was not full CLI/GUI validation of the INL workflow.

New regression coverage in `tests/test_inl_background_interfaces.py` exercises a real `python -m fluxforge.cli.app ingest` subprocess and a native Qt main window opening `Co-Cd-RAFM-1_25cm.ASC` with `background.ASC`, using the foreground/background selectors. Both rejection checks pass: CLI exits unsuccessfully without writing requested spectrum/adjusted CSV artifacts; GUI displays “background not applied” with the energy-grid reason and displays the original foreground counts. These are rejection tests, not successful subtraction or unfolding. They are automated GUI checks, not a manual desktop walkthrough. Receipts: `inl-interfaces.log` and `inl-interfaces.xml` in the evidence directory above.

Fresh diagnostics of all ten bundled raw sample files show 8192 channels each and identical sample calibration coefficients `[0.541, 0.498, 2.605e-7]`. The background uses `[-1.502, 0.4991, 2.239e-7]`. Background minus sample channel energy ranges from -2.043 to +4.511515 keV; all ten grids differ. Per-file details and live-time scaling factors are in `inl-grid-diagnostics.json`. These values come from the file headers via the existing reader; they do not qualify the calibrations experimentally.

Next implementation gate: integrate conservative background rebinning with covariance retained through spectrum storage, sessions, ROI estimators and fit consumers, with an explicit bin-edge/coverage policy. Validate independent numerical oracles before rerunning the real-data pipeline through CLI and GUI. Do not overwrite calibration coefficients or remove the grid guard to make this regression pass. Background measurement suitability and physical calibration/timing acceptance remain separate open gates.

- G1: physical calibration, timing, nuclear data, shared covariance and external dosimetry parity remain open.
- G2: accepted activity/reaction-rate reductions and all-method measured INL unfolding validation remain open. The current INL regression cannot reach a solver.
- G3: these 39 automated GUI checks do not cover every feature, manual usability, DPI/multiple monitors, endurance or a newly built installed package.
- G4: full cleanup/archive acceptance remains governed by the recovery checklist; this run adds no cleanup evidence.

Use [the recovery checklist](RECOVERY_TODO_20260916.md) for goal gates and [the feature matrix](FEATURE_VALIDATION_MATRIX.md) for the inventory requiring reconciliation. Test-file listings and checked-box counts are not completion percentages. SLOWPOKE capability coverage is not established by this test selection.

The local review index `D:/FluxForge-validation-resume-20260918/DOCUMENTATION_INDEX.md` lists documentation locations, including original handoff snapshots, current plans, study contracts and historical archives. It is an inventory, not a claim that every document or feature has been reviewed. Original handoffs remain outside this checkout and were not staged or changed.
