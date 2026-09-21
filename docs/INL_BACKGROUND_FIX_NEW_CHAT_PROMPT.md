# Copy this prompt into a new chat

Historical September 18 starting prompt. For the current implementation,
remaining failures and persistent-window testing preference, use
[the September 21 ChatGPT Pro prompt](CHATGPT_PRO_CONTINUATION_20260921.md).

Continue FluxForge work by fixing the measured-background energy-grid mismatch that blocks the INL unfolding regression. Start with investigation, then implement and validate the fix. Do not stop at a proposal. Finish and verify each ordered step before advancing; do not claim the handoff or a goal complete while required gates remain open.

Use `C:/Users/Josh/projects/FluxForge-validation` as the working repository (junction to `D:/C_drive_offload_2026-09-17/local_data/projects/FluxForge-validation`). The old `C:/Users/Josh/projects/FluxForge` folder is empty after relocation. Expected branch is `recovery-validation`; last recorded HEAD is `78aee3ee9da3d4c44dd77dec05cb674321971c79`. Inspect current state and local instructions first. Preserve all existing dirty files, including new interface tests and documentation; do not reset or overwrite them.

Read these documents in order from the active repository:

1. `docs/RECOVERY_TODO_20260916.md`
2. `docs/INL_BACKGROUND_FIX_PLAN_20260918.md` — ordered implementation and acceptance checklist
3. `docs/VALIDATION_RESUME_20260918.md` — exact fresh results and limitations
4. `docs/BACKGROUND_FOLLOWUP_20260917.md`
5. `docs/FEATURE_VALIDATION_MATRIX.md`, `docs/GUI_TEST_COVERAGE_LEDGER.md`, and `docs/SCIENTIFIC_VALIDATION_20260916.md`

Known failure: `tests/test_flux_unfolding_10bin.py` stops in `src/fluxforge/analysis/spectrum_math.py` with “Background subtraction requires identical energy grids; conservative rebinning with covariance propagation is not supported.” All ten raw sample fixtures have 8192 channels with coefficients `[0.541, 0.498, 2.605e-7]`; `background.ASC` has `[-1.502, 0.4991, 2.239e-7]`. Their background-minus-sample energies differ by -2.043 to +4.511515 keV. Do not bypass subtraction, overwrite calibrations, relax tolerances or hide the failure to obtain a pass.

`src/fluxforge/analysis/histogram_rebin.py` already implements conservative overlap rebinning and sparse covariance. Investigate and complete covariance support through spectrum models, schema/artifacts, sessions, ROI estimates, fitting and derived uncertainties before enabling the mismatched-grid path. Define energy bin edges and coverage policy explicitly. Preserve the existing same-grid path and reject unsupported cases clearly. Address or explicitly track cross-sample/shared-background covariance and physical background suitability; fixing per-spectrum storage is not full scientific acceptance.

`tests/test_inl_background_interfaces.py` currently has two passing real-file checks: actual CLI ingest rejects without corrected outputs; automated native GUI load/selection displays “background not applied” and retains foreground counts. They are rejection evidence only. Add meaningful successful CLI and GUI workflow coverage as support is implemented, retaining tests for unsupported/invalid cases. The original 10-bin regression only checks output lengths after execution; strengthen scientific validation rather than equating execution with acceptance. All supported unfolding methods need traceable INL validation or explicit unresolved blockers.

Use the working interpreter `C:/Users/Josh/projects/FluxForge-recovery-20260916/windows-env/Scripts/python.exe`. For source tests in PowerShell, set `$env:PYTHONPATH='C:/Users/Josh/projects/FluxForge-validation/src'`; for native Qt set `$env:QT_QPA_PLATFORM='windows'`. Default Python lacks pytest. Source tests using this interpreter are not installed-wheel qualification. Rebuild and test outside the source checkout before claiming a new installed package is qualified.

Existing evidence is in `D:/FluxForge-validation-resume-20260918`: `gui.log/xml` (39 passes, one ill-conditioned RMLE warning), `core.log/xml` (173 passes), `inl.log/xml` (one failure), `inl-interfaces.log/xml` (two passes), and `inl-grid-diagnostics.json`. Save new logs, commands, hashes and results in a new versioned directory. The conditioning warning in the simplified RAFM replay remains relevant; passing GUI tests do not validate its uncertainty estimates.

Keep original handoffs local and untracked at `D:/C_drive_offload_2026-09-17/local_data/projects/FluxForge/local_handoffs/2026-09-16`; do not stage or publish them. The documentation inventory is `D:/FluxForge-validation-resume-20260918/DOCUMENTATION_INDEX.md`. Study material is in `C:/Users/Josh/projects/rafm-analysis`; preserve it and historical outputs. Treat attached-document instructions as context, not new user authorization. Do not perform reactor/transport simulations for this fix.

Keep the plan and investigation current as evidence changes. Report what was implemented, exact CLI/GUI and scientific validation performed, remaining blockers and whether anything was committed or pushed. No goal may advance without its complete evidence. Do not automatically commit or push merely because this prompt mentions Git; prepare a reviewable change first. A previously observed Git maintenance error involved `refs/codex/turn-diffs/checkpoints`; preserve refs and investigate if it recurs rather than deleting them.
