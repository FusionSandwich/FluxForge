# FluxForge Phase 3 LLM Execution Handoff

Status: active reusable handoff
Last Updated: 2026-04-17 (phase6 workflow persistence + handoff refresh)
Scope: complete remaining Phase 3 work in strict sequence with full backend, CLI, modern Qt GUI, and testing parity.

This handoff is written so a new LLM session can start from this file and execute without guessing.

Current execution update (2026-04-17):
- Phase 2 checkpoint review is complete through `2.23` and `2.24` in `docs/ROADMAP_EXECUTION_STATUS.md`.
- Phase 3 checkpoint review confirms `3.1` through `3.17` are complete in-repo and formally complete in sequence.
- `3.26` and `3.27` are implemented in-repo with CLI/GUI/test evidence and release-checklist coverage, but remain sequence-pending while earlier Phase 3 parity gates are still open.
- Sequence control remains unchanged: `3.18` is still the active next gate and `3.19` through `3.24` remain in-progress/planned.
- This handoff now includes an explicit implementation pack for Phase 6 irradiation optimization capability (mapping to `3N.6` plus related `3N.1`-`3N.8` and ML/library additions) so future sessions can execute without re-deriving scope.
- Full-suite verification baseline for this continuation remains `1279 passed, 2 skipped`.
- Latest continuation verification (same workspace baseline):
	- parity/fixture/CLI targeted slice: `10 passed, 67 deselected`
	- Qt GUI targeted slice (`test_analysis_workspace_qt`, `test_module3_workflows_qt`, `test_modern_gui_shell`): `49 passed`
	- broad Phase 3 regression slice (unfolding/module3/analysis/calibration/CLI): `204 passed`
	- full modern-shell regression file after workflow-preset persistence + shell split: `14 passed`
	- Phase 6 CLI slice against the real RAFM/LDRD corpus: `3 passed`
	- Phase 6 Qt slice after the shell split: `3 passed, 29 deselected`
	- full suite: `1279 passed, 2 skipped`
	- CLI release check command executed: `gui-acceptance-check` wrote `artifacts/gui_review/phase327_probe/gui_acceptance_check.json`
	- native probe galleries refreshed: `artifacts/gui_review/phase326_probe/index.html`, `artifacts/gui_review/phase317_calibration_probe/index.html`, `artifacts/gui_review/phase31x_unfolding_probe/index.html`, `artifacts/gui_review/phase327_probe/index.html`, and `artifacts/gui_review/phase6_optimization_probe/index.html`
- GUI recovery note (important for the next session): the default Miniforge interpreter (`/groupspace/cnerg/users/smandych/miniforge3/bin/python`) does not include `PySide6` in this workspace, so native Qt launch/probes must use `/usr/bin/python`; use `PYTHONPATH=src /usr/bin/python -m fluxforge.gui.app --project-dir .` for the app and run probes with `/usr/bin/python` under `tests/gui_*_probe.py`.
- Browserless runtime note: FluxForge GUI runs natively and does not require a browser; browser use is only for optional HTML artifact-gallery inspection. If a `file:///filespace/...` gallery URL fails, resolve/open the real path under `/groupspace/...` or skip browser review when only native GUI execution is needed.
- Do not reopen `3.18` / `3.19` solely because older tracker summaries lag the codebase; only reopen them for concrete missing capabilities or regressions.
- `3.26` is now implemented in-repo:
	- saved theme profiles and persistence in `src/fluxforge/gui/mode_manager.py` + `src/fluxforge/gui/widgets/mode_switcher.py`
	- stronger graph-table synchronization in `src/fluxforge/gui/backends/pyqtgraph_backend.py` + `src/fluxforge/gui/panels/modern_shell.py`
	- clearer launch/discovery paths via the `Workspaces` menu and dock focus actions in `src/fluxforge/gui/main_window.py`
- `3.27` is now implemented in-repo:
	- expanded verification coverage in `tests/test_analysis_workspace_qt.py`, `tests/test_module3_workflows_qt.py`, `tests/test_modern_gui_shell.py`, and `tests/test_cli_app.py`
	- native release probe at `tests/gui_phase327_release_probe.py` with generated evidence under `artifacts/gui_review/phase327_probe/`
	- release-blocking checklist at `docs/PHASE3_27_RELEASE_CHECKLIST.md`
	- CLI readiness report command `gui-acceptance-check` in `src/fluxforge/cli/app.py`
- Phase 6 continuation refresh is now implemented in-repo:
	- deterministic irradiation-optimization support artifacts in `src/fluxforge/workflows/irradiation_optimization.py`
	- benchmark experimental bundle packaging in `src/fluxforge/io/artifacts.py`
	- modern Qt Phase 6 surfaces in `src/fluxforge/gui/panels/phase6.py`
	- persisted GUI workflow/workspace presets in `src/fluxforge/gui/workflow_presets.py` + `src/fluxforge/gui/main_window.py`
	- built-in saved workflows `quantumgold-workflow` and `astm-ldrd-irradiation`
	- shell modularization via `src/fluxforge/gui/panels/modern_shell_center.py`, `modern_shell_sidebar.py`, `modern_shell_context.py`, and `modern_shell_shared.py`
- Blockers called out in the prior refresh have been resolved in this pass:
	- CLI continuum-driver import/runtime compatibility restored in `src/fluxforge/core/analysis_workspace.py`
	- parity-fixture expectation mismatches reconciled across current fixture files and parity comparator handling
	- full-suite verification completed in this workspace: `1279 passed, 2 skipped`

Current restart snapshot (authoritative continuation state as of this handoff refresh):
- Active branch is `optimization-workflows`.
- Working tree is intentionally dirty with parity-fixture and parity-runner work in progress. Do not discard these edits unless explicitly directed.
- Modified/tracked files at handoff refresh include: `src/fluxforge/cli/app.py`, `src/fluxforge/core/analysis_workspace.py`, `src/fluxforge/gui/main_window.py`, `src/fluxforge/gui/library_manager.py`, `src/fluxforge/gui/mode_manager.py`, `src/fluxforge/gui/panels/modern_shell.py`, `src/fluxforge/io/artifacts.py`, `src/fluxforge/validation/__init__.py`, `src/fluxforge/workflows/__init__.py`, parity manifests under `tests/spectra/reference_parity/cases/`, activation fixture manifests under `tests/activation_inventory/fixtures/`, and parity/GUI tests including `tests/test_cli_app.py`, `tests/test_analysis_workspace_qt.py`, `tests/test_modern_gui_shell.py`, `tests/test_parity_fixture_manifests.py`, and `tests/test_parity_phase3_scaffolding.py`.
- Untracked/new files at handoff refresh include: `src/fluxforge/gui/workflow_presets.py`, `src/fluxforge/gui/panels/phase6.py`, `src/fluxforge/gui/panels/modern_shell_center.py`, `src/fluxforge/gui/panels/modern_shell_sidebar.py`, `src/fluxforge/gui/panels/modern_shell_context.py`, `src/fluxforge/gui/panels/modern_shell_shared.py`, `src/fluxforge/workflows/irradiation_optimization.py`, `src/fluxforge/validation/reference_parity.py`, `tests/_phase6_real_data.py`, `tests/gui_phase327_release_probe.py`, `tests/gui_phase6_optimization_probe.py`, `tests/test_reference_parity_runner.py`, activation fixture schema/data materializations, and multiple parity fixture inputs/expected outputs under `tests/spectra/reference_parity/cases/` and `tests/activation_inventory/fixtures/`.
- The current coding session completed the `3.26` and `3.27` patch sets plus the Phase 6 workflow/workspace persistence refresh, modern-shell split, targeted regression tests, probe generation, and documentation updates.

Known blockers and unresolved verification risks (must be resolved before claiming any step complete):
- `3.24` direct-manipulation canvas parity remains a sequence dependency if strict roadmap order enforcement is required before final release sign-off.

Immediate next-action map (resume exactly here):
1. Keep sequence lock on `3.18` as the active gate.
	- Land remaining identification/activity/reference parity capabilities.
	- Preserve existing `3.26`/`3.27` repo implementations as regression-protected while `3.18` closes.
2. Continue `3.21` -> `3.23` parity expansion.
	- Add missing algorithm/workflow fixture families and tighten tolerances/provenance.
	- Extend Qt parity surfaces and artifact-review evidence where still missing.
3. Land `3.24` direct-manipulation canvas parity.
	- Right-click peak actions, ROI/background handles, and plot-driven edits with Qt + probe evidence.
4. Reconcile `3.25` evidence in roadmap vs implementation docs before declaring sequence closure.
	- Treat `3.25` as open until explicit in-repo tests/artifacts are re-verified and linked in status docs.
5. Run closure evidence cycle for each completed gate.
	- Targeted suites -> full suite.
	- Native GUI launch/sizing validation.
	- Probe/artifact review.
	- Status-doc updates.
6. Continue the Phase 6 irradiation-optimization closure pack in Section 10A.
	- Preserve the implemented CLI/GUI/export/workflow-preset paths while adjacent parity work lands.
	- Expand source-linked fixtures and real-mouse/artifact evidence before claiming broader sequence closure.
	- Layer remaining ML/surrogate additions only as additive accelerators over the current deterministic path.

## 1. Mission and Current Baseline

Primary mission:
- Finish all remaining Phase 3 items in sequence, with no skipped quality gates.
- Ensure every adopted capability is implemented in core logic, CLI, and modern Qt GUI.
- Ensure every adopted capability has automated tests plus GUI interaction verification.

Current baseline (already complete):
- Step 3.18.1 (Unfolding and regularized inversion parity) is complete in-repo.
- Step 3.19.1 (Flux-wire and activation analysis parity) is complete in-repo.
- Full-suite baseline in this workspace has been validated at 1279 passed, 2 skipped.

Do not re-open completed steps unless a regression is found.

## 2. Non-Negotiable Execution Rules

1. Sequence lock:
- Do not move to the next roadmap step until the current one is fully implemented, tested, GUI-integrated, GUI-tested, and documented.

2. Accuracy-first:
- No temporary patches.
- No fake pass behavior.
- No knowingly incorrect approximations to move forward.
- No tolerance loosening without scientific justification and recorded rationale.

3. Modern GUI only for new parity work:
- Implement new GUI features in src/fluxforge/gui.
- Do not land new parity features in src/fluxforge_gui (legacy Tk path).

4. Open-source and reproducibility goals:
- Keep workflows offline-first and reproducible.
- Preserve provenance in outputs and reports.
- Keep methods additive (new methods do not delete valid existing methods by default).

5. Testing is mandatory, not optional:
- Every feature change must include backend tests, CLI coverage, and GUI coverage where user-facing.
- Interaction-heavy GUI surfaces require Qt tests plus review artifacts.

## 3. Source-of-Truth Reading Order (Required)

Read these in order at the start of every new chat:

1. docs/ROADMAP_EXECUTION_STATUS.md
2. docs/FLUXFORGE_CONSOLIDATED_MASTER.md
3. docs/GUI_PLAN.md
4. docs/FluxForge_Testing_Master.md
5. docs/RAFM_UNFOLDING_SOLVER_FIXES.md
6. docs/FluxForge_NAA_Optimization_FISPACT_Addition.md
7. docs/FluxForge_HPGe_Optimization_and_Masking_Methodology.md
8. docs/FluxForge_ML_Libraries_Addition.md
9. docs/FluxForge_Irradiation_Design_Literature_and_Handoff.md
10. docs/REPO_CLEANUP_WORKSTREAM.md
11. docs/optimization_of_irradiation/irradiation_optimization_master_plan.md

If details appear missing in consolidated docs, mine additional guidance from:
- docs/optimization_of_irradiation/
- docs/archive/planning_snapshot_2026-04-06/
- docs/archive/legacy_plans/
- docs/archive/roadmap.md

Important: older archived docs can add missing detail but do not override newer consolidated policy.

## 4. Start-Here Checklist for a New LLM Session

Run this sequence every time:

1. Confirm repo status and current branch state.
2. Read the source-of-truth docs in the required order above.
3. Confirm current sequence gate from docs/ROADMAP_EXECUTION_STATUS.md.
4. Identify the exact next incomplete step (do not skip ahead).
5. Identify reference behavior/data in ../testing repos for that step.
6. Implement backend first, then CLI, then modern Qt GUI.
7. Add/expand tests (backend, CLI, Qt GUI).
8. Generate GUI review artifacts and perform Playwright/browser artifact review.
9. Launch modern GUI and verify sizing/usability manually.
10. Run targeted suites first, then run full suite before declaring completion.
11. Update roadmap and status docs with evidence and residual risks.
12. Confirm no unresolved dirty-worktree files remain unexplained in the final handoff note.

## 5. Implementation Boundaries and Code Locations

Primary implementation targets:
- src/fluxforge/analysis
- src/fluxforge/core
- src/fluxforge/workflows
- src/fluxforge/unfolding
- src/fluxforge/standards
- src/fluxforge/data
- src/fluxforge/io
- src/fluxforge/cli/app.py
- src/fluxforge/gui
- src/fluxforge/gui/dialogs
- src/fluxforge/gui/panels

Primary test locations:
- tests/
- tests/data/
- tests/data/neutron_reference/
- tests/spectra/reference_parity/
- tests/activation_inventory/

GUI review/probe artifact locations:
- tests/gui_unfolding_workspace_probe.py
- tests/gui_module3_workflows_probe.py
- tests/gui_calibration_workspace_probe.py
- tests/gui_predictive_dashboard_probe.py
- artifacts/gui_review/

Legacy GUI path (avoid for new parity features):
- src/fluxforge_gui/

## 6. Reference Repositories in testing/ and Usage Policy

Reference root:
- ../testing

Reference writeup:
- ../testing/writeup.md

Current audited top-level reference repositories:
- ../testing/actigamma
- ../testing/activation
- ../testing/becquerel
- ../testing/curie
- ../testing/Gamma-MCA
- ../testing/gamma_spec_analysis
- ../testing/gammaspectroscopy
- ../testing/gmapy
- ../testing/GSA-v2
- ../testing/GSA-v4
- ../testing/hdtv
- ../testing/INAA-INRIM 3.1
- ../testing/InterSpec
- ../testing/irrad_spectroscopy
- ../testing/KayWinV410
- ../testing/NAA-ANN-1
- ../testing/NASA-gamma
- ../testing/Neutron-Spectrometry
- ../testing/Neutron-Unfolding
- ../testing/npat
- ../testing/peakingduck
- ../testing/prospect_trial_installation
- ../testing/py-findpeaks
- ../testing/PyGammaSpec
- ../testing/pyunfold
- ../testing/radioactivedecay
- ../testing/SpecKit

Hard policy for testing/ repos:
- Use them as inspiration, algorithm references, and sources of fixture data/examples.
- Do not make FluxForge runtime depend on code imported from ../testing.
- Port needed fixture data into FluxForge tests/data or examples with provenance notes.
- Re-implement methods natively in FluxForge architecture.

## 7. Strict Feature Delivery Lifecycle (Use for Every New Capability)

For each feature, execute this exact lifecycle:

1. Reference extraction:
- Identify matching implementation/data in ../testing and relevant docs.

2. Native adaptation:
- Implement or refine native FluxForge logic in src/fluxforge.

3. Mathematical verification:
- Add pytest coverage proving expected behavior against reference fixtures/tolerances.

4. CLI integration:
- Expose feature in src/fluxforge/cli/app.py with tests in tests/test_cli_app.py.

5. Modern Qt integration:
- Expose feature in src/fluxforge/gui and related dialogs/panels.

6. GUI automation and artifact review:
- Add/expand Pytest-Qt coverage.
- Generate review artifacts via probe scripts.
- Perform Playwright/browser artifact review for interaction evidence.

7. Manual GUI launch and usability check:
- Launch the modern Qt app and verify layout sizing and usability.

8. Documentation updates:
- Update docs/ROADMAP_EXECUTION_STATUS.md and related planning docs with evidence.

No progression to a new step is allowed until all 8 lifecycle items are complete.

## 8. Remaining Phase 3 Sequence and Gate Criteria

Formal sequence status remains controlled by `docs/ROADMAP_EXECUTION_STATUS.md` and proceeds in this exact order:

1. Step 3.18 remaining work (in-progress):
- Finish remaining identification/activity/reference parity items not covered by 3.18.1.
- Complete relative-activity, source-age/decay-chain review surfaces, deeper library/reference behavior, and isotopics parity in modern Qt + CLI.

2. Step 3.19 remaining work (in-progress/planned mix):
- Complete operational calculators and workflows (detection limits, dose/attenuation, shielding/source-fit, optimization planners, masking guidance, long-horizon dose reviews).

3. Step 3.20 (repo status: complete):
- Keep regression coverage and doc evidence current; reopen only on concrete regression.

4. Step 3.20.1:
- Complete advanced GUI decay tracking and identification confidence overlays.

5. Step 3.21 (repo status: in-progress):
- Complete repo-backed fixture manifests with provenance and tolerances.

6. Step 3.22 (repo status: in-progress):
- Complete algorithm-level parity tests across all required families.
- Close strict exit gates by adding modern Qt parity surfaces, Qt interaction tests, and Playwright/browser artifact-review evidence for those surfaces.

7. Step 3.23 (repo status: scaffolded/in-progress):
- Complete workflow-level end-to-end parity suites after `3.22` mandatory exit gates are closed.

8. Step 3.23.1:
- Complete formal testing/ integration pipeline from audited components.

9. Step 3.24:
- Complete direct-manipulation canvas parity in modern Qt.

10. Step 3.25 (repo status: unresolved in status docs):
- Re-verify dedicated Qt parity-workspace evidence and keep status docs synchronized before treating this step as closed.

11. Step 3.26:
- Preserve existing repo implementation and keep regression coverage current while earlier sequence gates close.

12. Step 3.27:
- Preserve existing repo implementation and keep release-acceptance artifacts/checklist coverage current while earlier sequence gates close.

Execution reality note:
- If local implementation is already underway for a later item (for example `3.26`/`3.27`) while earlier items remain open, do not mark later items complete until earlier sequence gates are closed. You may continue implementation work in the same branch, but final status claims must preserve roadmap order.

Per-step mandatory exit gate:
- Backend implementation complete.
- CLI interface complete.
- Modern Qt GUI interface complete.
- Backend/CLI/GUI tests passing.
- GUI probe artifacts generated and reviewed.
- Modern GUI launched and visually validated for sizing/usability.
- Status docs updated with test evidence and known limitations.
- Any temporary step-specific probe/helper scripts added only to close the current step are either promoted into the durable probe set for `3.27` or removed before the next roadmap step begins.

## 9. Unfolding Workflow Hardening Checklist

Even though 3.18.1 is marked complete, keep this hardening checklist active for regressions and parity expansion:

Core unfolding paths:
- src/fluxforge/unfolding/
- src/fluxforge/workflows/spectrum_unfolding.py
- src/fluxforge/cli/app.py
- src/fluxforge/gui/dialogs/unfolding_dialog.py

Required method surfaces to keep consistent:
- Registry entry
- Workflow support
- CLI parser/dispatch
- Modern Qt selection and diagnostics
- Artifact/report outputs

Critical unfolding tests:
- tests/test_unfolding_registry.py
- tests/test_unfolding_workflows.py
- tests/test_unfolding_external_examples.py
- tests/test_unfolding_reference_parity.py
- tests/test_unfolding_workspace_qt.py
- tests/test_unfolding_diagnostics.py

Reference data sources:
- ../testing/Neutron-Unfolding
- ../testing/SpecKit
- ../testing/gmapy
- ../testing/pyunfold

Policy reminder:
- Keep examples/unfolding_external fixture-only.
- Do not couple FluxForge runtime to external repo code.

## 10. Irradiation Optimization Workstream Map (Implementation + Tests)

Primary optimization plans:
- docs/optimization_of_irradiation/irradiation_optimization_master_plan.md
- docs/optimization_of_irradiation/method1_difom_workflow.md
- docs/optimization_of_irradiation/method2_fim_workflow.md
- docs/optimization_of_irradiation/method3_mwdcs_workflow.md
- docs/optimization_of_irradiation/method_n1_bassd_workflow.md
- docs/optimization_of_irradiation/method_n2_stbdmr_workflow.md
- docs/optimization_of_irradiation/method_n3_spectral_feature_overlay_workflow.md
- docs/optimization_of_irradiation/isotope_priority_workflow.md
- docs/optimization_of_irradiation/isotopes_of_interest_filter_workflow.md
- docs/optimization_of_irradiation/rafm_second_irradiation_baseline.md

Primary code modules:
- src/fluxforge/analysis/optimization_difom.py
- src/fluxforge/analysis/optimization_fim.py
- src/fluxforge/analysis/optimization_mwdcs.py
- src/fluxforge/analysis/optimization_bassd.py
- src/fluxforge/analysis/optimization_stbdmr.py
- src/fluxforge/analysis/optimization_schedule_builder.py
- src/fluxforge/cli/app.py (optimization-sweep, isotope-priority, masking-review)
- src/fluxforge/core/analysis_workspace.py and modern Qt panels for GUI previews

Optimization test modules:
- tests/test_optimization_difom.py
- tests/test_optimization_fim.py
- tests/test_optimization_mwdcs.py
- tests/test_optimization_bassd.py
- tests/test_optimization_stbdmr.py
- tests/test_optimization_schedule_builder.py
- tests/test_isotope_priority.py
- tests/test_masking_review.py
- tests/test_cli_app.py
- tests/test_analysis_workspace_qt.py

Required behavior for this workstream:
- First and second irradiation handling.
- Multi-window schedule support.
- Masking-aware diagnostics and recommendations.
- Exportable machine-readable outputs.
- GUI and CLI parity for each objective family.

### 10A. Phase 6 Implementation Pack (Irradiation Optimization + NAA + ML)

Interpretation rule for "Phase 6" in this repository:
- Treat "Phase 6" as the irradiation-optimization capability family anchored on `3N.6` (plus `3N.1`-`3N.8` dependencies) and mapped onto roadmap execution steps `3.19`, `3.20`, and `3.25`.
- Keep sequence lock in force: this capability can be implemented incrementally in-branch, but formal sequence completion claims must respect open earlier gates.

Primary source docs for this pack:
- `docs/FluxForge_NAA_Optimization_FISPACT_Addition.md`
- `docs/FluxForge_ML_Libraries_Addition.md`
- `docs/GUI_PLAN.md`
- `docs/FluxForge_Testing_Master.md`

#### 10A.0 Current repo status for the next LLM (authoritative as of 2026-04-17)

Implemented Phase 6 code paths already present:
- `src/fluxforge/workflows/irradiation_optimization.py` builds masking, inventory, optimization-grid, recommended-schedule, dose-endpoint, and second-irradiation support artifacts from measured activity-review inputs.
- `src/fluxforge/io/artifacts.py` now packages benchmark experimental bundles via `.ffexp`.
- `src/fluxforge/cli/app.py` now exposes `activity-review`, `inventory-review`, `isotope-priority`, `masking-review`, `optimization-sweep`, `second-irradiation-plan`, and `ffexp-export`.
- `src/fluxforge/gui/panels/phase6.py` now provides `MaskingReviewPanel`, `OptimizationWorkspacePanel`, and `SecondIrradiationPlannerPanel`, including `.ffexp` export from the optimizer panel.
- `src/fluxforge/gui/panels/modern_shell.py` wires those Phase 6 panels into the bottom workspace tabs and round-trips their state through `workflow_state()` / `apply_workflow_state()`.
- `src/fluxforge/gui/workflow_presets.py` + `src/fluxforge/gui/main_window.py` now persist GUI workflow/workspace state across sessions, with built-in presets `quantumgold-workflow` and `astm-ldrd-irradiation`.
- `src/fluxforge/gui/panels/modern_shell_center.py`, `modern_shell_sidebar.py`, `modern_shell_context.py`, and `modern_shell_shared.py` now hold extracted pieces of the former oversized `modern_shell.py`; do not collapse them back into one file.
- `tests/_phase6_real_data.py` intentionally uses the real RAFM/LDRD irradiation corpus and aligned testing-repo inputs rather than synthetic optimization-only fixtures.

Implemented verification/evidence already present:
- `tests/test_cli_app.py` covers the current Phase 6 CLI/export slice.
- `tests/test_analysis_workspace_qt.py` covers masking, optimization, `.ffexp`, and second-irradiation Qt workflows.
- `tests/test_modern_gui_shell.py` covers workflow preset persistence and active-workflow restore across GUI sessions.
- `tests/gui_phase6_optimization_probe.py` writes the native review gallery at `artifacts/gui_review/phase6_optimization_probe/`.
- The generated Phase 6 gallery was already reviewed in the browser/Playwright lane and matched the native row counts.

#### 10A.1 Required data-model prerequisites (must exist and be exercised)

- `IrradiationSchedule` with irradiation, cooldown, and count segments.
- `InventoryState` for atoms/activity/mass at explicit times.
- `ObservableTimeSeries` for activity/heat/dose and related metrics.
- `LineMaskingResult` with ranked masking candidates.
- `OptimizationScenario` describing objective, grids, and constraints.

Implementation rule:
- These objects are not optional scaffolding; they are the contract for CLI, GUI, exports, and tests.

#### 10A.2 Physics-first optimization engine requirements

Mandatory objective modes:
1. maximize target counts
2. maximize signal-to-background
3. maximize signal-to-mask ratio
4. minimize relative uncertainty on target activity
5. minimize required count time for fixed detectability margin
6. minimize shutdown dose while preserving detectability
7. maximize target-vs-mask separation
8. optimize for chosen endpoint (`shutdown`, `1 d`, `1 wk`, `1 y`, `100 y`, custom)

Mandatory sweep parameters:
- irradiation time
- cooldown time
- count time
- target isotope and candidate lines
- optional second-irradiation duration
- pulse delay between irradiations
- flux/power state assumptions where available

Required deterministic outputs:
- `optimization_grid.csv`
- `recommended_schedules.csv`
- `dose_endpoints.csv`
- `masking_candidates.csv`
- `inventory_timeseries.csv`
- `activities_at_irradiation.csv`

#### 10A.3 ML and surrogate policy (additive, never replacement)

Hard policy:
- ML must not replace Bateman/activation/counting physics.
- ML is allowed only as surrogate/ranking/acquisition acceleration over a physics-grounded forward model.

Required ML-capable features for this pack:
- multi-objective Bayesian optimization over expensive schedule spaces
- expected information gain support for second-irradiation planning
- multi-line/multi-isotope joint objective handling
- optional full-spectrum Bayesian activity inference mode for unstable single-line cases

Good ML uses (allowed):
- proposing high-value schedule regions
- surrogate objective prediction
- ranking likely dominant/masking isotopes
- reducing expensive posterior evaluations

Disallowed ML uses:
- opaque recommendation without provenance
- bypassing uncertainty propagation
- replacing decay-chain/activation equations

#### 10A.4 Nuclear library stack requirements for optimization/NAA validity

Required layered data architecture for this pack:
- ENDF/B-VIII base decay layer
- ENSDF evaluated backbone
- IAEA LiveChart sync layer
- NUBASE2020 half-life/isomer overlay
- DDEP/LNHB/IAEA standards override subset
- optional SandiaDecay-compatible interoperability layer
- coincidence/cascade extension layer
- reaction-gamma extension layer

Conflict-resolution contract (required):
- record `display_value`, `display_uncertainty`, `source_name`, `source_version`, `source_priority`, `alternate_values`, and bibliography/provenance entries for surfaced values.

#### 10A.5 Required CLI, GUI, and export surfaces

CLI requirements:
- keep `activity-review`, `inventory-review`, `isotope-priority`, `masking-review`, `optimization-sweep`, `second-irradiation-plan`, and `ffexp-export` green as adjacent parity work lands
- if additional schedule/objective variants are added, route them through the existing Phase 6 artifact builder rather than creating one-off export paths

GUI requirements:
- `Inventory / Time Evolution` workspace
- `Line Interference / Masking` workspace/panel
- irradiation schedule optimizer workspace with heatmap + Pareto + recommendation card views
- second-irradiation planner views (timeline + comparison)
- launch path for benchmark bundle export (`.ffexp`)

Machine-readable export contract:
- benchmark experimental bundle `.ffexp` containing metadata, activity/inventory/dose/masking/optimization products and plot manifests

#### 10A.6 Mandatory test matrix for phase6 capability

Core/analysis tests:
- `tests/test_optimization_difom.py`
- `tests/test_optimization_fim.py`
- `tests/test_optimization_mwdcs.py`
- `tests/test_optimization_bassd.py`
- `tests/test_optimization_stbdmr.py`
- `tests/test_optimization_schedule_builder.py`
- `tests/test_isotope_priority.py`
- `tests/test_masking_review.py`

Activation/NAA inventory test family (must be present and green):
- decay-only and Bateman-chain regressions
- EOI reconstruction tests
- observable trend regressions
- CSV export tests
- plot smoke tests

Workflow and interface tests:
- `tests/test_cli_app.py` coverage for all phase6-facing commands
- Qt tests for optimization/masking/time-evolution workspaces in `tests/test_*_qt.py`
- `tests/test_modern_gui_shell.py` coverage for saved-workflow persistence and session restore
- native GUI probe evidence under `artifacts/gui_review/` for each new interaction-heavy surface

Current targeted-green evidence in this workspace:
- `PYTHONPATH=src pytest -q tests/test_cli_app.py -k "second_irradiation_plan_writes_json_and_csv_outputs or optimization_sweep_builds_candidates_from_activity_review or ffexp_export_packages_phase6_products"` -> `3 passed`
- `PYTHONPATH=src pytest -q tests/test_analysis_workspace_qt.py -k "masking_review_panel_runs_and_exports_tables or optimization_workspace_panel_runs_and_exports_phase6_bundle or optimization_workspace_panel_advanced_guard_and_second_irradiation_panel"` -> `3 passed, 29 deselected`
- `PYTHONPATH=src pytest -q tests/test_modern_gui_shell.py` -> `14 passed`
- `PYTHONPATH=src /usr/bin/python tests/gui_phase6_optimization_probe.py artifacts/gui_review/phase6_optimization_probe` -> generated gallery + `phase6_probe.ffexp` (mouse-driven panel actions via `QTest.mouseClick`)

#### 10A.7 Phase6 acceptance gate (do not claim done until all pass)

1. Measured spectrum -> activity at count time and EOI is reproducible.
2. Time-evolution and daughter-chain views are available with uncertainty-aware outputs.
3. Masking candidates are ranked with alternate-line/cooldown guidance.
4. One-pulse vs two-pulse schedules are comparable by objective and constraints.
5. Heatmap/Pareto/recommended-schedule outputs are generated and exported.
6. Shutdown and long-term endpoints (`shutdown`, `1 y`, `100 y`) are available in exports/plots.
7. CLI + GUI + tests + probes are all green for the implemented slice.
8. `.ffexp` experimental bundle export contains required optimization and provenance artifacts.

Current status note:
- The implemented Phase 6 slice satisfies the targeted acceptance gate above for the currently landed deterministic workflows, saved-workflow recall, and `.ffexp` export path. Broader sequence closure is still blocked by the remaining roadmap parity gates outside this slice.

#### 10A.8 Remaining continuation tasks for the next LLM

1. Keep the current deterministic Phase 6 path stable while `3.18` through `3.25` parity work continues.
- Do not regress `activity-review` -> `inventory-review` -> `masking-review` / `optimization-sweep` / `second-irradiation-plan` -> `.ffexp`.

2. Expand source-linked fixture depth using real LDRD/testing data, not synthetic-only optimization payloads.
- Prioritize richer cases under `tests/activation_inventory/fixtures/` and `tests/spectra/reference_parity/cases/` sourced from `fluxforge/ldrd_irradiation/` and `../testing/writeup.md`.

3. Add broader real-mouse GUI evidence for saved workflow load/save/delete paths and the Phase 6 tabs.
- Keep the native probe gallery path, then review the resulting `index.html` in the browser lane.

4. Continue ML/library additions only through the physics-first policy in Sections 10A.3 and 10A.4.
- Surrogates may rank or accelerate schedule search, but they must not bypass the existing Bateman/activity/inventory chain or provenance reporting.

5. Before any future "Phase 6 complete" claim, refresh:
- the targeted CLI slice,
- the targeted Qt Phase 6 slice,
- `tests/test_modern_gui_shell.py`,
- the Phase 6 native probe,
- and then the broader suite required by the current roadmap gate.

## 11. Modern GUI Verification Protocol (Required Every Step)

A. Automated Qt interaction tests:
- Use Pytest-Qt tests for each new workspace/action in tests/test_*_qt.py.

B. Native Qt probe evidence:
- Generate screenshot galleries using probe scripts under tests/gui_*_probe.py.
- Store outputs under artifacts/gui_review/ with per-surface index.html.

C. Playwright/browser artifact review lane:
- Use Playwright to review generated HTML artifact galleries and capture interaction evidence (click filters/tabs, verify state screenshots) when a closure gate explicitly requires browser-reviewed evidence.
- This lane is for repeatable interaction review artifacts and release evidence, not for running the Qt GUI itself.

C1. Real mouse interaction requirement for modern Qt:
- Keep Pytest-Qt coverage for widget-level regression, but also require at least one real desktop interaction run for each new major GUI surface.
- Implement/maintain a modern-Qt desktop driver lane (pyautogui/pywinauto style) that performs real mouse clicks and drags against src/fluxforge/gui surfaces.
- Save screenshots and action logs to artifacts/gui_review and review those artifacts with Playwright.
- Do not declare a GUI step complete without both widget-test evidence and real-mouse evidence.

D. Manual modern GUI launch and sizing review after each completed step:
- Launch: PYTHONPATH=src /usr/bin/python -m fluxforge.gui.app --project-dir .
- Validate usability at minimum laptop and desktop window sizes.
- Confirm no clipped controls, inaccessible actions, or unreadable tables.

D1. Current recommended probe refresh commands for continuation:
- `PYTHONPATH=src /usr/bin/python tests/gui_analysis_workspace_probe.py artifacts/gui_review/phase326_probe`
- `PYTHONPATH=src /usr/bin/python tests/gui_module3_workflows_probe.py artifacts/gui_review/phase327_probe`
- `PYTHONPATH=src /usr/bin/python tests/gui_phase327_release_probe.py artifacts/gui_review/phase327_probe`
- `PYTHONPATH=src /usr/bin/python tests/gui_phase6_optimization_probe.py artifacts/gui_review/phase6_optimization_probe`

E. Legacy path policy:
- Keep src/fluxforge_gui as legacy reference/fallback only.
- Do not satisfy new phase requirements by only modifying legacy GUI.

## 12. Required Test Commands and Interpretation

Baseline full-suite command:
- pytest -q -rs

Recommended continuation command order from the current state:
1. `pytest -q tests/test_reference_parity_runner.py tests/test_parity_fixture_manifests.py tests/test_parity_phase3_scaffolding.py tests/test_cli_app.py -k "parity or fixture or manifest"`
2. `pytest -q tests/test_analysis_workspace_qt.py tests/test_module3_workflows_qt.py tests/test_modern_gui_shell.py`
3. Run native probe commands from Section 11 (D1).
4. `pytest -q -rs`

Important interpretation rule:
- Deselected counts usually come from targeted commands (file subset or -k filtering), not from missing tests.

Phase-focused examples:
- Unfolding slice: pytest -q tests/test_unfolding_registry.py tests/test_unfolding_external_examples.py tests/test_unfolding_workflows.py tests/test_unfolding_workspace_qt.py tests/test_cli_app.py -k "unfold"
- Optimization slice: pytest -q tests/test_optimization_difom.py tests/test_optimization_fim.py tests/test_optimization_mwdcs.py tests/test_optimization_bassd.py tests/test_optimization_stbdmr.py tests/test_cli_app.py tests/test_analysis_workspace_qt.py -k "optimization_sweep or difom or fim or mwdcs or bassd or stbdmr"
- Phase6 core slice: pytest -q tests/test_optimization_difom.py tests/test_optimization_fim.py tests/test_optimization_mwdcs.py tests/test_optimization_bassd.py tests/test_optimization_stbdmr.py tests/test_optimization_schedule_builder.py tests/test_isotope_priority.py tests/test_masking_review.py tests/test_cli_app.py -k "optimization|isotope_priority|masking_review|second_irradiation"
- Current implemented Phase6 regression slice: pytest -q tests/test_cli_app.py -k "second_irradiation_plan_writes_json_and_csv_outputs or optimization_sweep_builds_candidates_from_activity_review or ffexp_export_packages_phase6_products"
- Current GUI workflow-preset slice: pytest -q tests/test_modern_gui_shell.py -k "workflow or main_window_restores_saved_workflow_state_across_sessions"

Skip/xfail policy:
- Do not add skip/xfail to bypass failing functionality.
- Environment-dependent skips are acceptable only when dependency/display constraints are real and documented.

## 13. Documentation Update Requirements (Per Completed Slice)

Always update, at minimum:
- docs/ROADMAP_EXECUTION_STATUS.md
- docs/FLUXFORGE_CONSOLIDATED_MASTER.md (status row if changed)
- docs/GUI_PLAN.md (if GUI scope/acceptance moved)
- docs/FluxForge_Testing_Master.md (if testing contract/coverage changed)
- docs/PHASE3_EXECUTION_HANDOFF.md (refresh branch state, blockers, and next concrete command sequence)

If consolidated docs are too sparse for implementation detail:
- Add pointers in this handoff or roadmap status to the exact supporting docs in docs/optimization_of_irradiation and docs/archive.

## 14. Definition of Done for Any Phase 3 Step

A step is done only when all are true:
- Feature behavior implemented in native FluxForge backend.
- CLI command path implemented and tested.
- Modern Qt GUI path implemented and tested.
- Parity/fixture tests added or updated with declared tolerances and provenance.
- GUI probe artifacts generated and reviewed.
- Playwright/browser artifact review evidence produced.
- Modern GUI launched and manually checked for sizing/usability.
- Full relevant test suite passing.
- Status docs updated with evidence and residual risk notes.

If any item above is missing, the step is not complete and work must continue.
