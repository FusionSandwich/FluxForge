# FluxForge GUI Master Plan

**Status:** active GUI source of truth  
**Last Updated:** 2026-04-19  
**Purpose:** consolidated GUI architecture, interaction, and workspace plan for the
modern Qt shell.

This file supersedes the older GUI planning files archived under
`docs/archive/planning_snapshot_2026-04-06/`.

## 1. GUI Source Order

Use the GUI docs set in this order:

1. `docs/ROADMAP_EXECUTION_STATUS.md` for live implementation state
2. `docs/GUI_PLAN.md` for GUI direction and interaction rules
3. `docs/FLUXFORGE_CONSOLIDATED_MASTER.md` for product scope and feature roadmap
4. `docs/FluxForge_Testing_Master.md` for GUI acceptance and parity requirements
5. `docs/FLUXFORGE_CONSOLIDATED_MASTER.md` section `10` for Phase 5 writeup-crosswalk implementation requirements
6. `../testing/writeup.md` section `0.7` for replay-oriented evidence and traceability methodology

## 2. Core GUI Rules

- FluxForge remains a native Windows/Linux desktop application without a browser dependency.
- The primary GUI surface is the Qt shell under `src/fluxforge/gui/`.
- The Tk shell under `src/fluxforge_gui/` is legacy reference material, not the target for new parity work.
- Every adopted GUI workflow must map cleanly onto reusable core logic, a CLI or scriptable API path, and a saved artifact/report/provenance record.
- The plot is a primary input surface, not a passive display.
- Automation never blocks manual analyst correction.
- Graphs and tables must stay synchronized.
- Standards mode must visibly lock governed workflows without destroying expert-state context.
- Library management must be provenance-aware: users choose which optional libraries are downloaded, external library locations can be registered from GUI and CLI, and bundled library IDs stay reserved against accidental shadowing.
- Prefer registry-driven or configuration-driven selectors over hardcoded workflow and value lists unless the lock is deliberate for QuantumGold parity, PeakEasy parity, governed standards workflows, or RAFM irradiation-analysis paths.

## 3. Implemented GUI Baseline

| Area | Implemented GUI Surface | Evidence |
|---|---|---|
| Shell architecture | Docked Qt shell, persisted layout, mode switcher, theme handling, hardware LED, dashboard reservation, and split shell containers in `modern_shell_center.py`, `modern_shell_sidebar.py`, `modern_shell_context.py`, and `modern_shell_shared.py` | Steps `1.3` through `1.18` in `docs/ROADMAP_EXECUTION_STATUS.md` plus `tests/test_modern_gui_shell.py` |
| Analysis workspace | Peak table, auto-review, background workflows, overlay roles, isotope browser, reassignment, cascade overlays, guide overlays | Steps `2.7` through `2.24`, `3.16` |
| Calibration and efficiency | Unified calibration dialog, quick slider, deviation pairs, ROI fitter, preserved/fine-tune flows, detector slots, all four efficiency models, NASA smart seed | Steps `2.1` through `2.6`, `2.11`, `3.17` |
| Standards and QA | Standards locks, QA history, ASTM checks, C1030 wizard, governed selectors and banners | Steps `3.7` through `3.12` |
| Reporting and batch | Report export dialog, batch queue, progress tracking, JSON/CSV outputs | Steps `3.13`, `3.14` |
| Workflow persistence | Saved workflow/workspace presets, active-workflow restore across sessions, and built-in `quantumgold-workflow` plus `astm-ldrd-irradiation` presets | `src/fluxforge/gui/workflow_presets.py`, `src/fluxforge/gui/main_window.py`, `tests/test_modern_gui_shell.py` |
| Phase 6 optimization surfaces | `Inventory / Time Evolution`, `Line Interference / Masking`, `Irradiation Optimizer`, and `Second Irradiation` tabs with `.ffexp` export from the optimizer workspace | `src/fluxforge/gui/panels/phase6.py`, `tests/test_analysis_workspace_qt.py`, `tests/gui_phase6_optimization_probe.py` |
| Phase 5 crosswalk review surface | `Phase 5 Parity` tab for testing-catalog crosswalk inspection, replay-state filtering, and parity-suite execution with workflow-state persistence | `src/fluxforge/gui/panels/phase5.py`, `src/fluxforge/gui/panels/modern_shell.py`, `tests/test_modern_gui_shell.py`, `tests/gui_phase5_parity_probe.py` |
| Predictive extras | ROI ETA, dead-time forecasting, QA recalibration forecasting, saved lists, mixtures, log-scale and peak-label toggles | Steps `4P.1` through `4P.7` |

## 4. Interaction Contract

The GUI must preserve the following behavior contracts:

- Explicit foreground, background, and secondary spectrum roles across loading, drag/drop, legends, and session provenance
- Click-table-to-zoom and plot-edit-to-update-table synchronization
- Manual peak assignment, deletion, replacement, and override after automated peak workflows
- Standards-aware method selectors, lock summaries, and governed data-source restrictions
- Artifact-backed review for interaction-heavy surfaces rather than screenshot-free unit tests only

## 5. Active GUI Roadmap

| Step | GUI Behavior | Status |
|---|---|---|
| 3.18 | Richer library/reference surfaces, source-age and decay-chain views, relative-activity and isotopics workflows, deeper identification context in the modern shell, and spectrum-level activation review/export surfaces | In Progress |
| 3.20 | File-query/archive workbench, exemplar batch reuse, detector-response lifecycle panels, multi-spectrum review, governed k0 characterization/report views, and saved workflow/workspace recall | In Progress (repo) |
| 3.24 | Direct-manipulation canvas parity: right-click peak editing, ROI/background drag handles, overlap actions, and explicit role actions | Planned |
| 3.25 | Dedicated Qt workspaces for ROI Statistics, Detection Limit, Relative Activity, Dose/Shielding, File Query/Batch Compare, Reference/Library Workbench, and k0 reporting; the first Phase 6 parity tabs are already live in the modern shell | In Progress (repo) |
| 3.26 | First-class dark mode, saved themes, stronger graph-table synchronization, and improved launch/discovery paths | Complete (repo) |
| 3.27 | Native probes, artifact reviews, and release-blocking GUI acceptance for every new parity workspace | Complete (repo) |
| 4.1-4.4 | Live-MCA device discovery, dashboard telemetry, and spectrogram surfaces | Deferred |
| 5.1 | testing/writeup crosswalk review and parity-state visibility in modern Qt | In Progress (repo) |
| 5.3 | GUI/workflow parity closure for the audited testing catalog: plot-controller actions, role-aware overlays, ROI/detection-limit/shielding/archive workspaces, saved analyst context, and report/export behavior parity | Planned |
| 5.6 | Phase 5 GUI acceptance gate: source-linked traceability, Qt interaction tests, native probe evidence, browser-lane artifact review, and manual GUI sizing validation | Planned |

## 6. Source Behaviors to Preserve

| Source Family | GUI Behaviors To Preserve | Planned Landing |
|---|---|---|
| `GSA-v4` | Plot-driven ROI/peak editing, detector-slot calibration workflows, separate ROI statistics surfaces, graph-plus-table review | `3.20`, `3.24`, `3.25` |
| `InterSpec` | Role-aware upload UX, plot-as-controller interaction, manual-first analyst overrides, detection-limit tools, file query/archive workbench, relative-activity and shielding workspaces, saved themes | `3.18` through `3.27` |
| `NASA-gamma` | Smart calibration seeding, broader peak-search families, advanced fit/review helpers, repeated-run diagnostics | `3.17`, `3.20`, `3.22` |
| `PeakEasy` / legacy desktop workflows | Dense but usable desktop-first interaction, strong calibration and peak-review ergonomics, saved analyst context | ongoing parity rule |

Current `3.18` GUI slice:
- Activity Results must review all matched isotope lines for the active spectrum, back-correct to EOI, and surface count-time plus irradiation-time activity summaries.
- The Qt shell must export isotope activity CSVs plus saved half-life decay and Bateman parent/daughter review plots without dropping uncertainty information from counting, efficiency, or emission probability inputs.
- Library/reference surfaces must preserve the bundled GSA-v4 edited and natural gamma libraries, NASA-gamma common-lab/natural/CapGam/IAEA capture/delayed-activation/inelastic families, and the ENDF/B-VIII supplement as selectable governed sources in the same provenance-aware picker model used for the existing bundled and actigamma sources.
- Future source-age and decay-chain GUI views must treat the bundled ICRP-107 decay network plus Kayzero 2020/2023 half-life-uncertainty overlays as first-class decay datasets rather than burying that provenance inside hardcoded assumptions.
- Library management surfaces must let users opt into downloadable library families, register their own library locations from the GUI, mirror the same capability in the terminal, and prevent built-in library-name collisions by reserving bundled IDs and forcing distinct user aliases when names conflict.
- Activity-facing GUI workspaces should expose an explicit activity-unit selector so analysts can switch between Bq-family and Ci-family views without changing the stored raw results.

Additive `3N` GUI overlay on the existing roadmap map:
- `3N.3` and `3N.4` extend the `3.18` and `3.25` GUI work with an `Inventory / Time Evolution` workspace, arbitrary-time solver controls, Bateman-chain plots, half-life plots, inventory evolution plots, and uncertainty-band export surfaces.
- `3N.8` extends the planned parity workspaces with a dedicated `Line Interference / Masking` panel, ranked masker tables, cooldown sensitivity plots, and line-choice comparisons.
- `3N.12` and `3N.13` extend the planned calculator and parity workspaces with optimization heatmaps, Pareto views, recommended-schedule summary cards, pulse timeline views, and second-irradiation comparison panels.
- `3N.14` adds long-term dose and hazard review surfaces, including shutdown-through-100-year endpoint presets, dominant-contributor plots, and domain quick-look presets for microreactor, fusion-material, and activation-experiment use cases.
- `3N.15` requires GUI launch points for benchmark experimental bundle export, including `.ffexp` packaging and plot-manifest review.

Current repo implementation note (2026-04-17):
- The additive `3N` GUI slice is no longer only planned: `src/fluxforge/gui/panels/phase6.py` now lands the first inventory, masking, optimization, second-irradiation, and `.ffexp` export surfaces in the modern shell.
- Workflow-state round-tripping is now part of the GUI contract for these panels and the surrounding shell containers so analysts can save a workspace and resume it in a later session.
- The default saved workflows intentionally preserve two governed starting points: `quantumgold-workflow` and `astm-ldrd-irradiation`.

## 7. Non-Negotiable GUI Acceptance Rules

- New GUI-only code is not enough; each adopted surface must have reusable backend logic and automated coverage.
- Interaction-heavy tools need Qt workflow tests plus native probe/artifact review.
- New parity work must land only in the Qt shell unless a deliberate archival/fallback reason is documented.
- Theme, overlay-role, and standards-lock behavior are part of product behavior, not optional polish.

## 8. Phase 5 GUI Traceability and Evidence Rules

- Every Phase 5 GUI parity claim must cite at least one concrete local path in
	`../testing/writeup.md` or the referenced source repository when such a path
	exists.
- Every GUI dataset or sample artifact cited for replay must be labeled as one
	of: `bundled locally`, `downloaded dynamically`, `generated during runtime`,
	or `docs-only / implied`.
- GUI behavior extraction must be explicit for each source family and include:
	plot-as-controller behavior, graph/table synchronization loops,
	foreground/background/secondary role handling, manual override availability,
	and saved theme/workspace/workflow state behavior.
- Phase 5 GUI slices are not complete until both automated Qt workflow tests and
	native probe artifacts are available, reviewed in the browser-lane artifact
	flow, and backed by manual modern-GUI sizing/usability checks.
- Classification guardrail remains in force for GUI library surfaces:
	prompt capture and reaction-gamma references must not be presented as decay
	emission-probability sources for activity calculations.

## 9. Archived GUI Inputs

The superseded GUI planning files now live in
`docs/archive/planning_snapshot_2026-04-06/`:

- `GUI_PLAN.md`
- `GUI_PLAN_old.md`
- `GUI_CAPABILITY_PROGRAM.md`
- `GUI_CAPABILITY_PROGRAM_old.md`
