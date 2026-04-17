# FLUXFORGE Consolidated Master

**Status:** active feature master  
**Last Updated:** 2026-04-17  
**Purpose:** consolidated source of truth for product scope, feature commitments, and
roadmap-level capability planning.

This file supersedes the older top-level planning inputs that were merged into
`docs/archive/planning_snapshot_2026-04-06/`.

## 1. Source of Truth Order

Use the docs set in this order when there is overlap:

1. `docs/ROADMAP_EXECUTION_STATUS.md` for live implementation and sequence state
2. `docs/FLUXFORGE_CONSOLIDATED_MASTER.md` for feature scope and roadmap intent
3. `docs/GUI_PLAN.md` for GUI architecture, interaction rules, and workspace direction
4. `docs/FluxForge_Testing_Master.md` for testing, parity, and acceptance rules
5. `docs/archive/planning_snapshot_2026-04-06/` for historical rationale and source-detail recovery

Conflict rules:

- Newer consolidated docs win over archived inputs.
- Additive delta documents add missing capabilities; they do not silently delete
  older valid requirements.
- The additive-capability rule remains in force: adding a method never justifies
  removing an existing valid one without an explicit correctness or safety reason.

## 2. Product Scope

FluxForge remains an offline-first native desktop toolkit for:

- HPGe gamma-spectroscopy analysis
- activity, reaction-rate, and standards-driven reduction workflows
- neutron dosimetry, response construction, and spectrum adjustment/unfolding
- reproducible experimental handoff bundles for downstream model-comparison tools

FluxForge owns the experimental side of the workflow. The separate comparison tool
continues to own modeled transport, modeled activation/photon sources, and final
experiment-vs-code comparison logic.

## 3. Non-Negotiable Rules

- Issue-first execution: significant work is tracked through GitHub issues and the ordered roadmap.
- Additive capability policy: new methods do not replace valid old ones by default.
- User-choice policy: scientifically defensible alternatives remain selectable and provenance-recorded.
- Standards-locked workflows: Standards mode constrains equations, defaults, and reports where required.
- Offline-first execution: supported workflows must not require network access.
- Provenance-first outputs: sessions, exports, and reports must record method, library, and parameter choices.
- Library-governance policy: optional downloadable libraries remain user-selected, external library locations must be registerable from both GUI and CLI, built-in FluxForge library IDs are reserved, and user-supplied libraries must never silently shadow or overwrite bundled libraries.
- Configuration-first policy: remove avoidable hardcoded workflows and values where practical, except where deliberate locking is required for QuantumGold or PeakEasy parity, governed standards workflows such as ASTM and k0-NAA, or RAFM irradiation-analysis paths.

## 4. Implemented Baseline

The repository is currently implemented through:

- Stage 0 governance and project-tracker scaffolding
- Phase 1 foundation in the modern Qt shell and shared package layout
- Phase 2 core analysis in the redesigned Qt path
- Phase 3.1 through 3.17, including unfolding, standards, reporting, batch, ROI parity,
  and calibration/efficiency parity
- the first landed Phase 6 irradiation-optimization slice, including deterministic
  masking/inventory/optimization/second-irradiation workflows, `.ffexp` bundle export,
  and saved GUI workflow/workspace presets for `quantumgold-workflow` and
  `astm-ldrd-irradiation`
- User-directed predictive items `4P.1` through `4P.7`

Live acquisition, HAL transport, and true MCA device surfaces remain deferred until
the offline-parity module closes.

## 5. Active Roadmap

| Step | Capability Family | Status | Required Deliverables |
|---|---|---|---|
| 3.18 | Identification / activity / reference parity | In Progress | Editable text/reference libraries; richer energy-window search; reference overlays; relative activity; source-age and decay-chain views; pinned mixtures; common-lab, natural, capture, inelastic, and delayed-activation library families; bundled GSA-v4 edited and natural gamma libraries; bundled NASA-gamma common-lab, natural-radiation, CapGam capture, IAEA capture, IAEA delayed-activation, Baghdad inelastic, and TALYS 14 MeV reaction-gamma libraries; bundled ICRP-107 decay-network dataset plus Kayzero 2020/2023 half-life-uncertainty overlays; bundled ENDF/B-VIII supplemental decay-line library; library install/register surfaces with opt-in downloads, GUI/CLI custom library-location entry, reserved built-in names and IDs, and collision-safe user aliases; arbitrary-time and EOI activity solving; GUI activity-unit selectors for activity-review and inventory plots/tables; `IrradiationSchedule`, `InventoryState`, `ObservableTimeSeries`, `LineMaskingResult`, and `OptimizationScenario` core models; GUI/CLI spectrum activation review; isotope-at-irradiation-time CSV export; uncertainty-bearing half-life decay plots and Bateman parent/daughter review plots; direct/comparator/relative-comparator/k0/activation-only workflows; experimental inventory object; isotopics-by-peaks and isotopics-by-nuclides workspaces |
| 3.19 | Operational calculator parity | Planned | Detection-limit workspaces; dose and attenuation calculators; shielding/source-fit tools; units conversion; FISPACT-style activity/heat/dose observables; dominant-contributor curves; irradiation/cooldown/count optimization; masking-isotope ranking; alternate-line and de-masking planning; second-irradiation planning; shutdown-through-100-year dose studies; in-GUI math/command terminal |
| 3.20 | Archive / batch / k0 parity | Planned | File-query and archive workbench; compact role-aware file management; session auto-store/resume; exemplar-driven batch analysis; detector-response lifecycle manager; multi-spectrum ROI/detector-consistency review; KayWin-style detector/facility characterization and governed k0 reporting; benchmark experimental bundle (`.ffexp`) export; library cache/install management and provenance-backed external-library registration |
| 3.21 | Repo-backed parity fixtures | Planned | Curated manifests under `tests/spectra/reference_parity/` plus `tests/activation_inventory/` fixtures with source repo, workflow, golden outputs, tolerances, provenance notes, and plot smoke inputs |
| 3.22 | Algorithm-level parity tests | Planned | Parser, calibration, peak search, fit, activity, detector-response, detection-limit, dose/shielding, masking, optimization, activation-inventory, and k0 parity checks against curated fixtures |
| 3.23 | Workflow-level parity tests | Planned | End-to-end parity suites for source families, tutorial/example datasets, NAA/activation inventory workflows, second-irradiation planners, and benchmark export bundles |
| 3.24 | Direct-manipulation canvas parity | Planned | Right-click peak actions, ROI/background drag handles, plot-driven edits, explicit foreground/background/secondary actions, overlap Gaussian insertion, and peak-label toggles |
| 3.25 | New parity workspaces in the Qt shell | Planned | ROI Statistics, Detection Limit, Relative Activity, Dose/Shielding, File Query/Batch Compare, Reference/Library Workbench, detector-response tools, and k0 characterization/report views |
| 3.26 | GUI polish and theme parity | Implemented (repo) | Shipped dark mode, saved theme profiles, stronger graph-table synchronization, clearer launch points, and role-aware overlay polish |
| 3.27 | GUI verification and release acceptance | Implemented (repo) | Qt workflow tests, native probes, artifact-gallery review states, and release-blocking checklists for new parity surfaces |
| 4.1 | HAL drivers | Deferred | First real MCA hardware drivers after offline-parity closure |
| 4.2 | Device discovery surfaces | Deferred | Device list, thumbnails, and discovery dialogs after HAL transport exists |
| 4.3 | Live Digital Twin dashboard | Deferred | Telemetry-driven dashboard once live acquisition is present |
| 4.4 | Live spectrogram panel | Deferred | Time-energy spectrogram after live acquisition lands |

The sequence map above remains the controlling roadmap map. The activation /
FISPACT-style, library-governance, and de-hardcoding additions below are
explicit additive overlays on that map, not replacements for it.

## 6. Feature Workstreams That Must Be Preserved

These workstreams came from the merged planning inputs and remain mandatory even when
they span multiple roadmap steps.

### 6.1 Experimental Analysis and NAA

- Support the full quantity ladder from spectrum to peak areas, net count rates,
  line activity, nuclide activity, arbitrary-time activity, production estimate,
  mass estimate, and concentration result.
- Keep direct activity, comparator NAA, relative comparator, k0-NAA, and
  activation-only workflows as explicit user-selectable modes.
- Maintain a first-class experimental inventory object containing activities,
  uncertainties, timestamps, daughter links, inferred masses/concentrations,
  sample metadata, irradiation metadata, geometry assumptions, and workflow mode.

### 6.2 Time-State, Decay, and Daughter Analysis

- Provide arbitrary-time activity solving at count start, midpoint, end, EOI, and future times.
- Support back-decay and forward-decay calculations with optional parent-daughter handling.
- Add daughter/ancestor exploration, line-emergence forecasting, and time-history visualization.
- Preserve spectrum-level activation review in both GUI and CLI, including isotope CSV export at EOI/irradiation time plus uncertainty-bearing decay and Bateman review plots.
- Keep the bundled `radioactivedecay` ICRP-107 decay network and the Kayzero `uT12` half-life-uncertainty overlays as first-class provenance-tracked sources for future source-age, daughter, and inventory workflows.
- Standardize time-evolution work around explicit `IrradiationSchedule`, `InventoryState`, and `ObservableTimeSeries` analysis objects rather than ad hoc per-workflow payloads.

### 6.3 Planning, Optimization, and Interference Analysis

- Add irradiation, cooldown, and count-time optimization against uncertainty, MDA,
  dead-time, and interference objectives.
- Add masking-isotope ranking, overlap provenance, alternate-line recommendations,
  and time-dependent de-masking guidance.
- Add staged and second-irradiation scenario planning, objective-based schedule comparison,
  and recommended schedule summaries with dose, detectability, and masking tradeoffs.
- Support FISPACT-style observable reviews for total activity, decay heat, gamma dose rate,
  beta heat, gamma heat, ingestion metric, inhalation metric, clearance index, and
  dominant-contributor timelines where source data support them.

### 6.4 InterSpec and GSA Capability Families

- Preserve all four registered efficiency-model families as first-class options.
- Keep detector slots, preserve/fine-tune calibration flows, and detector-response lifecycle management.
- Add file query, archive search, compact role-aware file staging, exemplar-driven batch reuse,
  dedicated detection-limit workspaces, relative-activity/isotopics workflows, and dose/shielding tools.
- Keep manual-first analyst control available after automation.
- Preserve the bundled GSA-v4 edited gamma library, NASA-gamma common-lab/natural/capture libraries, and the ENDF/B-VIII decay-line supplement as selectable provenance-bearing reference sources rather than collapsing back to one hardcoded line table.
- Let users choose which optional reference libraries are downloaded locally, and let them register external library locations from either the GUI or CLI without losing provenance or bundled defaults.
- Protect bundled library names and IDs from accidental user override; user-supplied libraries must be assigned distinct aliases when a collision would occur.

### 6.5 STAYSL-Grade Experimental Inverse Analysis

- Add corrected activation-rate workflows with explicit correction breakdowns.
- Provide a SigPhi-style workspace for corrected reaction observables.
- Keep generalized least-squares spectral adjustment as a governed, covariance-aware workflow.
- Plan for covariance outputs, run-family variants, reaction/covariance-library management,
  SHIELD-style self-shielding, BCF/flux-history correction handling, and monitor diagnostics.

### 6.6 Activation-Inventory and Benchmark Export Extension

- Adopt the additive `3N` workstream from the activation / FISPACT-style supplement as a preserved cross-cutting scope across `3.18` through `3.23`.
- Preserve the planned `Inventory / Time Evolution`, masking/interference, optimization, and long-term dose-study surfaces together with their machine-readable outputs.
- Add a benchmark experimental bundle export contract centered on `.ffexp` packaging for downstream comparison-tool ingestion.
- Repo status note: the first deterministic landing now exists in
  `src/fluxforge/workflows/irradiation_optimization.py`,
  `src/fluxforge/io/artifacts.py`, and `src/fluxforge/gui/panels/phase6.py`,
  with workflow-preset persistence in `src/fluxforge/gui/workflow_presets.py`
  and built-in `quantumgold-workflow` / `astm-ldrd-irradiation` presets.

### 6.7 De-Hardcoding and Configuration Cleanup

- Audit workflows and defaults for hardcoded values, special-case branches, and hidden source assumptions, then move removable cases behind registries, governed presets, or explicit user configuration.
- Keep explicit carve-outs only where deliberate hardcoding is part of required parity or governance: QuantumGold parity, PeakEasy parity, standards workflows including ASTM and k0-NAA, and RAFM irradiation-analysis paths.

## 7. Capabilities That Must Remain First-Class

- Simple, Expert, and Standards GUI modes
- Manual, Bayesian, and ML-assisted identification workflows
- Registry-backed peak-search families rather than one hardcoded algorithm
- GRAVEL, MAXED, RMLE, and ML-seed unfolding options
- Multiple background and counting methods where scientifically valid
- All registered efficiency models and calibration workflows
- Foreground, background, and secondary spectrum roles with explicit provenance
- Bundled, downloadable, and user-supplied nuclear-data libraries with explicit selection,
  provenance capture, collision-safe naming, and GUI/CLI parity

## 8. Experimental Handoff Contract

FluxForge must emit machine-readable handoff bundles that can include:

- sample, detector, and geometry metadata
- calibration and detector-response versions
- fitted peaks, line activities, nuclide activities, and EOI activities
- corrected activation rates
- inferred masses and concentrations
- uncertainty decomposition
- daughter-chain assumptions
- selected NAA and inverse-analysis modes
- nuclear-data provenance
- masking/interference outputs
- inventory timeseries, dominant-contributor tables, and chosen schedule definitions
- dose endpoints, optimization grids, recommended schedules, and staged-irradiation comparisons
- planned-vs-actual irradiation/cool/count schedules
- `.ffexp` benchmark experimental bundles and plot-manifest sidecars when the workflow requires them
- adjusted flux spectra and covariance when available
- readiness flags such as `peak_fit_complete`, `activity_complete`,
  `naa_quant_complete`, `inverse_adjustment_complete`, and `comparison_ready`

## 9. Archived Planning Inputs

The following superseded inputs were preserved in
`docs/archive/planning_snapshot_2026-04-06/`:

- `FLUXFORGE_CONSOLIDATED_MASTER.md`
- `FluxForge_Final_Additions.md`
- `FluxForge_Additions_v3_Final.md`
- `FluxForge_NAA_InterSpec_STAYSL_Addition.md`
- `GUI_PLAN.md`
- `GUI_PLAN_old.md`
- `GUI_CAPABILITY_PROGRAM.md`
- `GUI_CAPABILITY_PROGRAM_old.md`
