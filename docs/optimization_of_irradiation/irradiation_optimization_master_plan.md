# FluxForge Irradiation Optimization Master Plan

## Purpose
This document consolidates the strongest implementation-ready content from four planning sources into one execution blueprint for FluxForge irradiation optimization workflows:

1. `docs/optimization_of_irradiation/openmc_deep.md`
2. `docs/optimization_of_irradiation/deep-research-report (2).md`
3. `docs/FluxForge_NAA_Optimization_FISPACT_Addition.md`
4. `docs/FluxForge_ML_Libraries_Addition.md`

The plan is structured to be directly actionable in FluxForge code.

## Hard Method Coverage Gate
This plan explicitly extracts and preserves:

- **State-of-the-art Method 1:** DI-FOM schedule screening
- **State-of-the-art Method 2:** Poisson/FIM schedule optimization
- **State-of-the-art Method 3:** Multi-window scheduling with full-spectrum support
- **Novel Method 1:** Dose-weighted Bayesian adaptive sequential scheduling
- **Novel Method 2:** STBD-MR and differentiable interference-graph optimization

### Coverage Checklist
- [x] 3 state-of-the-art methods (Methods 1-3)
- [x] >=2 novel methods
- [x] First and second irradiation handling for all selected methods
- [x] Method-to-code mapping for FluxForge implementation

## Method Extraction Matrix

| Method | Source Priority | Why Selected | Maturity |
|---|---|---|---|
| Method 1: DI-FOM | `openmc_deep.md` + `deep-research-report (2).md` | Fast, transparent, auditable screening objective for schedule preselection | Baseline |
| Method 2: FIM/Poisson D-opt | `openmc_deep.md` + `deep-research-report (2).md` | Information-theoretic core for uncertainty-aware schedule optimization | Baseline |
| Method 3: MWDCS + full-spectrum support | `openmc_deep.md` + `deep-research-report (2).md` | Multi-window design is essential for mixed half-life inventories and masking control | Baseline |
| Novel 1: BASS-D | `openmc_deep.md` + `deep-research-report (2).md` | Sequential decision making using posterior updates and utility-based next action | Advanced |
| Novel 2: STBD-MR + differentiable interference graph | `openmc_deep.md` + `deep-research-report (2).md` + `FluxForge_ML_Libraries_Addition.md` | Joint spectro-temporal inversion and interpretable masking-aware optimization | Advanced |

## Common Physics and Inference Scaffold

### Core forward counts model
For isotope `i`, line `l`, window `k`:

$$
\mu_{ilk}(d,\theta)=A_i(t_k;\theta,d)\,I_{il}\,\epsilon(E_l)\,\int_0^{T_k}e^{-\lambda_i\tau}d\tau\,C^{corr}_{ilk}
$$

Where:
- `d` = irradiation, cooldown, count schedule variables
- `theta` = EOI activities + nuisance/calibration terms
- `C_corr` = deadtime, pile-up, summing, geometry, attenuation corrections

### Inventory coupling
Inventory propagation must support:
- Irradiation buildup
- Cooldown decay
- Count-window decay during acquisition
- Parent-daughter Bateman coupling
- Non-zero initial inventory for second irradiation

### Measurement model
- Poisson likelihood as default for peaks and channels
- Full-spectrum mode for high-overlap regions
- Joint multi-time fitting for short/medium/long cooldown counts

## State-of-the-Art Methods (Required)

## Method 1: DI-FOM (Dose-Importance Figure of Merit)

### Objective
Use DI-FOM as a screening stage before expensive optimization.

$$
J_{DI}(d)=\sum_i w_i(\tau)\sum_{l\in L_i}\frac{S_{il}(d)^2}{S_{il}(d)+B_{il}(d)+\sum_{j\neq i}I_{j\to l}(d)}
$$

### Use Cases
- Rapid ranking of candidate schedules
- Line inclusion/exclusion prefiltering
- Early-phase first-irradiation planning with uncertain priors

### Required Outputs
- Dose-ranked isotopes by time endpoint
- Predicted line signal/background/interference table
- Top schedule candidates with uncertainty-aware score

### Second Irradiation Behavior
Replace nominal prior inventory with measured posterior from campaign 1.

## Method 2: Poisson/FIM Optimization

### Objective
Use FIM to optimize identifiability and uncertainty.

$$
F(d)=R^T\,diag(1/\mu(d))\,R
$$

Optimization modes:
- D-opt: maximize `log det(F)`
- A-opt: minimize `tr(F^-1)`
- C-opt: minimize `c^T F^-1 c` for decision-oriented projections

### Required Nuisance Handling
Include at least:
- Efficiency normalization and curve uncertainty
- Energy calibration / width model uncertainty
- Background scaling terms
- Deadtime correction uncertainty

### Required Outputs
- CRLB/posterior variance approximations by nuclide
- Correlation and condition diagnostics
- Pareto surfaces for precision vs time vs dose constraints

### Second Irradiation Behavior
Warm-start FIM objective with posterior covariance from campaign 1.

## Method 3: Multi-Window Scheduling (MWDCS) with Full-Spectrum Support

### Objective
Design multiple cooldown/count windows that maximize aggregate information.

$$
\tilde{F}(d)=\sum_{m=1}^{M}F^{(m)}(t_{cool,m},t_{count,m})
$$

### Scheduling Principle
Use window families spanning:
- Very early
- Hour scale
- Day scale
- Week/month scale

### Full-Spectrum Extension
When peaks are heavily overlapped, switch to channel-wise objective:
- Poisson NLL with regularization
- Optional weighted least squares approximation

### Required Outputs
- Multi-window recommendations with marginal gain by window
- Identifiability diagnostics for overlapping isotopes
- Residual maps showing unresolved spectrum structure

### Second Irradiation Behavior
Treat residual inventory as initial state and rerun multi-window design.

## Novel Methods (Required)

## Novel Method 1: BASS-D (Dose-Weighted Bayesian Adaptive Sequential Scheduling)

### Utility
Select next action after each measurement by maximizing expected utility.

$$
U(a_k)=I(\theta_{ROI};y_k\mid a_k,\mathcal{D}_{<k})-\lambda_t\,Cost(a_k)-\lambda_d\,DoseRisk(a_k)-\lambda_m\,MaskPenalty(a_k)
$$

### Why Keep
- Explicitly adaptive
- Natural first-to-second irradiation transfer
- Supports "wait vs count now" decisions with traceable rationale

### Required Outputs
- Action sequence with utility deltas
- Posterior variance reduction trajectory
- Rationale log for each adaptive step

## Novel Method 2: STBD-MR and Differentiable Interference-Graph Optimization

### STBD-MR Core
Jointly infer activities from all windows and channels using spectro-temporal structure plus masking-aware regularization.

### Interference Graph Core
Define masking centrality and optimize schedule to reduce interference-driven uncertainty for high-priority nuclides.

### Why Keep
- Handles severe overlap and cascade complexity
- Produces interpretable masking communities
- Supports gradient-based optimization for advanced schedules

### Required Outputs
- Spectral coherence and temporal separability diagnostics
- Masking graph and critical pair list
- Schedule sensitivity against masking centrality penalty

## Unified Ranking and Masking Logic

## Isotope Priority Score

$$
W_i=\alpha_D D_i+\alpha_A A_i+\alpha_Q Q_i+\alpha_N N_i+\alpha_F F_i
$$

Components:
- `D_i`: dose relevance over selected time endpoints
- `A_i`: activation significance/pathway importance
- `Q_i`: assay/inference leverage
- `N_i`: NAA utility
- `F_i`: feasibility after masking/background constraints

## Predictive Masking Metrics
- Energy-resolution overlap score
- Interference-to-signal ratio (ISR)
- Sum-peak/coincidence risk
- Time-dependent masking centrality

## FluxForge Implementation Mapping

## Existing Reusable Foundations
- `src/fluxforge/core/inventory_timeline.py`: inventory state/time evolution
- `src/fluxforge/physics/activation.py`: line activity and irradiation buildup primitives
- `src/fluxforge/physics/dose.py`: dose-rate calculations
- `src/fluxforge/physics/decay_chain.py`: Bateman chain solver
- `src/fluxforge/cli/app.py`: CLI extension point
- `src/fluxforge/plots/activation.py`: existing decay plotting
- `src/fluxforge/io/artifacts.py`: bundle export framework
- `src/fluxforge/core/schemas.py`: schema registration/validation

## New/Extended Modules Required
- `src/fluxforge/analysis/optimization.py`
  - DI-FOM scorer
  - FIM scorer
  - MWDCS planner
  - Objective aggregation and constraints
- `src/fluxforge/analysis/masking.py`
  - Line interference ranking
  - Overlap/ISR/summing risk metrics
- `src/fluxforge/workflows/second_irradiation_optimizer.py`
  - Posterior-to-prior campaign transfer
  - Scenario comparison for pulse plans
- `src/fluxforge/plots/optimization.py`
  - Heatmaps, Pareto fronts, schedule cards
- `src/fluxforge/plots/masking.py`
  - ROI overlays and masking score trends

## Data Model Upgrades
Extend schedule representation in `src/fluxforge/core/inventory_timeline.py` to support:
- Multi-segment irradiation
- Multi-segment cooldown
- Multiple count windows
- Campaign metadata for first/second irradiation handoff

## CLI Additions
Add command families in `src/fluxforge/cli/app.py`:
- `optimization-sweep`
- `masking-analysis`
- `second-irradiation-plan`

## Export Contract
Add optimization and campaign artifacts using existing artifact infrastructure:
- `optimization_grid.csv`
- `recommended_schedules.csv`
- `masking_candidates.csv`
- `dose_endpoints.csv`
- benchmark experimental bundle (`.ffexp`-compatible schema and payload)

## Method-by-Method Delivery Protocol

Each method must be delivered in this strict order before starting the next method:

1. Implement core method math and API in analysis/core modules.
2. Add and pass method-level tests.
3. Integrate method into GUI surfaces.
4. Update documentation and examples for that method.
5. Run method gate checklist and sign off.

No new method starts until the current method gate is complete.

## Method 1 Delivery Track: DI-FOM

Status update (2026-04-06): Baseline DI-FOM implementation is now present in core analysis, CLI, GUI preview, and docs on branch `optimization-workflows`.

### M1-A Core Implementation
- Add DI-FOM scorer and schedule-grid evaluator in `src/fluxforge/analysis/optimization_difom.py`.
- Add data contracts needed by DI-FOM inputs in `src/fluxforge/core/inventory_timeline.py`.
- Expose CLI entry point for DI-FOM mode in `src/fluxforge/cli/app.py` (`optimization-sweep --objective di-fom`).

### M1-B Testing
- Add unit tests for DI-FOM objective correctness and ranking stability.
- Add CLI parser and command tests for DI-FOM mode.
- Add regression fixture for expected DI-FOM ordering on a known synthetic case.

### M1-C GUI Integration
- Add DI-FOM preview control in the inventory timeline panel (`src/fluxforge/gui/panels/modern_shell.py`).
- Render DI-FOM preview score and line-count summary card from activity-review inputs.

### M1-D Documentation
- Add DI-FOM workflow subsection to optimization docs (`docs/optimization_of_irradiation/method1_difom_workflow.md`).
- Add one end-to-end example (inputs, command, outputs, interpretation).

### M1 Gate (Must Pass)
- DI-FOM tests pass.
- DI-FOM GUI path is functional.
- DI-FOM docs and example are updated.

### M1 Verification Evidence (Completed 2026-04-06)
- Core implemented in `src/fluxforge/analysis/optimization_difom.py`.
- CLI command implemented in `src/fluxforge/cli/app.py` (`optimization-sweep`).
- GUI integration implemented in `src/fluxforge/gui/panels/modern_shell.py` (`InventoryTimelinePanel` DI-FOM preview).
- Tests run and passing:
  - `pytest -q tests/test_optimization_difom.py tests/test_cli_app.py -k "optimization_sweep or difom"`
  - `pytest -q tests/test_analysis_workspace_qt.py -k difom_preview`

## Method 2 Delivery Track: FIM / Poisson Information Optimization

Status update (2026-04-06): Baseline FIM implementation is now present in core analysis, CLI objective routing, GUI preview, and docs on branch `optimization-workflows`.

### M2-A Core Implementation
- M2-A1: Create `src/fluxforge/analysis/optimization_fim.py` with Fisher matrix assembly from line-level sensitivities and covariance models.
- M2-A2: Implement objective evaluators for `fim-d`, `fim-a`, and `fim-c` with stable fallback for singular/ill-conditioned matrices.
- M2-A3: Add nuisance-parameter support for efficiency, background, dead-time, and branch-ratio uncertainty.
- M2-A4: Add candidate parser/serializer compatible with M1 payload shape plus FIM-specific fields.
- M2-A5: Extend `optimization-sweep` objective routing in `src/fluxforge/cli/app.py` to include FIM objectives.
- M2-A6: Add artifact fields for matrix diagnostics (condition number, determinant/log-det, trace-inverse, dominant eigenvalues).

### M2-B Testing
- M2-B1: Unit tests for matrix construction using synthetic sensitivity fixtures.
- M2-B2: Unit tests for D/A/C objective monotonic behavior.
- M2-B3: Numerical-stability tests with nearly collinear sensitivity vectors.
- M2-B4: CLI tests for `--objective fim-d|fim-a|fim-c` parsing and output schema.
- M2-B5: Integration test that runs M1 and M2 on a shared fixture and verifies deterministic ranking outputs.
- M2-B6: Regression fixture with expected FIM diagnostics and ranking order.

### M2-C GUI Integration
- M2-C1: Add objective selector entries for FIM variants in the optimization controls.
- M2-C2: Add FIM diagnostics panel (condition number, determinant/log-det, effective rank).
- M2-C3: Add per-candidate matrix-inspection table and line-contribution summary.
- M2-C4: Add comparison toggle to overlay DI-FOM vs FIM top schedules in one view.

### M2-D Documentation
- M2-D1: Add FIM workflow document with input schema and examples for D/A/C objectives.
- M2-D2: Add interpretation guide for choosing D-opt vs A-opt vs C-opt.
- M2-D3: Add troubleshooting guide for ill-conditioned matrices and nuisance modeling.
- M2-D4: Add worked RAFM example showing DI-FOM vs FIM ranking differences.

Implemented doc artifact: `docs/optimization_of_irradiation/method2_fim_workflow.md`.

### M2 Gate (Must Pass)
- FIM tests pass.
- FIM GUI path is functional.
- FIM docs and example are updated.
- M1 vs M2 comparison report is generated for the shared benchmark fixture.

### M2 Verification Evidence (Completed 2026-04-06)
- Core implemented in `src/fluxforge/analysis/optimization_fim.py`.
- CLI objective routing implemented in `src/fluxforge/cli/app.py` (`fim-d`, `fim-a`, `fim-c`).
- GUI integration implemented in `src/fluxforge/gui/panels/modern_shell.py` (InventoryTimelinePanel FIM preview and diagnostics).
- Tests run and passing:
  - `pytest -q tests/test_optimization_fim.py tests/test_optimization_difom.py tests/test_cli_app.py -k "optimization_sweep or fim or difom"`
  - `pytest -q tests/test_analysis_workspace_qt.py -k "difom_preview or fim_preview"`
- RAFM M1-vs-M2 gate artifacts generated:
  - `examples/RAFM_irradiation/results/method_benchmark/m1_m2_schedule_comparison.csv`
  - `examples/RAFM_irradiation/results/method_benchmark/m1_m2_schedule_comparison.md`

## Method 3 Delivery Track: MWDCS with Full-Spectrum Support

Status update (2026-04-06): Baseline MWDCS implementation is now present in core analysis, CLI objective routing, GUI preview, tests, and docs on branch `optimization-workflows`.

### M3-A Core Implementation
- M3-A1: Create `src/fluxforge/analysis/optimization_mwdcs.py` for multi-window scheduler logic.
- M3-A2: Implement additive information objective across multiple cooldown/count windows.
- M3-A3: Add full-spectrum mode adapter for overlap-heavy scenarios using aggregated window evidence.
- M3-A4: Add schedule-candidate model supporting multiple count windows and window-level constraints.
- M3-A5: Extend CLI payload contract for window arrays and serialization.

### M3-B Testing
- M3-B1: Unit tests for additive window scoring and diminishing returns behavior.
- M3-B2: Fixture tests for early/mid/late window sensitivity changes on mixed half-life nuclides.
- M3-B3: Full-spectrum mode smoke tests against overlap-heavy synthetic spectra.
- M3-B4: CLI tests for multi-window payload parsing and ranking outputs.
- M3-B5: Integration test comparing single-window vs multi-window schedules on the same inventory state.

### M3-C GUI Integration
- M3-C1: Add multi-window editor UI with add/remove/reorder controls.
- M3-C2: Add timeline visualization showing irradiation, cooldown, and count windows.
- M3-C3: Add marginal information-by-window chart with uncertainty bars.
- M3-C4: Add side-by-side view for single-window baseline vs MWDCS recommendation.

### M3-D Documentation
- M3-D1: Add MWDCS workflow document with multi-window input schema.
- M3-D2: Add decision guidance for when to enable full-spectrum mode.
- M3-D3: Add RAFM case example interpreting window contributions by isotope family.
- M3-D4: Add performance notes and practical window-count recommendations.

### M3 Gate (Must Pass)
- MWDCS tests pass.
- MWDCS GUI path is functional.
- MWDCS docs and example are updated.
- M1 vs M2 vs M3 comparison report is generated on shared benchmark fixtures.

### M3 Verification Evidence (Completed 2026-04-06)
- Core implemented in `src/fluxforge/analysis/optimization_mwdcs.py`.
- CLI objective routing implemented in `src/fluxforge/cli/app.py` (`mwdcs` + window/full-spectrum options).
- GUI integration implemented in `src/fluxforge/gui/panels/modern_shell.py` (InventoryTimelinePanel MWDCS preview).
- Tests run and passing:
  - `pytest -q tests/test_optimization_mwdcs.py tests/test_cli_app.py -k "optimization_sweep or mwdcs"`
  - `pytest -q tests/test_analysis_workspace_qt.py -k "mwdcs_preview or fim_preview or difom_preview"`
- Cross-method benchmark artifacts generated (including legacy optimizer baseline):
  - `examples/RAFM_irradiation/results/method_benchmark/m1_m2_m3_legacy_schedule_comparison.csv`
  - `examples/RAFM_irradiation/results/method_benchmark/m1_m2_m3_legacy_schedule_comparison.md`

## Method N1 Delivery Track: BASS-D (Novel)

Status update (2026-04-06): Baseline BASS-D implementation is now present in core analysis, CLI advanced-objective routing, GUI preview, tests, and docs on branch `optimization-workflows`.

### N1-A Core Implementation
- N1-A1: Create `src/fluxforge/analysis/optimization_bassd.py` adaptive scheduler core.
- N1-A2: Implement expected-utility function with dose-weighted value-of-information terms.
- N1-A3: Add posterior update engine for line-activity uncertainty after each adaptive step.
- N1-A4: Add action log model and reproducibility seed handling.
- N1-A5: Add advanced-mode guard flags in CLI and GUI.

### N1-B Testing
- N1-B1: Deterministic posterior-update tests for fixed synthetic observations.
- N1-B2: Action-selection tests on adaptive sequences with known best next action.
- N1-B3: Seed-reproducibility tests for stochastic branches.
- N1-B4: Safety tests for advanced-mode enable/disable paths.
- N1-B5: Integration tests against M1/M2 static schedules as baselines.

### N1-C GUI Integration
- N1-C1: Add adaptive-step planner view with current posterior summary.
- N1-C2: Add rationale log panel showing why each action was selected.
- N1-C3: Add uncertainty-trend chart across adaptive iterations.
- N1-C4: Add export action for adaptive campaign trace.

### N1-D Documentation
- N1-D1: Add BASS-D workflow tutorial with advanced-mode activation steps.
- N1-D2: Add interpretation guide for posterior trends and action logs.
- N1-D3: Add caution notes for sparse-data and high-interference regimes.
- N1-D4: Add comparison example vs M3 static multi-window scheduling.

### N1 Gate (Must Pass)
- BASS-D tests pass.
- BASS-D GUI path is functional.
- BASS-D docs and example are updated.
- Comparative report includes M1, M2, M3, and N1 on shared fixtures.

### N1 Verification Evidence (Baseline 2026-04-06)
- Core implemented in `src/fluxforge/analysis/optimization_bassd.py`.
- CLI objective routing and advanced guard implemented in `src/fluxforge/cli/app.py` (`bass-d` + `--enable-advanced-objectives`).
- GUI integration implemented in `src/fluxforge/gui/panels/modern_shell.py` (InventoryTimelinePanel BASS-D preview and advanced guard checkbox).
- Baseline documentation added in `docs/optimization_of_irradiation/method_n1_bassd_workflow.md`.
- Tests run and passing:
  - `pytest -q tests/test_optimization_bassd.py tests/test_cli_app.py -k "optimization_sweep or bassd"`
  - `pytest -q tests/test_analysis_workspace_qt.py -k "bassd_preview or mwdcs_preview or fim_preview or difom_preview"`

## Method N2 Delivery Track: STBD-MR and Differentiable Interference Graph (Novel)

### N2-A Core Implementation
- N2-A1: Create `src/fluxforge/analysis/optimization_stbdmr.py` for spectro-temporal Bayesian inference.
- N2-A2: Implement masking-regularized objective terms and interference adjacency metrics.
- N2-A3: Add differentiable interference-graph objective mode for advanced optimization.
- N2-A4: Add graph-construction pipeline from line overlap and continuum burden metrics.
- N2-A5: Add advanced-mode objective routing and artifact serialization hooks.

### N2-B Testing
- N2-B1: Spectro-temporal consistency tests across multi-window synthetic datasets.
- N2-B2: Masking-graph construction and metric correctness tests.
- N2-B3: Differentiable-objective gradient sanity tests.
- N2-B4: Robustness tests for sparse graph and dense graph edge cases.
- N2-B5: Integration tests comparing N2 recommendations against N1 and M3 baselines.

### N2-C GUI Integration
- N2-C1: Add masking graph explorer with node/edge importance controls.
- N2-C2: Add sensitivity panel for regularization and graph-threshold sweeps.
- N2-C3: Add comparative ranking table versus non-graph objectives.
- N2-C4: Add advanced diagnostics export for publication-quality analysis.

### N2-D Documentation
- N2-D1: Add STBD-MR and interference-graph workflow document.
- N2-D2: Add guidance for regularization tuning and graph-threshold selection.
- N2-D3: Add caveats for identifiability and optimization instability risks.
- N2-D4: Add comparative case study against M1-M3-N1 outputs.

### N2 Gate (Must Pass)
- STBD-MR/interference-graph tests pass.
- Advanced GUI path is functional.
- Advanced docs and example are updated.
- Full cross-method comparison report (M1, M2, M3, N1, N2) is generated.

### N2 Verification Evidence (Baseline 2026-04-06)
- Core implemented in `src/fluxforge/analysis/optimization_stbdmr.py`.
- Analysis exports wired in `src/fluxforge/analysis/__init__.py`.
- CLI objective routing and advanced guard implemented in `src/fluxforge/cli/app.py` (`stbd-mr` + `--enable-advanced-objectives`).
- GUI integration implemented in `src/fluxforge/gui/panels/modern_shell.py` (InventoryTimelinePanel STBD-MR preview and differentiable-graph toggle).
- Baseline documentation added in `docs/optimization_of_irradiation/method_n2_stbdmr_workflow.md`.
- Cross-method benchmark script updated to include N2 in `examples/RAFM_irradiation/compare_m1_m2_m3_n1_legacy_schedule_objectives.py`.
- Generated benchmark artifacts:
  - `examples/RAFM_irradiation/results/method_benchmark/m1_m2_m3_n1_n2_legacy_schedule_comparison.csv`
  - `examples/RAFM_irradiation/results/method_benchmark/m1_m2_m3_n1_n2_legacy_schedule_comparison.md`
- Tests run and passing:
  - `pytest -q tests/test_optimization_stbdmr.py tests/test_cli_app.py tests/test_analysis_workspace_qt.py -k "stbdmr or optimization_sweep"`

## Cross-Method Rule

Method progression is strictly sequential:
- M1 complete before M2 starts.
- M2 complete before M3 starts.
- M3 complete before N1 starts.
- N1 complete before N2 starts.

## Cross-Method Comparison Protocol (Starts When M2 Exists)

### CMP-A Shared Benchmark Fixtures
- Maintain one synthetic benchmark fixture set used by all methods.
- Maintain one RAFM benchmark fixture set derived from `examples/RAFM_irradiation/results/analysis_json/`.

### CMP-B Required Metrics
- Rank correlation (Spearman) between methods.
- Top-k schedule overlap statistics.
- Objective-score spread and stability under uncertainty perturbation.
- Dose endpoint and detectability tradeoff deltas.

### CMP-C Required Deliverables Per New Method
- `method_comparison_<method>.csv` summary table.
- `method_comparison_<method>.md` interpretation memo.
- Overlay plot pack comparing all currently available methods.

### CMP-D User-Selected Isotope Focus (Baseline 2026-04-06)
- Cross-objective isotope filtering is available through `optimization-sweep --isotopes-of-interest`.
- Filter behavior is applied before objective scoring and reports before/after candidate and line-term counts.
- Baseline workflow note: `docs/optimization_of_irradiation/isotopes_of_interest_filter_workflow.md`.
- Verification coverage in `tests/test_cli_app.py` includes parser and scoring-path filtering assertions.

## RAFM Campaign Analysis Track for Second-Irradiation Conditions

Baseline campaign timing audit: `docs/optimization_of_irradiation/rafm_second_irradiation_baseline.md`.

### RAFM-A Data Inventory and Normalization
- RAFM-A1: Build campaign inventory table from all files in `examples/RAFM_irradiation/results/analysis_json/`.
- RAFM-A2: Normalize timing fields (`irradiation_phase`, `irradiation_time_s`, `decay_time_s`, `decay_label`) across RAFM1/RAFM3/RAFM4 and flux wires.
- RAFM-A3: Resolve RAFM1 records with missing phase labels using `examples/RAFM_irradiation/metadata/sample_schedules.json`.

### RAFM-B Feature Engineering for Optimization Inputs
- RAFM-B1: Build per-sample line-activity feature tables at each cooling window.
- RAFM-B2: Build isotope-family summaries (short, intermediate, long half-life groups).
- RAFM-B3: Build uncertainty-weighted signal/background proxies compatible with M1 and M2 objectives.

### RAFM-C Candidate Second-Irradiation Condition Grid
- RAFM-C1: Seed grid around observed phase-2 baseline (`2 h` irradiation, `~16 d` cooldown).
- RAFM-C2: Add practical variants around irradiation duration and cooldown duration.
- RAFM-C3: Enforce dose and dead-time constraints from existing RAFM workflow thresholds.

### RAFM-D Method Evaluation on All RAFM Irradiations
- RAFM-D1: Run M1 across all RAFM candidates and capture ranked schedules.
- RAFM-D2: Once M2 exists, run M2 on the same RAFM candidates and generate M1 vs M2 comparison outputs.
- RAFM-D3: Once M3 exists, extend to M1 vs M2 vs M3 comparison and stability checks.

### RAFM-E Recommendation for Second Irradiation
- RAFM-E1: Produce per-sample recommended second-irradiation conditions.
- RAFM-E2: Produce one global recommended condition set for campaign-wide use.
- RAFM-E3: Document why recommended conditions improve detectability vs dose/time constraints.
- RAFM-E4: Validate recommendation against RAFM4 `15dEOI` outcomes and flux-wire consistency checks.

### RAFM-F Full LDRD Gamma-Spec Validation Gate (Required)
- RAFM-F1: Run full campaign validation workflow over `examples/RAFM_irradiation/raw_gamma_spec/`:
  - `PYTHONPATH=src python examples/RAFM_irradiation/run_validation.py --no-fail`
- RAFM-F2: Confirm summary coverage in `examples/RAFM_irradiation/results/validation_summary.json`:
  - all RAFM raw spectra analyzed
  - matched-pair counts reported
  - unmatched lists explicitly reported
- RAFM-F3: Publish one method-comparison note per available optimization method against RAFM timing windows.
- RAFM-F4: Treat this as a mandatory gate before advancing from M2 to M3 and again before finalizing M3.

## Test and Validation Plan

## Unit Tests
- Method-specific objective correctness
- Constraint enforcement behavior
- Schedule scoring reproducibility

## Integration Tests
- CLI command parsing and output generation
- Cross-module pipeline from activity review to optimization outputs
- First/second irradiation state transfer

## Artifact Tests
- Schema validation in `src/fluxforge/core/schemas.py`
- Bundle read/write roundtrip in `src/fluxforge/io/artifacts.py`

## Initial Backlog (Implementation Order)
1. `analysis/optimization.py` + tests
2. CLI `optimization-sweep` + parser tests
3. `analysis/masking.py` + tests
4. CLI `masking-analysis` + parser tests
5. Schedule model extension + migration support
6. `workflows/second_irradiation_optimizer.py`
7. Plot modules and exports
8. Novel methods and advanced modes

## Definition of Done for This Plan
This plan is considered implemented when FluxForge can:
- Optimize irradiation/cooldown/count schedules with Methods 1-3
- Report masking-aware tradeoffs and recommended schedules
- Compare first and second irradiation scenarios
- Export machine-readable optimization artifacts
- Run >=2 advanced methods in a gated advanced mode
