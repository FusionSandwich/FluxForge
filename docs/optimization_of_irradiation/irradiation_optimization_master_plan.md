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

### M1-A Core Implementation
- Add DI-FOM scorer and schedule-grid evaluator in `src/fluxforge/analysis/optimization.py`.
- Add data contracts needed by DI-FOM inputs in `src/fluxforge/core/inventory_timeline.py`.
- Expose CLI entry point for DI-FOM mode in `src/fluxforge/cli/app.py` (`optimization-sweep --objective di-fom`).

### M1-B Testing
- Add unit tests for DI-FOM objective correctness and ranking stability.
- Add CLI parser and command tests for DI-FOM mode.
- Add regression fixture for expected DI-FOM ordering on a known synthetic case.

### M1-C GUI Integration
- Add DI-FOM option to optimization controls in the modern GUI optimization panel.
- Render DI-FOM heatmap and schedule summary card.

### M1-D Documentation
- Add DI-FOM workflow subsection to optimization docs.
- Add one end-to-end example (inputs, command, outputs, interpretation).

### M1 Gate (Must Pass)
- DI-FOM tests pass.
- DI-FOM GUI path is functional.
- DI-FOM docs and example are updated.

## Method 2 Delivery Track: FIM / Poisson Information Optimization

### M2-A Core Implementation
- Add FIM builder and D/A/C-opt scoring in `src/fluxforge/analysis/optimization.py`.
- Add nuisance-parameter hooks for efficiency/background/deadtime uncertainty.
- Extend CLI objective options to include FIM variants.

### M2-B Testing
- Add unit tests for matrix assembly and objective behavior.
- Add numerical-stability tests for near-singular cases.
- Add integration test comparing DI-FOM vs FIM outputs on a shared fixture.

### M2-C GUI Integration
- Add FIM objective selection and matrix diagnostics view.
- Add Pareto view support for FIM-based runs.

### M2-D Documentation
- Add FIM method section with objective choices and interpretation guidance.
- Add troubleshooting notes for conditioning and nuisance terms.

### M2 Gate (Must Pass)
- FIM tests pass.
- FIM GUI path is functional.
- FIM docs and example are updated.

## Method 3 Delivery Track: MWDCS with Full-Spectrum Support

### M3-A Core Implementation
- Add multi-window schedule planner in `src/fluxforge/analysis/optimization.py`.
- Add full-spectrum objective mode hook for overlap-heavy conditions.
- Add schedule serialization support for multiple cooldown/count windows.

### M3-B Testing
- Add unit tests for additive multi-window information behavior.
- Add fixture-based tests for early/mid/late window sensitivity.
- Add full-spectrum mode smoke tests.

### M3-C GUI Integration
- Add multi-window schedule editor and timeline visualization.
- Add marginal information-by-window chart.

### M3-D Documentation
- Add MWDCS method section and full-spectrum mode decision guidance.
- Add multi-window example with interpretation of window contributions.

### M3 Gate (Must Pass)
- MWDCS tests pass.
- MWDCS GUI path is functional.
- MWDCS docs and example are updated.

## Method N1 Delivery Track: BASS-D (Novel)

### N1-A Core Implementation
- Add adaptive scheduler state machine and expected utility function.
- Add posterior update hooks and action logging.

### N1-B Testing
- Add deterministic posterior-update tests.
- Add action-selection tests on synthetic adaptive sequences.

### N1-C GUI Integration
- Add adaptive-step planner view with rationale log.

### N1-D Documentation
- Add adaptive workflow tutorial and interpretation guide.

### N1 Gate (Must Pass)
- BASS-D tests pass.
- BASS-D GUI path is functional.
- BASS-D docs and example are updated.

## Method N2 Delivery Track: STBD-MR and Differentiable Interference Graph (Novel)

### N2-A Core Implementation
- Add STBD-MR inference mode and masking-regularization structures.
- Add interference-graph metric pipeline and differentiable objective mode.

### N2-B Testing
- Add spectro-temporal consistency tests.
- Add masking-graph metric correctness tests.

### N2-C GUI Integration
- Add masking graph explorer and sensitivity panels.

### N2-D Documentation
- Add advanced-method documentation and caveats.

### N2 Gate (Must Pass)
- STBD-MR/interference-graph tests pass.
- Advanced GUI path is functional.
- Advanced docs and example are updated.

## Cross-Method Rule

Method progression is strictly sequential:
- M1 complete before M2 starts.
- M2 complete before M3 starts.
- M3 complete before N1 starts.
- N1 complete before N2 starts.

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
