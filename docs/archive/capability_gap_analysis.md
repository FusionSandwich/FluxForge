# FluxForge Capability Gap Analysis

This document maps the capabilities defined in the FluxForge capability specification
to the current implementation status, identifies gaps, and provides implementation
priorities.

## Executive Summary

| Category | Implemented | Partial | Missing |
|----------|-------------|---------|---------|
| A: Spectrum Ingestion | 2 | 1 | 3 |
| B: Peak Detection | 4 | 1 | 0 |
| C: Activity Calculation | 1 | 0 | 3 |
| D: Reaction Rates | 3 | 0 | 0 |
| E: Response Matrix | 3 | 0 | 2 |
| F: Unfolding Solvers | 7 | 0 | 0 |
| G: Plotting | 5 | 0 | 0 |
| H: Model Comparison | 0 | 0 | 4 |
| I: TRIGA/k0-NAA | 5 | 0 | 0 |
| J: Artifacts | 3 | 0 | 0 |
| K: Reactor Dosimetry (INL) | 7 | 0 | 1 |
| **Total** | **40** | **2** | **13** |

**Implementation Coverage: 75%** (up from 62%)

---

## INL Reactor Dosimetry Workflow (INL/EXT-21-64191)

Reference: T. Holschuh et al., "Impact of Flux Wire Selection on Neutron 
Spectrum Adjustment", INL/EXT-21-64191, August 2021.

### K: Reactor Dosimetry 7-Step Workflow

| Step | Capability | Status | Notes |
|------|------------|--------|-------|
| K1 | Flux Wire Selection |  | `fluxforge.analysis.flux_wire_selection` - IRDFF reactions, threshold energies, INL combos |
| K2 | Irradiation Modeling |  | `fluxforge.physics.activation.IrradiationSegment` |
| K3 | Flux Wire Measurement |  | `fluxforge.io.flux_wire`, HPGe processing, activity determination |
| K4 | A Priori Spectrum |  | `SpectrumUnfolder.set_mcnp_initial_guess()`, 640/725 group structures |
| K5 | Spectrum Unfolding |  | GLS, GRAVEL, MLEM, MCMC solvers |
| K6 | Adjusted Spectrum |  | `UnfoldingResult`, `compare_with_mcnp()` |
| K7 | Fluences of Interest |  | `calculate_1mev_equivalent_fluence()`, `calculate_dpa()` |
| K8 | A Priori Covariance |  | STAYSL-style spectrum covariance not implemented |

**New modules added:**
- [flux_wire_selection.py](src/fluxforge/analysis/flux_wire_selection.py): Wire advisor, IRDFF database, 1-MeV eq, DPA
- [reactor_dosimetry_workflow.py](examples/reactor_dosimetry_workflow.py): Complete 7-step example

---

## Detailed Gap Analysis

### Stage A: HPGe Spectrum Ingestion

| ID | Capability | Status | Notes | Reference Implementation |
|----|------------|--------|-------|--------------------------|
| A1.1 | SPE format reader |  | `fluxforge.io.spe.read_spe_file()` | - |
| A1.2 | CHN format reader |  | Partial in `fluxforge.io.hpge` | `testing/irrad_spectroscopy` |
| A1.3 | CNF (Canberra) format |  | Not implemented | `testing/hdtv` has CNF support |
| A1.4 | N42/IEC XML format |  | Not implemented | - |
| A1.5 | Background estimation (SNIP) |  | `fluxforge.analysis.peakfit.estimate_background()` | - |
| A1.6 | Dead-time validation |  | `GammaSpectrum.dead_time_fraction` | - |

**Priority Gaps:**
1. **CNF format** - Needed for Canberra detector support (common in labs)
2. **N42/IEC format** - Needed for RIID and standards compliance

---

### Stage B: Peak Detection & Fitting

| ID | Capability | Status | Notes | Reference Implementation |
|----|------------|--------|-------|--------------------------|
| B1.1 | Automated peak finding |  | Multiple methods: `scipy`, `window`, `chunked` | `testing/peakingduck` |
| B1.2 | Gaussian peak fitting |  | `fluxforge.analysis.peakfit.GaussianPeak` | - |
| B1.3 | Hypermet peak shapes |  | `HypermetPeak`, `fit_hypermet_peak()` | `testing/hdtv` |
| B1.4 | Multiplet handling |  | Basic awareness, needs constrained fitting | `testing/hdtv` |
| B1.5 | Peak fit covariance |  | `PeakFitResult.covariance` | - |

**Priority Gaps:**
1. ~~**Hypermet model**~~  DONE - Implemented with left tail, step function

---

### Stage C: Efficiency & Activity Calculation

| ID | Capability | Status | Notes | Reference Implementation |
|----|------------|--------|-------|--------------------------|
| C1.1 | Efficiency curve fitting |  | `fluxforge.data.efficiency` | - |
| C1.2 | Activity from peak area |  | `GammaLineMeasurement` exists but needs integration | - |
| C1.3 | Weighted activity (multi-line) |  | Function exists but check needed | - |
| C1.4 | Coincidence summing corrections |  | Not implemented | `testing/actigamma` |

**Priority Gaps:**
1. **Coincidence summing** - Required for close-geometry measurements

---

### Stage D: Reaction Rates

| ID | Capability | Status | Notes | Reference Implementation |
|----|------------|--------|-------|--------------------------|
| D1.1 | Multi-segment irradiation history |  | `IrradiationSegment`, `irradiation_buildup_factor()` | - |
| D1.2 | Decay corrections |  | `irradiation_buildup_factor()` handles decay | - |
| D1.3 | EOI reaction rate |  | `reaction_rate_from_activity()` | - |

**Status: Fully implemented!**

---

### Stage E: Response Matrix

| ID | Capability | Status | Notes | Reference Implementation |
|----|------------|--------|-------|--------------------------|
| E1.1 | Energy group structure |  | `get_flux_wire_energy_groups()`, 640/725 groups | - |
| E1.2 | Response matrix construction |  | `build_response_matrix()` | - |
| E1.3 | IRDFF cross sections |  | `fluxforge.data.irdff.IRDFFDatabase` | - |
| E1.4 | Self-shielding corrections |  | Not implemented | STAYSL documentation |
| E1.5 | Cadmium cover corrections |  | Not implemented | - |

**Priority Gaps:**
1. **Self-shielding** - Important for thick foils
2. **Cd cover corrections** - Required for thermal/epithermal separation

---

### Stage F: Unfolding Solvers

| ID | Capability | Status | Notes | Reference Implementation |
|----|------------|--------|-------|--------------------------|
| F1.1 | GLS adjustment |  | `fluxforge.solvers.gls.gls_adjust()` | - |
| F1.2 | GRAVEL iterative |  | `fluxforge.solvers.iterative.gravel()` | `testing/Neutron-Unfolding` |
| F1.3 | MLEM iterative |  | `fluxforge.solvers.iterative.mlem()` | `testing/Neutron-Unfolding` |
| F1.4 | Gradient descent |  | `fluxforge.solvers.iterative.gradient_descent()` | `testing/SpecKit` |
| F1.5 | Bayesian MCMC |  | `fluxforge.solvers.mcmc.mcmc_unfold()` | - |
| F1.6 | Positivity constraints |  | Enforced in all solvers | - |
| F1.7 | Chi-square diagnostics |  | Available in solution objects | - |

**Status: Fully implemented!**

---

### Stage G: Plotting & Reporting

| ID | Capability | Status | Notes |
|----|------------|--------|-------|
| G1.1 | Spectrum with uncertainty bands |  | `examples/generate_plots.py` |
| G1.2 | Prior vs posterior overlay |  | `examples/generate_plots.py` |
| G1.3 | Residual/pull plots |  | `examples/generate_plots.py` |
| G1.4 | Covariance/correlation heatmaps |  | `examples/generate_plots.py` |
| G1.5 | Parity plot |  | `examples/generate_plots.py` |

**Status: Fully implemented!**

---

### Stage H: Model Comparison (OpenMC/MCNP)

| ID | Capability | Status | Notes | Reference Implementation |
|----|------------|--------|-------|--------------------------|
| H1.1 | OpenMC statepoint ingestion |  | Not implemented | OpenMC Python API |
| H1.2 | MCNP tally ingestion |  | Not implemented | PyNE |
| H1.3 | ALARA input generation |  | Not implemented | Main ALARA repo |
| H1.4 | ALARA output parsing |  | Not implemented | Main ALARA repo |

**Priority Gaps:** All high priority for model validation workflow.

---

### Stage I: TRIGA / k₀-NAA

| ID | Capability | Status | Notes |
|----|------------|--------|-------|
| I1.1 | Cd-ratio calculations |  | `fluxforge.triga.cd_ratio` - CdRatioAnalyzer, calculate_cd_ratio() |
| I1.2 | f and α parameter fitting |  | `fluxforge.triga.cd_ratio` - estimate_f(), estimate_alpha_multi() |
| I1.3 | k₀-standardization module |  | `fluxforge.triga.k0` - TRIGAk0Workflow, calculate_sdc_factors() |
| I1.4 | Triple-monitor method |  | `fluxforge.triga.k0` - triple_monitor_method() for Zr-94/Zr-96/Au-197 |
| I1.5 | TRIGA flux validation |  | `fluxforge.triga.k0` - validate_triga_flux_params() |

**Status: Fully implemented!** See `examples/triga_k0naa_workflow.py` for demonstration.

---

### Stage J: Artifacts & Provenance

| ID | Capability | Status | Notes |
|----|------------|--------|-------|
| J1.1 | JSON artifact output |  | `fluxforge.io.artifacts` |
| J1.2 | Provenance metadata |  | `fluxforge.core.provenance` |
| J1.3 | Unit metadata validation |  | `fluxforge.core.schemas` |

**Status: Fully implemented!**

---

## Implementation Priorities

### Phase 1: Critical (Weeks 1-2)
1. **Verify D1.x capabilities** - May already be implemented
2. **Verify E1.x capabilities** - May already be implemented
3. **Add Hypermet peak shape** (B1.3)

### Phase 2: High (Weeks 3-4)
1. **CNF format reader** (A1.3) - Reference: `testing/hdtv`
2. **Coincidence summing** (C1.4) - Reference: `testing/actigamma`
3. **Self-shielding corrections** (E1.4)

### Phase 3: Medium (Weeks 5-6)
1. ~~**Bayesian MCMC solver** (F1.5)~~  DONE - `fluxforge.solvers.mcmc.mcmc_unfold()`
2. **OpenMC integration** (H1.1)
3. **MCNP tally parsing** (H1.2)

### Phase 4: ALARA Integration (Weeks 7-8)
1. **ALARA input generation** (H1.3)
2. **ALARA output parsing** (H1.4)

### Phase 5: TRIGA/k₀ Module  COMPLETE
1. ~~**Cd-ratio calculations** (I1.1)~~  DONE - `fluxforge.triga.cd_ratio`
2. ~~**Spectral parameter fitting** (I1.2)~~  DONE - estimate_f(), estimate_alpha_multi()
3. ~~**k₀-standardization** (I1.3)~~  DONE - `fluxforge.triga.k0`

---

## Reference Implementations in testing/

| Repository | Capabilities to Port |
|------------|---------------------|
| `peakingduck` |  Window/chunked peak finders (DONE) |
| `hdtv` |  Hypermet peak shapes (DONE), CNF reader |
| `actigamma` | Coincidence summing corrections |
| `SpecKit` |  Gradient descent solver (DONE) |
| `Neutron-Unfolding` |  GRAVEL/MLEM reference (verified) |
| `Neutron-Spectrometry` |  Bayesian MCMC unfolding (DONE - pure Python) |
| `irrad_spectroscopy` | CHN format improvements |
| `gamma_spec_analysis` | General gamma analysis utilities |

---

## Test Coverage Summary

**Current: 136 tests passing**

New implementations in this session:
- `HypermetPeak` dataclass with evaluate() method
- `fit_hypermet_peak()` function for Hypermet fitting
- `mcmc_unfold()` Bayesian MCMC solver with:
  - Metropolis-Hastings sampling
  - Log-normal proposals (positivity preserving)
  - Smoothness prior
  - Adaptive step size
  - Convergence diagnostics

New tests added:
- `test_hypermet.py`: 15 tests (peak model, fitting)
- `test_mcmc.py`: 18 tests (solver, diagnostics)

Recommended additional tests:
- OpenMC/MCNP round-trip tests
- k₀-NAA validation tests
- Coincidence summing tests
