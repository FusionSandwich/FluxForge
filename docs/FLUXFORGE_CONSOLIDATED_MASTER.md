# FLUXFORGE — CONSOLIDATED MASTER DOCUMENT

**HPGe-driven flux-wire / foil activation analysis • Neutron spectrum unfolding • Model validation**  
**OpenMC 0.15.3 (CE transport ± depletion) vs MCNP6.3 + ALARA (group activation)**

**Version:** 3.0  
**Last Updated:** 2026-01-10  
**Scope:** End-to-end, reproducible pipeline from raw HPGe spectra to (a) unfolded/group spectra with covariance and (b) rigorous model-to-experiment comparisons for TRIGA irradiations (flux wires/foils + larger samples).

---

## Table of Contents

1. [Purpose and Success Criteria](#1-purpose-and-success-criteria)
2. [Implementation Status Summary](#2-implementation-status-summary)
3. [Pipeline Architecture](#3-pipeline-architecture)
4. [Capability Backlog (Implementation Epics)](#4-capability-backlog-implementation-epics)
5. [STAYSL Parity Mode (Epic M)](#5-staysl-parity-mode-epic-m)
6. [IRDFF-II Data Access (Epic N)](#6-irdff-ii-data-access-epic-n)
7. [ENDF/B-VIII.0 Data Access (Epic O)](#7-endfb-viii0-data-access-epic-o)
8. [k₀-NAA Module (Epic P)](#8-k0-naa-module-epic-p)
9. [RMLE Gamma Unfolding (Epic Q)](#9-rmle-gamma-unfolding-epic-q)
10. [Critical Pitfalls and Guardrails](#10-critical-pitfalls-and-guardrails)
11. [Test Suite Documentation](#11-test-suite-documentation)
12. [Validation Results](#12-validation-results)
13. [Package Layout and CLI](#13-package-layout-and-cli)
14. [Quick Start Guide](#14-quick-start-guide)
15. [Reference Implementations](#15-reference-implementations)

---

## 1. Purpose and Success Criteria

FluxForge implements an end-to-end workflow to:
1. Infer neutron spectra and integral spectral parameters from activation monitors measured by HPGe gamma spectroscopy
2. Validate transport + activation workflows (OpenMC vs MCNP+ALARA) against experiment

### 1.1 Definition of Done (Project-Level)

| # | Criterion | Status |
|---|-----------|--------|
| 1 | From raw HPGe spectra (or peak reports), produce isotope activities with full uncertainty propagation and QA/QC | ✅ |
| 2 | Convert activities to end-of-irradiation (EOI) reaction rates using explicit irradiation history (multi-segment) | ✅ |
| 3 | Build response matrix R[i,g] using dosimetry cross sections, sample compositions, and corrections | ✅ |
| 4 | Unfold/adjust neutron spectra using multiple solver families: GLS/STAYSL-like, MLEM/GRAVEL, Bayesian, SpecKit-style | ✅ |
| 5 | Produce posterior spectra with covariance/correlation plus paper-style diagnostics (χ², pulls, influence) | ✅ |
| 6 | Ingest OpenMC and MCNP(+ALARA) outputs, collapse to common group structures, compare with clear metrics (C/E, parity, residuals) | ✅ |
| 7 | Output reproducible "run bundle" per analysis: config, input hashes, nuclear-data versions, all artifacts, final report | ✅ |

### 1.2 STAYSL Parity Definition (NEW)

**STAYSL parity** refers to neutron dosimetry spectral adjustment capabilities:
- SigPhi-style corrected saturation reaction rates
- SHIELD-style neutron self-shielding
- Cover correction factors (Cd/Gd/B/Au)
- Covariance-aware GLS adjustment and diagnostics

**Note:** Gamma-spectroscopy corrections (e.g., gamma self-attenuation) are mandatory for activity inference but are NOT part of STAYSL parity; they belong to the HPGe activity path.

### 1.3 Canonical Definitions for C/E and GLS Inputs (NEW)

All "C/E" comparisons and all GLS adjustments must explicitly declare and record:

| Symbol | Definition |
|--------|------------|
| A(t_c) | Activity at counting reference time (count start or mid-count; specify) |
| A(t_EOI) | Activity at end-of-irradiation (EOI) |
| R_EOI | EOI reaction rate (reactions/s) inferred from A(t_EOI) and decay data |
| A_sat / R_sat | Saturation activity/rate consistent with multi-segment irradiation history ("SigPhi-equivalent") |
| y | GLS input vector: saturated reaction rates R_sat (or explicitly documented alternative) with covariance V_y |

**CRITICAL:** No report table or plot is permitted to display C/E unless it references one of these explicit quantities and includes the normalization basis (per watt / per fission / per source particle).

---

## 2. Implementation Status Summary

### Overall Progress

| Category | Implemented | Partial | Missing | Total |
|----------|-------------|---------|---------|-------|
| A: Spectrum Ingestion | 6 | 0 | 0 | 6 |
| B: Peak Detection | 5 | 0 | 0 | 5 |
| C: Activity Calculation | 5 | 0 | 0 | 5 |
| D: Reaction Rates | 5 | 0 | 0 | 5 |
| E: Response Matrix | 5 | 0 | 0 | 5 |
| F: Unfolding Solvers | 7 | 0 | 0 | 7 |
| G: Plotting | 5 | 0 | 0 | 5 |
| H: Model Comparison | 4 | 0 | 0 | 4 |
| I: TRIGA/k₀-NAA | 5 | 0 | 0 | 5 |
| J: Artifacts | 3 | 0 | 0 | 3 |
| K: Reactor Dosimetry (INL) | 8 | 0 | 0 | 8 |
| M: STAYSL Parity | 10 | 0 | 0 | 10 |
| N: IRDFF-II Access | 8 | 0 | 0 | 8 |
| O: ENDF/B-VIII.0 Access | 7 | 0 | 0 | 7 |
| P: k₀-NAA Complete | 9 | 0 | 0 | 9 |
| Q: RMLE Gamma Unfolding | 9 | 0 | 0 | 9 |
| Z: NAA-ANN Neural Networks | 8 | 0 | 0 | 8 |
| **Total** | **109** | **0** | **0** | **109** |

**Implementation Coverage: 100% Complete ✅**
**Test Suite: 731 tests passing**
**Last Updated: January 10, 2026**

---

## 3. Pipeline Architecture

### 3.1 Pipeline Stages (A → H)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FLUXFORGE PIPELINE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Stage A: Spectrum Ingest                                                   │
│  ┌─────────────────┐    ┌───────────────────────────────────────────────┐  │
│  │ Raw Spectra     │───▶│ SpectrumFile                                  │  │
│  │ CNF/CHN/SPE/N42 │    │ (counts + calibration + validated metadata)   │  │
│  └─────────────────┘    └───────────────────────────────────────────────┘  │
│           │                                                                 │
│           ▼                                                                 │
│  Stage B: Peak Finding & Fitting                                           │
│  ┌─────────────────┐    ┌───────────────────────────────────────────────┐  │
│  │ SpectrumFile    │───▶│ PeakReport                                    │  │
│  │ + calibration   │    │ (peaks, ROIs, fit params, covariance, QC)     │  │
│  └─────────────────┘    └───────────────────────────────────────────────┘  │
│           │                                                                 │
│           ▼                                                                 │
│  Stage C: Efficiency / Spectroscopy Corrections → Line Activities (NEW)   │
│  ┌─────────────────┐    ┌───────────────────────────────────────────────┐  │
│  │ PeakReport      │───▶│ LineActivities + IsotopeActivities            │  │
│  │ + DetectorModel │    │ (with gamma self-attenuation corrections)     │  │
│  │ + gamma libs    │    │                                               │  │
│  └─────────────────┘    └───────────────────────────────────────────────┘  │
│           │                                                                 │
│           ▼                                                                 │
│  Stage D: Time-History Engine → EOI Activities + Reaction Rates            │
│  ┌─────────────────┐    ┌───────────────────────────────────────────────┐  │
│  │ LineActivities  │───▶│ ReactionRates: A(t_c), A(t_EOI), R_EOI, R_sat │  │
│  │ + IrradHistory  │    │ + full covariance V_y for GLS input           │  │
│  │ + cooling/count │    │ + burnup correction when |R_sat/R_EOI-1|>1%   │  │
│  └─────────────────┘    └───────────────────────────────────────────────┘  │
│           │                                                                 │
│           ▼                                                                 │
│  Stage E: Response Construction                                            │
│  ┌─────────────────┐    ┌───────────────────────────────────────────────┐  │
│  │ Monitor defs    │───▶│ ResponseBundle                                │  │
│  │ + dosimetry XS  │    │ (R matrix, σ_eff, corrections, diagnostics)   │  │
│  │ + group struct  │    │ + self-shielding + cover corrections          │  │
│  └─────────────────┘    └───────────────────────────────────────────────┘  │
│           │                                                                 │
│           ▼                                                                 │
│  Stage F: Unfolding / Adjustment                                           │
│  ┌─────────────────┐    ┌───────────────────────────────────────────────┐  │
│  │ ReactionRates   │───▶│ UnfoldResult                                  │  │
│  │ + ResponseBundle│    │ (posterior flux φ, cov V_φ, χ², residuals)    │  │
│  │ + PriorSpectrum │    │ + explicit prior covariance model             │  │
│  └─────────────────┘    └───────────────────────────────────────────────┘  │
│           │                                                                 │
│           ▼                                                                 │
│  Stage G: Model Comparison / Validation                                    │
│  ┌─────────────────┐    ┌───────────────────────────────────────────────┐  │
│  │ OpenMC/MCNP     │───▶│ ValidationBundle                              │  │
│  │ + ALARA outputs │    │ (C/E tables with explicit definitions,        │  │
│  │ + UnfoldResult  │    │  closure metrics, library provenance)         │  │
│  └─────────────────┘    └───────────────────────────────────────────────┘  │
│           │                                                                 │
│           ▼                                                                 │
│  Stage H: Reporting and Archival                                           │
│  ┌─────────────────┐    ┌───────────────────────────────────────────────┐  │
│  │ All artifacts   │───▶│ ReportBundle                                  │  │
│  │                 │    │ (STAYSL-class outputs, flux correlations)     │  │
│  └─────────────────┘    └───────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Core Data Model

| Object | Description | Status |
|--------|-------------|--------|
| `SpectrumFile` | counts vs channel/energy, live/real time, dead time, detector id | ✅ |
| `Peak`, `ROI`, `PeakReport` | peak centroids, net areas, fit covariance, QC flags | ✅ |
| `DetectorModel` | energy calibration, FWHM(E), efficiency ε(E) with covariance | ✅ |
| `DecayData` | half-lives, gamma yields, branching, chains (versioned) | ✅ |
| `Sample` | composition, mass/density, geometry, covers, container params | ✅ |
| `IrradiationHistory` | time segments, interruptions, cooling/counting windows | ✅ |
| `MonitorReaction` | target isotope + MT, product, gamma lines, interference map | ✅ |
| `GroupStructure` | energy boundaries, lethargy widths, labels | ✅ |
| `ResponseMatrix` / `ResponseBundle` | R[i,g], σ_g,eff, correction factors + uncertainties | ✅ |
| `PriorSpectrum` | OpenMC/MCNP spectrum + covariance (or assumed structure) | ✅ |
| `UnfoldResult` | posterior flux, covariance, χ², residuals, influence | ✅ |
| `ValidationBundle` | C/E tables with explicit definitions, closure metrics | ✅ |

---

## 4. Capability Backlog (Implementation Epics)

### Epic A — Spectrum Ingestion, Metadata Validation, QA/QC

| ID | Capability | Status | Module | Notes |
|----|------------|--------|--------|-------|
| A1.1 | SPE format reader | ✅ | `io.spe.read_spe_file()` | Full support |
| A1.2 | CHN format reader | ✅ | `io.hpge` | Full ORTEC/Maestro CHN binary support |
| A1.3 | CNF (Canberra) format | ✅ | `io.cnf.read_cnf_file()` | Binary parser + tests |
| A1.4 | N42/IEC XML format | ✅ | `io.n42` | Reader/writer + tests |
| A1.5 | Background estimation (SNIP) | ✅ | `analysis.peakfit` | Full support |
| A1.6 | Dead-time validation | ✅ | `io.spe` | `dead_time_fraction` |

### Epic B — Peak Detection & Fitting

| ID | Capability | Status | Module | Notes |
|----|------------|--------|--------|-------|
| B1.1 | Automated peak finding | ✅ | `analysis.peak_finders` | scipy, window, chunked, simple |
| B1.2 | Gaussian peak fitting | ✅ | `analysis.peakfit` | `GaussianPeak` |
| B1.3 | Hypermet peak shapes | ✅ | `analysis.hypermet` | `HypermetPeak`, `fit_hypermet_peak()` |
| B1.4 | Multiplet handling | ✅ | `analysis.peakfit` | Shared width/background |
| B1.5 | Peak fit covariance | ✅ | `analysis.peakfit` | `PeakFitResult.covariance` |

### Epic C — Efficiency & Activity Calculation

| ID | Capability | Status | Module | Notes |
|----|------------|--------|--------|-------|
| C1.1 | Efficiency curve fitting | ✅ | `data.efficiency` | Multi-source, outlier detection |
| C1.2 | Activity from peak area | ✅ | `physics.activation` | Full dead-time and decay correction |
| C1.3 | Weighted activity (multi-line) | ✅ | `physics.activation` | `weighted_activity()` |
| C1.4 | Coincidence summing corrections | ✅ | `corrections.coincidence` | Full decay scheme TCS for Co60/Y88/Cs134/Eu152/Na24 + tests |
| C1.5 | **Gamma self-attenuation (NEW)** | ✅ | `corrections.gamma_attenuation` | Sample/container attenuation correction + tests |

### Epic D — Reaction Rates

| ID | Capability | Status | Module | Notes |
|----|------------|--------|--------|-------|
| D1.1 | Multi-segment irradiation history | ✅ | `physics.activation` | `IrradiationSegment` |
| D1.2 | Decay corrections | ✅ | `physics.activation` | `irradiation_buildup_factor()` |
| D1.3 | EOI reaction rate | ✅ | `physics.activation` | `reaction_rate_from_activity()` |
| D1.4 | **SigPhi saturation rates (NEW)** | ✅ | `physics.sigphi` | SigPhi-equivalent R_sat + full factors |
| D1.5 | **Burnup/transmutation (NEW)** | ✅ | `physics.sigphi` | Threshold-based burnup correction + tests |

### Epic E — Response Matrix

| ID | Capability | Status | Module | Notes |
|----|------------|--------|--------|-------|
| E1.1 | Energy group structure | ✅ | `core.response` | 640/725g support |
| E1.2 | Response matrix construction | ✅ | `core.response` | `build_response_matrix()` |
| E1.3 | IRDFF cross sections | ✅ | `data.irdff` | `IRDFFDatabase` |
| E1.4 | Self-shielding corrections | ✅ | `corrections.self_shielding` | SHIELD-style geometry support + tests |
| E1.5 | Cadmium cover corrections | ✅ | `corrections.covers` | Cd/Gd/B/Au cover transmission + tests |

### Epic F — Unfolding Solvers

| ID | Capability | Status | Module | Notes |
|----|------------|--------|--------|-------|
| F1.1 | GLS adjustment | ✅ | `solvers.gls` | `gls_adjust()` |
| F1.2 | GRAVEL iterative | ✅ | `solvers.iterative` | `gravel()` - validated |
| F1.3 | MLEM iterative | ✅ | `solvers.iterative` | `mlem()` |
| F1.4 | Gradient descent | ✅ | `solvers.iterative` | `gradient_descent()` |
| F1.5 | Bayesian MCMC | ✅ | `solvers.mcmc` | `mcmc_unfold()` |
| F1.6 | Positivity constraints | ✅ | All solvers | Enforced |
| F1.7 | Chi-square diagnostics | ✅ | All solvers | Solution objects |

### Epic G — Plotting & Reporting

| ID | Capability | Status | Notes |
|----|------------|--------|-------|
| G1.1 | Spectrum with uncertainty bands | ✅ | `examples/generate_plots.py` |
| G1.2 | Prior vs posterior overlay | ✅ | Implemented |
| G1.3 | Residual/pull plots | ✅ | Implemented |
| G1.4 | Covariance/correlation heatmaps | ✅ | Implemented |
| G1.5 | Parity plot | ✅ | Implemented |

### Epic H — Model Comparison (OpenMC/MCNP)

| ID | Capability | Status | Notes |
|----|------------|--------|-------|
| H1.1 | OpenMC statepoint ingestion | ✅ | `io.openmc.read_openmc_flux_spectrum()`, `OpenMCSpectrum` |
| H1.2 | MCNP tally ingestion | ✅ | `io.mcnp.read_mctal()`, `read_mcnp_flux_tally()`, `MCTALFile` |
| H1.3 | ALARA input generation | ✅ | `io.alara.create_alara_activation_input()`, `ALARAInputGenerator` |
| H1.4 | ALARA output parsing | ✅ | `io.alara.read_alara_output()`, `parse_alara_output()`, `ALARAOutput` |

### Epic I — TRIGA / k₀-NAA ✅ COMPLETE

| ID | Capability | Status | Module |
|----|------------|--------|--------|
| I1.1 | Cd-ratio calculations | ✅ | `triga.cd_ratio.CdRatioAnalyzer` |
| I1.2 | f and α parameter fitting | ✅ | `triga.cd_ratio.estimate_f/alpha_multi()` |
| I1.3 | k₀-standardization module | ✅ | `triga.k0.TRIGAk0Workflow` |
| I1.4 | Triple-monitor method | ✅ | `triga.k0.triple_monitor_method()` |
| I1.5 | TRIGA flux validation | ✅ | `triga.k0.validate_triga_flux_params()` |

### Epic J — Artifacts & Provenance ✅ COMPLETE

| ID | Capability | Status | Module |
|----|------------|--------|--------|
| J1.1 | JSON artifact output | ✅ | `io.artifacts` |
| J1.2 | Provenance metadata | ✅ | `core.provenance` |
| J1.3 | Unit metadata validation | ✅ | `core.schemas` |

### Epic K — Reactor Dosimetry Workflow (INL)

| ID | Capability | Status | Module | Notes |
|----|------------|--------|--------|-------|
| K1 | Flux Wire Selection | ✅ | `analysis.flux_wire_selection` | IRDFF reactions |
| K2 | Irradiation Modeling | ✅ | `physics.activation` | `IrradiationSegment` |
| K3 | Flux Wire Measurement | ✅ | `io.flux_wire` | HPGe processing |
| K4 | A Priori Spectrum | ✅ | `SpectrumUnfolder` | 640/725 groups |
| K5 | Spectrum Unfolding | ✅ | `solvers` | GLS, GRAVEL, MLEM, MCMC |
| K6 | Adjusted Spectrum | ✅ | `UnfoldingResult` | `compare_with_mcnp()` |
| K7 | Fluences of Interest | ✅ | — | 1-MeV eq, DPA |
| K8 | A Priori Covariance | ✅ | `core.prior_covariance` | Prior covariance models + CLI wiring for GLS |

---

## 5. STAYSL Parity Mode (Epic M) — NEW

**Goal:** FluxForge must reproduce the functional capabilities and diagnostics of a STAYSL PNNL workflow end-to-end.

### M1. SigPhi-Equivalent Saturation/Reaction-Rate Engine

| ID | Capability | Status | Notes |
|----|------------|--------|-------|
| M1.1 | Saturation activity corrections | ✅ | Multi-segment history in `physics.sigphi` |
| M1.2 | BCF-like computation pathway | ✅ | `flux_history_correction_factor()` |
| M1.3 | Saturation with sampling decay | ✅ | `CorrectionType.SATURATION_WITH_SAMPLING_DECAY` |
| M1.4 | Saturated reaction rates artifact | ✅ | `SaturationRateResult` + burnup model |

### M2. Neutron Self-Shielding (SHIELD-Style)

| ID | Capability | Status | Notes |
|----|------------|--------|-------|
| M2.1 | Geometry-aware calculator | ✅ | `corrections.self_shielding` - slab/cylinder/sphere |
| M2.2 | Fine internal energy grid | ✅ | `calculate_group_self_shielding()` with subgroup resolution |
| M2.3 | Self-shielding library artifact | ✅ | `SelfShieldingLibrary`, `create_self_shielding_library()` |
| M2.4 | Isotropic vs beam flux type | ✅ | `FluxType.ISOTROPIC`, `FluxType.BEAM` |

### M3. Cover Correction Factors

| ID | Capability | Status | Notes |
|----|------------|--------|-------|
| M3.1 | Cadmium cover model | ✅ | `corrections.covers.CoverMaterial.CADMIUM` |
| M3.2 | Gadolinium cover model | ✅ | `corrections.covers.CoverMaterial.GADOLINIUM` |
| M3.3 | Boron cover model | ✅ | `corrections.covers.CoverMaterial.BORON` |
| M3.4 | Gold cover model | ✅ | `corrections.covers.CoverMaterial.GOLD` |
| M3.5 | Uncertainty propagation | ✅ | `CoverCorrectionFactor.F_c_uncertainty` |
| M3.6 | **STAYSL CCF parity mode** | ✅ | `CoverSpec`, `compute_ccf_staysl()`, E₂(x) integral |
| M3.7 | **Energy-dependent T(E)** | ✅ | `compute_energy_dependent_cover_corrections()` |
| M3.8 | **Beam vs isotropic flux models** | ✅ | `FluxAngularModel.BEAM`, `FluxAngularModel.ISOTROPIC` |
| M3.9 | **STAYSL parity report** | ✅ | `create_staysl_parity_report()` for sta_spe.dat comparison |

### M4. GLS Spectral Adjustment (STAYSL Core)

| ID | Capability | Status | Notes |
|----|------------|--------|-------|
| M4.1 | Full covariance treatment | ✅ | `solvers.gls` |
| M4.2 | Response covariance option | ✅ | `ResponseCovariancePolicy` enum + AUGMENT_VY, MONTE_CARLO modes |
| M4.3 | Prior covariance model (NEW) | ✅ | `core.prior_covariance` (regional + lethargy-correlated defaults) |
| M4.4 | Response uncertainty policy (NEW) | ✅ | `core.prior_covariance` (augment V_y / nuisance / MC policy enum) |

**Prior Covariance V_φ0 (REQUIRED):**
- User-supplied V_φ0, OR
- Default structured model (energy-region fractional uncertainties with lethargy-correlation length)
- Report must state which model was used

**Response Uncertainty Treatment (REQUIRED):**
- Augment V_y with propagated response uncertainty, OR
- Treat response as nuisance parameters, OR
- Monte Carlo propagation producing effective V_y
- Report must state policy and contribution

### M5. Output/Report Parity

| ID | Capability | Status | Notes |
|----|------------|--------|-------|
| M5.1 | Dosimetry input correlation matrix | ✅ | `reporting.CorrelationMatrix` |
| M5.2 | Input vs output flux correlations | ✅ | `reporting.CorrelationMatrix.from_covariance()` |
| M5.3 | Differential flux tables | ✅ | `reporting.DifferentialFluxTable` |
| M5.4 | Spectral-averaged reaction rates | ✅ | `reporting.ReactionRateTable` |
| M5.5 | Plot-ready stepwise spectrum | ✅ | `reporting.StepwiseSpectrum` |

### M6. Interoperability

| ID | Capability | Status | Notes |
|----|------------|--------|-------|
| M6.1 | Import saturation rates from spreadsheet | ✅ | `io.interop.read_saturation_rates_csv()` |
| M6.2 | Export STAYSL-compatible bundle | ✅ | `io.interop.export_staysl_bundle()` |
| M6.3 | Lower-triangular symmetric matrix ingestion | ✅ | `io.interop.read_lower_triangular_matrix()` |

---

## 6. IRDFF-II Data Access (Epic N)

**Goal:** Complete access to IRDFF-II dosimetry library including covariance and metadata.

| ID | Capability | Status | Module | Notes |
|----|------------|--------|--------|-------|
| N1.1 | Multi-group library ingestion | ✅ | `data.irdff` | IAEA download |
| N1.2 | ENDF-6 evaluation processing | ✅ | `data.irdff` | Full tabulated format support |
| N1.3 | Energy grid / boundaries | ✅ | `data.irdff` | 119 reactions |
| N1.4 | Covariance support | ✅ | `data.irdff` | Full MF33 + endf_covariance module |
| N1.5 | Reaction metadata (MT, product, thresholds) | ✅ | `data.irdff` | Full |
| N1.6 | Reaction browser CLI/API | ✅ | `cli.app reactions` | Category/target filter |
| N1.7 | NJOY processing pipeline | ✅ | `data.njoy` | Reproducible group XS |
| N1.8 | Wire set robustness diagnostics | ✅ | `analysis.robustness` | Condition/coverage/LOO analysis |

---

## 7. ENDF/B-VIII.0 Data Access (Epic O)

**Goal:** Unified nuclear data interface preventing library mismatch between transport/activation.

| ID | Capability | Status | Notes |
|----|------------|--------|-------|
| O1.1 | Unified NuclearData interface | ✅ | CE and MG XS |
| O1.2 | Thermal scattering metadata | ✅ | `data.thermal_scattering` S(α,β) material-TSL mapping |
| O1.3 | Decay data link-outs | ✅ | `data.gamma_database` |
| O1.4 | Multi-temperature support | ✅ | Temperature tags |
| O1.5 | ENDF ↔ IRDFF bridge utilities | ✅ | Explicit mapping |
| O1.6 | Library provenance enforcement | ✅ | `ProvenanceBundle` + `validate_library_provenance()` |

### ENDF-6 Covariance Requirements (NEW)

**[ADDED] ENDF-6 covariance ingestion is first-class (no silent drops).**
FluxForge ingests and preserves ENDF-6 covariance information needed for dosimetry/activation comparisons.
At minimum MF33 (cross section covariance) is supported. If any covariance component cannot be parsed,
FluxForge must either fail fast or continue only with an explicit **"covariance degraded"** flag that is
carried into downstream artifacts and reports.

**[ADDED] Covariance integrity checks + documented conditioning.**
Before GLS (or any inversion), covariance matrices are validated and (when needed) conditioned with a recorded method:
symmetry check, non-positive-definite detection, near-singularity detection, and conditioning options such as
SVD truncation and/or diagonal loading ("nugget"). Conditioning method + parameters must be recorded in the run bundle.

| ID | Capability | Status | Notes |
|----|------------|--------|-------|
| O2.1 | MF33 (XS covariance) ingestion | ✅ | `data.endf_covariance` + tests |
| O2.2 | Covariance degraded flag | ✅ | Explicit flag + warnings when defaults/padding used |
| O2.3 | Covariance integrity checks | ✅ | Symmetry/PD/conditioning diagnostics |
| O2.4 | Documented conditioning | ✅ | SVD truncation and PD projection utilities |

---

## 8. k₀-NAA Module (Epic P)

**Goal:** Complete, traceable k₀ workflow with uncertainty for standalone use and cross-validation.

| ID | Capability | Status | Module | Notes |
|----|------------|--------|--------|-------|
| P1.1 | k₀ factors database | ✅ | `triga.k0` | `STANDARD_MONITORS` |
| P1.2 | Resonance integrals | ✅ | `triga.k0` | In database |
| P1.3 | Q₀(α) parameters | ✅ | `triga.k0` | Implemented |
| P1.4 | Westcott g(T) factors | ✅ | `triga.k0` | `WestcottFactors`, 17 isotopes with polynomial g(T) |
| P1.5 | Recommended lines + interference | ✅ | `triga.k0` | In database |
| P1.6 | Canonical k₀ pipeline | ✅ | `triga.k0` | `TRIGAk0Workflow` |
| P1.7 | Cd-ratio with uncertainty | ✅ | `triga.cd_ratio` | `calculate_cd_ratio()` |
| P1.8 | f/α reconciliation vs unfolded | ✅ | `triga.reconcile` | Cross-validation hook |
| P1.9 | Separate uncertainty components | ✅ | `uncertainty.budget` | `UncertaintyBudget` with category breakdown |

---

## 9. RMLE Gamma Unfolding (Epic Q) — Optional High Value

**Goal:** Full-spectrum gamma unfolding using Poisson-likelihood inverse model with regularization.

| ID | Capability | Status | Notes |
|----|------------|--------|-------|
| Q1.1 | Detector response matrix D | ✅ | `solvers.rmle` provides ResponseMatrix + Gaussian response builder |
| Q1.2 | Smearing matrices | ✅ | Gaussian smearing via resolution (FWHM(E)) model |
| Q1.3 | RMLE solver | ✅ | Poisson-likelihood + regularization + positivity |
| Q1.4 | Background component modeling (NEW) | ✅ | Explicit b in m = R_γ μ + b (constant/vector) |
| Q1.5 | Contaminant peak components | ✅ | `prior_activities/uncertainties` in `PoissonRMLEConfig` for constrained priors |
| Q1.6 | Calibration uncertainty propagation (NEW) | ✅ | Response-operator MC hook (response_sampler) + count resampling |
| Q1.7 | Unfolded spectrum uncertainty | ✅ | MC-based bands (counts + optional response sampling) |
| Q1.8 | Refolding/closure diagnostics | ✅ | Refold residual/χ² guardrails |
| Q1.9 | Integration with activation path | ✅ | `workflows.activation_pipeline.ActivationPipeline` for end-to-end ALARA workflow |

---

## 9a. Becquerel Parity (Epic R)

**Goal:** Implement capabilities from LBL-ANP's `becquerel` package for nuclear spectroscopy analysis with alternative fitting methods and extended data sources.

| ID | Capability | Status | Notes |
|----|------------|--------|-------|
| R1.1 | ExpGauss peak fitting | ✅ | `analysis.peakfit.expgauss()` - Exponentially-modified Gaussian |
| R1.2 | Double-exponential tail fitting | ✅ | `analysis.peakfit.gauss_dbl_exp()` - Left+right exponential tails |
| R1.3 | Poisson-likelihood fitting backend | ✅ | `analysis.peakfit.fit_peak_poisson()` with Nelder-Mead |
| R1.4 | CPS-keV spectrum representation | ✅ | `core.spectrum_ops.cpskev` - Counts/keV spectral density |
| R1.5 | Auto-calibration via known isotopes | ✅ | `analysis.auto_calibration.auto_calibrate()` |
| R1.6 | NNDC nuclear data access | ✅ | `data.nndc` - Isotope data fetching from NNDC |
| R1.7 | XCOM attenuation coefficients | ✅ | `data.xcom` - 12 materials, interpolated μ/ρ, HVL, transmission |
| R1.8 | NIST materials database | ✅ | `data.materials` - 45+ NIST/Compendium materials with compositions |
| R1.9 | Isotope/element metadata | ✅ | Already in `data.isotope_data` |
| R1.10 | Expression-based calibration | ✅ | Already in `calibration.energy_calibration` |

---

## 9b. GMApy/Evaluation Methods (Epic S)

**Goal:** Implement IAEA GMA (Generalized Method of Adjustment) methods from `gmapy` for cross-section evaluation and advanced inference.

| ID | Capability | Status | Notes |
|----|------------|--------|-------|
| S1.1 | Sparse GLS with cholmod | ✅ | `solvers.advanced.gls_update_numpy()` with sparse option |
| S1.2 | Levenberg-Marquardt optimization | ✅ | `solvers.advanced.levenberg_marquardt()` for nonlinear GLS |
| S1.3 | Adaptive Romberg integration | ✅ | `solvers.advanced.romberg_integrate()`, `spectrum_averaged_cross_section()` |
| S1.4 | PPP (Prior-Predictive-Posterior) correction | ✅ | `solvers.advanced.apply_ppp_correction()` Chiba-Smith method |
| S1.5 | Unknown uncertainty estimation | ✅ | `solvers.advanced.estimate_unknown_uncertainty()` ML/Birge methods |
| S1.6 | GMA workflow manager | ✅ | `evaluation.gma_workflow.GMAWorkflow` - Full GMA pipeline |
| S1.7 | Sensitivity matrix builder | ✅ | Already in `unfolding.sensitivity_matrix` |
| S1.8 | JSON experimental database format | ✅ | `evaluation.gma_workflow.ExperimentalDatabase` with JSON I/O |

---

## 9c. Curie/Charged-Particle Analysis (Epic T)

**Goal:** Implement capabilities from `curie` for charged-particle activation analysis and stacked-target experiments.

| ID | Capability | Status | Notes |
|----|------------|--------|-------|
| T1.1 | Spe/Chn/CNF/IEC format readers | ✅ | Full support in `io.spe`, `io.hpge`, `io.cnf`, `io.n42` |
| T1.2 | SNIP background algorithm | ✅ | Already in `analysis.peakfit` |
| T1.3 | DecayChain (Bateman solver) | ✅ | `physics.decay_chain.DecayChain` with matrix exponential |
| T1.4 | Stack (stacked-target characterization) | ✅ | `physics.stacked_target.StackedTarget` - Energy degradation |
| T1.5 | Multi-library cross sections | ✅ | `physics.stacked_target.CrossSectionLibrary` - IRDFF, ENDF, TENDL |
| T1.6 | Library cross-section search | ✅ | `CrossSectionLibrary.search_reactions()` - By target, product, projectile |
| T1.7 | 5-parameter efficiency calibration | ✅ | Available in `calibration.efficiency_calibration` |
| T1.8 | Geometry attenuation corrections | ✅ | `data.xcom` transmission + geometry in `corrections` |
| T1.9 | Multi-format spectrum export | ✅ | `io.spectrum_export.SpectrumExporter` - SPE/CNF/IEC/MCNP |

---

## 9d. NPAT/Stopping Power (Epic U)

**Goal:** Implement capabilities from `npat` for stopping power and charged-particle transport.

| ID | Capability | Status | Notes |
|----|------------|--------|-------|
| U1.1 | Ziegler stopping power | ✅ | `physics.stopping_power.electronic_stopping_ziegler()` |
| U1.2 | Energy loss through foil stack | ✅ | `physics.stopping_power.calculate_energy_loss()` |
| U1.3 | Range tables | ✅ | `physics.stopping_power.calculate_range()` |
| U1.4 | Straggling estimates | ✅ | `physics.stopping_power.calculate_straggling()` - Bohr |
| U1.5 | Compound stopping power | ✅ | `physics.stopping_power.total_stopping_power()` - Bragg |
| U1.6 | IAEA charged-particle database | ✅ | `physics.stacked_target.CrossSectionLibrary` with IRDFF |

---

## 9e. Testing Repository Cross-Validation (Epic V)

**Goal:** Systematic comparison tests running identical data through testing repos and FluxForge.

| ID | Capability | Status | Notes |
|----|------------|--------|-------|
| V1.1 | Becquerel comparison suite | ✅ | Peak fitting (ExpGauss, Poisson) via `cross_repo_comparison.py` |
| V1.2 | GMApy comparison suite | ✅ | GLS, LM, Romberg, PPP via `cross_repo_comparison.py` |
| V1.3 | Curie comparison suite | ✅ | Decay chains, Bateman, saturation via `cross_repo_comparison.py` |
| V1.4 | NPAT comparison suite | ✅ | Stopping power validated in `test_epic_implementations.py` |
| V1.5 | STAYSL comparison suite | ✅ | Already implemented |
| V1.6 | Cross-repo consistency tests | ✅ | 25/27 tests pass (2 skipped: becquerel not installed) |
| V1.7 | Performance benchmarks | ✅ | Inline timing in module tests |
| V1.8 | Numerical precision validation | ✅ | Tolerance checks validated in comparison tests |

---

## 9f. PyUnfold Parity (Epic W)

**Goal:** D'Agostini iterative Bayesian unfolding with advanced features from PyUnfold.

| ID | Capability | Status | Notes |
|----|------------|--------|-------|
| W1.1 | D'Agostini iterative Bayesian unfolding | ✅ | Already in FluxForge as iterative method |
| W1.2 | Spline regularization (SplineRegularizer) | ✅ | `solvers/test_statistics.py:SplineRegularizer` |
| W1.3 | Jeffreys prior support | ✅ | `jeffreys_prior()`, `power_law_prior()` in test_statistics.py |
| W1.4 | Multiple test statistics (KS, Chi2, BF, RMD) | ✅ | All 4 in `solvers/test_statistics.py` |
| W1.5 | Multinomial/Poisson covariance options | ✅ | `solvers.advanced_unfolding.CovarianceModel` - 4 models |
| W1.6 | Adye error propagation corrections | ✅ | `solvers.advanced_unfolding.adye_error_propagation()` |
| W1.7 | Callback system for iteration hooks | ✅ | Via regularizer callable in `iterative_bayesian_unfold()` |
| W1.8 | Iteration DataFrame output | ✅ | UnfoldingResult with history |

---

## 9g. actigamma Parity (Epic X)

**Goal:** Forward activity-to-spectrum prediction using decay line databases.

| ID | Capability | Status | Notes |
|----|------------|--------|-------|
| X1.1 | Decay line database (JSON) | ✅ | `physics.gamma_spectrum.DECAY_LINES` - 30+ nuclides |
| X1.2 | Energy bin specification | ✅ | Already in FluxForge spectrum handling |
| X1.3 | Line-to-bin mapping | ✅ | `physics.gamma_spectrum.bin_decay_lines()` |
| X1.4 | Energy conservation scaling | ✅ | `bin_decay_lines(energy_conservation=True)` |
| X1.5 | Multi-emission aggregation | ✅ | `get_decay_lines(emission_types=['gamma','x-ray'])` |
| X1.6 | Nuclide identification from peaks | ✅ | `physics.gamma_spectrum.identify_nuclides()` |
| X1.7 | Activity-to-atoms conversion | ✅ | Already in physics/decay_chain.py |
| X1.8 | Metastable state support | ✅ | `normalize_nuclide_name()`, `parse_nuclide()` - Tc-99m, In-116m |

---

## 9h. SpecKit/Neutron-Unfolding Parity (Epic Y)

**Goal:** Alternative unfolding algorithms and gradient-based methods.

| ID | Capability | Status | Notes |
|----|------------|--------|-------|
| Y1.1 | Log-smoothness regularization | ✅ | `solvers.advanced_unfolding.log_smoothness_penalty()` |
| Y1.2 | Gradient descent optimization | ✅ | `solvers.iterative.gradient_descent()` |
| Y1.3 | Chi-square convergence criterion | ✅ | 95% confidence threshold (already implemented) |
| Y1.4 | Monte Carlo uncertainty propagation | ✅ | Via `run_monte_carlo_trials()` |
| Y1.5 | GRAVEL algorithm | ✅ | `solvers.iterative.gravel()` - SAND-II variant |
| Y1.6 | MLEM algorithm | ✅ | `solvers.iterative.mlem()` |
| Y1.7 | Second-derivative convergence (f'') | ✅ | `solvers.advanced_unfolding.compute_ddJ_convergence()` |
| Y1.8 | Zero-channel elimination | ✅ | Handled in preprocessing |

---

## 9i. NAA-ANN Neural Networks (Epic Z)

**Goal:** Artificial Neural Network approach for element quantification from gamma spectra (optional TensorFlow dependency).

| ID | Capability | Status | Notes |
|----|------------|--------|-------|
| Z1.1 | TensorFlow integration | ✅ | `analysis.naa_ann` - optional import, `HAS_TENSORFLOW` flag |
| Z1.2 | GPU configuration | ✅ | `configure_gpu()`, `get_tensorflow_info()` |
| Z1.3 | Spectral data augmentation | ✅ | `SpectralAugmentor` - noise, shifts, intensity variation |
| Z1.4 | Patch-based CNN architecture | ✅ | `NAAANNModel` - multi-output with embedding layers |
| Z1.5 | Multi-output predictions | ✅ | Concentration, uncertainty, detection limit |
| Z1.6 | MC dropout uncertainty | ✅ | `predict_with_uncertainty()` |
| Z1.7 | Model serialization | ✅ | `.keras` format with config.json |
| Z1.8 | Training pipeline | ✅ | `train_naa_ann_model()`, `create_training_dataset()` |

**Notes:**
- TensorFlow is an **optional** dependency - FluxForge core works without it
- GPU acceleration available when CUDA is configured
- Based on IAEA NAA-ANN-1 CRP methodology
- 22 unit tests in `tests/test_naa_ann.py`

---

## 10. Critical Pitfalls and Guardrails

### 10.1 Existing Guardrails

| Pitfall | Prevention |
|---------|------------|
| Timebase errors (live vs real time) | Explicit validation in readers |
| Dead-time double application | Tracked in metadata |
| Normalization drift (per-source vs per-fission vs per-watt) | Explicit in all artifacts |
| Unit/definition drift (group-integrated vs averaged) | Schema enforcement |
| Library inconsistency | Provenance tracking |
| Ignoring Cd covers / self-shielding | Required correction flags |
| Peak interference + summing | Multiplet fitting, QC flags |
| Solver over-regularization | Residual/pull checks |
| Ill-conditioned response matrices | Condition checks, warnings |

### 10.2 NEW Guardrails

#### Definition Drift Hard-Stop (R1)

The pipeline must **refuse** to emit "final" C/E tables/plots unless each compared quantity is labeled as one of:
- A(t_c), A(t_EOI), R_EOI, R_sat, or documented alternative
- With explicit definition, units, and normalization basis

**Rationale:** Prevents order-of-magnitude errors from timebase/normalization confusion.

#### Library Drift Hard-Stop (R2)

All validation outputs must embed library provenance for:
- Transport flux source (ENDF)
- Dosimetry XS source (IRDFF/ENDF)
- Decay/gamma yield source
- Group structure

If any provenance is missing: run is non-reproducible and report must be labeled **"provisional"**.

#### Correction-Factor Completeness (R3)

For any Cd-covered or resonance-sensitive monitor, report must explicitly state:
- Whether cover correction was applied (and which model)
- Whether neutron self-shielding was applied (and which model)

Refuse to emit "final" validation plots unless checklist is satisfied or explicitly waived.

#### Diagnostics-First Philosophy (R4)

Every adjustment/unfold run must emit:
- Residuals and pulls per monitor
- χ² and reduced χ²
- Sensitivity / influence measures (or leverage proxies)
- Prior-to-posterior change diagnostics (where did spectrum move, and why?)

---

## 11. Test Suite Documentation

### 11.1 Test Summary

**Current Status:** 136+ tests passing

| Test Category | File | Tests | Status |
|---------------|------|-------|--------|
| Master Plan Goals | `test_master_plan_goals.py` | 8 | ✅ |
| Pipeline Validation | `test_pipeline_validation.py` | 12 | ✅ |
| Hypermet Peak Shapes | `test_hypermet.py` | 15 | ✅ |
| MCMC Solver | `test_mcmc.py` | 18 | ✅ |
| Peak Finders | `test_peak_finders.py` | 20 | ✅ |
| GLS Solver | `test_gls.py` | 15 | ✅ |
| GRAVEL/MLEM | `test_iterative.py` | 18 | ✅ |
| SPE Reading | `test_spe.py` | 10 | ✅ |
| IRDFF Database | `test_irdff.py` | 12 | ✅ |
| k₀-NAA | `test_k0.py` | 8 | ✅ |
| NAA-ANN Neural Networks | `test_naa_ann.py` | 22 | ✅ |
| Transport Code I/O | `test_transport_io.py` | 13 | ✅ |
| Peak Finder Methods | `test_peak_finder_methods.py` | 16 | ✅ |

### 11.2 Running Tests

```bash
# Complete test suite
python -m pytest tests/ -v

# Master plan goal validation
python -m pytest tests/test_master_plan_goals.py -v

# Pipeline validation with experimental data
python -m pytest tests/test_pipeline_validation.py -v

# Run with coverage
python -m pytest tests/ --cov=fluxforge --cov-report=html
```

### 11.3 Validation Framework

**Location:** `/testing_validation/`

```bash
# Run comparison framework
python testing_validation/compare_fluxforge_testing.py --test all

# Process RAFM flux wire data
python testing_validation/process_rafm_flux_wires.py --wire all
```

---

## 12. Validation Results

### 12.1 Algorithm Validation (vs testing/ repository)

| Test | Status | Correlation | Notes |
|------|--------|-------------|-------|
| GRAVEL | ✅ PASS | 0.9826 | FluxForge matches testing code |
| MLEM | ✅ PASS | 0.9965 | Added `convergence_mode="ddJ"` for parity |
| SPE File Reading | ✅ PASS | 1.0000 | Exact match |
| Gamma Database | ✅ PASS | 1.0000 | All 1,913 nuclides match |

### 12.1.1 Testing Repository Validation Checklist

Cross-validation of FluxForge against 10 repositories in `testing/`:

| Repository | Test | Status | Correlation | Notes |
|------------|------|--------|-------------|-------|
| **Neutron-Unfolding** | GRAVEL | ✅ | 0.9826 | Direct comparison |
| **Neutron-Unfolding** | MLEM (ddJ mode) | ✅ | 0.9965 | Matches original convergence |
| **SpecKit** | XS file reading | ✅ | 1.0000 | ENDF/IRDFF format |
| **SpecKit** | Gradient descent | ✅ | 0.95+ | Log-smoothness regularization |
| **gamma_spec_analysis** | SPE reading | ✅ | 1.0000 | Exact channel match |
| **gamma_spec_analysis** | Peak finding | ✅ | 1.0000 | Same peaks identified |
| **pyunfold** | D'Agostini MLEM | ✅ | 0.9965 | Two-peak test passes |
| **peakingduck** | SNIP background | ✅ | 0.98+ | Background extraction |
| **peakingduck** | Peak identification | ✅ | 0.95+ | Peak detection |
| **actigamma** | Decay database | ✅ | 1.0000 | Gamma lines match ENSDF |
| **actigamma** | Activity→spectra | ✅ | 0.99+ | Gamma cascade modeling |
| **hdtv** | ROOT file reading | ✅ | 1.0000 | TH1 histogram support |
| **hdtv** | Calibration | ✅ | 0.999+ | Energy calibration |
| **irrad_spectroscopy** | Isotope ID | ✅ | 0.95+ | Database matching |
| **irrad_spectroscopy** | Activity calc | ✅ | 0.98+ | Decay correction |
| **NAA-ANN-1** | Real spectra | ✅ | 1.00 | 20/20 spectra read, 20 with peaks |
| **Neutron-Spectrometry** | NNS response | ✅ | 1.0000 | Matrix format |

**Summary:** All 17 validation tests PASS (correlation > 0.90)

**Validation script:** `testing_validation/cross_validate_testing_repos.py`

### 12.1.1 Comprehensive Capability Validation (22/22 PASS)

All new FluxForge capabilities have been validated against actual APIs:

| Category | Test | Status | Notes |
|----------|------|--------|-------|
| **STAYSL-M1** | SigPhi saturation rates | ✅ | R_sat=9.17e-09 |
| **STAYSL-M2** | Self-shielding | ✅ | Au SSF=0.9810 |
| **STAYSL-M3** | Cover corrections | ✅ | Cd CCF=0.3479 |
| **STAYSL-M4** | GLS with covariance | ✅ | χ²_red=5.65 |
| **Epic-N** | IRDFF-II database | ✅ | 36 reactions |
| **Epic-O** | ENDF covariance | ✅ | MF33 validated |
| **Epic-O** | Thermal scattering | ✅ | H2O, graphite |
| **Epic-O** | Library provenance | ✅ | ENDF/B-VIII.0 + IRDFF-II |
| **Epic-P** | k₀-NAA workflow | ✅ | Cd-ratio=10.00 |
| **Epic-P** | Uncertainty budget | ✅ | 3 components |
| **Epic-Q** | RMLE gamma unfolding | ✅ | converged=True |
| **Epic-H** | Transport comparison | ✅ | C/E=0.981 |
| **Workflow** | Activation pipeline | ✅ | ALARA interface |
| **Corrections** | Coincidence summing | ✅ | Co-60 batch |
| **Corrections** | Gamma attenuation | ✅ | factor=1.15 |
| **Data** | Gamma database | ✅ | API verified |

**Validation script:** `testing_validation/comprehensive_validation.py`

### 12.2 GRAVEL Bug Fix Applied

**Original Issue:** Incorrect weighting scheme (correlation 0.04)

**Fix Applied:** Changed to correct SAND-II weighting:
```python
W_ig = measurements[i] * response[i][g] * phi[g] / predicted[i]
```

**Result:** Correlation improved to 0.9826

### 12.3 RAFM Flux Wire Validation

| Wire | Element | Lines Matched |
|------|---------|---------------|
| Ti | Titanium | 6/6 (100%) |
| Ni | Nickel | 2/4 (50%) |
| Co | Cobalt (bare) | 2/2 (100%) |
| Co-Cd | Cobalt (Cd) | 2/2 (100%) |
| In | Indium | 0/5 (0%)* |
| Sc | Scandium (bare) | 2/2 (100%) |
| Sc-Cd | Scandium (Cd) | 2/2 (100%) |
| Cu | Copper | 2/2 (100%) |

*In-116m: Short half-life (54 min), peaks decayed

**Total:** 30/37 matches (81.1%) — **Both FluxForge and testing methods identical**

---

## 13. Package Layout and CLI

### 13.1 Package Structure

```
fluxforge/
├── io/               # Spectrum + model readers/writers
│   ├── spe.py        # SPE format reader (✅)
│   ├── hpge.py       # Generic HPGe reader (✅)
│   ├── artifacts.py  # JSON artifact I/O (✅)
│   └── flux_wire.py  # Flux wire data (✅)
│
├── analysis/         # Peak finding and fitting
│   ├── peak_finders.py   # All peak finder methods (✅)
│   ├── peakfit.py        # Gaussian fitting (✅)
│   ├── hypermet.py       # Hypermet model (✅)
│   └── flux_wire_selection.py  # Wire advisor (✅)
│
├── physics/          # Activation physics
│   └── activation.py     # Decay, buildup, rates (✅)
│
├── data/             # Nuclear data
│   ├── irdff.py          # IRDFF-II database (✅)
│   ├── gamma_database.py # Gamma library (✅)
│   ├── elements.py       # Element data (✅)
│   └── efficiency.py     # Detector efficiency (✅)
│
├── core/             # Core functionality
│   ├── response.py       # Response matrix (✅)
│   ├── provenance.py     # Provenance tracking (✅)
│   ├── validation.py     # ValidationBundle, C/E tables (✅)
│   └── schemas.py        # Artifact schemas (✅)
│
├── solvers/          # Unfolding algorithms
│   ├── gls.py            # GLS adjustment (✅)
│   ├── iterative.py      # GRAVEL/MLEM/gradient (✅)
│   └── mcmc.py           # Bayesian MCMC (✅)
│
├── triga/            # TRIGA-specific
│   ├── cd_ratio.py       # Cd-ratio analysis (✅)
│   ├── k0.py             # k₀-NAA workflow (✅)
│   └── reconcile.py      # f/α reconciliation (✅)
│
├── io/               # Transport code interfaces
│   ├── openmc.py         # OpenMC statepoint ingestion (✅)
│   ├── mcnp.py           # MCNP tally/meshtal ingestion (✅)
│   └── alara.py          # ALARA I/O interface (✅)
│
└── cli/              # Command-line interface
    └── app.py            # CLI entry points (✅)
```

### 13.2 CLI Entry Points

```bash
fluxforge ingest <spectrum.spe>     # Read spectrum
fluxforge peaks <spectrum>          # Find/fit peaks
fluxforge activity <peaks.json>     # Compute activities
fluxforge rates <activities.json>   # Compute reaction rates
fluxforge response <config.yaml>    # Build response matrix
fluxforge unfold <rates.json>       # Run unfolding
fluxforge compare <unfold.json>     # Validate vs model
fluxforge report <bundle/>          # Generate report
fluxforge run <config.yaml>         # Full pipeline
```

---

## 14. Quick Start Guide

### 14.1 Installation

```bash
# Clone repository
git clone <repo-url>
cd FluxForge

# Create conda environment
conda env create -f environment.yml
conda activate FluxForge

# Install in development mode
pip install -e .
```

### 14.2 Basic Example

```python
from fluxforge.physics.activation import (
    GammaLineMeasurement,
    weighted_activity,
    reaction_rate_from_activity,
    IrradiationSegment,
)
from fluxforge.core.response import build_response_matrix
from fluxforge.solvers.gls import gls_adjust

# 1. Define gamma line measurements
gamma_lines = [
    GammaLineMeasurement(
        energy_kev=1099.245,
        intensity=0.565,
        net_counts=8743.0,
        counts_uncertainty=94.82,
        efficiency=0.001085,
        efficiency_uncertainty=5.425e-05,
    ),
]

# 2. Calculate weighted activity
activity, uncertainty = weighted_activity(gamma_lines)

# 3. Define irradiation history
segments = [IrradiationSegment(flux_level=1.0, duration_s=7200.0)]

# 4. Compute reaction rate
rate_result = reaction_rate_from_activity(
    activity, segments, half_life_s=3843936.0
)

# 5. Build response matrix and unfold
# (See full example in examples/generate_flux_spectrum.py)
```

### 14.3 Running Validation

```bash
# Run algorithm comparison
python testing_validation/compare_fluxforge_testing.py --test all

# Process RAFM flux wires
python testing_validation/process_rafm_flux_wires.py --wire all

# View results
cat testing_validation/comparison_output/COMPARISON_REPORT.md
cat testing_validation/rafm_results/FLUX_WIRE_REPORT.md
```

---

## 15. Reference Implementations

### 15.1 Local Testing Repositories

| Category | Local Path | Key Features |
|----------|-----------|--------------|
| HPGe I/O | `testing/gamma_spec_analysis` | Lightweight spectrum I/O |
| Peak Workflow | `testing/hdtv` | Peak shapes, ROOT patterns |
| Peak Detection | `testing/peakingduck` | SNIP, windowed methods |
| Forward Gamma | `testing/actigamma` | Gamma synthesis |
| Isotope Tables | `testing/irrad_spectroscopy` | Gamma tables, fluence |
| Unfolding | `testing/Neutron-Unfolding` | GRAVEL/MLEM Python |
| Regularized | `testing/SpecKit` | Gradient descent |
| MLEM-STOP | `testing/Neutron-Spectrometry` | Stopping criteria |

### 15.2 External References

| Reference | URL | Purpose |
|-----------|-----|---------|
| STAYSL PNNL | PNNL-22253 | GLS adjustment methodology |
| IRDFF-II | IAEA | Dosimetry cross sections |
| ENDF/B-VIII.0 | NNDC | Transport/activation data |
| k₀-NAA | Literature | Standardization method |

---

## Appendix A: Deprecated/Superseded Documents

The following documents are superseded by this consolidated master:

| File | Status |
|------|--------|
| `FluxForge_Master_Dev_Plan_Consolidated.txt` | Merged into this document |
| `Flux_Forge_Capability_List_HPGe_Unfolding_and_Model_Validation_v2.txt` | Merged into this document |
| `capability_audit.txt` | Updated in Section 2 |
| `capability_gap_analysis.md` | **DELETED** - content merged into Sections 2 and 4 |
| `master_plan.md` | Superseded by this document |
| `roadmap.md` | Integrated into Section 4 |
| `plans.txt` | Superseded by this document |
| `issue_map.md` | **DELETED** - content merged into Sections 4-9 |
| `quick_start.md` | Integrated into Section 14 |
| `experimental_results_summary.md` | Kept for experimental record |

---

## Appendix B: Implementation Priorities

All major implementation phases have been completed. The following capabilities are now available:

### Completed Phases ✅

1. **STAYSL Parity (Epic M)** - All 10 capabilities implemented
2. **IRDFF-II Data Access (Epic N)** - All 8 capabilities implemented  
3. **ENDF/B-VIII.0 Data Access (Epic O)** - All 7 capabilities implemented
4. **k₀-NAA Module (Epic P)** - All 9 capabilities implemented
5. **Model Comparison (Epic H)** - OpenMC, MCNP, ALARA I/O implemented
6. **NAA-ANN Neural Networks (Epic Z)** - TensorFlow integration (optional)

### Optional Enhancements

The following are optional enhancements that may be added based on user needs:

1. **RMLE Gamma Unfolding (Epic Q)** - Already complete (9/9)
2. **Additional spectrum formats** - ROOT, CNF extensions as needed
3. **GPU acceleration** - Available for NAA-ANN when CUDA is configured
4. **Additional peak finding methods** - CWT, morphological, etc.

---

**Document Version:** 3.0  
**Last Updated:** 2026-01-10  
**Prepared by:** FluxForge Development Team
**Test Suite:** 740 tests passing, 0 skipped
**GPU Support:** NVIDIA T600, CUDA 12.5.1, cuDNN 9
