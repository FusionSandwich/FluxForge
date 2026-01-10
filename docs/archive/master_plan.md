# FluxForge Master Development Plan (Consolidated)

**HPGe-driven flux-wire / foil activation analysis • Neutron spectrum unfolding • Model validation**  
**OpenMC 0.15.3 (CE transport ± depletion) vs MCNP6.3 + ALARA (group activation)**

**Prepared:** 2026-01-02  
**Scope:** End-to-end, reproducible pipeline from raw HPGe spectra to (a) unfolded/group spectra with covariance and (b) rigorous model-to-experiment comparisons for TRIGA irradiations (flux wires/foils + larger samples).

---

## Table of Contents

1. [How to Use This Document](#1-how-to-use-this-document)
2. [Purpose and Success Criteria](#2-purpose-and-success-criteria)
3. [Reference Implementations](#3-reference-implementations-local)
4. [End-to-End Architecture](#4-end-to-end-architecture)
5. [Capability Backlog (Epics)](#5-capability-backlog-implementation-epics)
6. [STAYSL Integration Requirements](#6-staysl-integration-requirements)
7. [TRIGA-Specific Modules](#7-triga-specific-modules)
8. [OpenMC/MCNP Integration](#8-openmcmcnp-integration)
9. [Critical Pitfalls and Guardrails](#9-critical-pitfalls-and-guardrails)
10. [Implementation Sequence](#10-implementation-sequence)
11. [Package Layout](#11-package-layout)
12. [CLI Entry Points](#12-cli-entry-points)
13. [Validation and QA](#13-validation-and-qa)

---

## 1. How to Use This Document

- **Single source of truth** for the FluxForge backlog, architecture, and QA criteria
- Each subsection maps cleanly into GitHub Epics/Issues
- **Artifact-driven pipeline:**
  - Every stage emits machine-readable outputs (JSON/YAML + CSV + NPZ/HDF5)
  - Plus a short human summary
  - Plus provenance (hashes, versions, settings)
- No GUI required for initial development; core APIs must be CLI-friendly

---

## 2. Purpose and Success Criteria

FluxForge implements an end-to-end workflow to:
1. Infer neutron spectra and integral spectral parameters from activation monitors measured by HPGe gamma spectroscopy
2. Validate transport + activation workflows (OpenMC vs MCNP+ALARA) against experiment

### 2.1 Definition of Done (Project-Level)

| # | Criterion |
|---|-----------|
| 1 | From raw HPGe spectra (or peak reports), produce isotope activities with full uncertainty propagation and QA/QC |
| 2 | Convert activities to end-of-irradiation (EOI) reaction rates using explicit irradiation history (multi-segment) |
| 3 | Build response matrix R[i,g] using dosimetry cross sections, sample compositions, and corrections |
| 4 | Unfold/adjust neutron spectra using multiple solver families: GLS/STAYSL-like, MLEM/GRAVEL, Bayesian, SpecKit-style |
| 5 | Produce posterior spectra with covariance/correlation plus paper-style diagnostics (χ², pulls, influence) |
| 6 | Ingest OpenMC and MCNP(+ALARA) outputs, collapse to common group structures, compare with clear metrics (C/E, parity, residuals) |
| 7 | Output reproducible "run bundle" per analysis: config, input hashes, nuclear-data versions, all artifacts, final report |

---

## 3. Reference Implementations (Local)

Policy: FluxForge implements capabilities internally (no copying UIs; avoid calling external repos at runtime).
These local repos are used strictly as references for algorithms, data-model patterns, and QA/plot conventions.

| Category | Local Path | Key Features to Adopt |
|----------|-----------|----------------------|
| HPGe I/O & Peak Workflow | `testing/gamma_spec_analysis` | Lightweight activation-style I/O, basic peak workflow |
| Peak Detection & Background | `testing/peakingduck` | SNIP, windowed methods, multiplet logic, chunked peak finding |
| Peak Fitting | `testing/hdtv` | Peak-shape breadth, robust fitting, ROOT histogram patterns |
| Gamma Libraries & ID | `testing/actigamma` | Forward gamma synthesis, decay database patterns |
| Isotope Tables | `testing/irrad_spectroscopy` | Gamma tables, fluence calculations, cooldown corrections |
| Spectrum Unfolding | `testing/Neutron-Unfolding` | GRAVEL/MLEM implementations, diagnostics |
| Regularized Solver | `testing/SpecKit` | Gradient-based minimization, χ² + smoothness penalties |
| MLEM-STOP Criteria | `testing/Neutron-Spectrometry` | C++ implementation of stopping criteria |

---

## 4. End-to-End Architecture

### 4.1 Pipeline Stages (A → H)

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
│  Stage C: Efficiency / Corrections → Line Activities                       │
│  ┌─────────────────┐    ┌───────────────────────────────────────────────┐  │
│  │ PeakReport      │───▶│ LineActivities                                │  │
│  │ + DetectorModel │    │ (activity per gamma line @ count time + unc)  │  │
│  │ + gamma libs    │    │                                               │  │
│  └─────────────────┘    └───────────────────────────────────────────────┘  │
│           │                                                                 │
│           ▼                                                                 │
│  Stage D: Time-History Engine → EOI Activities + Reaction Rates            │
│  ┌─────────────────┐    ┌───────────────────────────────────────────────┐  │
│  │ LineActivities  │───▶│ ReactionRates                                 │  │
│  │ + IrradHistory  │    │ (EOI activity + inferred RR per monitor)      │  │
│  │ + cooling/count │    │                                               │  │
│  └─────────────────┘    └───────────────────────────────────────────────┘  │
│           │                                                                 │
│           ▼                                                                 │
│  Stage E: Response Construction                                            │
│  ┌─────────────────┐    ┌───────────────────────────────────────────────┐  │
│  │ Monitor defs    │───▶│ ResponseBundle                                │  │
│  │ + dosimetry XS  │    │ (R matrix, σ_eff, corrections, diagnostics)   │  │
│  │ + group struct  │    │                                               │  │
│  └─────────────────┘    └───────────────────────────────────────────────┘  │
│           │                                                                 │
│           ▼                                                                 │
│  Stage F: Unfolding / Adjustment                                           │
│  ┌─────────────────┐    ┌───────────────────────────────────────────────┐  │
│  │ ReactionRates   │───▶│ UnfoldResult                                  │  │
│  │ + ResponseBundle│    │ (posterior flux, cov, diagnostics, χ²)        │  │
│  │ + PriorSpectrum │    │                                               │  │
│  └─────────────────┘    └───────────────────────────────────────────────┘  │
│           │                                                                 │
│           ▼                                                                 │
│  Stage G: Model Comparison / Validation                                    │
│  ┌─────────────────┐    ┌───────────────────────────────────────────────┐  │
│  │ OpenMC/MCNP     │───▶│ ValidationBundle                              │  │
│  │ + ALARA outputs │    │ (C/E tables, closure metrics, consistency)    │  │
│  │ + UnfoldResult  │    │                                               │  │
│  └─────────────────┘    └───────────────────────────────────────────────┘  │
│           │                                                                 │
│           ▼                                                                 │
│  Stage H: Reporting and Archival                                           │
│  ┌─────────────────┐    ┌───────────────────────────────────────────────┐  │
│  │ All artifacts   │───▶│ ReportBundle                                  │  │
│  │                 │    │ (plots, tables, configs, hashes, JSON summary)│  │
│  └─────────────────┘    └───────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Core Data Model

| Object | Description |
|--------|-------------|
| `SpectrumFile` | counts vs channel (and/or energy), live/real time, dead time, start time, detector id, sample id |
| `Peak`, `ROI`, `PeakReport` | peak centroids, net areas, fit model, covariance, multiplet memberships, QC flags |
| `DetectorModel` | energy calibration, resolution FWHM(E), efficiency curve ε(E) with parameter covariance |
| `DecayData` | half-lives, gamma yields, branching, parent-daughter chains (versioned) |
| `Sample` | composition, mass/areal density, geometry, covers (Cd), container/shielding parameters |
| `IrradiationHistory` | time segments (power/flux scaling), interruptions, cooling and counting windows |
| `MonitorReaction` | target isotope + MT, product nuclide(s), gamma lines used, interference map |
| `GroupStructure` | energy boundaries, lethargy widths, labels, plotting helpers |
| `ResponseMatrix` / `ResponseBundle` | R[i,g], σ_g,eff, correction factors and their uncertainties |
| `PriorSpectrum` | OpenMC/MCNP-derived group spectrum (plus covariance if available/assumed) |
| `AdjustmentResult` / `UnfoldResult` | posterior spectrum, covariance/correlation, residuals, χ², leverage/influence |
| `ValidationResult` / `ValidationBundle` | C/E tables, closure metrics, cross-workflow consistency checks |

### 4.3 Canonical I/O Formats (Strict)

- **JSON or YAML:** metadata (units, conventions, library versions, processing options)
- **CSV:** tabular artifacts (peaks, line activities, reaction rates, monitor tables)
- **NPZ or HDF5:** arrays (spectra vectors, response matrices, covariance/correlation matrices)

**Every artifact must embed:**
- Units and definitions (group-integrated vs group-averaged; per energy vs per lethargy)
- Normalization basis (per source particle / per fission / per watt, etc.)
- Provenance (hashes + version strings)

**Round-trip tests** (import → export) must preserve values within numerical tolerance.

---

## 5. Capability Backlog (Implementation Epics)

### Epic A — Spectrum Ingestion, Metadata Validation, QA/QC

#### A1. File Format Support
- **Read:** SPE/CHN/CNF, N42, common CSV exports; optional ROOT histograms via pure-Python reader
- **Parse and validate:** live/real time, dead-time fraction, start time, detector id, calibration provenance
- **Background spectrum** library support (blank runs) + subtraction rules
- **Reference:** `testing/gamma_spec_analysis`, `testing/hdtv`

#### A2. Automated QC Flags
- Gain shift / energy drift checks vs reference lines
- Saturation/pileup indicators; missing/invalid metadata detection
- Configurable acceptance rules (hard fail vs warn vs auto-exclude)

#### A3. Preprocessing Tools
- Rebinning, smoothing (controlled), dead-time corrections, energy calibration application
- Background estimation options: SNIP variants, AsLS/airPLS, morphological top-hat, wavelet baselines
- **Reference:** `testing/peakingduck/peakingduck/core/smoothing.py`

---

### Epic B — Peak Candidate Detection, ROI Segmentation, Multiplets

#### B1. Candidate Peak Detection
- **Algorithms:** derivative/DoG filters, CWT maxima, significance tests vs local noise; tunable false-positive control
- Windowed local methods (to capture peaks across dynamic ranges)
- Candidate scoring and ranking (deterministic)
- **Reference:** `testing/peakingduck` (WindowPeakFinder, ChunkedSimplePeakFinder, ScipyPeakFinder)

#### B2. ROI and Multiplet Handling
- Group candidates using resolution model FWHM(E); auto-build ROIs
- Global multiplet fitting: shared width/shape parameters; shared local background
- **Reference:** `testing/hdtv`

#### B3. "Peak Report" Artifact
- Machine-readable peak report: ROIs, fits, covariance, QC flags
- Used as downstream input and for CI tests

---

### Epic C — Peak Fitting, Efficiency Calibration, Activity Inference

#### C1. Peak-Shape Library + Fitting Backends
- **Peak shapes:** Gaussian, Voigt, EMG, Hypermet-like (tail + step), constrained multi-Gaussian for partial resolution
- **Backends:** Poisson likelihood (preferred at low counts), weighted least squares
- **Return:** fit covariance, residuals, goodness-of-fit; automatic instability flags
- **Reference:** `testing/hdtv/src/hdtv/peakmodels/`

#### C2. Detector Calibration Models
- Energy calibration model with uncertainty (poly or spline, with covariance)
- Resolution model FWHM(E) used as a constraint during fits
- Full-energy peak efficiency curve ε(E) with covariance; multi-source fitting + outlier logic

#### C3. Line Activity Computation
For each gamma line:
- Correct net counts for dead time and counting interval
- Compute activity at reference time using ε(E), I_γ, and decay corrections
- **Optional corrections (configurable):** coincidence summing, self-attenuation of gammas, geometry corrections

#### C4. Combine Multiple Lines per Isotope
- Weighted combination with consistency checks (pulls/outliers)
- Report per-line contributions
- **Reference:** `testing/irrad_spectroscopy`

---

### Epic D — Irradiation History Engine + Reaction Rate Inference

#### D1. Time-History Model (Piecewise-Constant, Multi-Segment)
- Supports arbitrary timestamps, interruptions/pulses, repeated counts per sample
- Exact decay during irradiation + cooling + counting (avoid one-factor shortcuts unless explicitly enabled)
- **Formula:**
  ```
  A_sat = A(t) / [(1 - e^{-λt_i}) × e^{-λt_c}]
  ```

#### D2. Reaction-Rate Inference
- Compute EOI activity and infer reaction rate per monitor reaction
- Support simple parent/daughter build-in/out where needed
- Optional target depletion/burnup logic (off by default)

#### D3. Uncertainty Propagation
- Propagate: peak fit + counting stats + efficiency covariance + nuclear data + time-history inputs
- Preserve correlation structure where available (e.g., shared efficiency systematic)

---

### Epic E — Nuclear Data Management

#### E1. Data Packages and Provenance
- Bundle decay + gamma libraries (ENSDF-derived or equivalent) with explicit versioning and checksums
- Bundle/produce dosimetry cross sections for monitor reactions (IRDFF-II style), with covariance support:
  - Minimum: diagonal uncertainties
  - Target: full covariance matrices (where available)
- **Reference:** `testing/actigamma/actigamma/database.py`

#### E2. Processing Utilities
- Tools to collapse continuous-energy cross sections to chosen group structures
- Utilities to manage temperature/processing assumptions; embed provenance in outputs

---

### Epic F — Group Structures, Self-Shielding, Covers, Response Matrix

#### F1. Group Structures
- **Built-ins:** 10g/31g/50g/100g/175g/640g/725g plus user-defined
- Conversions between structures must conserve integrals
- Plotting must be explicit about normalization

#### F2. Self-Shielding Corrections (SHIELD-style)
- Geometry-aware approximations (slab/cylinder minimum)
- Parameterized by thickness, density, composition
- Uncertainty models for correction factors (bounds, sampling)
- **STAYSL PNNL Reference:** Energy-dependent self-shielding for wire and foil geometries

#### F3. Cadmium Cover Corrections (BCF-style)
- Cd transmission/cutoff model; thickness/density/composition
- Uncertainty handling
- Explicit reporting of correction factors and their impact

#### F4. Response Matrix Construction
- Construct R[i,g] = N_i × σ_i,g,eff × (correction factors)
- Support multiple products/branches and multiple gamma lines; interference maps
- **Diagnostics:** matrix visualization, condition number/rank checks, stabilization options
- Response uncertainty handling (nuisance propagation or augmented covariance)

---

### Epic G — Unfolding / Spectrum Adjustment Solvers

#### G1. GLS / STAYSL-like Adjustment (Core "Gold Standard")

**Mathematical Framework:**
```
P' - P = N_p × G^T × (N_A + N_Ao)^{-1} × (A^o - A)
```

Where:
- N_p = covariance matrix of input parameters (group fluxes + cross sections)
- G = sensitivity matrix of calculated reaction rates
- N_A, N_Ao = covariance matrices of calculated and observed reaction rates
- (A^o - A) = residual vector

**Updated covariance:**
```
N_{P'} = N_p - N_p × G^T × (N_A + N_Ao)^{-1} × G × N_p
```

**Capabilities:**
- Full covariance treatment (measurement + prior; optional response as nuisance)
- Positivity enforcement (log-parameterization or constrained optimization)
- Smoothness regularization (optional) with documented priors/penalties

**Outputs:**
- Posterior flux φ, covariance V_φ, correlation matrix
- Diagnostics: χ², reduced χ², per-monitor pulls, influence/leverage, predicted vs measured rates

#### G2. Iterative Unfolding Cross-Checks (MLEM / GRAVEL)
- **Reference:** `testing/Neutron-Unfolding/gravel.py`, `testing/Neutron-Unfolding/mlem.py`
- Robust stopping criteria:
  - Fixed iteration caps
  - Relative change thresholds
  - Objective-based stopping (MLEM-STOP-like logic)
- Uncertainty estimation via Monte Carlo sampling and/or ensemble runs

#### G3. Bayesian Unfolding (Optional, High Value)
- Positivity + smoothness priors; MCMC (or equivalent) posterior sampling
- Convergence diagnostics (R-hat, ESS) + posterior predictive checks
- Outputs: posterior summaries (mean/median), credible intervals, covariance/correlation estimates

#### G4. SpecKit/SAND-II-Style Deterministic Regularized Solver
- **Reference:** `testing/SpecKit/src/neutron_spectrum_solver.py`
- Gradient-based minimization of χ²(y − Rφ) + log-smoothness penalties
- Line-search/backtracking and explicit stopping criteria
- Multi-start ensembles (flat, Maxwellian+1/E, OpenMC prior, MCNP prior) to detect local minima

---

## 6. STAYSL Integration Requirements

### 6.1 STAYSL PNNL Auxiliary Tools to Replicate

| Tool | Function | FluxForge Module |
|------|----------|-----------------|
| STAYSL PNNL | Core spectral adjustment using GLSQM | `solvers/staysl.py` |
| SHIELD | Calculates neutron self-shielding for wires | `response/shielding.py` |
| BCF | Correction for Cd, B, or Gd covers | `response/covers.py` |
| SigPhi | Spreadsheet-based reaction rate calculation | `physics/activation.py` |
| NJpp | Data extraction from IRDFF/NJOY libraries | `nuclear_data/processing.py` |

### 6.2 Required Capabilities
- Support for IRDFF v1.05 and v2.0 libraries up to 60 MeV
- Energy grids up to 725 groups
- Automated generation of a priori spectrum covariance matrices from MC relative errors
- Single-step (non-iterative) solver to avoid SAND-II-like divergence issues

---

## 7. TRIGA-Specific Modules

### H1. Cd-Ratio + (f, α) Characterization
- Compute Cd-ratios using bare vs Cd-covered monitors
- Infer thermal-to-epithermal ratio f and epithermal shape factor α with uncertainties
- Position-by-position plots and tables for reactor characterization

### H2. k₀-Standardization Module (k₀-NAA Style)
- Internal k₀ constants database for common monitors:
  - k₀ factors, resonance integrals, Q₀(α), Westcott g(T), recommended lines
- Canonical k₀ workflow: net peak → activity → saturation activity → reaction rate
- Cross-check hooks:
  - Compare k₀-derived parameters vs unfolded spectrum integrals
  - Compare vs OpenMC/MCNP group-collapsed thermal/epithermal components

### 7.1 Standard Flux Wire Reactions

| Reaction | Energy Region | Resonance/Threshold | Product | Half-life |
|----------|---------------|---------------------|---------|-----------|
| ¹⁹⁷Au(n,γ)¹⁹⁸Au | Thermal/Epithermal | 4.9 eV | ¹⁹⁸Au | 2.69 d |
| ⁵⁹Co(n,γ)⁶⁰Co | Thermal/Epithermal | 132 eV | ⁶⁰Co | 5.27 y |
| ¹¹⁵In(n,γ)¹¹⁶ᵐIn | Epithermal | Low-energy resonances | ¹¹⁶ᵐIn | 54 min |
| ²⁷Al(n,α)²⁴Na | Fast | 5-7 MeV threshold | ²⁴Na | 15 h |
| ⁵⁴Fe(n,p)⁵⁴Mn | Fast | ~2 MeV threshold | ⁵⁴Mn | 312 d |

---

## 8. OpenMC/MCNP Integration

### I1. OpenMC (0.15.3) Ingestion
- Read statepoint tallies (mesh/cell), uncertainties, normalization
- Collapse continuous-energy flux to selected group structures
- Compute predicted reaction rates/activities using same monitor definitions
- Export standardized PriorSpectrum and model-predicted ReactionRates

**Key Fixes:**
- HDF5-based tally parser with automated volume-normalization detection
- Correction for ~22% flux discrepancy vs MCNP F4 tallies
- Handle track-length vs collision estimators consistently

### I2. MCNP6.3 Ingestion
- Parse MCTAL/MESHTAL/OUTP flux tallies with energy bins
- Map to exact group boundaries
- Support direct ingestion of group flux files for ALARA

### I3. ALARA Interface
- Generate ALARA decks programmatically: materials, irradiation schedule, flux file refs, decay times
- Parse ALARA outputs (activities, inventories, uncertainties)
- Map to monitor products/lines

### I4. Cross-Workflow Consistency Checks

| Check Type | Description |
|------------|-------------|
| Transport-only | OpenMC vs MCNP group flux at wire/sample volumes |
| Activation-only | Given identical group flux, FluxForge activation math must match ALARA within tolerance |
| End-to-end closure | Forward-calc expected gamma lines from model inventories, compare to HPGe results |

### 8.1 Normalization Reconciliation

**MCNP:** F4 tally normalized by total source weight automatically  
**OpenMC:** Flux in [particle-cm / source-particle], requires:
```
φ = (tally / V) × S
```
Where V = cell volume, S = source strength

**FluxForge must implement:** "Source Weight Correction" module to reconcile variance reduction behaviors

---

## 9. Critical Pitfalls and Guardrails

| Pitfall | Guardrail |
|---------|-----------|
| **Timebase errors** (live vs real time) | Explicit validation, dead-time correction flags |
| **Normalization drift** (per-source vs per-fission vs per-watt) | Normalization tag required in all artifacts |
| **Unit/definition drift** (group-integrated vs group-averaged) | Explicit unit field in all arrays |
| **Library inconsistency** (XS libraries/temperatures) | Provenance hashes, version checks |
| **Ignoring Cd covers / self-shielding** | Correction factor reports, mandatory for resonance reactions |
| **Peak interference + summing** | Multiplet detection, summing correction flags |
| **Solver over-regularization** | Residual/pull checks required in reports |
| **Ill-conditioned response matrices** | Condition number checks, stabilization warnings |

---

## 10. Implementation Sequence

### Phase 0 — Foundation (Weeks 1-2)
1. Canonical artifact formats + provenance + unit/definition enforcement
2. Minimal CLI skeleton with all stage subcommands
3. Round-trip I/O tests for core artifacts

### Phase 1 — Spectrum Ingest and Peaks (Weeks 3-5)
1. SPE/CHN/CNF readers with metadata validation
2. QC flags for gain drift, dead time, missing metadata
3. Peak candidate detection (windowed + CWT methods)
4. ROI construction and multiplet handling
5. PeakReport artifact with CI regression tests

### Phase 2 — Activities and Reaction Rates (Weeks 6-8)
1. Detector calibration + efficiency models with covariance
2. Line activity computation with corrections
3. Multi-segment irradiation history engine
4. Reaction-rate inference with uncertainty propagation

### Phase 3 — Response and Adjustment (Weeks 9-12)
1. Group structures and response matrix construction
2. Self-shielding and Cd cover corrections
3. GLS solver (STAYSL-like) with full diagnostics
4. MLEM/GRAVEL cross-checks with stopping criteria

### Phase 4 — Validation and Reporting (Weeks 13-16)
1. OpenMC statepoint ingestion and group collapse
2. MCNP ingestion and ALARA interface
3. Cross-workflow consistency checks
4. Report bundles with required plots

### Phase 5 — Advanced Features (Weeks 17-20)
1. SpecKit-style regularized solver
2. Bayesian unfolding (optional)
3. TRIGA k₀ module
4. Experimental benchmark locking

---

## 11. Package Layout

```
fluxforge/
├── __init__.py
├── io/                    # Spectrum + model readers/writers; canonical formats
│   ├── spe.py            # SPE/CHN file I/O (existing)
│   ├── genie.py          # Genie-2000 format (existing)
│   ├── n42.py            # N42 XML format (new)
│   ├── root_reader.py    # Pure-Python ROOT histogram reader (new)
│   ├── artifacts.py      # Artifact serialization (existing)
│   └── metadata.py       # Metadata validation (existing)
├── hpge/                  # Peak finding, fitting, calibration, efficiency
│   ├── peaking.py        # Peak detection algorithms (new)
│   ├── fitting.py        # Peak fitting backends (enhance existing)
│   ├── multiplet.py      # Multiplet handling (new)
│   ├── calibration.py    # Energy + efficiency calibration (new)
│   └── qc.py             # QC flags and validation (new)
├── activation/            # Decay, irradiation history, reaction rates
│   ├── decay.py          # Decay calculations (new)
│   ├── history.py        # Irradiation history engine (enhance existing)
│   └── rates.py          # Reaction rate inference (enhance existing)
├── nuclear_data/          # Libraries, covariance handling, processing tools
│   ├── decay_lib.py      # Decay data library (new)
│   ├── gamma_lib.py      # Gamma line library (new)
│   ├── irdff.py          # IRDFF cross sections (existing)
│   └── processing.py     # XS collapsing, temperature handling (new)
├── response/              # Group structures, self-shielding, Cd covers, response matrix
│   ├── groups.py         # Group structure definitions (enhance existing)
│   ├── shielding.py      # Self-shielding corrections (new)
│   ├── covers.py         # Cd/B/Gd cover corrections (new)
│   └── matrix.py         # Response matrix construction (enhance existing)
├── solvers/               # GLS, MLEM, GRAVEL, Bayesian, gradient solver
│   ├── gls.py            # GLS solver (existing, enhance)
│   ├── staysl.py         # STAYSL-like full adjustment (new)
│   ├── iterative.py      # MLEM/GRAVEL (existing, enhance)
│   ├── bayesian.py       # Bayesian MCMC (new)
│   └── gradient.py       # SpecKit-style regularized (new)
├── validate/              # OpenMC/MCNP/ALARA comparisons; closure tests
│   ├── openmc.py         # OpenMC ingestion (new)
│   ├── mcnp.py           # MCNP ingestion (new)
│   ├── alara.py          # ALARA interface (new)
│   └── closure.py        # Consistency checks (new)
├── triga/                 # TRIGA-specific modules
│   ├── cd_ratio.py       # Cd-ratio analysis (new)
│   └── k0.py             # k₀ standardization (new)
├── report/                # Plots + report bundle generation
│   ├── plots.py          # Required plot set (new)
│   └── bundle.py         # Report bundle generation (new)
├── cli/                   # Command-line entry points
│   └── app.py            # CLI commands (existing, enhance)
├── core/                  # Core utilities
│   ├── linalg.py         # Linear algebra (existing)
│   ├── provenance.py     # Provenance tracking (existing)
│   ├── schemas.py        # Data schemas (existing)
│   └── units.py          # Unit handling (new)
└── examples/              # Worked examples + tutorial configs
```

---

## 12. CLI Entry Points

```bash
# Stage A - Spectrum ingestion
fluxforge ingest <spectrum_files> [--format spe|chn|cnf|n42] [--output dir]

# Stage B - Peak finding and fitting
fluxforge peaks <spectrum_file> [--detector-model file] [--output peak_report.json]

# Stage C - Activity calculation
fluxforge activity <peak_report> [--efficiency-model file] [--gamma-lib file] [--output activities.csv]

# Stage D - Reaction rates
fluxforge rates <activities> <irrad_history> [--output rates.csv]

# Stage E - Response matrix
fluxforge response <monitor_defs> <xs_library> <group_structure> [--output response.npz]

# Stage F - Unfolding
fluxforge unfold <rates> <response> [--prior flux.npz] [--method gls|mlem|gravel|bayesian] [--output result.npz]

# Stage G - Comparison
fluxforge compare <unfold_result> [--openmc statepoint.h5] [--mcnp mctal] [--alara output] [--output validation.json]

# Stage H - Reporting
fluxforge report <run_dir> [--format html|pdf] [--output report.html]

# End-to-end
fluxforge run <config.yaml> [--output-dir results/]
```

---

## 13. Validation and QA

### K1. Synthetic Benchmarks
- Generate synthetic rates from known spectrum; add noise; unfold; verify recovery
- Stress tests: ill-conditioned response matrices, missing monitors, wrong priors, sparse counts

### K2. Experimental Benchmark Locking
- Freeze at least one full irradiation dataset as regression target
- **Standard metrics:** per-monitor C/E, pulls, reduced χ², spectrum-integral comparisons

### K3. Engineering Requirements
- Deterministic outputs (seeded RNG for MC sampling)
- Comprehensive logging; clear error messages; strict unit checking
- CI must run:
  - Synthetic unfold benchmark
  - One locked experimental regression test
  - Round-trip I/O tests for canonical artifacts

---

## Appendix A — GitHub Reference Repositories

| Repository | URL | Key Features |
|------------|-----|--------------|
| SpecKit | https://github.com/lifangchen2021/SpecKit | Regularized adjustment, uncertainty bands |
| HDTV | https://github.com/janmayer/hdtv | Peak fitting, ROOT integration |
| gamma_spec_analysis | https://github.com/py1sl/gamma_spec_analysis | Lightweight HPGe I/O |
| actigamma | https://github.com/fispact/actigamma | Forward gamma synthesis |
| irrad_spectroscopy | https://github.com/SiLab-Bonn/irrad_spectroscopy | Isotope tables, fluence |
| peakingduck | https://github.com/fispact/peakingduck | AI-enhanced peak detection |
| Neutron-Unfolding | https://github.com/tylerdolezal/Neutron-Unfolding | GRAVEL/MLEM implementations |
| Neutron-Spectrometry | https://github.com/kildealab/Neutron-Spectrometry | MLEM-STOP criteria |
