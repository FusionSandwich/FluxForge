FluxForge Roadmap
=================

**HPGe-driven flux-wire / foil activation analysis • Neutron spectrum unfolding • Model validation**  
**OpenMC 0.15.3 (CE transport ± depletion) vs MCNP6.3 + ALARA (group activation)**

Last Updated: 2026-01-02

Purpose
-------
Deliver an end-to-end, reproducible pipeline from HPGe spectra to unfolded
neutron spectra with full uncertainty propagation, then validate OpenMC and
MCNP+ALARA against experiment with auditable artifacts.

For the complete master plan including mathematical framework and STAYSL
integration details, see [master_plan.md](master_plan.md).

Guiding Principles
------------------
- **Artifact-driven pipeline:** every stage emits machine-readable outputs plus a short human summary
- **Provenance first:** inputs, versions, and settings are always captured
- **Deterministic outputs:** seeded RNG for MC sampling; reproducible by design
- **CLI-friendly:** no GUI required; core APIs designed for automation
- **Reference-based development:** learn from local repos in `testing/` but implement internally

Phases and Milestones
---------------------

### Phase 0 - Foundation (M0) — Weeks 1-2
**Status:** IN PROGRESS

| Task | Status | Priority | Reference |
|------|--------|----------|-----------|
| Canonical artifact schemas with units + normalization |  | P0 | core/schemas.py |
| Provenance tracking (hashes, versions) |  | P0 | core/provenance.py |
| CLI skeleton with all stage subcommands |  | P0 | cli/app.py |
| Round-trip I/O tests for core artifacts | ⬜ | P0 | tests/ |
| Unit handling module (group-integrated vs averaged) | ⬜ | P0 | core/units.py (new) |

**Deliverables:**
- All artifacts include provenance + units + normalization
- CLI can run end-to-end on synthetic inputs

---

### Phase 1 - Spectrum Ingest and Peak Report (M1) — Weeks 3-5
**Status:** NOT STARTED

#### 1.1 File Format Support
| Format | Status | Reference Implementation |
|--------|--------|-------------------------|
| SPE (Ortec) |  | io/spe.py (existing) |
| CHN | ⬜ | testing/gamma_spec_analysis |
| CNF (Genie-2000) |  | io/genie.py (partial) |
| N42 (XML) | ⬜ | testing/hdtv patterns |
| CSV exports | ⬜ | io/csv_readers.py |
| ROOT histograms | ⬜ | testing/hdtv (pure-Python reader) |

#### 1.2 QC and Validation
| Feature | Status | Notes |
|---------|--------|-------|
| Gain drift detection | ⬜ | Compare vs reference lines |
| Dead-time validation | ⬜ | live/real time consistency |
| Metadata completeness | ⬜ | Hard fail vs warn vs auto-exclude |
| Saturation/pileup flags | ⬜ | High count rate indicators |

#### 1.3 Peak Detection (Reference: testing/peakingduck)
| Algorithm | Status | Notes |
|-----------|--------|-------|
| Derivative/DoG filters |  | analysis/peakfit.py (partial) |
| CWT maxima | ⬜ | scipy.signal.find_peaks_cwt |
| Windowed local methods | ⬜ | peakingduck.WindowPeakFinder |
| Chunked detection | ⬜ | peakingduck.ChunkedSimplePeakFinder |
| SNIP background | ⬜ | peakingduck.core.smoothing |

#### 1.4 Peak Fitting (Reference: testing/hdtv)
| Feature | Status | Notes |
|---------|--------|-------|
| Gaussian |  | analysis/peakfit.py |
| Voigt | ⬜ | hdtv.peakmodels |
| EMG / Hypermet | ⬜ | hdtv.peakmodels |
| Poisson likelihood | ⬜ | Low-count preference |
| Multiplet fitting | ⬜ | Shared width/background |

**Deliverables:**
- PeakReport artifact with covariance and QC flags
- CI regression test with fixed test spectra

---

### Phase 2 - Activities and Reaction Rates (M2) — Weeks 6-8
**Status:** NOT STARTED

#### 2.1 Detector Calibration
| Feature | Status | Reference |
|---------|--------|-----------|
| Energy calibration with uncertainty |  | io/spe.py (partial) |
| Resolution model FWHM(E) | ⬜ | data/efficiency.py |
| Efficiency curve ε(E) with covariance |  | data/efficiency_models.py |

#### 2.2 Activity Computation
| Feature | Status | Reference |
|---------|--------|-----------|
| Dead-time correction |  | physics/activation.py |
| Decay correction |  | physics/activation.py |
| Coincidence summing correction | ⬜ | New module |
| Self-attenuation correction | ⬜ | New module |
| Multi-line weighted combination |  | physics/activation.py |

#### 2.3 Irradiation History Engine
| Feature | Status | Reference |
|---------|--------|-----------|
| Multi-segment piecewise-constant |  | physics/activation.py |
| Interruptions/pulses | ⬜ | Enhance existing |
| Repeated counts per sample | ⬜ | New logic |
| Parent/daughter build-in/out | ⬜ | Complex chains |

**Deliverables:**
- ReactionRates artifact with full uncertainty propagation
- Analytic verification tests for known decay chains

---

### Phase 3 - Response and Adjustment (M3) — Weeks 9-12
**Status:** NOT STARTED

#### 3.1 Group Structures
| Feature | Status | Notes |
|---------|--------|-------|
| Built-in structures (10g/31g/50g/100g/175g/640g/725g) |  | core/response.py (partial) |
| User-defined structures | ⬜ | JSON/YAML input |
| Integral-conserving conversions | ⬜ | New utility |
| Lethargy plotting | ⬜ | plots/unfolding.py |

#### 3.2 Corrections (STAYSL PNNL Style)
| Feature | Status | Reference |
|---------|--------|-----------|
| Self-shielding (SHIELD-like) | ⬜ | response/shielding.py (new) |
| Cd cover corrections (BCF-like) | ⬜ | response/covers.py (new) |
| Uncertainty propagation | ⬜ | Bounds + sampling |

#### 3.3 Response Matrix
| Feature | Status | Reference |
|---------|--------|-----------|
| R[i,g] construction |  | core/response.py |
| Multiple products/branches | ⬜ | MonitorReaction class |
| Condition number diagnostics | ⬜ | New |
| Matrix stabilization | ⬜ | SVD-based |

#### 3.4 Solvers
| Solver | Status | Reference |
|--------|--------|-----------|
| GLS / STAYSL-like |  | solvers/gls.py (enhance for full GLSQM) |
| GRAVEL |  | solvers/iterative.py |
| MLEM |  | solvers/iterative.py |
| MLEM-STOP criteria | ⬜ | testing/Neutron-Spectrometry |
| Bayesian MCMC | ⬜ | solvers/bayesian.py (new) |
| SpecKit-style gradient | ⬜ | solvers/gradient.py (new) |

**Deliverables:**
- ResponseBundle with diagnostics
- UnfoldResult with χ², pulls, influence

---

### Phase 4 - Validation and Reporting (M4) — Weeks 13-16
**Status:** NOT STARTED

#### 4.1 OpenMC Integration
| Feature | Status | Notes |
|---------|--------|-------|
| Statepoint HDF5 reading | ⬜ | validate/openmc.py (new) |
| Group collapse | ⬜ | Energy-grid interpolation |
| Normalization reconciliation | ⬜ | ~22% flux fix |
| Volume filter handling | ⬜ | Wire geometry |

#### 4.2 MCNP Integration
| Feature | Status | Notes |
|---------|--------|-------|
| MCTAL parsing | ⬜ | validate/mcnp.py (new) |
| MESHTAL parsing | ⬜ | Mesh tally support |
| Group mapping | ⬜ | Exact boundaries |

#### 4.3 ALARA Interface
| Feature | Status | Notes |
|---------|--------|-------|
| Deck generation | ⬜ | validate/alara.py (new) |
| Output parsing | ⬜ | Activities, inventories |
| Product/line mapping | ⬜ | Monitor definitions |

#### 4.4 Closure Tests
| Test Type | Status | Notes |
|-----------|--------|-------|
| Transport-only (OpenMC vs MCNP) | ⬜ | Group flux comparison |
| Activation-only (FluxForge vs ALARA) | ⬜ | Given identical flux |
| End-to-end closure | ⬜ | Forward gamma synthesis |

#### 4.5 Reporting
| Feature | Status | Notes |
|---------|--------|-------|
| Prior vs posterior overlay | ⬜ | plots/unfolding.py |
| Residuals/pulls plot | ⬜ | Per-monitor χ² |
| Parity plot (predicted vs measured) | ⬜ | With error bars |
| Covariance heatmap | ⬜ | Correlation matrix |
| HTML/PDF report bundle | ⬜ | report/bundle.py (new) |
| Machine-readable JSON summary | ⬜ | CI gating |

**Deliverables:**
- ValidationBundle with C/E tables
- ReportBundle with all plots and provenance

---

### Phase 5 - Advanced Features (M5) — Weeks 17-20
**Status:** PARTIALLY COMPLETE

#### 5.1 TRIGA-Specific Modules  COMPLETE
| Feature | Status | Notes |
|---------|--------|-------|
| Cd-ratio + (f, α) characterization |  | `fluxforge.triga.cd_ratio` - CdRatioAnalyzer |
| k₀ standardization |  | `fluxforge.triga.k0` - TRIGAk0Workflow |
| k₀ constants database |  | STANDARD_MONITORS, TRIPLE_MONITOR_DATA |
| Triple-monitor method |  | `triple_monitor_method()` for bare Zr-94/Zr-96/Au-197 |
| SDC factors |  | `calculate_sdc_factors()` - Saturation, Decay, Counting |

See `examples/triga_k0naa_workflow.py` for complete demonstration.

#### 5.2 Advanced Solvers
| Feature | Status | Notes |
|---------|--------|-------|
| SpecKit-style multi-start | ⬜ | Local minima detection |
| Bayesian MCMC with R-hat/ESS |  | `fluxforge.solvers.mcmc` |

#### 5.3 Experimental Locking
| Feature | Status | Notes |
|---------|--------|-------|
| Freeze reference dataset | ⬜ | Thesis benchmark |
| CI regression metrics | ⬜ | Per-monitor C/E, reduced χ² |

---

Current Focus
-------------
**M0 (Foundation)** → then vertical slices through M1 and M2

Immediate next steps:
1. Complete artifact schemas with explicit units/normalization
2. Implement round-trip I/O tests
3. Add CLI subcommands for all stages
4. Begin SPE/CHN/CNF reader enhancements

Reference Implementations (Local)
---------------------------------
All reference repositories are cloned in `testing/`:

| Category | Local Path | Key Features |
|----------|-----------|--------------|
| HPGe I/O | testing/gamma_spec_analysis | Lightweight spectrum I/O |
| Peak Workflow | testing/hdtv | Peak shapes, ROOT patterns |
| Peak Detection | testing/peakingduck | SNIP, windowed methods, multiplets |
| Forward Gamma | testing/actigamma | Gamma synthesis from inventories |
| Isotope Tables | testing/irrad_spectroscopy | Gamma tables, fluence calcs |
| Unfolding | testing/Neutron-Unfolding | GRAVEL/MLEM Python implementations |
| Regularized Solver | testing/SpecKit | Gradient descent, χ² + smoothness |
| MLEM-STOP | testing/Neutron-Spectrometry | Stopping criteria (C++) |

Legend
------
-  Complete
-  In Progress / Partial
- ⬜ Not Started
