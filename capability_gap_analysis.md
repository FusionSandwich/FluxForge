# FluxForge Capability Gap Analysis

## Latest Update: Testing Repos Integration

The following capabilities have been added based on reviewing testing repositories:

### New Modules Added (from testing repos)

| Source Repo | Capability | FluxForge Module | Status |
|-------------|------------|------------------|--------|
| `actigamma` | Nuclide Database | `fluxforge.physics.nuclides` | ✅ Implemented |
| `actigamma` | Half-life Lookup | `get_half_life()`, `get_gamma_lines()` | ✅ Implemented |
| `irrad_spectroscopy` | Dose Rate Calc | `fluxforge.physics.dose.gamma_dose_rate()` | ✅ Implemented |
| `irrad_spectroscopy` | Isotope Dose | `fluxforge.physics.dose.isotope_dose_rate()` | ✅ Implemented |
| `irrad_spectroscopy` | Fluence Estimation | `fluxforge.physics.dose.fluence_from_activity()` | ✅ Implemented |
| `peakingduck` | SNIP Background | `fluxforge.analysis.peak_finders.snip_background()` | ✅ Implemented |
| `peakingduck` | Window Peak Finder | `fluxforge.analysis.peak_finders.WindowPeakFinder` | ✅ Implemented |
| `peakingduck` | Chunked Peak Finder | `fluxforge.analysis.peak_finders.ChunkedPeakFinder` | ✅ Implemented |
| `peakingduck` | Scipy Peak Finder | `fluxforge.analysis.peak_finders.ScipyPeakFinder` | ✅ Implemented |
| `SpecKit` | Tikhonov Regularization | `fluxforge.solvers.regularized.tikhonov_solve()` | ✅ Implemented |
| `SpecKit` | L-curve Selection | `fluxforge.solvers.regularized.l_curve_corner()` | ✅ Implemented |
| `SpecKit` | GCV Selection | `fluxforge.solvers.regularized.gcv_select_alpha()` | ✅ Implemented |
| `SpecKit` | Log-Smoothness | `fluxforge.solvers.regularized.log_smoothness_penalty()` | ✅ Implemented |
| `Neutron-Unfolding` | GRAVEL | `fluxforge.solvers.gravel()` | ✅ Already present |
| `Neutron-Unfolding` | MLEM | `fluxforge.solvers.mlem()` | ✅ Already present |

### Benchmark Results (run_comprehensive_benchmark.py)

| Test | Result | Details |
|------|--------|---------|
| Nuclide Database | ✅ PASS | 17 nuclides loaded, half-lives correct to 0.00% |
| Dose Calculations | ✅ PASS | Co-60 dose rate, Eu-152 multi-line, decay curves verified |
| Peak Finding | ✅ PASS | 100% detection rate, WindowPeakFinder found all 5 true peaks |
| Regularized Unfolding | ✅ PASS | Tikhonov 12.7% error, L-curve 8.2% error |
| Spectroscopy Data | ✅ PASS | irrad_spectroscopy example data readable |

---

## H. Hybrid Modeling (MCNP/OpenMC/ALARA Integration)

| ID | Capability | Status | Notes |
|----|------------|--------|-------|
| H1.1 | MCNP Input Parsing | **Implemented** | `fluxforge.io.mcnp.parse_mcnp_input` |
| H1.2 | MCNP Tally Ingestion | **Implemented** | `fluxforge.io.mcnp.read_meshtal_hdf5` (requires h5py) |
| H1.3 | ALARA Input Generation | **Implemented** | `fluxforge.io.alara.ALARAInputGenerator` |
| H1.4 | ALARA Output Parsing | **Implemented** | `fluxforge.io.alara.parse_alara_output` |
| H1.5 | OpenMC Tally Ingestion | **Implemented** | `fluxforge.io.openmc.read_openmc_tally` (requires h5py) |

## A. Analysis & Processing

| ID | Capability | Status | Notes |
|----|------------|--------|-------|
| A1.1 | Peak Fitting | Implemented | `fluxforge.analysis.peak_fitting` |
| A1.2 | Efficiency Calibration | Implemented | `fluxforge.analysis.efficiency` |
| A1.3 | CNF File Support | **Implemented** | `fluxforge.io.cnf.read_cnf_file` (Interface only, requires external tools) |
| A1.4 | k₀-NAA Analysis | **Implemented** | `fluxforge.analysis.k0_naa` |
| A1.5 | SNIP Background | **NEW** | `fluxforge.analysis.peak_finders.snip_background()` |
| A1.6 | Advanced Peak Finders | **NEW** | `WindowPeakFinder`, `ChunkedPeakFinder`, `ScipyPeakFinder` |

## C. Corrections & Physics

| ID | Capability | Status | Notes |
|----|------------|--------|-------|
| C1.1 | Self-Absorption | Implemented | `fluxforge.corrections.self_absorption` |
| C1.2 | Geometry Correction | Implemented | `fluxforge.corrections.geometry` |
| C1.3 | Decay Correction | Implemented | `fluxforge.corrections.decay` |
| C1.4 | Coincidence Summing | **Implemented** | `fluxforge.corrections.coincidence.CoincidenceCorrector` |
| C1.5 | Dose Rate Calculation | **NEW** | `fluxforge.physics.dose.gamma_dose_rate()` |
| C1.6 | Nuclide Database | **NEW** | `fluxforge.physics.nuclides.NuclideDatabase` |

## S. Solvers

| ID | Capability | Status | Notes |
|----|------------|--------|-------|
| S1.1 | GLS Adjustment | Implemented | `fluxforge.solvers.gls_adjust()` |
| S1.2 | GRAVEL | Implemented | `fluxforge.solvers.gravel()` |
| S1.3 | MLEM | Implemented | `fluxforge.solvers.mlem()` |
| S1.4 | MCMC | Implemented | `fluxforge.solvers.mcmc_unfold()` |
| S1.5 | Tikhonov Regularization | **NEW** | `fluxforge.solvers.regularized.tikhonov_solve()` |
| S1.6 | L-curve Parameter Selection | **NEW** | `fluxforge.solvers.regularized.l_curve_corner()` |
| S1.7 | GCV Parameter Selection | **NEW** | `fluxforge.solvers.regularized.gcv_select_alpha()` |

## Completion Status
All identified gaps have been addressed. New capabilities added from testing repos review:
- **Nuclide database** with 17 common calibration/activation nuclides
- **Dose rate calculations** for gamma radiation
- **Advanced peak finders** (SNIP, Window, Chunked, Scipy methods)
- **Regularized solvers** with automatic parameter selection

**Test Results: 146/146 unit tests passing**
