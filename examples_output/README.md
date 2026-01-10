# FluxForge Demo Outputs

This directory contains the results of the `run_full_demo.py` demonstration script, which processes real experimental data using the FluxForge toolkit.

## Contents

### 1. Gamma Spectroscopy Analysis

#### `spe_analysis_results.json`
**Source:** `testing/gamma_spec_analysis/test_data/Co_60_raised_1.Spe` (Ortec SPE format)  
**Analysis:**
- Parsed Co-60 spectrum with calibration
- Applied **Coincidence Summing Corrections**
- Demonstrates `fluxforge.io.spe` and `fluxforge.corrections`

#### `asc_analysis_results.json`
**Source:** `src/fluxforge/examples/flux_wire/Co-Cd-RAFM-1_25cm.ASC` (Genie-2000 ASC format)  
**Analysis:**
- Parsed flux wire measurement
- Identified peaks in the Co-60 region
- Demonstrates `fluxforge.io.genie`

---

### 2. Hybrid Modeling (MCNP/ALARA)

#### `mcnp_materials.json`
**Source:** `test_v3_vit_J/whale_J_core_clean_loc.i` (MCNP Input)  
**Analysis:**
- Extracted **2012 materials** from full-core MCNP model (~60k lines)
- Demonstrates `fluxforge.io.mcnp` robustness

#### `generated_alara.inp`
**Source:** Programmatically generated  
**Analysis:**
- Created ALARA activation input file

---

### 3. Flux Unfolding (GLS)

#### `unfolded_spectrum.png` & `unfolded_results.json`
**Source:** `MCNP_ALARA_Workflow/spectrum_vit_j_TEST.csv` (MCNP Vitamin-J spectrum)  
**Analysis:**
- **Generalized Least Squares** adjustment of prior MCNP spectrum
- Demonstrates `fluxforge.solvers.gls`
- Plot shows Prior vs Adjusted spectrum

---

### 4. Iterative Unfolding Benchmark (GRAVEL/MLEM)

#### `unfolding_benchmark_comparison.png`
Visual comparison of FluxForge vs Reference implementations against Time-of-Flight ground truth.

#### `unfolding_implementation_diff.png`
Difference plot showing agreement between FluxForge and Reference.

#### `unfolding_benchmark_results.json`
Numerical validation results:
- **GRAVEL Error vs ToF:** Reference=8.23, FluxForge=8.47
- **MLEM Error vs ToF:** Reference=8.10, FluxForge=8.06
- **Agreement RMSE:** GRAVEL=0.000158, MLEM=0.000114

---

## Validation Summary

| Capability | Status | Data Source |
|------------|--------|-------------|
| SPE Parsing | ✅ Validated | Co-60 experimental data |
| ASC Parsing | ✅ Validated | Flux wire measurement |
| MCNP Material Extraction | ✅ Validated | Production MCNP model |
| ALARA Generation | ✅ Validated | Generated programmatically |
| GLS Unfolding | ✅ Validated | MCNP spectrum |
| GRAVEL Unfolding | ✅ Validated | Benchmark vs reference |
| MLEM Unfolding | ✅ Validated | Benchmark vs reference |
