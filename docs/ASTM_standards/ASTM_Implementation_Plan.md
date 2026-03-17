# ASTM Standard Implementation Playbook

This document outlines the standard procedure for implementing new ASTM standards into FluxForge, modeled directly after the successful implementation and verification of **ASTM E261**. 

By following these steps, we ensure that every new mathematical standard is consistently modeled, cleanly integrated into both the CLI and GUI, computationally verified via unit tests, and cross-validated against our legacy QuantumGold (QG) pipelines.

## Step 1: Core Mathematical Module & Dictionary Schema
**Target Location:** `src/fluxforge/analysis/astm_<standard>.py`

1. **Define the Input Data Model:**
   * Create a standardized python dictionary schema (a "plan") that the standard requires (e.g., monitor dimensions, measurement counts, efficiencies, cross-sections, decay constants).
2. **Implement Core Equations:**
   * Write isolated functions for the specific intermediate physics math required by the standard (e.g., specific activity, saturation factors, reaction rates).
3. **Establish the Main API Entrypoint:**
   * Create a primary functional wrapper: `analyze_astm_<standard>_plan(plan_dict: dict) -> dict`.
   * This function should ingest the raw dictionary, run the variables through the standard's formulas, and return a dictionary containing both the echoed inputs and the computed outputs (e.g., End of Irradiation Activities, Fluences).

## Step 2: Unit Testing
**Target Location:** `tests/test_astm_<standard>.py`

1. **Write Core Logic Tests:**
   * Provide hardcoded inputs with known analytical answers (hand-calculated or reference data).
   * Use `pytest` to assert that calculations output intermediate and final values precisely within numerical tolerance.
2. **Test Edge Cases:**
   * Handle missing values or divide-by-zero potentials gracefully (e.g. 0% detector efficiency or 0 live time).

## Step 3: CLI Integration
**Target Location:** `src/fluxforge/cli.py` (or main execution script)

1. **Add Sub-Command Parser:**
   * Ensure `astm-<standard>` is a valid command parsing option.
2. **Connect Execution:**
   * Read the targeted JSON plan file.
   * Route it to `analyze_astm_<standard>_plan`.
   * Save the evaluated output dictionaries to `json` via standard standard output or designated file paths.

## Step 4: Graphical User Interface (GUI) Wiring
**Target Location:** `src/fluxforge_gui/app.py`

1. **Add Form/Preview Callbacks:**
   * Implement UI builder functions formatted like `_load_astm_<standard>_preview` to build tables showcasing the standard's required variables before execution.
2. **Add Execution Callbacks:**
   * Implement `_run_astm_<standard>`.
   * Verify it safely fetches the relevant files from the user interface context, packages them into subprocess CLI arguments or python API calls, and handles the output without throwing UI exceptions.

## Step 5: Raw Data Parity & QG Validation Scripting
**Target Location:** `examples/RAFM_irradiation/compare_astm_<standard>_to_qg.py`

*Note: This guarantees the new standard maps physically and functionally to the results obtained previously from legacy workflows like QuantumGold, starting securely from our RAW spectral pipeline.*
1. **Load Raw Spectrum and QG Reference Data:**
   * Load raw `.ASC` spectra from the `raw_gamma_spec` directories.
   * Load the matching processed `.txt` or `.ANS` reports from `QG_processed_gamma_data` for baseline truth.
2. **Raw Data Processing:**
   * Run the raw `.ASC` files through FluxForge's existing spectra peak analysis tools to determine observed net counts, background subtraction, and line measurements.
3. **Run the Standard API:**
   * Pass the extracted measurements from the raw data pipeline directly into `analyze_astm_<standard>_plan`.
4. **Compare Outputs:**
   * Compare the derived analytical outputs against the identically mapped parameters located in the QG processed files.
   * Calculate relative differences (`abs(ASTM_from_Raw - QG) / QG * 100`) and visually report parity.