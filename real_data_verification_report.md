# Real Data Verification Report

## Overview
This report documents the verification of FluxForge capabilities using real experimental gamma spectroscopy data found within the repository.

## Data Sources
*   **Co-60 Spectrum**: `testing/gamma_spec_analysis/test_data/Co_60_raised_1.Spe`
*   **Y-88 Spectrum**: `testing/gamma_spec_analysis/test_data/Y_88_raised_1.Spe`

## Verification Results

### 1. Spectrum I/O
*   **Status**: Verified
*   **Details**: Successfully parsed SPE files, including metadata (Live Time) and spectral data.
*   **Fixes**: Updated `fluxforge.io.spe` to handle non-numeric units (e.g., "keV") in calibration sections.

### 2. Peak Fitting
*   **Status**: Verified
*   **Details**:
    *   **Co-60**: Automatically identified the 1173 keV and 1332 keV doublet.
        *   1173 keV: Centroid ~3210 ch, FWHM ~5.2 ch
        *   1332 keV: Centroid ~3645 ch, FWHM ~5.5 ch
    *   **Calibration**: Derived `E = 0.3656*ch - 0.18 keV`, which is physically consistent.
    *   **Y-88**: Successfully identified the 1836 keV peak.

### 3. Coincidence Summing Correction
*   **Status**: Verified
*   **Details**:
    *   Applied `CoincidenceCorrector` to Co-60 and Y-88 peaks.
    *   Calculated correction factors ~1.02-1.03 (2-3% correction), which is typical for close geometries.
    *   Demonstrated integration of efficiency curves with correction logic.

## Conclusion
The FluxForge codebase has been validated against real experimental data. The core analysis pipeline (Load -> Find Peaks -> Fit -> Correct) is functional and robust.
