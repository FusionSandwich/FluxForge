# Issue 28: Crowded-Region Isotope Mismatch Near 1189 keV

## Problem
RAFM4 comparison reports show a repeated isotope mismatch where a QG `Ta182` line near `1189 keV` is being matched by FluxForge as `Tb154m`. This is a real identification issue in a crowded region and should be handled separately from the detector-efficiency problem.

## Evidence
- `results/reports/RAFM4-A_15dEOI_comparison.txt`
- `results/reports/RAFM4-B_15dEOI_comparison.txt`
- `results/reports/RAFM4-N_15dEOI_comparison.txt`
- `results/tables/line_diagnostics.csv`

## Required work
1. Review the bundled RAFM gamma library around `1180-1195 keV`.
2. Inspect the annotated spectrum plots for the RAFM4 `15d` spectra.
3. Tighten identification logic in crowded regions without using QG isotope selections as runtime inputs.

## Acceptance
- The repeated `Ta182` vs `Tb154m` misidentification is either corrected or explicitly justified by the line library and peak-fit evidence.
