# Issue 24: Line-Level Efficiency and Activity Diagnostics

## Problem
FluxForge still has honest measurement-time activity parity failures for several RAFM flux-wire and RAFM3/RAFM4 spectra. The sample-level pass/fail tables are not enough to separate:
- peak-area mismatch
- detector-efficiency bias
- branching-ratio / gamma-library mismatch
- processed-file inconsistency

## Required work
1. Persist one line-diagnostics CSV per matched raw/QG sample.
2. Persist one aggregate `results/tables/line_diagnostics.csv` file for the full workflow run.
3. Include at minimum:
   - sample ID
   - isotope
   - line energy
   - QG net counts / uncertainty
   - FluxForge net counts / uncertainty
   - relative count error and En-score
   - QG line activity at measurement time
   - FluxForge line activity at measurement time
   - QG-implied efficiency
   - FluxForge efficiency used
   - QG `rad_int`
   - bundled branching ratio and uncertainty
   - diagnostic bucket
4. Use the diagnostic bucket only for reporting. It must not affect FluxForge runtime analysis logic.

## Acceptance
- Every matched raw/QG RAFM sample has a line-diagnostics CSV.
- The per-spectrum comparison report references the same underlying line-level findings.
- The workflow summary can be used to count diagnostic buckets across the run.
