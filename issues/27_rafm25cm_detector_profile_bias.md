# Issue 27: RAFM 25 cm Detector Profile Bias

## Problem
The new line-diagnostics tables show a structured efficiency/activity conversion bias across many matched lines even when net-count parity is acceptable. Representative efficiency ratios (`FluxForge efficiency / QG-implied efficiency`) are often:
- around `0.58-0.65` in the low-energy region
- around `0.75-0.90` in the mid/high-energy region

This means the remaining measurement-time activity parity problem is not mainly a missing-peak problem.

## Evidence
- `results/tables/line_diagnostics.csv`
- `results/tables/line_diagnostics/Ti-RAFM-1_25cm_line_diagnostics.csv`
- `results/tables/line_diagnostics/RAFM4-A_15dEOI_line_diagnostics.csv`

## Required work
1. Audit the `rafm_25cm` efficiency implementation against the actual detector model intended by the QG/LabSOCS export.
2. Keep the audit independent from QG result values during runtime analysis.
3. Use the line-diagnostics tables only to validate candidate detector-model corrections after implementation.

## Acceptance
- The structured low/mid-energy bias is materially reduced.
- Flux-wire and RAFM measurement-time activity parity improves without using processed outputs to set runtime parameters.
