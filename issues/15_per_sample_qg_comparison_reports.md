# Per-Sample Raw-vs-QG Comparison Reports and Plots

## Problem
The earlier workflow produced aggregate comparison plots, which made it difficult to debug one raw spectrum against its own QG processed result.

## Required changes
- Produce one comparison plot per matched raw/QG pair.
- Produce one text report per raw spectrum.
- Each report should list:
  - QG peaks not identified by FluxForge
  - matched energies with isotope mismatches
  - missing QG nuclides
  - parity failures
  - remaining unidentified FluxForge peaks
  - FluxForge-only identified peaks
- Remove or rename stale global plot names that conflict with the per-sample workflow.

## Acceptance
- `results/plots/comparisons/<sample>_vs_qg.png` exists for each matched pair.
- `results/reports/<sample>_comparison.txt` exists for every analyzed raw spectrum.
