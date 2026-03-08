# Issue 29: QG Peak Gross/Net/Energy Side-by-Side Comparison

## Problem
The RAFM validation workflow needs direct peak-by-peak comparison against the QG processed outputs for:
- gross counts
- net counts
- peak energy
- line activity at measurement time

This has to stay comparison-only. QG must not be used to set FluxForge runtime peak-selection or activity-conversion logic.

## Required work
1. Persist gross-count comparison fields in the per-sample and aggregate comparison tables.
2. Ensure FluxForge `gross_counts` means raw ROI gross counts for QG parity, not background-adjusted gross.
3. Keep `background_adjusted_gross_counts` available separately for FluxForge internal diagnostics.
4. Add report sections and plots that expose gross-count parity failures independently from net-count and activity failures.

## Acceptance
- Every matched line comparison includes gross, net, energy, and activity side-by-side.
- Gross-count comparison uses raw counts from FluxForge and the QG gross column directly.
- The workflow still does not use QG data to influence the raw analysis path.
