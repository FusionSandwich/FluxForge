# Issue 25: QG Internal Consistency Audit

## Problem
Some processed QG exports contain header activities that do not agree with the set of per-line activities in the same file. `Ti` wire `Sc48` is the clearest current example. Without recording this explicitly, the parity report makes FluxForge look wrong even when the processed file is internally inconsistent.

## Required work
1. For each matched processed file, compute a per-isotope audit table comparing:
   - processed header activity
   - min/max/mean of processed per-line activities
   - maximum relative deviation from the header
2. Flag isotopes whose per-line activity spread or header mismatch exceeds a conservative threshold.
3. Write:
   - one per-sample CSV under `results/tables/qg_internal_consistency/`
   - one aggregate CSV under `results/tables/qg_internal_consistency.csv`
4. Add a dedicated section to the per-sample comparison text report.

## Acceptance
- `Sc48` in the Ti processed files is explicitly flagged by the audit output.
- The audit remains comparison-only and does not feed back into FluxForge analysis logic.
