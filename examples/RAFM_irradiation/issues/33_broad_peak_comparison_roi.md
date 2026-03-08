# Broad Peak Comparison ROI

## Problem
Using fit-sized ROIs for flux-wire QG count parity under-counted broad strong peaks, especially:
- `Co-Cd-RAFM-1_25cm Co60`
- `Sc-RAFM-1_25cm Sc46`

The fit width was too narrow to represent the broader shoulder/base used by QG for its reported `GROSS/NET` values.

## Change
- For strong peaks only, FluxForge now expands the count-comparison ROI using five-point-smoothed local minima inside the `32 channel` capture range.
- The background-adjusted activity path is unchanged.

## Outcome
- `Sc-RAFM-1_25cm` gross/net count parity is now close to QG.
- `Co-Cd-RAFM-1_25cm` improved substantially but is not yet fully matched.
- Weak-line Ti `Sc48 @ 175 keV` is unaffected and remains a separate problem.
