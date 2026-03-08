# Generic flux-wire comparison-window scoring

## Problem
The automatic flux-wire QG count-parity path still had a small set of lines where the comparison ROI support was wrong even though the underlying peak was found correctly. The two worst examples were:
- broad `Co60` lines, where the comparison gross window was too narrow
- weak `Ti Sc48 @ 175 keV`, where the comparison gross/net window was too broad or shifted

## Constraint
This must stay generic. We do not want sample-specific or isotope-specific branches for one RAFM case.

## Implemented direction
- add a compact comparison-window selector that searches candidate ROIs inside the standard `32 channel` capture range
- score candidates by:
  - local-background net area
  - edge-to-background continuity penalty
  - asymmetry penalty
  - width penalty
- keep the existing broad-window path for strong peaks only
- allow the broad-window path to override the compact window only when it preserves the compact-window net area within a small tolerance

## Why this is acceptable
- one generic rule is applied across the full flux-wire fixture set
- it improves weak-line support without hardcoding any one wire or isotope
- it keeps the activity path separate from the raw QG `GROSS/NET` comparison path

## Current outcome
- `Ti-RAFM-1a Sc48 @ 175 keV` improved from roughly `+33% gross / -29% net` to about `+11% gross / -4% net`
- `Ti-RAFM-1a Sc46 @ 889 keV` improved from roughly `+19% net` error to about `+8%`
- broad `Co` and `Sc` lines still use the broader support when that support does not distort net area

## Remaining work
- reduce the remaining `Co-Cd 1332 keV` gross/net gap without regressing the improved Ti lines
- re-audit the activity-conversion path only after the automatic count path is stable
