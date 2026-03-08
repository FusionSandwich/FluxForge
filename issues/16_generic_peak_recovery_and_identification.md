# Improve Generic RAFM Peak Recovery and Identification

## Problem
Generic RAFM spectra are still missing some QG peaks and leaving some detected peaks unidentified.

## Current mitigation
- Use exploratory peak search plus targeted recovery from the committed RAFM gamma library.
- Deduplicate near-duplicate gamma lines in the library before targeted recovery.
- Preserve unidentified FluxForge peaks in per-sample reports so they can be reviewed.

## Remaining work
- Audit recurring misses in RAFM3 and RAFM4 discrepancy reports.
- Tighten line matching and overlapping-peak handling for crowded low-energy regions.
- Review FluxForge-only IDs such as `Sb124`, `Co58`, and `V52` in late-time RAFM spectra.

## Acceptance
- Per-sample reports show fewer missing QG peaks and fewer unidentified FluxForge peaks in known problem spectra.
