# Audit QG Sc48 Export Inconsistency in Ti Flux-Wire Files

## Problem
The QG processed Ti-wire exports appear internally inconsistent for `Sc48`: the nuclide header activity and some per-line activity values disagree by orders of magnitude.

## Current evidence
- `Ti-RAFM-1_25cm.txt` reports `Sc48` header activity `0.424 uCi`.
- The same file contains per-line `Sc48` activities around `4.70E-03`, `7.52E-03`, `0.646`, and `0.691 uCi`, which are not self-consistent.
- The current FluxForge peak matching finds the expected `Sc48` lines, but activity parity still fails badly because the exported QG values do not agree internally.

## Remaining work
- Check whether the QG text export is truncating scientific notation for the `983.5 keV` and `1312.1 keV` `Sc48` lines.
- Compare the QG header activity, per-line activities, and reported line intensities for all Ti-wire processed files.
- Decide which QG field is the authoritative gold-standard comparator for `Sc48`.

## Acceptance
- The `Sc48` comparison rule is documented and reproducible.
- Ti-wire validation is no longer blocked by ambiguous QG export formatting.
