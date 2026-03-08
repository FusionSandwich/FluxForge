# Flux-Wire Gross Count Protocol Gap

## Problem
After switching flux-wire count parity to a QG-style local ROI, several lines still show large `GROSS` disagreement even when `NET` is much closer.

Current examples:
- `Co-Cd-RAFM-1_25cm Co60 @ 1173.13 keV`: gross about `-28.5%`, net about `+0.3%`
- `Co-Cd-RAFM-1_25cm Co60 @ 1332.44 keV`: gross about `-23.4%`, net about `-3.7%`
- `Sc-RAFM-1_25cm Sc46 @ 889.36 keV`: gross about `-16.5%`, net about `-12.5%`

## Interpretation
QG `GROSS` is not behaving like a simple fixed `4.0 * FWHM` raw ROI for every flux-wire line. There is still a protocol mismatch in how QG defines or stores gross counts for some regions.

## Remaining work
- Audit whether QG gross counts are tied to a larger fit region, asymmetric ROI, or another internal rule.
- Keep count-parity reporting explicit so this mismatch is not hidden inside activity errors.
