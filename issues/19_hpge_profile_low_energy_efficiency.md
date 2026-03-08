# Refine Low-Energy HPGe Profile Efficiency for RAFM 25 cm Data

## Problem
The shared `rafm_25cm` profile no longer shows the large global efficiency bias it had before the detector-model attenuation fix, but low-energy lines are still not matching QG measurement-time activities closely enough.

## Current evidence
- Interpreting the detector-model attenuation terms with mass attenuation coefficients removed the previous factor-of-2 to factor-of-3 high-energy efficiency error.
- High-energy lines (`~900-1330 keV`) are now much closer to QG than before.
- Low-energy and mid-energy lines (`~159-725 keV`) still show systematic activity mismatch in flux-wire validation.

## Remaining work
- Review whether the detector-model absorption term needs a more appropriate Ge interaction component than the current total-mass-coefficient approximation.
- Compare against FluxForge's semi-empirical HPGe efficiency tooling and local validation material before changing the profile calculation again.
- Re-run the RAFM and flux-wire validation bundle after each model update and capture the delta in the reports.

## Acceptance
- The shared `rafm_25cm` profile produces materially better low-energy measurement-time parity without using QG outputs to set isotope targets or tune per-spectrum logic.
- The efficiency-model interpretation is documented in the RAFM example notes.
