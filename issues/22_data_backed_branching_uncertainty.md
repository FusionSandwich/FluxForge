# Use Data-Backed Branching-Ratio Uncertainty in Flux-Wire Activity Calculations

## Problem
FluxForge was using a fixed global emission-probability uncertainty assumption in the flux-wire activity calculation path. That is not defensible when committed decay data already exists locally.

## Current mitigation
- Bundled RAFM decay data now includes `intensity_uncertainty` per gamma line.
- Flux-wire activity calculations now use those committed line-by-line uncertainties instead of a fixed `1%` branch-ratio guess.
- The hidden global `5%` efficiency uncertainty term was also removed; efficiency uncertainty is now explicit/configurable rather than invented at runtime.

## Remaining work
- Confirm the bundled actigamma-derived uncertainty values against the laboratory/QG database assumptions for the RAFM validation set.
- Decide whether detector-profile files should carry explicit efficiency uncertainty metadata for RAFM 25 cm counting geometry.
- Re-run the remaining parity failures and document where uncertainty discrepancies persist after the data-backed update.

## Acceptance
- Flux-wire line-activity uncertainty uses committed per-line decay data.
- No global hardcoded branch-ratio uncertainty remains in the activity calculation path.
- Any nonzero efficiency uncertainty included in results must come from explicit profile/user metadata.
