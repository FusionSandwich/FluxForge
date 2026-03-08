# Audit RAFM Peak and Activity Uncertainty Propagation

## Problem
Background-adjusted targeted fits were under-reporting uncertainty when the Gaussian fit uncertainty was smaller than the propagated counting/background uncertainty.

## Current mitigation
- Keep the propagated ROI uncertainty when it exceeds the fit uncertainty.
- Preserve subtraction uncertainty channel-by-channel in the background-adjusted spectrum.
- Suppress overflowed EOI back-corrections for extremely short-lived isotopes counted long after irradiation.

## Remaining work
- Compare FluxForge line-by-line net-count uncertainties against QG for the worst RAFM3/RAFM4 parity failures.
- Review whether efficiency and emission-probability uncertainty assumptions are still too optimistic.
- Document any sample-specific systematic uncertainty terms that are missing from the current workflow.

## Acceptance
- Line and isotope uncertainty discrepancies are documented for the remaining failing samples.
- No targeted-fit path can silently report smaller uncertainty than the propagated subtraction/statistical term.
