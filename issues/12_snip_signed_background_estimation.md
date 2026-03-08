# [Analysis] Use signed-count SNIP handling for background-subtracted spectra

## Goal
Keep measured-background-subtracted spectra signed for accounting and uncertainty propagation, while avoiding pre-SNIP clipping warnings in RAFM and flux-wire workflows.

## Scope
- Shift SNIP input by `offset = max(0, -min(counts))`.
- Run SNIP on the shifted working copy.
- Shift the estimated background back and floor it at `0.0`.
- Continue computing ROI integrals and uncertainties from the original signed spectrum.

## Acceptance Criteria
- RAFM background-subtracted SNIP workflows stop emitting the prior negative-channel clipping warning.
- Stored spectra still retain signed counts after subtraction.
- Peak/ROI uncertainty calculations continue to use propagated subtraction variance.
