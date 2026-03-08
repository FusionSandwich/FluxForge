# [Analysis] Integrate measured background into flux-wire analysis

## Goal
Apply shared measured background to raw flux-wire/RAFM analysis before peak extraction.

## Scope
- Add background parameters to flux-wire analysis entrypoints.
- Use propagated per-channel uncertainties in ROI uncertainty estimation.
- Preserve compatibility with targeted parity workflows.

## Acceptance Criteria
- Flux-wire analysis supports shared background input and scale modes.
- ROI uncertainty reflects subtraction variance contribution.
- Existing parity workflows still execute.
