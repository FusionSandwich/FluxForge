# [Core] Extend GammaSpectrum with per-channel uncertainty

## Goal
Store and serialize per-channel uncertainties for downstream propagation.

## Scope
- Add optional `counts_uncertainty` to `GammaSpectrum`.
- Default to `sqrt(max(counts,0))` if not supplied.
- Include in `to_dict`/`from_dict` and spectrum schema.

## Acceptance Criteria
- Artifact read/write preserves `counts_uncertainty`.
- Existing spectrum operations remain backward compatible.
- Tests verify default and explicit uncertainty behavior.
