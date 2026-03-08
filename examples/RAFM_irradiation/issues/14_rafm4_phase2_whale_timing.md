# RAFM4 15-Day Timing Must Stay Anchored to Whale-Tube Phase 2

## Problem
`RAFM4-*_15dEOI` validation depends on using the 2-hour whale-tube irradiation end time, not the earlier rabbit-tube timing used for RAFM3 short-cooldown spectra.

## Required changes
- Keep `RAFM4-*_15dEOI` timing tied to `sample_schedules.phase2`.
- Record the phase explicitly in workflow artifacts as `irradiation_phase = phase2_whale_tube`.
- Keep the per-sample comparison reports and docs explicit about this distinction.

## Acceptance
- `RAFM4-*_15dEOI` artifacts show `phase2_whale_tube` in timing metadata.
- Validation docs in `examples/RAFM_irradiation/` state that RAFM4 15-day spectra are post-whale-tube counts.
