# Luna activation uncertainty audit

## Accepted findings

- Absolute count sigma is propagated through the activity derivative and is then propagated through E261/E262 rate calculations. ASTME262's clipped Cd-subtracted thermal rate exposes the unconstrained result, clipping flag, and the stated pre-clipping count-sigma interpretation. The standard-comparison path keeps unknown, standard, and reference-field terms separate.
- Generic activity-to-rate conversion leaves rate uncertainty unavailable when activity sigma is absent. The artifact validator requires an unavailable scope, reason, and `scientific_admission=false`; unfolding and scientific plot consumers reject unavailable uncertainty. The master example supplies weighted activity sigma to the rate calculation.
- The uncertainty-budget RSS calculations use `hypot` for improved numerical stability.
- RAFM replay reports 12 QG reports, 19 header observations, and 40 line observations; maximum relative line-sigma error is 1.7763568394002505e-15. All 19 recorded input hashes and 247 runtime source hashes in `rafm_final.json` match current files.
- Final focused suite: 227 passed, 0 failed, 0 errors, 0 skipped; 17 existing datetime deprecation warnings. Luna's independently run canonical batch passed 61 tests and integration batch passed 48 tests after the plotting consumer changes.

## Rejected findings

- The earlier `rafm_replay.json` hash mismatch was resolved by the final replay: `rafm_final.json` has no input or runtime hash mismatches.
- No correctness defect was found in the reviewed uncertainty propagation or unavailable-uncertainty handling.

## Remaining limitations

The Poisson fallback for absent `net_counts_unc` is explicitly an assumption about net counts; it does not model background-subtraction uncertainty. Reported budgets are conditional and exclude calibration, nuclear-data, timing, material, and other stated inputs; they are marked `scientific_admission=false`. The RAFM QG comparison validates propagation behavior only and is not physical truth or qualification evidence.
