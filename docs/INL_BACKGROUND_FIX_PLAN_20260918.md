# INL background-grid investigation and implementation plan

Status: OPEN. Owner: G1 scientific correctness; blocks G2 measured-data reduction and relevant G3 CLI/GUI qualification. Finish and verify each step before advancing it. This issue is not the only remaining handoff gate.

Current implementation: see [the implementation receipt](INL_BACKGROUND_IMPLEMENTATION_RECEIPT.md)
and [covariance contract](BACKGROUND_COVARIANCE_CONTRACT.md). The investigation
below records the starting state. Storage, conservative subtraction and supported
ROI/Gaussian consumers are implemented; real CLI/native GUI success receipts
exist. Acceptance remains open for legacy QG reaction reduction, shared covariance,
broad GUI failures/crash investigation and physical/all-method INL qualification.

## Confirmed investigation

- All ten bundled raw samples in `tests/data/flux_wires/raw` have 8192 channels and energy coefficients `[0.541, 0.498, 2.605e-7]`; background coefficients are `[-1.502, 0.4991, 2.239e-7]`.
- Background minus sample channel energies range from -2.043 to +4.511515 keV. Matching channel counts do not imply matching energy bins.
- `tests/test_flux_unfolding_10bin.py` fails in `analysis/spectrum_math.py` before reaching unfolding. Do not hide this with xfail, skipping background, calibration replacement or relaxed grid tolerance.
- Two new real-file interface checks pass in `tests/test_inl_background_interfaces.py`: CLI ingest exits unsuccessfully without writing corrected artifacts; the native GUI labels background as not applied and displays foreground counts. These prove rejection, not successful analysis.
- `analysis/histogram_rebin.py` already supplies sparse overlap weights, propagated covariance, coverage and discarded counts. It is not integrated through spectrum/session storage and downstream analysis.
- Existing broad source receipts: 39 native GUI passes (one RMLE conditioning warning), 173 backend passes, one failing INL regression. No claim of all-method INL acceptance or complete manual GUI testing.

Evidence and precise scope: [validation receipt](VALIDATION_RESUME_20260918.md), [background follow-up](BACKGROUND_FOLLOWUP_20260917.md). Local logs, JUnit XML and per-file grid diagnostics: `D:/FluxForge-validation-resume-20260918`.

## Ordered implementation and acceptance

1. [ ] Reproduce current failure and both interface rejection checks; record branch, revision, dirty diff and fixture hashes in a new evidence directory. Trace actual calibration and normalization paths in readers, CLI and GUI. Verify bin-edge conventions and background provenance; document unresolved physical suitability separately.
2. [ ] Define the covariance and coverage contract before integration. Specify how channel boundaries map to energies, monotonicity checks, strict/partial coverage behavior and cropped counts. Define signed-count handling, sparse representation, symmetry/positive-semidefinite validation, diagonal uncertainty consistency, schema compatibility and any scale-uncertainty assumptions.
3. [ ] Implement spectrum and serialization support with independent roundtrip tests, including sessions and CLI artifacts. Legacy diagonal-only data must remain valid when appropriate. No reader, copy, export or consumer may silently discard supplied correlations; unsupported paths must reject explicitly.
4. [ ] Integrate conservative measured-background subtraction using the existing primitive. For independent sample/background counts and fixed scale a, verify net counts s - a Wb and covariance Cs + a² W Cb Wᵀ against hand-calculated cases. Preserve originals and normalization metadata. Keep identical-grid behavior correct and expose coverage decisions.
5. [ ] Propagate covariance through affected ROI and fitting consumers and derived activity/rate uncertainties. Verify ROI variance wᵀ C w, covariance-aware fit weighting, singular cases and no double counting. Inventory shared-background correlations between samples/peaks; implement required support or retain explicit acceptance blockers. Do not describe per-spectrum covariance alone as full shared-covariance acceptance.
6. [ ] Exercise real-file CLI and native GUI success paths after integration, including selection, subtraction, error/recovery behavior, save/reopen and export. Update rejection tests only where support is implemented; retain invalid-grid/coverage rejection tests. Confirm displayed and exported values agree and document any diagonal-only export limitations.
7. [ ] Rerun the INL raw/processed regression and record every subsequent blocker. Enumerate every supported unfolding method from source and the feature matrix; validate each applicable method on traceable INL inputs against independent expectations, with tolerances, conditioning, residuals and uncertainty evidence. A run completing or returning the expected number of bins is insufficient. Keep physical calibration, efficiency, timing and nuclear-data gates open until independently qualified.
8. [ ] Run affected numerical, reader, schema, session, CLI and native GUI regressions. Build and verify a fresh installed package outside the checkout if claiming installed qualification. Update the recovery checklist, feature matrix and GUI ledger with exact evidence. Mark this issue complete only when all required acceptance checks are satisfied; do not promote G1/G2/G3 solely because this issue is fixed.

## Boundaries

Preserve existing working changes and original measurements. Keep original handoff copies untracked. Do not perform reactor/transport runs as part of this software fix. Source execution, installed-package execution, automated GUI tests and manual usability checks must be reported separately. Record remaining limitations instead of claiming certainty unsupported by tests.

New-chat starting instructions: [continuation prompt](INL_BACKGROUND_FIX_NEW_CHAT_PROMPT.md).
