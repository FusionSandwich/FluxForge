# Activation uncertainty issue batch

Repository: FusionSandwich/FluxForge. The supplied workspace is FluxForge, not ParaStell. This branch builds on the latest pushed analysis branch at `6eb68a0`; it does not change the other agent's checkout or unpublished work.

## Changes and issue scope

Addresses concrete remaining defects in #199 and the activity-to-rate propagation portion of #25. Activity in Bq no longer supplies an invented Poisson error: generic conversion accepts `activity_uncertainty_bq`, propagates it linearly, and records missing uncertainty as unavailable. Review exports preserve zero activity and remove a hidden buildup-factor floor. JSON preserves null uncertainty, and unfolding, scientific plots, and rate import require supplied valid uncertainty.

Gamma-line and E261/E262 paths accept `net_counts_unc`, including independent Cd/unknown/standard fields. Absolute derivatives preserve finite uncertainty for zero counts, clipped Cd subtraction, and a zero unknown in standard comparison. Per-row named conditional components disclose omitted inventory, calibration, timing, nuclear and shared covariance terms. The legacy missing-count-sigma approximation is explicitly an assumed Poisson net-count model with background uncertainty unavailable. Library k0 uncertainty remains an unqualified proxy unless explicit cross-section uncertainty is supplied. Cd sigma describes propagation before clipping, not a censored posterior or detection limit.

Existing source-covariance budgets retain signed shared effects; stable norms avoid overflow/underflow of otherwise representable uncertainties. No measured physical covariance is invented. These changes do not close the full scientific qualification requirements of #199 or #25.

## Verification

The final combined regression run passes **227 tests** covering ASTM, independent analytic derivatives, uncertainty contracts, finite/shared covariance, activation, the Fe-Cd example, the complete pipeline, CLI, interoperability and scientific plots. Exact test identities and hashes are in `validation_receipt.json`; output is in `final_tests.xml` and `final_tests.log`. Existing datetime deprecation warnings remain. An expensive whole-repository run was stopped before completion; it is not counted as acceptance.

Independent ASTM antagonist review accepted 24 analytic/adversarial cases; contract and numerical reviewers supplied the canonical adversarial tests. Luna independently reran canonical and integration checks and reviewed the final source/input hash scope; see `antagonist/luna/REVIEW.md`.

## RAFM results and limits

The final replay uses **12 processed QG wire reports, 19 nuclide headers, and 40 supplied peak errors**. Generic and E261/E262 propagated errors agree with independent chronological activation/count-integral calculations within **1.78e-15 maximum relative error**. Supplied peak errors range approximately 1.07–10.46 times sqrt(counts); Co-Cd 1173 keV is 4.91 times sqrt(counts). Three signed shared-covariance unit-rescaling probes pass. Source data are unchanged and hashed.

`rafm_final.json` is the accepted frozen replay. `rafm_replay.json` is the initial replay before the consumer migration/source freeze and its runtime hashes are historical. All final runtime hashes and 19 input hashes are verified in the receipt. Reproduce with `tools/validate_activation_uncertainty_rafm.py --root . --out <fresh-file.json>` and the isolated `src` on PYTHONPATH.

These are conditional propagation diagnostics. Unit efficiency, gamma yield, inventory and cross section in the line tests are deliberately fixed, not measured sample activities. QG header errors have unknown total composition and are never combined again with their constituent line errors. Some filename-to-schedule joins require an explicitly labelled common-phase assumption. The frozen 13x20 unfolding response has a diagonal input sigma vector, not a qualified full physical covariance. Missing calibration, nuclear, material and timing uncertainty remain open; the other agent's source-record qualification work was left untouched.
