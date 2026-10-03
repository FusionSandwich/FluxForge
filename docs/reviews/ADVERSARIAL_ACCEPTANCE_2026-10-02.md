# Adversarial acceptance follow-up — 2026-10-02

The audited software defects have fixes and regression evidence. **Qualified
UWNR/INL scientific acceptance remains false.** The independent full-corpus
replay completed, but 22 samples failed the existing thresholds and three had
insufficient comparison evidence. Successful software tests do not supersede
those failures.

## Integration and verification scope

The local integration branch is `codex/adversarial-acceptance-20261002`. It starts
at `registry-uncertainty-qualification`, frozen at
`1148e18a53a156d74029e84ed528e5fcd9cfd263`, and merges:

- `proposal/run-readiness-manifest-preflight-20260925` at `4ac58c9`, including
  `JS/hpge-03-calibration-efficiency` at `cd9c055`;
- `JS/issue-audit-9-up` at `50e9635426cc7336bb923ab69500255aa7d36278`.

All 32 remote branches and tags were fetched and inventoried. This work tests
the integrated feature selection above; it does not claim every historical
branch has been tested or merged. The integration is published in
[draft PR #216](https://github.com/FusionSandwich/FluxForge/pull/216). The PR
retains the scientific blockers below; publication does not imply acceptance.

Across 2,026 distinct collected test cases, consolidated terminal outcomes are
**2,009 passed, 17 skipped, zero failed, and zero missing**. Core, GUI, optional
ML, and focused repair checks ran in separate processes/environments. This is
case-level consolidated evidence, not one monolithic final-commit invocation.
Run receipts retain their individual commits and superseded failures.

The main environment used Python 3.12.14. TensorFlow-dependent checks ran with
Python 3.11.16 and TensorFlow 2.15.1 on the CPU: 24 passed, two CUDA-specific
checks skipped. Ten external solver parity cases ran against frozen upstream
checkouts. All 304 GUI cases received terminal outcomes through fresh process
batches; the final consolidated suite retains one unavailable legacy desktop
dependency skip. These automated checks do not establish manual interactive
or hardware-GPU acceptance.

## Implemented fixes

| Audit finding | Change and evidence |
| --- | --- |
| Background interpolation changed integrated counts and ROI variance | Replaced count-height interpolation with a sparse bin-overlap transformation `W`. Counts propagate as `W c`; covariance as `W C Wᵀ`. Midpoint-derived bin edges and overlap-only coverage are explicit. Fifteen independent histogram contracts cover split/merge, nonuniform and narrowly shifted grids, partial coverage, correlated source covariance, invalid axes, signed serialization, and ROI/sideband covariance. |
| Invalid normalization and lost explicit energy axes | Integrated finite, nonnegative scale and positive acquisition-time validation; preserved the explicit sample axis. Identity optimization requires exactly equal energy axes. |
| Inconsistent workspace subtraction and uncertainty | The canonical workspace uses the same scientific transformation. Signed counts and sparse covariance survive session save/reload; ROI and sideband weights retain cross-covariance; peak fits receive the covariance. Display/search copies can be nonnegative without changing scientific counts. |
| Full sample replay stalled in Hypermet fitting | Fit only enabled Hypermet parameters, retain zero covariance for fixed terms, use active-parameter degrees of freedom, and enforce an explicit evaluation budget. Hypermet is restricted to applicable unambiguous low-energy singlets. Both full 29-spectrum replays now finish. |
| Installed profile could not find its background | Packaged the measured example background as a package resource, with a source-tree fallback only when present. An isolated wheel passed 33 package/registry checks outside the checkout, plus one real CLI ingestion using the packaged default background and confirming serialized shared covariance. |
| Windows CSV decoding / SQLite locator failures | Added explicit UTF-8 handling and Windows SQLite URI normalization, including percent-encoded paths containing spaces. |
| Detector settings disappeared across save/reload | Canonical profile persistence now retains geometry-only detector settings even without an efficiency fit, and switches profiles with the active spectrum. Regression checks exercise save/reload and spectrum switching. |
| GUI tests depended on empty production startup or stale control catalogs | Supplied explicit spectrum fixtures for spectrum-dependent actions; reconciled inventory counts with existing demo slots; included editable `QTableWidget` controls in the strict production catalog scan. |
| Missing transport files caused failures with no format coverage | Added four mandatory synthetic MCNP format contracts and compatible HDF5 layout handling. Missing real fixtures are explicit skips; an explicitly configured but missing fixture fails. Real-reactor transport validation remains unverified. |
| External-reference/provenance guards | Added configurable, strict solver-reference binding. Historical source paths are permitted only as provenance in identified vendored fixture manifests; runtime dependencies remain forbidden. |
| CI did not cover current work | Added pull-request and relevant branch/path triggers, mandatory scientific contracts, frozen upstream solver references, and a Python 3.11 CPU-ML job. Existing Linux/Windows GUI coverage remains configured. |

The final focused scientific/persistence recheck passed 204 cases. CI's existing
Black and Flake8 scopes passed locally. Diff whitespace checks passed for code
changes; the copied raw background retains its original padded ASCII header.

## Repeating software acceptance

Install the development dependencies in the chosen Python environment, then
run from the repository root. Use a new or empty output directory:

```powershell
python tools/run_acceptance.py --output artifacts/validation/my_acceptance
```

The runner records dependency versions, commit/diff/source hashes, fixture
bindings, per-case outcomes, XML, logs, and bounded worker receipts. GUI batches
default to five cases in fresh processes to avoid cumulative Qt styling cost.
Missing cases, worker timeouts, failed cases, and nonzero worker exits fail the
aggregate. `--require-no-skips` additionally makes any skip fail acceptance.

Bind external solver checkouts explicitly when exercising their ten parity
cases. The root must contain `Neutron-Unfolding` and `pyunfold`:

```powershell
$env:FLUXFORGE_REFERENCE_ROOT = 'C:\path\to\references'
python tools/run_acceptance.py --output artifacts/validation/reference_checks --tests tests/test_unfolding_reference_parity.py
```

The audit used Neutron-Unfolding `d5e377b7bc3f6ac01d59a9ebcdd1bdc32f4ab85b`
and pyunfold `0b50d43d17380d2663c3d1a8c3356fadde4917aa`. Bind real transport
fixtures with `FLUXFORGE_TRANSPORT_FIXTURE_DIR`; see the required filenames and
tallies in `tests/test_transport_io.py`. Explicit invalid bindings fail rather
than silently skipping. The current TensorFlow 2.15 optional dependency was
validated under Python 3.11; use that environment for `.[dev,ml]` checks.

The remaining 17 skips are 13 real MCNP/OpenMC fixture cases, two CUDA-specific
environment cases, one POSIX directory-fsync case on Windows, and one legacy
desktop dependency case. Synthetic transport contracts do not substitute for
the 13 real-file checks.

## Full UWNR/INL example results

Public ingestion and peak processing completed for all 29 committed raw
spectra, with per-input/output hashes retained. Full reductions used thresholds
enforced and plotting disabled:

| Full replay | Completion | Acceptance |
| --- | --- | --- |
| Default workflow | 29/29; 627.297 seconds; no timeout | Gate exit 1. Twenty-six samples used report reproduction; all 29 lack an independently passing validation state. |
| `iec_tiered` for flux wires and generic targeting | 29/29; 616.344 seconds; no timeout | Gate exit 1. Four passed configured report-comparison thresholds, 22 failed, three unvalidated; zero reference-reproduction samples. |

These full reductions use the configured profile energy-calibration override.
Their 29 saved energy grids exactly match the run's background grid; the final
identity-guard correction leaves their alignment choice unchanged. The earlier
public ingestion exercised raw-header energy grids. Both calibration paths
still require measurement-specific qualification.

The four comparison passes are Co-Cd, Co, CU, and Sc-Cd. RAFM1, Long144h, and
Long72h are unvalidated. Twenty-seven raw/report pairs were found; six processed
reports have no raw counterpart. Independent end-of-irradiation truth is absent.

The independent replay retains the configured limits: 25% relative count
error, 20% relative activity error, and an En-score limit of three. It records
147 matched count failures including ambiguity/isotope mismatch, of which
142 exceed numeric count thresholds, plus 52 isotope-activity failures and
eight reported unidentifiable fit groups. These are counts of comparisons,
not additional test-suite failures. No thresholds were relaxed.

For example, RAFM3-C at 300 seconds EOI has a W187 comparison near 133.69 keV:
processed gross counts 27,213 and raw-estimated gross 27,108.58 nearly agree,
but processed net counts 1,603 and raw-estimated net 16,064.14 disagree. This
identifies peak-area/continuum recovery as a separate investigation from
efficiency calibration; it does not prove which estimate is correct. Other
cases also flag emission-yield conventions, assignments, exports, and activity
conversion. The processed reports remain unqualified absolute ground truth.

Repeat a threshold-enforced independent comparison with:

```powershell
python -m fluxforge.cli.app rafm-validate --example-root examples/RAFM_irradiation --results-root artifacts/validation/my_independent_rafm --flux-wire-counting-method iec_tiered --generic-counting-method iec_tiered
```

The recorded runs called `run_rafm_validation(..., enforce_thresholds=True,
generate_plots=False)` directly. Do not use `--no-fail`, report-derived count
substitution, or comparison success as evidence of qualified physical accuracy.

## Remaining scientific blockers

1. **Peak recovery and comparison completeness:** review the retained failing
   line windows, continuum/ROI models, ambiguous joint fits, and missing raw
   counterparts against original vendor exports/settings. Correct demonstrated
   defects with independent fixtures, then repeat the enforced comparison.
   Calibration alone cannot resolve the count-domain failures.
2. **Traceable efficiency and geometry:** obtain the detector-specific absolute
   energy and efficiency calibration/certificate, source activity and uncertainty, fit
   covariance, validity dates, and matching sample geometry. The historical
   South Small Vial export still has an unqualified percent convention/error
   column and conflicting geometry factors. Generic uncertainty values cannot
   be silently bound to this detector or these spectra.
3. **Measured rate covariance and irradiation history:** qualify detector
   efficiency, gamma yield, half-life, target mass, isotopic abundance, and
   irradiation history components for each admitted rate, including correlation
   and coverage-factor conventions. Recover weighing and power/time records.
4. **Unfolding admission:** all 13 source-joined saved rates were replayed across
   20 groups. The evaluated response matched its frozen reference at `1e-12`,
   with rank seven. GRAVEL, MLEM, and ML_SEED did not converge at 1,000 iterations;
   MAXED converged but is not physically admitted. The manufactured prior is
   unqualified, and strict complete-covariance validation correctly rejects
   the six missing measured components. This replay is not a fresh raw refit.

A local calibration email and four INL procedure memos were recovered and
hashed. The dosimetry procedure reports a balance uncertainty of 5.7 micrograms
and an HPGe efficiency uncertainty of 1.2% at one sigma for that program; the
email describes its source-calibration approach. These are useful records to
trace, not sample-specific qualification. The two original sample workbooks
are unreadable OneDrive placeholders. At the initial audit, email search was
unavailable. The record-access follow-up below supersedes that access status.

The evidence bundle retains original failures, corrected-case observations,
replay receipts, raw comparison exceptions, package tests, and record-search
receipts separately. The earlier sealed adversarial audit remains unchanged.

## Review and publication follow-up — 2026-10-03

A `gpt-6-luna` subagent reviewed `5c4b852a` against the frozen integration base
and reported no actionable regressions in the reviewed fixes. It ran 84 focused
background/covariance/signed-fit/Hypermet/profile/transport/readiness checks,
eight HPGe dialog/profile and readiness-CLI checks, and a four-case transport
acceptance-runner smoke check, all passing. These repeat existing cases and are
not added to the distinct-case totals above. The review did not qualify the
remaining physical measurements or missing real transport fixtures.

Tracing the uncertainty gaps also exposed an adapter defect: generic gamma
metadata could supply `intensity_uncertainty`, but constructing the analysis
library discarded it. The adapter now preserves that absolute emission-
probability uncertainty. Two additional analytic activity contracts cover the
provided and absent-field cases: 1,000 counts with a 10-count uncertainty,
100-second live time, 0.1 efficiency, and 0.5 yield give 200 Bq; supplying a
0.05 yield uncertainty gives 20.099751 Bq uncertainty rather than the previous
2 Bq. These two new cases are separate from the 2,026-case historical evidence.
All 209 lines and half-lives produced from the current example metadata are
identical before and after this fix, because that metadata omits the uncertainty
field. Thus the recorded sample replay results are unaffected. Missing values
remain unqualified; the change does not manufacture source uncertainties.

The first remote PR workflow passed CPU-ML, frozen external-reference parity,
and both Linux and Windows modern-GUI jobs. Its core job exposed an optional-Qt
test collection defect: `QApplication` was imported without checking whether
Qt was installed. The same file also imported Qt-dependent shell helpers in an
unmarked test. The follow-up guards the import and marks that runtime helper
test with the existing optional-Qt condition; both checks remain exercised by
the GUI jobs. Later workflow outcomes are visible in the PR Checks tab. Legacy
desktop jobs are dispatch-only and were not executed by this PR event.

Connected Outlook search located the original sample-workbook and South
calibration-export attachments. Materialization succeeded at the connector,
but downloading the returned links failed with HTTP 403, including one fresh
retry. Their contents have not been read or qualified. The readable local
six-page Quantum operations guide describes adjacent-channel continuum
subtraction and resolution-based ROI sizing, but does not supply the original
efficiency-error convention, calibration certificate, fit covariance, or
sample-specific measurement uncertainty. Private email search results and
attachment receipts remain local and are excluded from this PR.

The original sealed evidence remains unchanged. Review and remote-check
receipts are retained separately under `artifacts/validation/review_20261003`,
and record-access receipts under `artifacts/validation/email_qualification_20261003`.
