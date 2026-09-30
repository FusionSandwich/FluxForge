# Independent RAFM recovery review

## Corrected validation classification

The default RAFM `qg` mode imports reference peak counts and can substitute
reference isotope activities. Reproducing these numbers tests report handling;
it does not independently validate raw peak recovery or activity calculation.
The artifact now records `comparison_basis`, `reference_used_for_analysis` and
`comparison_passed`. Its raw-validation `passed` is null for reference reproduction.
Reports and the overall summary distinguish failed, unchecked and independently
compared samples. A run without a reference is unchecked.

Matched finite count evidence is required. RAFM3, RAFM4 and flux-wire comparisons
also require matched finite activity evidence. Missing metrics, absent domains,
unmatched reference files and unsupported status values cannot pass. An explicit
failure takes precedence over missing evidence. An enabled enforcement gate
accepts only an explicit overall true value. Passing these configured comparison
thresholds alone does not qualify detector calibration or physical normalization.

Independent Sol review found and verified corrections for two initial gaps:
all-unmatched/partial-domain evidence and references lacking raw partners.
Counterexamples also cover copied identical values, NaN/infinite/missing metrics,
empty runs, invalid thresholds and unsupported statuses. The affected suite ran
with existing local Python 3.12 and the Agg backend:

```powershell
python -m pytest tests/test_rafm_validation_independence.py tests/test_rafm_workflow.py tests/test_rafm_background_integration.py -q
```

Result: **57 passed**. No dependencies were acquired or changed.

## Actual UWNR replay

```powershell
python tools/audit_rafm_raw_recovery.py --output-root <new-output-directory>
```

The audit replays the twelve committed RAFM3 and four RAFM4 ASC spectra with the
existing profile/background, selecting `iec_tiered` instead of QG reproduction.
Reference-substitution entry points are replaced by functions that raise if called.
The first actual specimen is also analyzed with its QG report withheld: every
predicted peak and isotope result must have the same hash. QG remains a comparison
report, not independently certified ground truth.

The receipt records source, metadata, background and individual input hashes,
effective configuration, prediction hashes and per-sample validation. The terminal
status `RUN_COMPLETE_REVIEW_REQUIRED` describes execution, not measurement accuracy.
Full spectra, tables and plots remain in local output directories.

Final baseline evidence is preserved in
[`RAFM_RAW_RECOVERY_2026-09-30.json`](RAFM_RAW_RECOVERY_2026-09-30.json).
All four recorded source byte hashes matched the reviewed live files after
completion; the source content was committed as `156d924`. The Git HEAD recorded
at launch precedes that commit because the reviewed change was initially
uncommitted. The final receipt is
`C:\Users\joshu\Documents\UWNR_work\composition_review\raw_recovery_audit_2026-09-30_final\raw_recovery_receipt.json`.
The earlier audit directory is development evidence superseded by this replay.

### Per-sample comparison results

All sixteen specimens fail at least one configured criterion. Count failures
include isotope mismatches. Line-consistency flags are reported separately and
the committed configuration does not use them as pass/fail gates.

| Sample | Detected peaks | Unidentified | Count failures | Activity failures | Missing report peaks |
| --- | ---: | ---: | ---: | ---: | ---: |
| RAFM3-B_24hrEOI | 46 | 8 | 8 | 2 | 1 |
| RAFM3-B_2hrEOI | 24 | 2 | 6 | 2 | 0 |
| RAFM3-B_300sEOI | 21 | 0 | 6 | 3 | 1 |
| RAFM3-B_4dEOI | 39 | 10 | 5 | 2 | 1 |
| RAFM3-C_24hrEOI | 33 | 5 | 6 | 2 | 0 |
| RAFM3-C_2hrEOI | 26 | 3 | 4 | 2 | 0 |
| RAFM3-C_300sEOI | 23 | 0 | 7 | 3 | 2 |
| RAFM3-C_4dEOI | 26 | 7 | 4 | 1 | 1 |
| RAFM3-N_24hrEOI | 33 | 6 | 6 | 2 | 1 |
| RAFM3-N_2hrEOI | 20 | 3 | 5 | 2 | 0 |
| RAFM3-N_300sEOI | 23 | 1 | 8 | 4 | 1 |
| RAFM3-N_4dEOI | 39 | 11 | 6 | 1 | 1 |
| RAFM4-A_15dEOI | 39 | 2 | 16 | 4 | 0 |
| RAFM4-B_15dEOI | 40 | 1 | 19 | 4 | 0 |
| RAFM4-C_15dEOI | 42 | 3 | 18 | 4 | 0 |
| RAFM4-N_15dEOI | 49 | 3 | 20 | 5 | 0 |

Across the 212 report-line comparisons, the diagnostic buckets contain 142
count failures, 40 efficiency/activity-conversion discrepancies, two isotope
mismatches, six gamma-library discrepancies, nine missing lines and thirteen
matched lines. These buckets are ordered diagnostics: a count failure can also
have an activity-conversion discrepancy. They do not isolate a unique cause.

## Discrepancies and limits

The replay exposes peak-area and activity-conversion discrepancies that QG
reproduction hid. RAFM3-B at 24 hours misses Mn-56 at 2112.67 keV and assigns the
625.50 keV QG W-187 entry to Tb-154m. Its W-187 activity is about 65% higher and
Cr-51 about 194% higher. The spectrum also contains several unidentified peaks
and mutually inconsistent Tb-154m line activities. These assignments need review;
neither QG labels nor a nearest-energy match establish the isotope by themselves.

RAFM4-A at 15 days has no missing comparison peaks but fails sixteen count and
four isotope-activity comparisons. Ta-182 counts near 99.58 and 152.10 keV are
about 206% and 331% above the report. Some other lines have closer count agreement
while their activities differ, consistent with an additional conversion/efficiency
discrepancy. QG-implied efficiency is an inferred diagnostic, not a measured
detector calibration. The B-24-hour and A-15-day comparison plots were inspected.

The separate Co-Cd HPGe replay still has elevated residuals. Raw candidate Co-60
background centroids under stored header calibrations lie near 1171.00/1330.37
keV, compared with sample centroids near 1172.65/1331.72 keV. This suggests a
background energy-alignment problem requiring source/calibration review. No
empirical energy shift was applied. Header dates also differ. These observations
do not establish that the background activity remained constant.

Efficiency, emission-probability and other systematic budgets remain incomplete.
The 0.46 wt% Co case continues to use already adjusted element masses; no second
0.0046 factor is applied. #24, #25 and #26 remain open. Draft-only corrections
remain open until integration and issue-specific acceptance.
