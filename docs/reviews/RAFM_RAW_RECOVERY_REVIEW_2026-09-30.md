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
