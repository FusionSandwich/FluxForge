# RAFM issues 24–26: independent raw analysis

## Corrections

For #24, indistinguishable targeted hypotheses now share one observed peak
component. Alternative isotope/transition assignments are preserved; activity
is withheld when the assignment is unresolved. The conservative collapse rule
uses half the smallest candidate FWHM over the entire cluster, without chained
merging. It does not establish that other candidate components are resolved.
Failed joint fits preserve the signed window observation and its counting
standard deviation as a diagnostic, and suppress exploratory labels in that
window. The window sum is not a physical net peak area.

For #25, joint and single targeted fits use signed counts and the full propagated
fit-window covariance. Joint area errors include amplitude/width covariance
with explicit parameter layouts. Invalid covariance, insufficient fit bins,
ill-conditioned parameter covariance and optimizer failure cannot silently
fall back to multiple independent estimates of the same area. The final physical
net-count uncertainty retains the propagated ROI floor.

For #26, each sample receives a measurement-time audit separating peak-area,
assignment/library, activity-conversion/efficiency and export-data discrepancies.
Multiple causes can coexist. The activity ratio divided by the area ratio tests
conversion disagreement; it is not a measured efficiency certificate. Reports
and the summary explicitly state the comparison basis and that independent EOI
parity and absolute accuracy are unqualified. Missing evidence remains unknown.

## Tests and adversarial review

Synthetic counterexamples cover stronger/weaker overlapping isotope hypotheses,
same-isotope unresolved transitions, unknown labels, stale positive activity,
exploratory-label laundering, failed-fit windows, exact correlated Gaussian
area covariance, duplicate components and malformed comparison evidence.
The final focused suite passed 42 tests. An earlier affected integration suite
passed 93 tests before the last suppression/summary changes; the final broader
recheck is recorded with the replay evidence below.

A separate Sol reviewer challenged the implementation and found two initial
activity-laundering paths and an equal-energy unknown-label sorting defect.
All were corrected with regression tests. The final changed failure paths were
accepted without a remaining blocker. The reviewer independently checked B24h:
the crowded 846 keV region retains one ambiguous 964.5 ± 69.0 count peak; the
1810 keV estimate is 85.0 ± 31.45 counts (2.70 sigma), below the configured
3-sigma threshold and not claimed as recovered.

## Reproducible real-data audit

```powershell
$env:MPLBACKEND = 'Agg'
$env:OPENBLAS_NUM_THREADS = '1'
python tools/audit_rafm_raw_recovery.py --all-raw --output-root <new-output-directory>
```

This runs all 29 committed UWNR/INL raw spectra with IEC counting, the committed
detector profile and measured background. Reference substitution is forbidden;
the first specimen is repeated without its QG report, requiring identical
prediction hashes. The receipt binds eight source files, metadata, background,
individual raw/reference files and predictions. Completion describes execution,
not a pass of the scientific comparison criteria.

The Co assumption remains 0.46 wt%; the listed Co masses are already element
masses and receive no second composition factor.

## Uncertainty and reference limitations

QG is a comparison export, not independently certified ground truth. A default
zero emission-probability uncertainty or absent efficiency uncertainty does not
prove either term is exact. Detector efficiency calibration, geometry/material
corrections, temporal background representativeness and continuum/peak-model
residuals need external evidence. Within-spectrum shared-background covariance
is propagated; cross-sample shared-background activity covariance is not yet
qualified. Deriving EOI activity from the measurement-time estimate does not
provide an independent EOI reference. No unfolding logic is changed here.

Issues remain open until issue-specific acceptance evidence is reviewed and the
draft changes are integrated. Pending replay results are recorded separately.
