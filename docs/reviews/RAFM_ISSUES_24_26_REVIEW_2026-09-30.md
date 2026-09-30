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

The completed first 29-spectrum audit revealed two false Sb124 assignments in
RAFM4-B: library 602.73/1690.98 keV versus fitted 605.774/1695.866 keV. The generic
workflow now withholds a single known assignment when the final centroid differs
by more than one nominal profile FWHM at the library energy. Measured counts and
the candidate survive, with state `withheld_energy_mismatch`. The fitted width
does not set this threshold; these two fitted widths were anomalously narrow.
A nearby exploratory label cannot restore the withheld activity: merging uses
the stored nominal physical resolution. The flux-wire API default is unchanged.
This withholding criterion does not prove closer assignments correct or qualify
the detector calibration. A final-source replay supersedes the first audit.

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
The final affected workflow/integration suite passed 80 tests, and the all-raw
wire/generic dispatcher and checkpoint checks passed 17 additional tests. An earlier affected suite
passed 93 tests before the last suppression/summary changes; these overlapping
counts are not added together. No dependencies were acquired or changed.

```powershell
python -m pytest tests/test_targeted_assignment_ambiguity.py tests/test_joint_peak_covariance.py tests/test_measurement_time_audit.py tests/test_rafm_validation_independence.py tests/test_rafm_workflow.py tests/test_rafm_background_integration.py -q
python -m pytest tests/test_raw_recovery_dispatch.py -q
```

A separate Sol reviewer challenged the implementation and found two initial
activity-laundering paths and an equal-energy unknown-label sorting defect.
All were corrected with regression tests. The final changed failure paths were
accepted without a remaining blocker. The reviewer independently checked B24h:
the crowded 846 keV region retains one ambiguous 964.5 ± 69.0 count peak; the
1810 keV estimate is 85.0 ± 31.45 counts (2.70 sigma), below that bounded
check's 3-sigma threshold. The actual generic workflow uses the committed
2-sigma targeted threshold and can retain this as an ambiguous observation,
without qualifying its isotope/activity. Flux-wire replay retains positive
targeted estimates with threshold zero; these are not all significant detections.

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

The first launch stopped on an incomplete audit dispatch before processing;
the second stopped on a report without usable comparison rows. Those development
directories are superseded. The third run's execution session disappeared after
24 samples. Its completed artifacts were preserved and checked before resuming
the last five: seven scientific code hashes, all metadata/background/input
hashes, effective configuration, ordered sample identities, predictions,
validation/audit states and cached receipt fields must agree. Both the old and
reviewed current driver hashes are pinned; prior source/Git/time bindings are
retained in `resumed_from`. The external detector profile is now byte-hashed,
and the reused artifacts' effective detector values are checked against it.

The Co assumption remains 0.46 wt%; the listed Co masses are already element
masses and receive no second composition factor.

## Interpreting line and isotope discrepancies

The comparison estimator can use a narrower raw local-counting window than the
physical measured-background ROI. Its standard deviation (`raw_net_unc` in the
line table) is preserved to explain comparison metrics. The activity uses the
larger physical uncertainty (`raw_physical_net_unc` in the audit table), including
the propagated ROI floor. These are different estimator conventions. Neither
the comparison window nor its smaller error replaces the physical activity
uncertainty. QG line-activity errors are inferred from exported line count errors;
the separately exported isotope/header activity error may include other terms.

For example, Ti Sc48 at 175.23 keV has comparison net uncertainty 126.40 counts
and physical net uncertainty 238.97 counts; its exported activity is
321.88 ± 232.30 Bq. Ti Sc47 at 159.29 keV differs in area by only +1.45% but in
activity by +74.55%, leaving a conversion ratio of 1.721 after cancelling area.
At 983.36 keV, the Sc48 area differs by +7.37% while the conversion ratio is
0.01173. QG Sc48 line activities span 173.9–25570 Bq against a 15688 Bq header.
These inconsistencies prevent assigning the discrepancy solely to detector
efficiency. The Ti comparison plot and text report were inspected directly.

## Uncertainty and reference limitations

QG is a comparison export, not independently certified ground truth. A default
zero emission-probability uncertainty or absent efficiency uncertainty does not
prove either term is exact. Detector efficiency calibration, geometry/material
corrections, temporal background representativeness and continuum/peak-model
residuals need external evidence. Within-spectrum shared-background covariance
is propagated; cross-sample shared-background activity covariance is not yet
qualified. Deriving EOI activity from the measurement-time estimate does not
provide an independent EOI reference. No unfolding logic is changed here.

The effective `rafm_25cm` efficiency model records relative uncertainty 0.0:
no certified coefficient covariance is supplied by this profile. Library
emission uncertainties are propagated where supplied (e.g. Sc47 0.02 absolute,
Sc48 983 keV 0.03 absolute), but missing values default to zero and remain an
unqualified budget term. Ti/Cd model uncertainty additions do not validate the
underlying efficiency calibration or resolve inconsistent report exports.

Issues remain open until issue-specific acceptance evidence is reviewed and the
draft changes are integrated. Pending replay results are recorded separately.
