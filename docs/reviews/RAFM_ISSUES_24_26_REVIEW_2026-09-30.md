# RAFM issues 24–26: independent raw analysis

## Corrections

For #24, indistinguishable hypotheses within each targeted group share one observed peak
component. Alternative isotope/transition assignments are preserved; activity
is withheld when the assignment is unresolved. The conservative collapse rule
uses half the smallest candidate FWHM over the entire cluster, without chained
merging. It does not establish that other candidate components are resolved.
Failed joint fits preserve the signed window observation and its counting
standard deviation as a diagnostic, and suppress exploratory labels in that
window. The window sum is not a physical net peak area.
Continuum padding can overlap a separately fitted neighboring targeted group;
the diagnostic does not imply that every neighboring peak in that window failed.

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
The final affected workflow/integration, dispatcher and checkpoint suite passed
106 tests. Earlier 80/88/93-test runs overlap this coverage and are not added
together. Existing Python 3.12.10, NumPy 2.5.1 and SciPy 1.18.0 were used;
no dependencies were acquired or changed.

```powershell
python -m pytest tests/test_targeted_assignment_ambiguity.py tests/test_joint_peak_covariance.py tests/test_measurement_time_audit.py tests/test_rafm_validation_independence.py tests/test_rafm_workflow.py tests/test_rafm_background_integration.py tests/test_raw_recovery_dispatch.py -q
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
All ten final-source wire predictions were compared with the first full audit:
peak estimates, measurement/EOI activities and reaction results are unchanged
apart from the added nominal-resolution metadata. Co-Cd and Co target atoms
remain 3.750528264446526e19 and 4.154979967868026e19, respectively.

## Final replay results

The authoritative final run used source commit `a72cc4a` and completed all
29 raw spectra without resuming. It supersedes the resumed third-run baseline.
Execution exited zero; overall comparison remains false: **4 true, 23 false,
2 unknown** validation states. The two unknowns are RAFM1 and RAFM1_Long_72h_EOI,
which have no QG partner. RAFM1_Long_144h_EOI has a report without usable
comparison rows and one failed fit: basis `not_evaluated`, validation false.
That is a fit failure, not evidence of an independently measured activity bias.
Six other QG exports have no committed raw partner and remain unvalidated.

- [Final source/input-bound receipt](RAFM_ISSUES_24_26_REPLAY_2026-09-30.json)
- [All 29 sample outcomes and categories](RAFM_ISSUES_24_26_SAMPLE_AUDIT_2026-09-30.csv)
- [247 line comparison and uncertainty rows](RAFM_ISSUES_24_26_LINE_AUDIT_2026-09-30.csv)
- [76 isotope measurement-time comparison rows](RAFM_ISSUES_24_26_ISOTOPE_AUDIT_2026-09-30.csv)
- [Superseded full baseline receipt](RAFM_ISSUES_24_26_BASELINE_REPLAY_2026-09-30.json)

Eight source byte hashes, the runtime profile hash, six metadata hashes,
background, all raw/reference inputs and predictions were checked. A separate
Sol reviewer independently rehashed all 29 prediction payloads, checked the
withheld-report prediction and absence of reference substitutions, and reviewed
the decisive actual spectra. All 130 unverified assignments export null activity
and EOI values; 92 of them are centroid mismatches. All 21 failed joint-group
diagnostics explicitly decline net-area/activity estimation.

| Actual example | Result and remaining limitation |
| --- | --- |
| B24h | Missing report observations 1 → 0 versus the earlier raw baseline; 47 peaks, 7 unidentified and 22 unverified assignments. Still 9 count and 3 activity failures. |
| B24h 846/1810 keV | 964.5 ± 68.82 and 85.0 ± 31.65 counts retained as ambiguous observations, with no assigned activity. |
| B24h Mn56 aggregate | 388.75 Bq from the single 2112.89 keV line, which fails count parity; activity remains unqualified. |
| RAFM4-B Sb124 | Both distant-centroid candidates withheld; no Sb124 isotope aggregate remains. |
| RAFM4-B Co58 | Single 586 ± 266.83 count observation (2.20 sigma), 372.14 Bq estimate with no Co58 reference line; tentative identification. |
| Late RAFM4 V52 | No V52 isotope aggregate in the four final 15-day spectra; absence of an aggregate does not prove physical absence. |

There are still nine missing reference peaks across the supplied raw specimens.
An increased number of explicit unverified assignments is honest withholding,
not evidence of improved identification accuracy. The B24h, Ti and RAFM4-B
comparison plots and reports were inspected directly.

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
The isotope combiner uses scalar line uncertainties; correlated efficiency
parameters and joint inter-line/model contributions are not a qualified full
covariance budget. The counting corrections do not establish that missing terms
are zero.

The effective `rafm_25cm` efficiency model records relative uncertainty 0.0:
no certified coefficient covariance is supplied by this profile. Library
emission uncertainties are propagated where supplied (e.g. Sc47 0.02 absolute,
Sc48 983 keV 0.03 absolute), but missing values default to zero and remain an
unqualified budget term. Ti/Cd model uncertainty additions do not validate the
underlying efficiency calibration or resolve inconsistent report exports.

## Issue status

| Issue | Implemented and reviewed | Remaining acceptance work |
| --- | --- | --- |
| #24 | Peak-component preservation, alternative assignments, failed-fit diagnostics, centroid withholding and exploratory-bypass regressions; real-data replay reviewed. | Missing observations, weak/ambiguous identifiers and suspect fitted widths still need investigation. |
| #25 | Signed counting covariance, joint fit/area error handling, physical ROI uncertainty floor, line/isotope discrepancy ledger and explicit missing budgets. | Calibration/model and correlated systematic uncertainty remain unqualified; integration pending. |
| #26 | All supplied spectra audited with measurement-time categories and separate EOI status; report/summary code and examples reviewed. | Remaining activity differences are documented rather than corrected by copying QG; independent EOI/calibration evidence and integration pending. |

All three issues remain open. The draft PR has not been merged; passing execution
and regression tests does not close unresolved scientific acceptance criteria.
