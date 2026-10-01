# QG report source QC (#32, related #30)

This follow-up is based on reviewed PR212 head `da0f068`. It changes reporting
and importer provenance, preserving that PR's unfolding repairs and original
measured inputs. It does not correct a measured activity or reaction rate.

## Diagnosis

The existing source-QC code already identifies the Sc48 line-versus-summary
activity inconsistency in all three original Ti reports. The missing diagnostic
is the competing gamma-yield interpretation: the legacy comparison helper picks
whichever interpretation of RAD INT is closest to bundled decay data. Printed
`1.00` then appears as fraction `1.0`, hiding the alternative `0.01` percent
hypothesis and making the legacy gamma-library parity comparison look consistent.
RAD INT alone does not declare a unit. The helper is not proof of how QG used
its library, and its result must not justify a physical correction.

The report importer retains original numerical values and now also records
printed tokens, report-byte SHA256, source line and decoded line text. Nuclide
summary provenance is separate from ROI line provenance. The ROI activity
column unit is parsed independently of the summary activity unit. Conversion
for line diagnostics uses that declared unit; missing/unknown units remain
unknown instead of being silently treated as uCi. Existing manually constructed
records without a column-unit field retain an explicitly labeled summary-unit
assumption. The report decoder remains UTF-8 with replacement; byte hashes and
line numbers provide access to the original bytes.

## Report-only fields

The new source QC records raw RAD INT with unit `unspecified`, both percent and
fraction hypotheses, and separate conditional implied efficiencies. It compares
them with a nearby bundled decay line, bounded by a 1 keV match tolerance. An
absent/distant or ambiguous line remains unqualified. Matching provenance names
the bundled decay_2012/actigamma source, exact file hash, line energy, intensity
and intensity uncertainty. A reporting heuristic of the larger of 5% of the
bundled intensity or three bundled uncertainties is recorded explicitly; it is
not a significance test under unknown vendor/calibration uncertainty.

`yield_convention_discrepancy` means the percent hypothesis disagrees with the
bundled reference while the fraction hypothesis agrees. It is a source review
lead, not a demonstrated vendor mechanism. Cu64's printed 0.47 is consistent
with a 0.0047 photons/decay percent hypothesis and is not promoted to 0.47.

The separate `source_qc_bucket` combines yield and line-summary source findings
without replacing raw count/activity parity buckets. CSVs, per-spectrum reports
and summary source bucket counts use the same findings. Buckets are used only
for reporting and never feed activity calculation, rate construction, numerical
solver choices or the workflow's existing acceptance logic. Legacy comparison
values are retained and explicitly labeled as diagnostic and unverified.

## Actual-report evidence and limits

`tools/validate_qg_report_qc.py` reads the three original Ti reports and original
Cu control from the publication audit's source manifest. It compares imported
summary and line values, legacy comparison fields, and existing inconsistency
rows with a pre-change baseline. It checks all 32 original PR212 input hashes
and writes additive per-report/aggregate CSVs and a source-bound receipt.
No raw-spectrum re-fit or transport/inversion run is performed in that replay;
the empty raw-match sequence is marked `raw_parity_evaluated=False` and is not
evidence that an actual raw-analysis line was missing.

The [NNDC ENSDF Sc48 beta-decay evaluation](https://www.nndc.bnl.gov/ensnds/48/Ti/beta_decay.pdf)
(November 2021, Jun Chen NDS179/2022, pages 2–3) independently supports near-unit
absolute photon yields at 983.526 and 1312.120 keV. Its relative intensities
1000 multiply by 0.100 to obtain 100 photons per 100 parent decays. This reference
is separate from the bundled older decay data and the unknown original
GammaLib.mdb. Neither source establishes the vendor's actual processing convention.

Ti-RAFM-1 prints Sc48 line activities 0.646 and 0.691 uCi at these lines, versus
0.00752 uCi at 1037.5 keV, while its summary is 0.424 uCi. All values are retained.
Correcting two line yields cannot justify dividing the whole summary or rate by
100. The audit's hypothetical whole-rate sensitivity remains exploratory.

Original gamma-library/settings, dated calibration, irradiation history and
their uncertainty remain unqualified. No source, covariance or publication
admission gate from PR212 is relaxed. Scientific admission remains false.

## Additional source investigation — September 30, 2026

The historical [Quantum 4.04.00 manual](https://ludlums.com/images/product_manuals/QTMmanual.pdf)
(PDF pages 46, 118–119 and 126) describes intensity per 100 decays, percent
efficiency, activity/uncertainty summary weighting and an activity-reference
date. Applicability to the deployed version, active GammaLib, per-line summing
factors and count-decay processing remains unverified. The [IAEA historical
tabulation](https://nds.iaea.org/sgnucdat/safeg2008.pdf), Table D-2, PDF page 116,
corroborates percent yields; it does not replace current evaluated covariance
or provide independent detector calibration.

Replaying conditional N/u(N) weights from the original reports gives:

| Report | Printed summary uCi | Three stronger lines | All four lines |
| --- | ---: | ---: | ---: |
| Ti-RAFM-1 | 0.424 | 0.4241238300 | 0.4101672072 |
| Ti-RAFM-1a | 0.408 | 0.4416682062 | 0.4078967833 |
| Ti-RAFM-1b | 0.234 | 0.2456386056 | 0.2342877053 |

These different matching subsets do not establish a vendor inclusion rule or
reconstruct summary uncertainty. They give no basis for scaling whole summaries.
With printed half-life 43.700 h and real duration 172935.45 s, Ti_b has a
conditional uniform-live-fraction start/average factor 1.4288928812; the first
four-hour count gives 1.0320851574. Applying this factor requires establishing
vendor processing first. A common timing factor cannot explain a selective
factor-100 line discrepancy.

Current FluxForge already requires a count-decay declaration for QG activities:
`report_count_real_time_s` rejects an absent declaration, returns zero duration
for an already-corrected report and real duration otherwise. The example config
explicitly identifies its false setting as prior behavior awaiting verification.
The rate path preserves this distinction. Report summary activities remain
imported values, separate from raw peak consensus and conditional reconstructions.
No additional runtime defect is demonstrated by this evidence, so no new
calculation repair or configuration change is made. Existing count-decay tests
cover missing declarations, real-versus-live duration and both processing cases.

Additive source-bound arithmetic and test evidence is in
`artifacts/validation/qg_source_followup_20260930/`. The original PR213 replay
receipt remains unchanged. Private correspondence provenance remains local;
full measured uncertainty, active calibration identity and reactor history
remain unqualified. Scientific admission remains false.
