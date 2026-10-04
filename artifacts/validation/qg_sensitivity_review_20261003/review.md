# Peak sensitivity and acquisition coverage review, 2026-10-03

The six `reference_nondetection` rows are original Quantum Gold ROIs with zero NET and blank activity. They are not six positive vendor detections that FluxForge lost. Positive-reference coverage, native significance, candidate identity, and physical activity qualification must be reported separately.

This review uses the current scientific engine and PySide6 GUI integrated from `origin/codex/archive-legacy-gui` (b618f5e) and `origin/codex/validation-three-failures-20261003` (4615e61), together with the previous independent native-identity work. Integration commits are c640308 and 7c5cc8b. The other agent's checkout and environment were not changed. Newer South-background, working-efficiency and portable-replay branches were inspected for additional updates; their comparison scenarios were not substituted for this engine or its physical background convention.

## Source zero-net ROIs

| Report | Identity | QG center (keV) | QG NET ± uncertainty |
|---|---|---:|---:|
| RAFM3-A_4dEOI | Fe59 | 1098.84 | 0 ± 82 |
| RAFM3-B_300sEOI | Mn56 | 2532.86 | 0 ± 162 |
| RAFM3-C_300sEOI | W187 | 551.64 | 0 ± 240 |
| RAFM3-C_300sEOI | Mn56 | 2530.61 | 0 ± 123 |
| RAFM3-N_300sEOI | W187 | 551.58 | 0 ± 262 |
| RAFM3-N_300sEOI | Mn56 | 2530.36 | 0 ± 134 |

The Mn56 evaluated line is 2523.06 keV, so these reported centers differ by 7.30–9.80 keV. Native analysis finds strong peaks near the evaluated energy in all three available B/C/N spectra. This warrants a calibration/ROI-center audit; it does not justify shifting the nuclear library or widening an identity guard to force agreement. Native ANS and ASC calibrations/clocks can differ. The new recovered A data uses native channel arrays and native unzoned clocks, with the existing workflow profile explicitly retained for this diagnostic analysis. [NNDC evaluated Mn56 decay data](https://www.nndc.bnl.gov/nudat3/getdecaydataset.jsp?dsid=56mn+bM+decay&nucleus=56FE).

## Why weak positive lines remain tentative

The previous complete 27-acquisition replay retained seven positive associations below the common 2-sigma confirmation threshold: three Sc48 wire lines near 175 keV, W187 at 773 keV in C-300s, Fe59 at 1099 keV in C/N-4d, and Mn56 at 2113 keV in N-24h. A wire fit returned by the threshold-zero production path is not automatically a confirmed 2-sigma detection. Quantum Gold printing a positive area/activity also does not imply that its uncertainty excludes zero; for example C-300s W187 is 15 ± 201 counts.

The targeted physical estimator uses a four-FWHM ROI and only one continuum channel on each side. For a locally flat continuum, variance of the integrated continuum estimate grows approximately as ROI-channel-count squared divided by the total sideband-channel count. This can dominate the peak uncertainty. The existing maximum of fit-area and ROI uncertainty preserves a conservative physical counting floor; deleting that floor would hide the problem rather than improve the measurement.

Area selection also depends on the decision threshold: the targeted path selects fitted area for a multiplet, a nonpositive ROI indication, or ROI significance below `peak_threshold`. Changing the threshold can therefore change the estimator as well as the decision. The study holds threshold at 2 sigma and records fit/ROI components and `used_fit_area`. Wire production uses threshold zero, so even the study's “baseline” is not an exact replay of that wire production path.

The parameter study tests four fixed settings: `(ROI FWHM, sideband channels each, gap FWHM)` = `(4,1,0)`, `(4,8,0.5)`, `(2.5,1,0)`, `(2.5,8,0.5)`. It retains measured-background normalization, physical variance and the one-FWHM assignment guard. Cases were selected from known reference weak/zero rows, and all independent-library neighbors within 12 keV were fitted. This deliberately tests interference sensitivity; it is not identical to the production library-target selection and is not a blind sensitivity benchmark.

In the fresh pre-recovery sweep, C-300s W187 at 551.49 keV rose from 1.651 to 4.545 sigma under the combined narrower/wider setting; W187 at 772.91 rose from 1.104 to 2.580, and N-24h Mn56 at 2113.09 rose from 1.889 to 4.275. Three wire Sc48 lines rose from 1.546/1.990/1.621 to 3.897/4.804/2.107. These are diagnostic configurations, not accepted changes to production defaults. The corresponding JSON preserves the native/audit/tool/source hashes.

The all-neighbor control around Mn56 846.764 keV returns a 4.857-sigma centroid at 846.8848 keV with ambiguous identity (Tb154m 845.16 and Mn56 846.764 are candidate assignments). Its activity is correctly withheld. The all-neighbor Fe59 C-4d fit returns no accepted peak under any ROI variant. A separate baseline support probe records fitted components at 1094.501 and 1098.520 keV for targets 1099.25 and 1102.43; both fail the absolute two-keV centroid guard. This differs from the production-selected Fe59 candidate at 1.330 sigma. A sideband change cannot repair a displaced or ambiguous fit. Missing accepted rows in this stress test must not be replaced by a nearest-neighbor identity.

## New GUI issue fixed

The production PySide6 `Auto Find Peaks` action called a helper capped at the twelve strongest candidates. Valid weaker candidates were silently excluded from the review dialog. The GUI now passes `max_peaks=None`; the helper keeps its existing numerical default for callers requesting a bounded list. Detection method and threshold are unchanged. A regression test fits 21 supported synthetic peaks, exercises the actual modern-shell review action, and checks that all 21 reach the dialog and workspace.

The GUI still has different search defaults from the targeted scientific workflow: significance threshold 4 and minimum spacing 18 channels, while Mariscotti also applies a minimum internal threshold. These differences should be exposed and tested using calibrated FWHM, rather than assuming the GUI search is equivalent to the targeted 2-sigma analysis.

## Acquisition recovery and remaining data

The initial filesystem/OneDrive search examined 734 readable spectra and 20 relevant ZIPs and found no credible acquisition-clock match. Checking updated immutable Git objects subsequently found five original ANS acquisitions on commit 46096eb1d8f760645b4c498b2a7bb50c9b63f262. That supersedes the earlier claim that all six acquisitions were unavailable.

The bounded recovery tool verifies source-file hashes and sizes, rev4 structure, channel bounds, file-length closure, native energy coefficients, all 8192 observed integer channels, and native/report clocks and durations. Four reports match the archive byte for byte; Cu-Cd differs only by five cp1252 plus/minus signs previously converted to UTF-8 replacement characters. No ROI area, activity or reference identity enters the exported channel array. Original source bytes and the immutable source manifest are retained with a recovery receipt. The three A ASCII exports independently match the ANS channel arrays, but their clocks differ by one or three hours and their real times are rounded; those conflicts remain explicit.

Recovery adds 35 positive rows and one source zero-net row. Only RAFM1_Long_70d_EOI remains without raw data: July 1, 2025, 14:57:31 (timezone unspecified), LT 3600 s, RT 3848.07 s, 12 positive rows. Any original spectrum format with this acquisition identity is sufficient. Cloud placeholders and two unhydrated UW archives were not claimed to be read. Full local inventory paths are retained in scratch; the published search receipt contains coverage counts and manifest hashes.

Recovery resolves source coverage, not all activity-input limitations. In particular Fe-Cd was measured at near-contact geometry; the diagnostic workflow's common profile is not an independently qualified efficiency calibration for that geometry. Shared efficiency/calibration covariance across lines is also not carried by the inverse-variance isotope activity combiner. Passing peak-count tests does not qualify those missing activity inputs.

## Next methods to test

1. Choose a fixed background-support policy using separate blanks and injections, including real continuum slopes, tails, nearby lines and measured-background covariance. Compare the eight-channel and narrower-ROI variants without selecting whichever setting makes each reference row pass. IAEA guidance describes the ROI capture/background tradeoff and recommends a 2.5-FWHM ROI; reduced windows require response/capture assessment. [IAEA gamma spectrometry guidance](https://www-pub.iaea.org/MTCD/Publications/PDF/AQ-48_web.pdf).
2. Decouple area-estimator selection from the significance decision. Fit once with a declared estimator, preserve signed indications, then calculate uncertainty, decision threshold and detection limit consistently. Test continuity across the current threshold switch.
3. Extend the existing Poisson fit to joint original sample/background observations with nuisance continuum and background normalization. Do not apply a Poisson count likelihood directly to signed background-subtracted bins. Retain centroid/assignment guards and expose rejected components as diagnostics.
4. Audit calibration residuals against independently supported lines, including the high-energy Mn56 transitions. Resolve native-versus-ASCII calibration differences before treating the reference centers as identity errors. Use corroborating transitions and decay information to constrain ambiguous nuclides without injecting vendor labels.
5. Carry common efficiency/calibration covariance through multi-line activity combination. Qualify each geometry and source-background applicability separately. Current updated Git evidence also records saved Quantum ambient-off/continuum-on flags; those are a source-supported comparison scenario, not proof of the final report processing state. [Quantum Software manual](https://ludlums.com/images/product_manuals/QTMmanual.pdf); [pinned saved-settings investigation](https://github.com/FusionSandwich/FluxForge/blob/46096eb1d8f760645b4c498b2a7bb50c9b63f262/artifacts/validation/quantumgold_documentation_20261003/FINDINGS.txt).
6. Expose calibrated GUI search parameters and test dense RAFM spectra and resolved doublets. The twelve-candidate cap is fixed; spacing/threshold/search-wide false-positive behavior still needs separate validation.

The seeded 4000-trial flat-Poisson fixed-energy study improved recovery of an injected 200-count peak from 34.225% to 99.175% with the combined setting; every fixed setting recovered the strong injection. Blank false-positive fractions were 2.325–2.750% at the nominal one-sided 2-sigma decision, with uncertainty close to the empirical spread and approximately 68% one-sigma coverage. Choosing the best of four settings per spectrum raised blank false positives to 8.55%. These results exclude blind-search selection, interferences, tails, drift and measured-background covariance; they establish a useful proposed test direction, not an accepted production decision rule.

## Validation and reproducibility

The complete focused post-recovery suite passed 154 tests. Luna independently passed 39 GUI/matching/sensitivity tests and all 13 recovery tests. The existing unfolding GUI fixture emits one ill-conditioned-matrix warning; passing that integration test does not establish stable unfolding for the warning's matrix. The final source-bound replay and parameter-study status are recorded in `final_results.md` and `validation_receipt.json`. Strict reference agreement remains separate from tests passing, and `scientific_admission` stays false.

Run from the isolated repository with its `src` on PYTHONPATH and the existing project Python environment. Use new output paths:

```text
python tools/extract_rafm_native_peaks.py --root . --out NEW_NATIVE.json
python tools/validate_qg_peak_identifications.py --root . --native NEW_NATIVE.json --out NEW_AUDIT.json
python tools/review_qg_sensitivity.py --root . --audit NEW_AUDIT.json --native NEW_NATIVE.json --out NEW_SWEEP_DIRECTORY
python tools/check_roi_sensitivity.py --trials 4000 --out NEW_POISSON.json
python tools/plot_qg_reference_zero_regions.py --root . --audit NEW_AUDIT.json --out NEW_PLOT.png
```

`recover_qg_acquisitions.py --root .` is intended for the pre-recovery tree with the pinned Git source object available. It refuses existing source/output paths. All archived originals and converted raw files are already supplied in this branch.
