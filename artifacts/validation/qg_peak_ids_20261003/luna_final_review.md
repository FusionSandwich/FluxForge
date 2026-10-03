# Final QG native peak-ID antagonist review

Review ran in the isolated `codex/qg-peak-identifications` worktree. The validator used the frozen native extraction `artifacts/validation/qg_peak_ids_20261003/native_final.json`, current source-bound correction manifest, bundled QG reports, and raw spectra. No source files were changed by this review.

The validator reports **33 reports / 300 ROI rows** and the following final statuses:

- 239 `same_id`
- 1 `corrected_reference_id`
- 3 `tentative_native_same_id` (native fits below the 2σ confirmation threshold)
- 4 `tentative_same_id` (low-significance candidate fits)
- 47 `missing_raw_spectrum`
- 6 `reference_nondetection` (zero-net QG ROIs)

There are 243 matched native fits, 240 confirmed detections, and seven tentative identity associations. The validator correctly returns `passed: false`: 47 positive QG rows lack a paired raw spectrum, and seven same-label associations remain below the confirmation threshold. Do not describe all 294 positive QG rows as independently verified.

The sole corrected row is `RAFM4/RAFM4-N_15dEOI.txt`, source line 92: reported `Tb154m` at 264.28 keV, compared against `Ta182` using the manifest’s exact source hash. The artifact regression also confirms the Tb activity is excluded from the isotope payload and Ta activity remains equal to Ta’s own QG summary activity; the correction does not synthesize a Ta peak at 264.28 keV.

One separate source discrepancy remains visible: unpaired `RAFM3-A_300sEOI.txt` reports Mn56 at 2529.1 keV, 6.04 keV from the supported 2523.06-keV line. The inventory test keeps that discrepancy explicit and does not widen matching tolerance or invent a library line.

Validation: `tests/test_qg_validation_tool.py` passed **24 tests**. The validator independently reproduced the exact status counts above and verified the corrected row’s report, line number, and original/corrected isotope labels. Focused matcher, ROI inventory, correction, candidate, and artifact suites previously passed **70 tests**.
