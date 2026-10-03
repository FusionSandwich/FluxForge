# Quantum Gold peak identity audit — 2026-10-03

All 33 bundled reports and all 300 ROI rows are accounted for. The strict all-reports identity acceptance is **false**. This audit concerns peak identities, not a qualification of the complete uncertainty budget or unfolding result.

| Outcome | Rows |
| --- | ---: |
| Same identity in independently fitted native peaks at ≥2σ | 239 |
| Source-bound corrected Tb154m → Ta182 identity | 1 |
| Same-identity wire measurements below 2σ | 3 |
| Additional same-identity material fits below the 2σ threshold | 4 |
| Positive report rows without paired raw acquisitions | 47 |
| Zero-net reference nondetections, including blank activities | 6 |
| Total | 300 |

The 27 paired reports contain 247 positive rows. Every one has the expected identity in a supported fit, including four explicitly tentative material candidates. Of the 243 normally returned native fits, 240 reach 2σ. Three Sc48 wire ROIs near 175.3 keV use the existing zero-threshold wire measurement path and are also below 2σ. Thus **seven matched identity labels are below 2σ**; identity agreement does not promote them to confirmed detections. The four additional material candidates never enter detected-peak results or activity calculations. The wire threshold was not lowered for this audit.

The sole corrected source row is RAFM4/RAFM4-N_15dEOI.txt line 92, 264.28 keV, original Tb154m, expected Ta182 (library line 264.076 keV). The exception is bound to the original report SHA256, source line, energy and ID. Original text and assignment are retained. A changed source or duplicated exception fails. Wrong-nuclide Tb activity is excluded, never renamed or copied into the native Ta fit.

Default QG compatibility mode previously inserted missing report peaks and changed native isotope labels. The workflow now records an independent native identity comparison before that overlay. Matching uses one physical channel at most once, preserves ambiguous labels, and optimizes the full assignment instead of reusing a peak separately for each report row. Tests with empty native extraction show that a report-filled compatibility artifact cannot pass the independent identity check.

## Weak material candidates

- RAFM3/RAFM3-C_300sEOI.txt: W187, report 773.21 keV; native 772.91200 keV, significance 1.10477σ; report net 15 ± 201.
- RAFM3/RAFM3-C_4dEOI.txt: Fe59, report 1098.75 keV; native 1098.69773 keV, significance 1.33047σ; report net 62 ± 47.
- RAFM3/RAFM3-N_24hrEOI.txt: Mn56, report 2112.80 keV; native 2112.94345 keV, significance 1.39813σ; report net 60 ± 15.
- RAFM3/RAFM3-N_4dEOI.txt: Fe59, report 1099.10 keV; native 1099.00577 keV, significance 1.77990σ; report net 93 ± 43.

## Unavailable acquisitions

Six reports lack matching raw spectra; they contain 47 positive rows and one of the six nondetections. Different cooling-time acquisitions are not substituted:

- examples/RAFM_irradiation/QG_processed_gamma_data/flux_wires/Cu-Cd-RAFM-1_25cm.txt
- examples/RAFM_irradiation/QG_processed_gamma_data/flux_wires/Fe-Cd-RAFM-1_0cm.txt
- examples/RAFM_irradiation/QG_processed_gamma_data/RAFM1/RAFM1_Long_70d_EOI.txt
- examples/RAFM_irradiation/QG_processed_gamma_data/RAFM3/RAFM3-A_24hrEOI.txt
- examples/RAFM_irradiation/QG_processed_gamma_data/RAFM3/RAFM3-A_300sEOI.txt
- examples/RAFM_irradiation/QG_processed_gamma_data/RAFM3/RAFM3-A_4dEOI.txt

The unpaired RAFM3-A_300sEOI report also places an Mn56 ROI at 2529.1 keV. The bundled/evaluated Mn56 line is 2523.06 keV, a 6.04-keV difference outside the existing match tolerance. Its original value is preserved among the unavailable rows. Its raw acquisition is required to distinguish a report-centroid error from a misassignment. No additional exception or invented nuclear line was applied. Evaluated source: [NNDC ENSDF Mn56 decay dataset](https://www.nndc.bnl.gov/nudat3/getdecaydataset.jsp?dsid=56mn+bM+decay&nucleus=56FE).

Fe59 wire target selection now includes the already bundled 142.651 and 192.349 keV lines. Their energies are also present in [INL evaluated Fe59 spectrum](https://gammaray.inl.gov/SiteAssets/catalogs/ge/pdf/fe59.pdf). No paired Fe-Cd raw acquisition exists, so library availability is tested separately from recovery.

## Reproduction and provenance

Run tools/extract_rafm_native_peaks.py --root . --out <fresh-file.json> with this checkout's src on PYTHONPATH, then tools/validate_qg_peak_identifications.py --root . --native <fresh-file.json> --out <fresh-validation.json>. Existing output files are refused. Use the installed repository dependencies; no shared checkout or environment was modified.

native_baseline.json and native_baseline_extractor.py preserve the original independent 27-acquisition extraction from e794af6. Source hashes identify that extraction separately from the final audit code. The baseline input hashes were attached after extraction, with unchanged spectral inputs verified; this timing is explicitly recorded in the cache lineage. candidate_replay.json contains four full independent replays using the candidate collector. All returned native identities, channels, energies, net counts, uncertainties and significances were identical to baseline within 1e-10. native_final.json adds their tentative candidates. Final audit checks every paired input's hash, exact report/raw pairing, complete unique report coverage, and the source-bound correction. Current audit hashes do not imply the baseline was run with later code.

all_roi_identifications.csv and final_validation.json contain the complete row audit, confidence classification and original-source bindings. independent_source_inventory.csv is the antagonist's text-based inventory, independent of the parser. Baseline and reviewer files are historical evidence; final_tests.xml and validation_receipt.json record the final checks.

Final regression execution: **162 passed in 159.34 seconds** across ten suites. After making the three low-significance wire statuses explicitly tentative, the validator suite passed **24 tests in 4.21 seconds**. These runs contain **163 distinct passing test cases**, with overlapping reruns counted once. Changed-line formatting of rafm_workflow.py was verified to preserve its Python AST. Luna independently checked the source-bound activity artifact and confidence classification.
