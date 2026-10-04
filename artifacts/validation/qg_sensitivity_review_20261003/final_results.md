# Final native replay and diagnostic results

Fresh complete replay: **32 raw/report pairs, 33 reports, 300 reference ROI rows**. All raw channels were independently extracted before any reference overlay. Native input and package source hashes match the final tree.

| Status | Rows |
|---|---:|
| Confirmed same identity | 268 |
| Confirmed reviewed Tb-to-Ta correction | 1 |
| Positive tentative associations | 11 |
| Ambiguous identity | 1 |
| Reference-center matching failure | 1 |
| Positive rows lacking raw data | 12 |
| Original zero-net/blank-activity rows | 6 |

**Strict all-report acceptance remains false.** Recovery reduces missing positive source coverage from 47 to 12; it does not promote tentative peaks or resolve ambiguous identities. There are 282 positive rows with raw data: 269 confirmed, 11 tentative and two identity/energy discrepancies.

The ambiguous A-300s 846.679-keV centroid has 207.093 sigma, but no accepted isotope label. The A-300s QG Mn56 row at 2529.10 keV is classified `missing_peak` by the strict energy matcher; the raw native Mn56 fit at 2522.338 keV is present with 7.345 sigma, net 376.229 ± 51.222 counts. This is a reference-center discrepancy, not absent physical signal. No extra reference correction was admitted.

## Positive tentative rows

| Report | Identity / QG center keV | Native association sigma | Diagnostic combined-setting sigma |
|---|---|---:|---:|
| flux_wires/Ti-RAFM-1_25cm.txt | Sc48 / 175.23 | 0.497 | 3.897 |
| flux_wires/Ti-RAFM-1a_25cm.txt | Sc48 / 175.28 | 1.708 | 4.804 |
| flux_wires/Ti-RAFM-1b_25cm.txt | Sc48 / 175.26 | 0.232 | 2.107 |
| RAFM3/RAFM3-A_24hrEOI.txt | Mn56 / 2112.88 | 1.919 | 5.277 |
| RAFM3/RAFM3-A_300sEOI.txt | W187 / 552.01 | 1.838 | 5.505 |
| RAFM3/RAFM3-A_300sEOI.txt | W187 / 618.32 | 1.410 | 3.706 |
| RAFM3/RAFM3-A_300sEOI.txt | W187 / 773.35 | 1.396 | 2.981 |
| RAFM3/RAFM3-C_300sEOI.txt | W187 / 773.21 | 1.104 | 2.580 |
| RAFM3/RAFM3-C_4dEOI.txt | Fe59 / 1098.75 | 1.330 | No accepted identity/centroid |
| RAFM3/RAFM3-N_24hrEOI.txt | Mn56 / 2112.80 | 1.986 | 4.275 |
| RAFM3/RAFM3-N_4dEOI.txt | Fe59 / 1099.10 | 1.780 | 5.192 |

The diagnostic uses all library neighbors within 12 keV and threshold 2; wire production uses threshold zero. These values therefore are not interchangeable with native production scores. Ten of eleven tentative associations cross 2 sigma with the combined configuration; the all-neighbor Fe59 C-4d components fail centroid support. No parameter changes were applied to production defaults.

## Six source zero rows

Native analysis detects four at evaluated energies: B/C/N Mn56 near 2523.06 keV and N W187 near 551.49 keV. C W187 is a 1.651-sigma candidate; A Fe59 has no accepted native fit near 1099.25 keV. All six original vendor zero rows remain recorded as such. The fresh diagnostic improves C W187 to 4.545 sigma with the combined configuration; the A Fe59 all-neighbor case remains without an accepted result.

Full sweep: 20 cases × four configurations = 80 experiments. Cr51 and corrected Ta182 controls remain detected. The Mn56 846-keV all-neighbor control retains a strong but ambiguous centroid and withholds activity.

## Verification

154 tests passed in the final focused suite; Luna independently passed 39 GUI/identity/sensitivity tests and 13 recovery tests. The independent acquisition agent verified all five immutable native clock/count sources. Luna reached its usage limit before reviewing the final 32-spectrum replay; the root agent checked the final counts, source hashes and spectrum figure. The separate workflow checks are recorded in the receipt.

Plot: `reference_zero_regions.png`. Original acquisition bytes: `examples/RAFM_irradiation/recovered_qg_sources`. Scientific admission remains false; shared efficiency covariance, calibration/background applicability and near-contact geometry qualification remain separate.
