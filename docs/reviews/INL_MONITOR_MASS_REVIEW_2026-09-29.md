# INL RAFM-1 monitor mass review

## Source and corrections

The local original `CCN 257311 (2023) MEMO TVH Metrology Package_RAFM_Wisconsin.docx`
has SHA-256 `e6e0697f7e9fbba457a9700a4a4831efb5fad68b7eb1e6e47ef72866b981b652`.
Its designation table records these prepared RAFM-1 specimens:

| Identifier | Listed mass (mg) | Meaning | QA identifiers |
| --- | ---: | --- | --- |
| Co-RAFM-1 | 4.0661 | Adjusted Co element mass | 376298 |
| Co-Cd-RAFM-1 | 3.6703 | Adjusted Co element mass | 376298, 368486 |
| Cu-RAFM-1 | 1.3748 | Cu wire mass | 376301 |
| Cu-Cd-RAFM-1 | 12.9738 | Cu wire mass | 376301, 368486 |

Both RAFM-1 Co rows say: "Wire (0.46 wt% Co), mass adjusted accordingly".
The prepared-row dates are July 29, 2025 for Co and July 30, 2025 for Cu.
The source file's historical 2023 filename does not date these prepared rows.

Therefore the Co values must not receive another factor 0.0046. Metadata now
separates `alloy_co_mass_fraction` (physical composition) from
`element_mass_fraction` (the factor applied to the supplied mass). With
`mass_basis=element_mass`, the latter must equal 1; conflicting values fail.
This also removes the unsupported 0.9999 pure-Co default from these two inputs.
The mass basis is saved in raw analysis JSON and QG benchmark sample outputs.
Whole-specimen specific activity and radioactive mass fraction are omitted for
Co because the supplied values are element masses, not measured specimen masses.

Cu-Cd metadata incorrectly repeated the bare Cu mass. It now uses 12.9738 mg.
No other monitor masses were changed. The default Cu purity of 0.9999 remains an
explicitly unverified assumption; this source row does not establish purity.

## Verification performed

`PYTHONUTF8=1 MPLBACKEND=Agg python -m pytest tests/test_rafm_workflow.py tests/test_no_silent_defaults.py tests/test_irradiation_history.py -q`

Result: **43 passed**. Tests include independent Co target-atom arithmetic,
rejection of a second alloy correction (including NaN), invalid mass basis,
Cu-Cd versus bare-Cu mass identity, and suppression of misleading specimen metrics
in the actual Co raw-spectrum artifact.

The committed UWNR Co and Co-Cd raw spectra were replayed with the current
`analyze_flux_wire_sample` workflow. Output review checked the saved mass source,
target atom counts, unchanged activities and absence of specimen-specific fields:

| Spectrum | Target atoms | Rate (per target atom per second) |
| --- | ---: | ---: |
| Co-RAFM-1_25cm | 4.154979967868026e19 | 1.6461629758728343e-12 |
| Co-Cd-RAFM-1_25cm | 3.750528264446526e19 | 2.1065749846281812e-13 |

The 0.01% change from the earlier Co rate is removal of the bundled 0.9999
purity multiplier. It is not a factor-217 change. The default counting method
uses QG information, so QG agreement is not an independent activity validation.

`run_qg_benchmark(..., max_spectra=3)` processed Co-Cd, Co and Cu-Cd, producing
three sample rows and three reaction rows with mass provenance in the summary.
Cu-Cd has a committed processed QG report but no committed raw spectrum or
resolved irradiation/decay timing. Its measured Cu-64 activity is 1,090,982 Bq.
With natural Cu-63 abundance 0.6917 and the existing assumed Cu purity, its
target count is 8.50362748019437e19. Holding all other inputs fixed, correcting
the mass multiplies target-normalized activity by 1.3748/12.9738 =
0.10596741124419987. This checks the normalization change; it does not establish
a Cu-Cd absolute reaction rate.

## Remaining limits

This metrology record establishes the intended composition and adjusted mass
basis of the named Co specimens. It does not supply a composition certificate,
composition uncertainty, full measured body geometry or complete detector/rate
uncertainty budget. Nominal Cd thickness in related INL memos is not proof of
as-built shielding geometry. Cu attenuation through its vanadium tube, Cu purity,
Cu-Cd timing and count-specific efficiency still need source-bound review.
No issue is closed on the strength of this mass-input correction alone.
