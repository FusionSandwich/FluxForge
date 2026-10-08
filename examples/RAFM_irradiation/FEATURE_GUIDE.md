# UWNR / INL RAFM feature guide

Use this dataset to explore the spectroscopy, activation-review and planning
features below. A successfully executed example does not certify its activities
or neutron spectrum. The independent raw-recovery results and the historical
QG-derived planning example have different evidence bases.

## Data added from the October 2026 downloads

- **32 canonical ASC acquisitions:** the existing 29 plus three RAFM2 spectra
  measured in April/July 2025. RAFM2 has no source-backed irradiation schedule,
  specimen mass/material mapping or independent QG report in these downloads.
- **87 converted files:** 29 each CHN, SPE and binary SPC. CHN/SPE preserve every
  channel count but lose acquisition timing and usable calibration; use them
  for reader/count-integrity examples. FluxForge currently rejects binary SPC.
  Use the corresponding canonical ASC for activity analysis.
- **Four material compositions:** EUROFER97_2, _3, _4 and CNA, extracted from
  `RAFM_Composition.xlsx` with worksheet coordinates, units and source hash in
  `metadata/material_compositions.json`. These are nominal/averaged values.
  Missing entries remain null. The workbook's isotope columns include a Mn54
  label and were deliberately excluded; element Mn is retained.
- **Historical comparison inputs:** one QG-derived activation CSV and eleven
  simulated activation CSVs in `reference_data/`. Simulations are predictions,
  and the QG-derived table is not an independent measurement validation set.
- **Provenance:** `metadata/archive_manifest.json` binds inputs and conversions.
  Measured/archive files use exact byte hashes. Existing runtime JSON/profile
  hashes normalize CRLF to LF for checkout portability; no values are changed.

The original PDFs, metrology memo, spreadsheets, reactor photograph/background
and all archive contents are retained in the local extraction directory. The
example contains curated numerical inputs, rather than copies of correspondence
or the historical source programs.

## Runnable checks

From the repository root, using the existing Python environment:

```powershell
$env:PYTHONPATH = (Join-Path $PWD 'src')
$env:PYTHONUTF8 = '1'
$env:MPLBACKEND = 'Agg'
$env:OPENBLAS_NUM_THREADS = '1'
python -m fluxforge.examples.rafm_feature_example --output-root C:/Temp/rafm_features
```

This checks the complete file inventory, parses all 32 raw acquisitions, performs
signed measured-background subtraction, round-trips counting covariance through
JSON, checks all 87 conversions, sums two distinct raw Co acquisitions, and
round-trips a Co spectrum through CSV. It also produces a conditional Co1173 ROI
and Currie sensitivity calculation using the recorded detector profile.

Run fresh peak/activity extraction without QG substitution:

```powershell
python tools/audit_rafm_raw_recovery.py --all-raw --output-root C:/Temp/rafm_raw
```

For the newly added spectra only, append:

```powershell
--sample RAFM2_Long_72h_EOI --sample RAFM2_Long_144h_EOI --sample RAFM2_Long_70d_EOI
```

Each new RAFM2 output explicitly withholds EOI correction and reports its QG
comparison as unknown. Even count-start activities are conditional on the shared
profile; the old acquisition dates do not establish a matched efficiency setup.

Run the historical planning demonstration:

```powershell
python examples/RAFM_irradiation/run_phase6_ldrd_worked_example.py --sample-id RAFM4-C_15dEOI --output-root C:/Temp/rafm_planning
```

Read `qualification_receipt.json` and `WORKED_EXAMPLE_SUMMARY.md`. This uses the
bundled historical analysis and unfolding output; it does not rerun or qualify
the upstream activity measurements. Its inferred half-lives and optimized
schedules remain demonstration results.

## Feature coverage and evidence

| Feature | Example input/entrypoint | Evidence and limits |
| --- | --- | --- |
| Spectrum ingestion, calibration inspection, acquisition QC | All 32 ASC; feature runner | Executed; records embedded coefficients and real/live times |
| Alternate-format readers | 29 CHN + 29 SPE | Exact bin identity checked; timing/calibration lost |
| Binary SPC import | 29 SPC | Unsupported, explicitly rejected |
| Background subtraction, negative bins, counting covariance | `background.ASC`; feature runner | All 32 processed; signed counts and sparse covariance survive JSON |
| Manual ROI/statistics and detector sensitivity | Co1173 ROI in feature runner; GUI ROI tools | Covariance-aware ROI sum; MDA conditional on unqualified profile uncertainty |
| Spectrum addition and CSV interchange | Co and Co-Cd raw acquisitions | Count/variance arithmetic checked; CSV energies and supplied uncertainties retained at export precision |
| Peak search, targeted fits, overlap/assignment diagnostics | Raw-recovery driver | Existing 29-spectrum September review plus three new RAFM2 extractions |
| Isotope ID and measurement-time activity | Raw driver, per-spectrum reports | Diagnostic results; ambiguous or unsupported assignments withheld |
| Decay correction and irradiation history | RAFM3/4 schedules; existing workflow | Source-backed timing for supported samples; RAFM2 EOI unavailable |
| Monitor atoms, reaction rates and Cd ratios | `flux_wire_metadata.json`, raw workflow | Co masses already adjusted to element mass; existing uncertainty/ratio limits still apply |
| Material composition review | `metadata/material_compositions.json` | Source coordinates and nulls preserved; no automatic isotopic or concentration inference |
| Measured-versus-predicted activation review | `reference_data/` plus raw/QG results | Historical data supplied; reconcile schedules, volume and normalization before quantitative comparison |
| Activity review and decay inventory | Phase 6 script | Executed using historical bundled analysis; twelve inventory rows |
| Line masking/interference review | Phase 6 script | Executed; six masking candidates |
| Irradiation/cooling/count-time optimization | Phase 6 script | Five objectives executed; conditional planning, not validated optimal experimental settings |
| Second irradiation | Phase 6 script | Four candidates executed; same conditional input basis |
| Experimental bundle export | Phase 6 script | `.ffexp` and related outputs produced, source/output hashes recorded |
| Neutron unfolding and response/covariance review | Existing raw workflow and unfolding tools | RAFM inputs exist; this archive integration makes no new physical unfolding qualification claim |
| ASTM and k0 comparisons | Existing `compare_astm_*` scripts/workflows | Available demonstrators; standards-specific and calibration evidence remain necessary |
| GUI overlays, ROI editing, project review and report export | Canonical ASC + measured background + existing reports | Inputs supplied; automated desktop interaction was not performed in this intake |

This dataset cannot substantiate live detector acquisition, scintillator-specific
response, certified-source calibration, trained ANN accuracy, or plutonium
isotopics. Use appropriate separate examples for those features.

## Reconciliation decisions

1. Sixteen downloaded RAFM3/4 ASC files have alternative A/B/C calibration header
   coefficients but identical channel counts and acquisition times. Their hashes
   and both coefficient sets are recorded in `header_variants`. Committed
   headers were preserved. The full validation workflow also enables an explicit
   profile energy override; inspect effective calibration in its outputs.
2. Four reports named RAFM3-{A,B,C,N}_15dEOI are byte-identical to existing RAFM4
   reports. They do not create four additional acquisitions.
3. Co is **0.46 wt%**, with **4.0661 mg** and **3.6703 mg already Co element mass**.
   Do not apply 0.0046 again. The workbook's geometry calculations use pure-Co
   density on adjusted element mass, so those wire lengths are not certified
   specimen geometry. Keep the existing mass-basis safeguard.
4. Workbook material density 7.7 g/cm³ is an assumption and differs from 7.87
   used by the existing RAFM geometry example. The new composition table does
   not overwrite that model or certify either density.
5. `South 4hr Background Terminal.ANS` is retained locally. Matching detector
   setup/geometry is unproven, so it does not replace `background.ASC`.
6. OneDrive_2 contains download-error files only; the downloaded tables ZIP
   contains seven empty CSVs. They contribute no usable measurements.

Six QG records still lack canonical raw partners: Cu-Cd, Fe-Cd, RAFM1 70d and
three RAFM3-A acquisitions. These downloads did not fill those gaps.

The October execution receipts are in `docs/reviews/RAFM_*2026-10-03.json` and
`docs/reviews/RAFM_ARCHIVE_INTAKE_2026-10-03.md`. The September raw accuracy review
remains applicable to the unchanged 29 earlier acquisitions.
