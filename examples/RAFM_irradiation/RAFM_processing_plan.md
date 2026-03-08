# RAFM Irradiation Processing Plan (FluxForge)

This runbook defines how to process RAFM irradiation gamma spectra in FluxForge using one shared measured background spectrum and the explicit `rafm_25cm` detector profile.

## Inputs
- `examples/RAFM_irradiation/background.ASC` (shared measured background)
- `examples/RAFM_irradiation/raw_gamma_spec/**/*.ASC` (RAFM samples and flux wires)
- `--profile rafm_25cm` for the shared 25 cm detector efficiency/resolution/background defaults
- Optional user-provided calibration and efficiency overrides
- Recommended results root: `examples/RAFM_irradiation/results/`

## Default Result Locations
- Workflow analysis artifacts: `examples/RAFM_irradiation/results/analysis_json/`
- Per-spectrum count tables: `examples/RAFM_irradiation/results/counts/`
- Per-spectrum comparison reports: `examples/RAFM_irradiation/results/reports/`
- Validation tables: `examples/RAFM_irradiation/results/tables/`
- Per-spectrum line diagnostics: `examples/RAFM_irradiation/results/tables/line_diagnostics/`
- Per-spectrum QG internal consistency audits: `examples/RAFM_irradiation/results/tables/qg_internal_consistency/`
- Validation and unfolding plots: `examples/RAFM_irradiation/results/plots/`
- Unfolding method artifacts: `examples/RAFM_irradiation/results/unfolding/`

## Processing Sequence
1. Ingest each sample spectrum (`.ASC/.TXT/.txt`) and parse header metadata.
2. Resolve calibration and efficiency values.
   - If user overrides are provided, they take precedence.
   - Raw `.ASC` spectra keep their per-file header calibration `Energy = A + B*Ch + C*Ch^2`.
   - The RAFM example workflow now explicitly overrides raw-file energy calibration with the shared `rafm_25cm` profile calibration `Energy = -1.694 + 0.4996*Ch + 6.710E-08*Ch^2` because the RAFM `@25 cm` QG exports use one common detector calibration for this campaign.
   - `--profile rafm_25cm` fills the shared 25 cm efficiency, resolution, and background defaults for all RAFM samples and flux wires.
   - The bundled `rafm_25cm` profile matches the QG detector-model coefficients from `RAFM-1_25cm.ANS`:
     - `C1=-20.26`, `C2=10.29`, `C3=-1.655`, `C4=0.08666`
     - `Geometry Factor A=3.48E-03`, `T1=1000 um`, `DI=6.45 cm`, `DL=700 um`, `AI=0 deg`
     - `Resolution = 1.389 + 7.800E-04*E - 4.072E-08*E^2`
   - Processed `.txt` spectra keep file-derived energy calibration unless explicitly overridden.
3. Apply measured background subtraction before peak/activity analysis.
   - Default scale mode: `live` (sample live time / background live time)
   - Alternate modes: `real`, `manual`
   - FluxForge now calibrates/resamples the measured background onto the sample energy grid before subtraction when the sample and background use different energy calibrations.
4. Propagate subtraction uncertainty per channel:
   - `net_i = sample_i - f * background_i`
   - `var_i = sample_i + f^2 * background_i`
5. Continue analysis using signed counts for storage and uncertainty propagation.
   - SNIP uses an internal offset working copy for background estimation, then shifts the background estimate back to physical space.
6. Perform peak detection, fitting, isotope assignment, and activity calculations.
   - Generic RAFM samples now use an exploratory peak search plus targeted recovery against the committed RAFM gamma library.
   - The exploratory search stays at `3 sigma`.
   - The targeted recovery pass now uses a separate threshold of `2 sigma` so weak expected lines can be recovered without lowering the general unidentified-peak threshold.
   - The generic RAFM comparison library now collapses effectively identical cross-isotope line collisions inside `0.15 keV` by keeping the stronger authoritative line. This removes the known `Ta182` vs `Tb154m` ambiguity at `1189 keV` without using QG to choose isotopes at runtime.
   - This is intended to reduce missed QG peaks while still preserving unidentified FluxForge peaks for review.
   - Peaks below `80 keV` are excluded from RAFM validation and reporting; this matches the current RAFM LDRD activation-side utilities.
- Flux-wire expected products and search-line targets are loaded from `src/fluxforge/data/flux_wire_catalog.json`, not hardcoded in `flux_wire_analysis.py`.
- Flux-wire half-lives, emission probabilities, and emission-probability uncertainties are loaded from `src/fluxforge/data/rafm_decay_data.json`, which is a committed subset of the local `testing/actigamma` decay library.
- Flux-wire activities are computed directly from FluxForge peak fits and ROI uncertainty propagation; the analysis path no longer scales raw net counts or efficiencies from QG processed outputs.
- Simplified flux-wire unfolding defaults are loaded from `src/fluxforge/data/flux_wire_unfolding_defaults.json` instead of large inline dictionaries in `flux_unfold.py`.
 - Flux-wire QG `GROSS/NET` parity now uses raw-spectrum local-ROI counts:
   - shared `@25 cm` energy calibration
   - ROI width `4.0 * FWHM`
   - background width `1 channel`
   - background gap `0 * FWHM`
 - For strong broad peaks, FluxForge now expands the flux-wire QG count-comparison ROI using five-point-smoothed local minima within the `32 channel` capture range. This improves raw `GROSS/NET` parity on broad `Co` and `Sc` lines without changing the background-adjusted activity path.
 - Flux-wire activity calculations still use the background-adjusted analysis path; QG count parity and FluxForge activity parity are now tracked separately.
7. Export analysis artifacts and validation outputs.
   - Write one raw-vs-QG comparison plot per matched sample.
   - Write one text report per raw sample listing missing QG peaks, isotope mismatches, remaining unidentified FluxForge peaks, and parity failures.
   - Write one line-diagnostics CSV per matched sample with:
     - QG and FluxForge gross counts
     - QG and FluxForge net counts
     - FluxForge raw-spectrum count parity fields separated from the background-adjusted activity path
     - QG and FluxForge line activities at measurement time
     - QG-implied efficiency from the processed file
     - FluxForge efficiency used for activity conversion
     - bundled branching ratio vs QG `rad_int`
     - a simple diagnostic bucket (`count_parity_failure`, `efficiency_or_activity_conversion_bias`, `gamma_library_mismatch`, etc.)
   - Write one per-sample QG internal-consistency CSV that compares each processed-file nuclide header activity against the set of per-line activities in the same file.
8. Optionally export channel-by-channel CSV tables:
   - background-adjusted counts
   - background-adjusted, calibrated, efficiency-corrected counts
9. For manual inspection before the GUI exists, optionally export calibrated spectrum plots and manual ROI peak reports.
   - `python -m fluxforge.cli.app spectrum-plot ...` writes a headless-safe spectrum-vs-energy plot
   - `--background-subtracted` plots the measured-background-subtracted spectrum
   - `--manual-peaks-file` overlays user-supplied ROIs in CSV or JSON
   - `--save-peak-report` writes a FluxForge peak-report artifact from those manual ROIs
   - `python -m fluxforge.cli.app peaks --manual-peaks-file ...` can write the same manual peak report without making a plot
9. Run the RAFM validation workflow to compare FluxForge outputs to the committed QG gold-standard files and to unfold the flux-wire spectrum with all supported methods.

## RAFM4 Timing Rule
- `RAFM4-*_15dEOI` spectra are interpreted as counts after the phase-2 whale-tube irradiation.
- FluxForge now records this explicitly in artifact timing metadata as `irradiation_phase = phase2_whale_tube`.
- This is separate from the RAFM3 phase-1 rabbit-tube timing used for the short-cooldown spectra.

## Per-Sample Validation Outputs
- `results/plots/comparisons/<sample>_vs_qg.png` compares each raw sample only to its paired QG processed result.
- `results/reports/<sample>_comparison.txt` summarizes:
  - QG peaks not identified by FluxForge
  - matched energies with isotope mismatches
  - QG nuclides missing from FluxForge activity results
  - peak gross-count parity failures
  - peak-count and activity parity failures
  - line-level diagnostic flags
  - QG internal consistency flags
  - FluxForge peaks still left unidentified
  - FluxForge-only identified peaks with no QG counterpart
- `results/tables/line_diagnostics/<sample>_line_diagnostics.csv` is the main forensic table for remaining parity failures.
- `results/tables/qg_internal_consistency/<sample>_qg_consistency.csv` flags processed-file isotopes whose header activity disagrees strongly with their own line activities.
- `results/tables/flux_wire_count_disagreement.csv` gives the direct per-line QG-vs-FluxForge gross/net deltas for the matched flux-wire peaks.
- `results/tables/flux_wire_count_disagreement_summary.md` summarizes the worst remaining flux-wire count mismatches after the latest count-path changes.
- Flux-wire activity comparisons use the QG activity values reported at measurement time, not EOI-corrected activities.
- Flux-wire QG parity is always performed on raw FluxForge peak/activity results before any Cd-ratio post-processing. Cd ratios are derived only after the raw-vs-QG comparison is written.

## Uncertainty Handling Notes
- Background subtraction uncertainty is propagated channel-by-channel before peak fitting.
- The RAFM background subtraction path is now energy-aligned: if the raw sample and `background.ASC` have different calibrations, the background is resampled onto the sample energy grid before subtraction.
- Targeted peak fitting now keeps the propagated ROI uncertainty when it exceeds the Gaussian fit uncertainty, so background-adjusted spectra do not under-report net-count uncertainty.
- Emission-probability uncertainty for bundled flux-wire isotopes now comes from committed decay data instead of a fixed global `1%` assumption.
- Efficiency uncertainty is now zero unless the detector profile or user override explicitly provides one; FluxForge no longer injects a hidden global `5%` efficiency uncertainty term.
- Very short-lived isotopes counted long after irradiation can produce unstable back-corrections to EOI; FluxForge now records `None` instead of overflowing those EOI values.

## Current Open Validation Problems
- The workflow still reports failing parity cases for several RAFM3, RAFM4, and Ti wire spectra.
- The detailed discrepancy source for each spectrum is now in `results/reports/<sample>_comparison.txt`.
- The line-by-line source of each parity failure is now in `results/tables/line_diagnostics/<sample>_line_diagnostics.csv`.
- The largest remaining issues are measurement-time count/activity parity differences, not missing paired-plot/report generation.
- Current flux-wire count forensic read:
  - `Sc-RAFM-1` is now close on both major `Sc46` lines after switching strong broad peaks to smoothed-valley comparison bounds
  - `Co-Cd` improved substantially, but `1332 keV` still carries a remaining `~7%` net deficit and `~14%` gross deficit
  - `Ti-RAFM-1a` still matches `Sc47 @ 159 keV` well, but `Sc48 @ 175 keV` remains a real weak-line count problem
  - applying the shared calibration to `background.ASC` and energy-aligning background subtraction was necessary for the corrected activity path, but it does not explain the remaining raw `GROSS/NET` parity failures
- The detector-efficiency path still needs a separate audit after the count-path discrepancies are reduced further; this runbook now treats count parity and efficiency/activity parity as separate problems.
- The flux-wire analysis module no longer contains the old hardcoded product catalog or reference-derived scaling helpers; the remaining parity failures are now real model/data issues rather than hidden QG coupling.
- `flux_unfold.py` now loads its simplified sample-property and reaction-default dictionaries from bundled data files, but the workflow still needs experiment-specific sample metadata to replace the remaining generic defaults cleanly.
- `testing_validation/rafm_results/FLUX_WIRE_REPORT.md` shows that expected flux-wire peaks are being found reliably; the remaining flux-wire gap is quantitative activity parity, especially below about `725 keV`.
- `Ti` wire `Sc48` comparisons still require manual audit because the QG processed export contains internally inconsistent peak-vs-header activity values for that nuclide.
- Current forensic read of the parity gap:
  - peak-area extraction is usually close to QG for the main flux-wire lines
  - the dominant remaining bias for many wires is efficiency/activity conversion, not missed peaks
  - `Sc48` is a separate blocker because the processed export itself is internally inconsistent

## Local Reference Review Notes
- `testing/actigamma/` is the local authoritative decay-data reference that FluxForge now mirrors into committed data files for RAFM/flux-wire work.
- `testing/irrad_spectroscopy/` is useful for spectrum-analysis structure:
  - keep isotope tables in data files
  - validate isotope assignments against expected stronger lines
  - preserve local-background and fit-quality checks
- `testing/gamma_spec_analysis/gs_analysis.py` is useful as a simple peak-area reference:
  - explicit trapezoid background subtraction
  - explicit ROI net-count calculation
  - smoothing plus peak-finder preprocessing
- FluxForge should continue using those repositories only as design references and validation comparators, never as runtime dependencies.

## Important Warnings and Interpretation
- If background subtraction is enabled and no background spectrum is supplied, FluxForge warns and proceeds with raw counts.
- `--profile rafm_25cm` avoids that warning for RAFM example data by supplying the shared background file automatically.
- Background subtraction can produce negative bins. In `hybrid` mode this is expected and retained for uncertainty accounting.
- SNIP no longer clips RAFM background-subtracted spectra before background estimation; it uses an internal offset instead.
- Final corrected CSV export requires usable efficiency coefficients. If they are missing, FluxForge warns and skips that export.

## Recommended RAFM Batch Command
```bash
python -m fluxforge.cli.app ingest-batch \
  --input-dir examples/RAFM_irradiation/raw_gamma_spec \
  --profile rafm_25cm \
  --background-scale-mode live \
  --output-dir examples/RAFM_irradiation/results/spectrum_artifacts \
  --background-adjusted-dir examples/RAFM_irradiation/results/background_adjusted \
  --final-corrected-dir examples/RAFM_irradiation/results/final_corrected
```

This command applies the same `background.ASC` file and shared 25 cm detector efficiency profile to every raw spectrum in `raw_gamma_spec/`, while still keeping each raw file's own `A,B,C` energy calibration.

## Full RAFM Validation Workflow
```bash
cd FluxForge
PYTHONPATH=src python examples/RAFM_irradiation/run_validation.py
```

If you want the workflow to generate the full result bundle even when the QG parity thresholds are violated, use:

```bash
cd FluxForge
PYTHONPATH=src python examples/RAFM_irradiation/run_validation.py --no-fail
```

The workflow will:
- analyze every raw spectrum in `raw_gamma_spec/`
- apply the shared measured background from `background.ASC`
- use the `rafm_25cm` detector profile for efficiency and resolution defaults
- compare matched raw/QG pairs and list unmatched raw/QG files explicitly
- write one raw-vs-QG comparison plot and one text discrepancy report for each analyzed raw sample
- compute flux-wire reaction rates, Cd ratios, and unfolding results for `DISCRETE`, `GLS`, `GRAVEL`, and `MLEM`
- write summary files at `examples/RAFM_irradiation/results/validation_summary.json` and `examples/RAFM_irradiation/results/validation_summary.md`

## Single-Spectrum Export Example
```bash
python -m fluxforge.cli.app ingest \
  --input examples/RAFM_irradiation/raw_gamma_spec/RAFM4/RAFM4-B_15dEOI.ASC \
  --profile rafm_25cm \
  --output examples/RAFM_irradiation/results/spectrum_artifacts/RAFM4/RAFM4-B_15dEOI.json \
  --save-background-adjusted examples/RAFM_irradiation/results/background_adjusted/RAFM4/RAFM4-B_15dEOI_background_adjusted.csv \
  --save-final-corrected examples/RAFM_irradiation/results/final_corrected/RAFM4/RAFM4-B_15dEOI_final_corrected.csv
```

## Explicit Override Example
```bash
python -m fluxforge.cli.app ingest \
  --input examples/RAFM_irradiation/raw_gamma_spec/RAFM4/RAFM4-B_15dEOI.ASC \
  --profile rafm_25cm \
  --efficiency-coefficients "-3.743,2.167,-0.3724,0.02036,0.296" \
  --output examples/RAFM_irradiation/results/spectrum_artifacts/RAFM4/RAFM4-B_15dEOI.json \
  --save-final-corrected examples/RAFM_irradiation/results/final_corrected/RAFM4/RAFM4-B_15dEOI_final_corrected.csv
```

This override command replaces the profile efficiency with the user-supplied coefficients while still using the raw file's own energy calibration unless `--energy-calibration` is also provided.

## Manual ROI Inspection Example
Manual ROI CSV format:

```csv
label,left_keV,right_keV,isotope
Sc47_main,158.6,160.1,Sc47
Sc48_175,174.7,176.2,Sc48
```

Headless spectrum plot with manual ROI overlays:

```bash
python -m fluxforge.cli.app spectrum-plot \
  --input examples/RAFM_irradiation/raw_gamma_spec/flux_wires/Ti-RAFM-1a_25cm.ASC \
  --profile rafm_25cm \
  --background-subtracted \
  --manual-peaks-file examples/RAFM_irradiation/manual_peaks_ti.csv \
  --output examples/RAFM_irradiation/results/plots/manual/Ti-RAFM-1a_25cm_manual.png \
  --save-peak-report examples/RAFM_irradiation/results/manual/Ti-RAFM-1a_25cm_manual_peaks.json
```

Manual ROI peak report only:

```bash
python -m fluxforge.cli.app peaks \
  --spectrum-file examples/RAFM_irradiation/raw_gamma_spec/flux_wires/Ti-RAFM-1a_25cm.ASC \
  --profile rafm_25cm \
  --background-subtracted \
  --manual-peaks-file examples/RAFM_irradiation/manual_peaks_ti.csv \
  --output examples/RAFM_irradiation/results/manual/Ti-RAFM-1a_25cm_manual_peaks.json
```

Supported manual ROI fields:
- `label`
- `isotope`
- `left_keV`, `right_keV`
- or `left_channel`, `right_channel`

The manual peak report keeps:
- raw gross counts inside the chosen ROI
- integrated counts from the selected analysis spectrum
- ROI endpoints in both channels and keV
- a `background_subtracted` flag so the user can track whether the ROI area came from raw or background-adjusted counts

## Validation Checklist
- Shared background subtraction applied for all RAFM runs.
- Scale factor recorded in artifact metadata.
- Per-channel uncertainties present after subtraction.
- Export paths are explicit and do not depend on external repositories.
- FluxForge results reproducible from committed inputs and metadata.
