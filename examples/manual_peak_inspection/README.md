# Manual Peak Inspection Example

This example is separate from the main RAFM validation workflow.

Use it when you want to inspect calibrated gamma spectra over SSH, save a spectrum-vs-energy plot, and define peak endpoints manually before the GUI is ready.

Do not use this example to replace the automatic RAFM or flux-wire validation path in `examples/RAFM_irradiation/`. The main RAFM example should continue using FluxForge's automatic peak identification and counting.

## Included examples
- Flux wire example:
  - input: `examples/RAFM_irradiation/raw_gamma_spec/flux_wires/Ti-RAFM-1a_25cm.ASC`
  - ROI file: `examples/manual_peak_inspection/manual_flux_wire_ti_rafm_1a.csv`
- RAFM sample example:
  - input: `examples/RAFM_irradiation/raw_gamma_spec/RAFM4/RAFM4-B_15dEOI.ASC`
  - ROI file: `examples/manual_peak_inspection/manual_rafm4_b_15d.csv`

## Manual ROI file format
Supported columns:
- `label`
- `isotope`
- `left_keV`, `right_keV`
- or `left_channel`, `right_channel`

The ROI bounds are only examples. Users should inspect the plotted spectrum and adjust them as needed.

## Install profile

Use the CLI-only install unless you also want the GUI:

```bash
pip install -e .
```

## Save a calibrated plot with ROI overlays
Flux-wire example:

```bash
fluxforge spectrum-plot \
  --input examples/RAFM_irradiation/raw_gamma_spec/flux_wires/Ti-RAFM-1a_25cm.ASC \
  --profile rafm_25cm \
  --background-subtracted \
  --manual-peaks-file examples/manual_peak_inspection/manual_flux_wire_ti_rafm_1a.csv \
  --output examples/manual_peak_inspection/output/Ti-RAFM-1a_25cm_manual.png \
  --save-peak-report examples/manual_peak_inspection/output/Ti-RAFM-1a_25cm_manual_peaks.json
```

RAFM sample example:

```bash
fluxforge spectrum-plot \
  --input examples/RAFM_irradiation/raw_gamma_spec/RAFM4/RAFM4-B_15dEOI.ASC \
  --profile rafm_25cm \
  --background-subtracted \
  --manual-peaks-file examples/manual_peak_inspection/manual_rafm4_b_15d.csv \
  --output examples/manual_peak_inspection/output/RAFM4-B_15dEOI_manual.png \
  --save-peak-report examples/manual_peak_inspection/output/RAFM4-B_15dEOI_manual_peaks.json
```

## Create a manual peak report without plotting
```bash
fluxforge peaks \
  --spectrum-file examples/RAFM_irradiation/raw_gamma_spec/flux_wires/Ti-RAFM-1a_25cm.ASC \
  --profile rafm_25cm \
  --background-subtracted \
  --manual-peaks-file examples/manual_peak_inspection/manual_flux_wire_ti_rafm_1a.csv \
  --output examples/manual_peak_inspection/output/Ti-RAFM-1a_25cm_manual_peaks.json
```

## What the manual peak report contains
- raw gross counts inside the selected ROI
- integrated counts from the chosen analysis spectrum
- ROI endpoints in channels and keV
- a `background_subtracted` flag

## Why this example exists
- It gives users an SSH-safe inspection path now.
- It provides a direct target for future GUI behavior.
- It keeps the manual workflow out of the main automatic RAFM validation path.
