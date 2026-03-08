# Manual ROI Spectrum Inspection

## Problem
The current RAFM/flux-wire workflow still needs a user-driven inspection path for peak windows before the GUI is ready. Users working over SSH need to:
- export a calibrated spectrum-vs-energy plot
- view either raw counts or measured-background-subtracted counts
- mark ROI endpoints manually
- feed those manual ROI definitions back into FluxForge without depending on the GUI

## Required capability
- Add a headless-safe CLI plot export for calibrated gamma spectra
- Support plotting raw or background-subtracted counts
- Support overlaying manual ROI definitions from a simple CSV/JSON file
- Support turning the same manual ROI definitions into a FluxForge peak-report artifact

## Implemented in this pass
- `fluxforge.cli.app spectrum-plot`
- `fluxforge.cli.app peaks --manual-peaks-file ...`
- `fluxforge.plots.spectrum_inspection.plot_gamma_spectrum`

## Manual ROI file fields
- `label`
- `isotope`
- `left_keV`, `right_keV`
- or `left_channel`, `right_channel`

## Follow-up work
- Add optional manual linear/trapezoid local-background choices for manual ROI reports
- Add manual ROI editing directly in the GUI
- Add click-to-select ROI endpoints in the GUI spectrum viewer
