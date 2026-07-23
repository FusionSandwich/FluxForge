# Tutorial 1: Getting Started with FluxForge

## Overview

FluxForge has two main operating modes:

- CLI for reproducible workflows and explicit artifact generation
- GUI for interactive spectrum review, ROI editing, and linked analysis panels

This tutorial walks through a first session in both modes using bundled RAFM
example data.

## 1. Install and Verify

If you want both the CLI and the GUI, install:

```bash
pip install -e '.[native-gui,reporting]'
```

Then verify:

```bash
fluxforge --help
fluxforge commands
fluxforge gui --help
```

If another machine reports `ModuleNotFoundError: No module named 'fluxforge.gui'`,
the fix is usually to install FluxForge into an active Python 3.11 or 3.12 environment
and launch it with `fluxforge gui` or `fluxforge-gui` rather than trying to run
`python -m fluxforge.gui.app` directly.

## 2. First GUI Session

Launch the GUI:

```bash
fluxforge gui --project-dir .
```

Use these first files:

- foreground spectrum:
  `examples/RAFM_irradiation/raw_gamma_spec/RAFM4/RAFM4-B_15dEOI.ASC`
- background spectrum:
  `examples/RAFM_irradiation/background.ASC`

Recommended first interactions:

1. Choose **File > Open Example**, or load the foreground and background
   spectra above and assign their roles in the left sidebar.
2. Use the mouse wheel or **Zoom + / Zoom -** to inspect a photopeak region;
   left-drag pans and **Reset View** restores the full range.
3. Choose **Select ROI** and drag the displayed ROI bounds.
4. Choose the sideband or SNIP method in the ROI tools when you want a
   background model, then run ROI analysis or ROI statistics.
5. Run **Auto Find Peaks**, accept the reviewed proposals, and select a row to
   synchronize the plot, peak tools, and nuclide browser.
6. Use the peak-table controls to assign or clear an isotope, add a tag, or pin
   a nuclide.
7. Choose **File > Save Session** (`Ctrl+S`) to persist spectra, roles, ROIs,
   peaks, detector state, and plot ranges in a validated `.ffs` file.

Direct plot-sideband handles, centroid dragging, and peak context menus are not
part of the current production workflow. They remain tracked direct-manipulation
work rather than being presented as working controls.

## 3. First CLI Session

Run a short end-to-end analysis chain:

```bash
fluxforge ingest \
  --input examples/RAFM_irradiation/raw_gamma_spec/RAFM4/RAFM4-B_15dEOI.ASC \
  --profile rafm_25cm \
  --output /tmp/rafm4_b_ingest.json

fluxforge peaks \
  --spectrum-file /tmp/rafm4_b_ingest.json \
  --output /tmp/rafm4_b_peaks.json
```

This gives you:

- an ingested spectrum artifact
- a peak-report artifact

Move on to `activity-review` once you have a line-assigned peak report, or use
`fluxforge rafm-validate ...` or `fluxforge phase6-ldrd-worked-example ...`
when you want a bundled end-to-end activity and planning workflow.

## 4. First Maintained Replay Workflow

Run the bundled RAFM validation workflow:

```bash
fluxforge rafm-validate \
  --example-root examples/RAFM_irradiation \
  --results-root /tmp/rafm_validation \
  --no-fail
```

If you want the planning-oriented example next, run:

```bash
fluxforge phase6-ldrd-worked-example \
  --sample-id RAFM4-C_15dEOI \
  --output-root /tmp/phase6_ldrd_worked_example
```

## 5. Continue Exploring

- Installation details:
  [docs/INSTALLATION.md](/groupspace/cnerg/users/smandych/projects/ALARA/FluxForge/docs/INSTALLATION.md:1)
- Full CLI catalog:
  [docs/CLI_REFERENCE.md](/groupspace/cnerg/users/smandych/projects/ALARA/FluxForge/docs/CLI_REFERENCE.md:1)
- Example cookbook:
  [docs/EXAMPLE_WORKFLOWS.md](/groupspace/cnerg/users/smandych/projects/ALARA/FluxForge/docs/EXAMPLE_WORKFLOWS.md:1)
- Full example inventory:
  [examples/README.md](/groupspace/cnerg/users/smandych/projects/ALARA/FluxForge/examples/README.md:1)
