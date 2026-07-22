# FluxForge Example Workflows

This document is the copy-paste cookbook for the best supported starter
workflows. For the full example inventory, including research demos and
data-only benchmark assets, use
[examples/README.md](/groupspace/cnerg/users/smandych/projects/ALARA/FluxForge/examples/README.md:1)
and
[examples/example_inventory.json](/groupspace/cnerg/users/smandych/projects/ALARA/FluxForge/examples/example_inventory.json:1).

Run the commands below from the repository root after installation.

## 1. Before You Start

### Install the CLI

```bash
pip install -e .
```

### Install the GUI and reporting extras

```bash
pip install -e '.[native-gui,reporting]'
```

### Verify command discovery

```bash
fluxforge --help
fluxforge commands
```

## 2. Workflow Selection Guide

| Workflow | Use it when | Install profile | Main command |
|---|---|---|---|
| CLI discovery | You want to see the full command surface | CLI-only | `fluxforge commands` |
| First single-spectrum analysis | You want the shortest verified artifact chain from a real spectrum | CLI-only | `ingest`, `peaks` |
| First GUI session | You want interactive ROI editing and linked review panels | Full user install | `fluxforge gui --project-dir .` |
| Manual peak inspection | You want an SSH-safe plot and manual ROI workflow | CLI-only | `spectrum-plot` |
| RAFM validation | You want a maintained replay workflow with committed inputs | CLI-only | `fluxforge rafm-validate ...` |
| Phase 6 worked example | You want a bundled planning workflow | CLI-only | `fluxforge phase6-ldrd-worked-example ...` |
| Unfolding benchmark | You want to compare FluxForge against included reference implementations | CLI-only | `python examples/unfolding_benchmark/run_benchmark.py` |
| ASTM example | You want a standards-oriented dosimetry run | CLI-only | `fluxforge astm-e261 ...` |

## 3. Recipes

### A. CLI Discovery

```bash
fluxforge --help
fluxforge commands
fluxforge commands --family spectrum
```

Use this first if you do not yet know which command family you need.

### B. First Single-Spectrum Analysis

Input:

- `examples/RAFM_irradiation/raw_gamma_spec/RAFM4/RAFM4-B_15dEOI.ASC`

Commands:

```bash
fluxforge ingest \
  --input examples/RAFM_irradiation/raw_gamma_spec/RAFM4/RAFM4-B_15dEOI.ASC \
  --profile rafm_25cm \
  --output /tmp/rafm4_b_ingest.json

fluxforge peaks \
  --spectrum-file /tmp/rafm4_b_ingest.json \
  --output /tmp/rafm4_b_peaks.json
```

Expected outputs:

- `/tmp/rafm4_b_ingest.json`
- `/tmp/rafm4_b_peaks.json`

Use `activity-review` after you have a line-assigned peak report, or move to a
maintained replay workflow such as `fluxforge rafm-validate ...` when you want
an end-to-end activity and planning chain backed by committed reference assets.

### C. First GUI Session

Install profile:

- `pip install -e '.[native-gui,reporting]'`

Launch:

```bash
fluxforge gui --project-dir .
```

Recommended first files:

- foreground: `examples/RAFM_irradiation/raw_gamma_spec/RAFM4/RAFM4-B_15dEOI.ASC`
- background: `examples/RAFM_irradiation/background.ASC`

Recommended first interactions:

1. Use **File > Open Spectrum** to load the foreground `.ASC` file.
2. Use the mouse wheel to zoom and left-drag to pan the calibrated energy axis.
3. Click **Select ROI**, then drag either boundary on the plot.
4. Open **ROI Tools** to select the background method and run the bounded analysis.
5. Click **Reset View** to show the full spectrum or **Clear ROI** to remove the selection.
6. Use the spectrum tabs and background controls to compare foreground, background, and overlay spectra.

### D. Manual Peak Inspection

This is the best SSH-safe review workflow.

Inputs:

- spectrum: `examples/RAFM_irradiation/raw_gamma_spec/flux_wires/Ti-RAFM-1a_25cm.ASC`
- ROI CSV: `examples/manual_peak_inspection/manual_flux_wire_ti_rafm_1a.csv`

Command:

```bash
fluxforge spectrum-plot \
  --input examples/RAFM_irradiation/raw_gamma_spec/flux_wires/Ti-RAFM-1a_25cm.ASC \
  --profile rafm_25cm \
  --background-subtracted \
  --manual-peaks-file examples/manual_peak_inspection/manual_flux_wire_ti_rafm_1a.csv \
  --output /tmp/Ti-RAFM-1a_25cm_manual.png \
  --save-peak-report /tmp/Ti-RAFM-1a_25cm_manual_peaks.json
```

See the full walkthrough in
[examples/manual_peak_inspection/README.md](/groupspace/cnerg/users/smandych/projects/ALARA/FluxForge/examples/manual_peak_inspection/README.md:1).

### E. RAFM Validation Workflow

Inputs:

- `examples/RAFM_irradiation/raw_gamma_spec/`
- `examples/RAFM_irradiation/results/`

Preferred CLI command:

```bash
fluxforge rafm-validate \
  --example-root examples/RAFM_irradiation \
  --results-root /tmp/rafm_validation \
  --no-fail
```

Equivalent script entrypoint:

```bash
python examples/RAFM_irradiation/run_validation.py --no-fail
```

Typical outputs:

- analysis JSON bundles
- line and isotope comparison tables
- validation summaries
- report text under the selected results root

### F. Phase 6 Worked Example

Command:

```bash
fluxforge phase6-ldrd-worked-example \
  --sample-id RAFM4-C_15dEOI \
  --output-root /tmp/phase6_ldrd_worked_example
```

Expected artifact families:

- activity review
- inventory review
- masking review
- optimization outputs
- second-irradiation outputs
- `.ffexp` export bundle

Detailed reference:

- `docs/optimization_of_irradiation/phase6_ldrd_worked_example.md`

### G. Unfolding Benchmark

Inputs:

- `examples/unfolding_benchmark/response-matrix.txt`
- `examples/unfolding_benchmark/reduced_data.csv`
- `examples/unfolding_benchmark/energy-spectrum.txt`

Command:

```bash
python examples/unfolding_benchmark/run_benchmark.py
```

Typical outputs:

- `examples_output/unfolding_benchmark_comparison.png`
- `examples_output/unfolding_implementation_diff.png`
- `examples_output/unfolding_benchmark_results.json`

See the local benchmark notes in
[examples/unfolding_benchmark/README.md](/groupspace/cnerg/users/smandych/projects/ALARA/FluxForge/examples/unfolding_benchmark/README.md:1).

### H. ASTM E261 Example

Input:

- `examples/astm_e261_plan.json`

Command:

```bash
fluxforge astm-e261 \
  --plan-file examples/astm_e261_plan.json \
  --output /tmp/astm_e261_result.json
```

### I. Governance and Release Checks

Use these when you need parity or release evidence rather than ordinary user
analysis output:

```bash
fluxforge parity-check --scope all --output /tmp/parity_check.json
fluxforge phase5-crosswalk-report --include-parity-summary --output /tmp/phase5_crosswalk_report.json --markdown-output /tmp/phase5_crosswalk_report.md
fluxforge phase5-release-gate --output /tmp/phase5_release_gate.json
```

## 4. Where to Find the Rest of the Example Surface

- Full example inventory:
  [examples/README.md](/groupspace/cnerg/users/smandych/projects/ALARA/FluxForge/examples/README.md:1)
- Structured machine-readable example list:
  [examples/example_inventory.json](/groupspace/cnerg/users/smandych/projects/ALARA/FluxForge/examples/example_inventory.json:1)
- Installation and environment setup:
  [docs/INSTALLATION.md](/groupspace/cnerg/users/smandych/projects/ALARA/FluxForge/docs/INSTALLATION.md:1)
