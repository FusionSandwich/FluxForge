# FluxForge Examples

This directory is the canonical user-facing example surface for FluxForge.
Use this file when you want to know:

- which examples are starter workflows versus research demos
- what each example needs to run
- where the bundled input data lives
- what outputs to expect

For the grouped command catalog, use
[docs/CLI_REFERENCE.md](/groupspace/cnerg/users/smandych/projects/ALARA/FluxForge/docs/CLI_REFERENCE.md:1)
or run `fluxforge commands`.

The machine-readable inventory is
[examples/example_inventory.json](/groupspace/cnerg/users/smandych/projects/ALARA/FluxForge/examples/example_inventory.json:1).

## 1. Before Running Examples

Run all commands from the repository root.

Install profiles:

- CLI-only examples:
  `pip install -e .`
- GUI examples:
  `pip install -e '.[native-gui,reporting]'`

Use these discovery commands first:

```bash
fluxforge --help
fluxforge commands
```

## 2. Starter Workflows

These are the most stable and easiest examples to start with because they are
backed by committed inputs and clear output targets.

### `RAFM_irradiation/run_validation.py`

- Purpose: main end-to-end RAFM replay and validation workflow
- Install profile: CLI-only
- Preferred command:
  `fluxforge rafm-validate --example-root examples/RAFM_irradiation --results-root /tmp/rafm_validation --no-fail`
- Script entrypoint:
  `python examples/RAFM_irradiation/run_validation.py --no-fail`
- Inputs:
  `examples/RAFM_irradiation/raw_gamma_spec/`
  and
  `examples/RAFM_irradiation/results/`
- Outputs: analysis JSON bundles, comparison tables, validation summaries, report text
- Local notes:
  [examples/RAFM_irradiation/README.md](/groupspace/cnerg/users/smandych/projects/ALARA/FluxForge/examples/RAFM_irradiation/README.md:1)

### `RAFM_irradiation/run_phase6_ldrd_worked_example.py`

- Purpose: planning-oriented RAFM worked example with optimization and second-irradiation outputs
- Install profile: CLI-only
- Preferred command:
  `fluxforge phase6-ldrd-worked-example --sample-id RAFM4-C_15dEOI --output-root /tmp/phase6_ldrd_worked_example`
- Script entrypoint:
  `python examples/RAFM_irradiation/run_phase6_ldrd_worked_example.py --sample-id RAFM4-C_15dEOI --output-root /tmp/phase6_ldrd_worked_example`
- Inputs: committed RAFM raw spectra and Phase 6 support assets under `examples/RAFM_irradiation/results/`
- Outputs: activity review, inventory review, masking review, optimization results, second-irradiation bundle, `.ffexp`
- Local notes:
  [examples/RAFM_irradiation/README.md](/groupspace/cnerg/users/smandych/projects/ALARA/FluxForge/examples/RAFM_irradiation/README.md:1)

### `manual_peak_inspection/README.md`

- Purpose: SSH-safe or plot-first manual ROI inspection workflow
- Install profile: CLI-only
- Command:
  `fluxforge spectrum-plot --input examples/RAFM_irradiation/raw_gamma_spec/flux_wires/Ti-RAFM-1a_25cm.ASC --profile rafm_25cm --background-subtracted --manual-peaks-file examples/manual_peak_inspection/manual_flux_wire_ti_rafm_1a.csv --output /tmp/Ti-RAFM-1a_25cm_manual.png --save-peak-report /tmp/Ti-RAFM-1a_25cm_manual_peaks.json`
- Inputs:
  `examples/RAFM_irradiation/raw_gamma_spec/`
  plus ROI CSV files under
  `examples/manual_peak_inspection/`
- Outputs: saved PNG plot and manual peak JSON
- Local notes:
  [examples/manual_peak_inspection/README.md](/groupspace/cnerg/users/smandych/projects/ALARA/FluxForge/examples/manual_peak_inspection/README.md:1)

### `unfolding_benchmark/run_benchmark.py`

- Purpose: benchmark FluxForge GRAVEL and MLEM against included reference implementations
- Install profile: CLI-only
- Command:
  `python examples/unfolding_benchmark/run_benchmark.py`
- Inputs:
  `examples/unfolding_benchmark/response-matrix.txt`,
  `reduced_data.csv`,
  and
  `energy-spectrum.txt`
- Outputs:
  `examples_output/unfolding_benchmark_comparison.png`,
  `examples_output/unfolding_implementation_diff.png`,
  and
  `examples_output/unfolding_benchmark_results.json`
- Local notes:
  [examples/unfolding_benchmark/README.md](/groupspace/cnerg/users/smandych/projects/ALARA/FluxForge/examples/unfolding_benchmark/README.md:1)

## 3. Supported Advanced Examples

These are useful, documented examples, but they are not the default onboarding
path.

### `cadmium_cover_correction_example.py`

- Purpose: STAYSL-style cadmium cover correction and parity reporting demo
- Install profile: CLI-only
- Command: `python examples/cadmium_cover_correction_example.py`
- Inputs: built-in cover-correction models and constants
- Outputs: stdout summaries and optional matplotlib figures

### `endf_mf33_covariance_demo.py`

- Purpose: self-contained ENDF MF33 covariance ingest, validation, and conditioning demo
- Install profile: CLI-only
- Command: `python examples/endf_mf33_covariance_demo.py`
- Inputs: synthetic temporary ENDF-like section created by the script
- Outputs: stdout covariance diagnostics

### `gamma_poisson_rmle_demo.py`

- Purpose: Poisson RMLE gamma unfolding demo with response sampling
- Install profile: CLI-only
- Command: `python examples/gamma_poisson_rmle_demo.py`
- Inputs: synthetic response and counts generated by the script
- Outputs: stdout solver and uncertainty summary

### `generate_flux_spectrum.py`

- Purpose: advanced unfolding demo using package-bundled Fe-Cd-RAFM-1 inputs
- Install profile: CLI-only
- Command: `python examples/generate_flux_spectrum.py`
- Inputs:
  `src/fluxforge/examples/fe_cd_rafm_1/`
- Outputs: unfold artifact plus stdout reaction-rate and solver summary

### `generate_plots.py`

- Purpose: publication-style plotting demo driven by package-bundled Fe-Cd-RAFM-1 inputs
- Install profile: CLI-only
- Command: `python examples/generate_plots.py`
- Inputs:
  `src/fluxforge/examples/fe_cd_rafm_1/`
- Outputs: plot files written by the script

### `nuclear_data_interface_example.py`

- Purpose: unified nuclear-data interface and temperature-aware data container demo
- Install profile: CLI-only
- Command: `python examples/nuclear_data_interface_example.py`
- Inputs: synthetic reaction and cross-section data defined by the script
- Outputs: stdout nuclear-data interface summary

### `sigphi_saturation_demo.py`

- Purpose: saturation-rate and burnup-correction demo
- Install profile: CLI-only
- Command: `python examples/sigphi_saturation_demo.py`
- Inputs: script-defined irradiation history and monitor measurement
- Outputs: stdout saturation-rate summary

### `staysl_interop_demo.py`

- Purpose: STAYSL CSV, covariance-matrix, and bundle interoperability demo
- Install profile: CLI-only
- Command: `python examples/staysl_interop_demo.py`
- Inputs: synthetic STAYSL payloads created by the script
- Outputs: generated files under `examples/output/` plus stdout summaries

### `staysl_reporting_demo.py`

- Purpose: STAYSL-class reporting, correlation, and plotting demo
- Install profile: CLI-only
- Command: `python examples/staysl_reporting_demo.py`
- Inputs: synthetic flux, covariance, and cross-section data defined by the script
- Outputs: stdout report fragments and optional plots

### `triga_k0naa_workflow.py`

- Purpose: TRIGA-oriented k0-NAA worked example
- Install profile: CLI-only
- Command: `python examples/triga_k0naa_workflow.py`
- Inputs: script-defined TRIGA and k0 assumptions
- Outputs: stdout walkthrough and matplotlib figures

### `validation/attenuation_demo.py`

- Purpose: attenuation-material validation demo
- Install profile: CLI-only
- Command: `python examples/validation/attenuation_demo.py`
- Inputs: built-in attenuation material data
- Outputs: stdout transmission values

### `validation/calibration_fit_demo.py`

- Purpose: detector efficiency and resolution fitting demo
- Install profile: CLI-only
- Command: `python examples/validation/calibration_fit_demo.py`
- Inputs: synthetic efficiency and resolution points defined by the script
- Outputs: stdout fit coefficients

### `validation/cross_section_demo.py`

- Purpose: cross-section lookup and evaluation demo
- Install profile: CLI-only
- Command: `python examples/validation/cross_section_demo.py`
- Inputs: built-in placeholder IRDFF library
- Outputs: stdout reaction lookup result

### `validation/iec_read_demo.py`

- Purpose: IEC spectrum reader demo
- Install profile: CLI-only
- Command: `python examples/validation/iec_read_demo.py /path/to/file.iec`
- Inputs: user-supplied IEC file
- Outputs: stdout spectrum summary
- Local notes:
  [examples/validation/README.md](/groupspace/cnerg/users/smandych/projects/ALARA/FluxForge/examples/validation/README.md:1)

## 4. Research and Historical Demo Utilities

These examples are useful references, but they are not part of the starter
verification path and some rely on package-bundled or illustrative data rather
than top-level committed example assets.

### `complete_flux_wire_workflow.py`

- Purpose: historical end-to-end flux-wire workflow script
- Install profile: CLI-only
- Command: `python examples/complete_flux_wire_workflow.py`
- Inputs:
  `src/fluxforge/examples/flux_wire/`
  and
  `src/fluxforge/examples/fe_cd_rafm_1/`
- Outputs: `examples_outputs/flux_wire_analysis/`

### `complete_naa_workflow.py`

- Purpose: historical combined NAA and flux-characterization walkthrough
- Install profile: CLI-only
- Command: `python examples/complete_naa_workflow.py`
- Inputs: script-defined k0 and flux-characterization data
- Outputs: stdout walkthrough and figures

### `njoy_processing_example.py`

- Purpose: NJOY input generation and workflow-specification demo
- Install profile: CLI-only
- Command: `python examples/njoy_processing_example.py`
- Inputs: script-defined NJOY processing specification; external NJOY binary needed for real execution
- Outputs: stdout NJOY input-deck examples

### `reactor_dosimetry_workflow.py`

- Purpose: large illustrative dosimetry and unfolding workflow with simulated HFIR-like data
- Install profile: CLI-only
- Command: `python examples/reactor_dosimetry_workflow.py`
- Inputs: script-defined simulated spectrum and dosimetry data
- Outputs: stdout workflow summary

## 5. Shared Bundled Data and Data-Only Benchmarks

These paths are shipped for example workflows or future comparison work even if
they are not standalone entrypoints.

| Path | What it is for |
|---|---|
| `examples/RAFM_irradiation/raw_gamma_spec/` | committed RAFM raw gamma spectra |
| `examples/RAFM_irradiation/results/` | committed RAFM replay and validation outputs |
| `examples/RAFM_irradiation/background.ASC` | shared background spectrum for GUI and plotting workflows |
| `examples/manual_peak_inspection/*.csv` | manual ROI overlays for plot-based review |
| `examples/spectroscopy_data/` | small spectroscopy files used by validation demos |
| `examples/speckit_benchmark/` | data-only benchmark assets for future SpecKit-style comparisons |
| `src/fluxforge/examples/fe_cd_rafm_1/` | package-bundled Fe-Cd-RAFM-1 data used by advanced demos |
| `src/fluxforge/examples/flux_wire/` | package-bundled flux-wire data used by legacy research demos |

Directory notes:

- [examples/RAFM_irradiation/README.md](/groupspace/cnerg/users/smandych/projects/ALARA/FluxForge/examples/RAFM_irradiation/README.md:1)
- [examples/speckit_benchmark/README.md](/groupspace/cnerg/users/smandych/projects/ALARA/FluxForge/examples/speckit_benchmark/README.md:1)
- [examples/spectroscopy_data/README.md](/groupspace/cnerg/users/smandych/projects/ALARA/FluxForge/examples/spectroscopy_data/README.md:1)
- [examples/validation/README.md](/groupspace/cnerg/users/smandych/projects/ALARA/FluxForge/examples/validation/README.md:1)
