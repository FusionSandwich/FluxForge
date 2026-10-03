# RAFM Irradiation Examples

This directory contains the main maintained replay data and workflow entrypoints
for FluxForge.

Start with the [feature guide](FEATURE_GUIDE.md) for the expanded 32-acquisition
dataset, converted-format checks, material compositions, reproducible commands
and the qualification limits of each feature example.

For the complete, source-bound 2025 QuantumGold campaign, use the
[portable reference example](quantumgold_reference/README.txt). It bundles the
32 native spectra, 30 available ASC exports, 31 original reports and all extracted
peak/nuclide rows, with explicit cohort, timing, geometry and missing-file labels.
It needs no files from the original computer or QuantumGold installation:

```bash
python examples/RAFM_irradiation/run_portable_qg_example.py --verify-only
python examples/RAFM_irradiation/run_portable_qg_example.py --output replay_output
```

The portable replay audits all 32 counts, performs 30 available ASC reductions,
and runs the 12-monitor QG report benchmark. Software completion and physical
agreement are recorded separately; the native-only counts remain explicit.

The INL Co masses are already adjusted Co element masses, despite the wires
containing 0.46 wt% Co. Do not apply the alloy fraction a second time. See the
[source and replay review](../../docs/reviews/INL_MONITOR_MASS_REVIEW_2026-09-29.md)
for the mass basis, corrected Cu-Cd mass and remaining validation limits.

## What Is Here

- `raw_gamma_spec/`: committed RAFM raw gamma spectra
- `background.ASC`: shared background spectrum used by GUI and plot workflows
- `results/`: committed replay and validation outputs
- `run_validation.py`: end-to-end RAFM validation workflow
- `run_phase6_ldrd_worked_example.py`: planning-oriented worked example

## How to Run the Validation Workflow

Preferred CLI entrypoint:

```bash
fluxforge rafm-validate \
  --example-root examples/RAFM_irradiation \
  --results-root /tmp/rafm_validation \
  --no-fail
```

Equivalent script:

```bash
python examples/RAFM_irradiation/run_validation.py --no-fail
```

See the [RAFM processing runbook](RAFM_processing_plan.md) for single-spectrum
ingest, explicit overrides, output locations, and warning interpretation.

Typical outputs:

- analysis JSON bundles
- comparison tables
- validation summaries
- text reports

### Background subtraction and negative channels

The RAFM runs use the measured `background.ASC` spectrum, scaled by acquisition
time. A background-subtracted channel may be negative because the two measured
counts fluctuate. In the default `hybrid` mode, FluxForge retains that signed
value and the propagated uncertainty in the saved spectrum and ROI accounting.
It does not mean a negative physical count rate. An algorithm that requires
nonnegative input uses a separate working copy; when it clips negative channels,
FluxForge issues a warning and leaves the saved signed spectrum unchanged.
Inspect the `background_subtraction` metadata for the scale factor and number of
negative channels before interpreting a result.

### Calibration and efficiency overrides

Genie `.ASC` and `.txt` readers use energy and efficiency coefficients from the
file when present. A selected RAFM profile fills detector values that the file
does not supply; it preserves the file's energy calibration. Explicit
`--energy-calibration` and `--efficiency-coefficients` CLI values, or the
corresponding reader arguments, take precedence over file and profile values.
The saved spectrum records the effective coefficients, so check that artifact
before comparing an analysis with a processed report.

## How to Run the Phase 6 Worked Example

```bash
fluxforge phase6-ldrd-worked-example \
  --sample-id RAFM4-C_15dEOI \
  --output-root /tmp/phase6_ldrd_worked_example
```

Typical outputs:

- activity review
- inventory review
- masking review
- optimization outputs
- second-irradiation outputs
- `.ffexp` bundle

## Good First Files

- foreground spectrum:
  `examples/RAFM_irradiation/raw_gamma_spec/RAFM4/RAFM4-B_15dEOI.ASC`
- background spectrum:
  `examples/RAFM_irradiation/background.ASC`

## Source-bound UWNR curve diagnostic

The exact authorized South HPGe table and public-safe provenance are in [calibration/](calibration/). Use the [all-report diagnostic and covariance guide](../../docs/reviews/UWNR_SOURCE_DATA_USE_2026-09-30.md) to replay source QC. This historical table does not qualify physical calibration; negative source rows and unknown covariance remain explicit admission exclusions.
