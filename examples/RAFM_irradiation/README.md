# RAFM Irradiation Examples

This directory contains the main maintained replay data and workflow entrypoints
for FluxForge.

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
