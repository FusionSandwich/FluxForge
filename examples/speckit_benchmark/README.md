# SpecKit Benchmark Assets

This directory currently ships benchmark source assets rather than a dedicated
runner script.

## What Is Here

- `double_peak/`: benchmark inputs for overlapping-peak cases
- `quasi_single_peak/`: benchmark inputs for quasi-single-peak cases

Representative files:

- `Hypothetical Experiment Design.xlsx`
- `prior.txt`
- `MCNP_input/prior.txt`

## Current Status

This is a data-only benchmark directory. FluxForge does not currently ship a
top-level `python` or `fluxforge` entrypoint that consumes these assets
directly.

Use this directory when you need:

- source assets for future SpecKit-style comparison work
- benchmark inputs for manual inspection or downstream tool development
