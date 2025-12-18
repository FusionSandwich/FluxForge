# FluxForge

UWNR Flux-Wire–Driven Neutron Spectrum Reconstruction Tool. This repository
provides a pure-Python package and CLI that converts HPGe-derived wire activities
into reaction rates, constructs response matrices from dosimetry cross sections,
and infers neutron flux spectra with generalized least squares and Monte Carlo
uncertainty propagation.

## Features
- Activation and decay helpers for converting net peak areas to activities and
  reaction rates.
- Response matrix construction for groupwise cross sections and number densities.
- GLS spectrum adjustment with optional non-negativity enforcement.
- Iterative GRAVEL and MLEM unfolding options for prior-free spectrum recovery.
- Monte Carlo uncertainty propagation to estimate percentile bands on inferred
  spectra.
- Argparse-based CLI with ``build-response``, ``infer-spectrum``, ``validate``,
  and ``report`` commands.

## Getting started
The project avoids external dependencies to simplify offline execution. Install
it in editable mode (setuptools is bundled with Python) and invoke the CLI:

```bash
pip install -e .
python -m fluxforge.cli.app --help
```

Synthetic validation and inference routines expect JSON inputs; see
``src/fluxforge/cli/app.py`` for the expected schemas and adjust to your UWNR
data formats.

## Example data
The `src/fluxforge/examples/fe_cd_rafm_1` folder contains a concrete
Fe58(n,γ)→Fe59 activation-wire example derived from a Cd-covered RAFM wire
measurement (live time 43.2 ks, dead time 0.1%). The JSON files mirror the
CLI schemas for group boundaries, cross sections, number densities, measured
gamma lines, and an illustrative prior spectrum. Swap in dosimetry-grade
cross sections (e.g., IRDFF-II) and geometry-specific number densities before
using the example for physics conclusions.
