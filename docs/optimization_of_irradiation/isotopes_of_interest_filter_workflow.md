# Isotopes-of-Interest Filter Workflow

This workflow adds user-directed isotope selection to schedule optimization. It is implemented through `optimization-sweep --isotopes-of-interest` and applies to all supported objectives (`di-fom`, `fim-*`, `mwdcs`, `bass-d`, `stbd-mr`).

## Why this matters

Users often need a schedule optimized for a narrow assay objective, such as a shortlist of dose-relevant isotopes. The isotopes-of-interest filter removes non-target line terms before objective scoring so rankings are driven by the user-selected nuclides.

## CLI usage

```bash
PYTHONPATH=src python -m fluxforge.cli.app optimization-sweep \
  --input examples/RAFM_irradiation/results/analysis_json/optimization_candidates.json \
  --objective di-fom \
  --isotopes-of-interest Mo-99,Tc-99m \
  --output examples/RAFM_irradiation/results/method_benchmark/optimization_sweep_roi.json \
  --csv-output examples/RAFM_irradiation/results/method_benchmark/optimization_sweep_roi.csv
```

For advanced objectives, combine with the existing guard flag:

```bash
PYTHONPATH=src python -m fluxforge.cli.app optimization-sweep \
  --input examples/RAFM_irradiation/results/analysis_json/optimization_candidates.json \
  --objective stbd-mr \
  --enable-advanced-objectives \
  --isotopes-of-interest Mo-99,Tc-99m \
  --output examples/RAFM_irradiation/results/method_benchmark/optimization_sweep_stbdmr_roi.json
```

## Output metadata

Optimization output bundles now include:

- `isotopes_of_interest`: ordered isotope list requested by the user.
- `isotope_filter_summary`: before/after counts for candidates and line terms.

These fields make focused optimization runs auditable and reproducible.

## Verification pointers

- CLI implementation: `src/fluxforge/cli/app.py`
- Parser and behavior tests: `tests/test_cli_app.py`
