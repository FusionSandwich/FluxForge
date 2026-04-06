# Isotope Priority Workflow From Gamma Spectrum

This workflow ranks the most important isotopes directly from an activity-review gamma-spectrum artifact.

The command consumes `activity-review` output and produces a prioritized isotope list with component-level scoring diagnostics.

## Command

```bash
PYTHONPATH=src python -m fluxforge.cli.app isotope-priority \
  --activity-review-file examples/RAFM_irradiation/results/analysis_json/RAFM4-A_15dEOI.json \
  --isotopes-of-interest Mo-99,Sc-46,Co-60 \
  --top-n 10 \
  --output examples/RAFM_irradiation/results/method_benchmark/isotope_priority.json \
  --csv-output examples/RAFM_irradiation/results/method_benchmark/isotope_priority.csv
```

## Scoring model

Each isotope receives a composite score:

- activity component (EOI-referenced activity magnitude)
- detectability component (net-count strength)
- confidence component (inverse relative uncertainty)
- line-support component (multi-line support)
- dose component (dose-rate relevance when available)

Default weights:

- activity: 0.35
- detectability: 0.25
- confidence: 0.20
- line support: 0.10
- dose: 0.10

Weights are configurable through:

- `--weight-activity`
- `--weight-detectability`
- `--weight-confidence`
- `--weight-line-support`
- `--weight-dose`

## Outputs

JSON bundle schema:

- `fluxforge.isotope_priority.v1`

Key fields:

- `ranked_isotopes[]` with `priority_score`
- component contributions per isotope
- selected `isotopes_of_interest`
- scoring weights used for reproducibility

## Verification pointers

- Analysis core: `src/fluxforge/analysis/isotope_priority.py`
- CLI command: `src/fluxforge/cli/app.py` (`isotope-priority`)
- Tests: `tests/test_isotope_priority.py`, `tests/test_cli_app.py`
