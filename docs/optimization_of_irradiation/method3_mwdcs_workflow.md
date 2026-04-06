# Method 3 MWDCS Workflow (Implemented Baseline)

Method 3 adds a multi-window objective that accumulates line-level information across multiple cooldown/count windows. This objective is exposed in the CLI as `mwdcs` and in the GUI via the Inventory Timeline panel as `Preview MWDCS`.

## Scope of this baseline

- Core workflow: additive multi-window score with diminishing returns by repeated nuclide observations
- Optional full-spectrum mode: overlap penalty for near-energy lines from different nuclides
- CLI workflow: `optimization-sweep --objective mwdcs`
- GUI workflow: Inventory Timeline MWDCS preview controls (window count + full-spectrum toggle)

## Objective summary

For each window, each line contributes an information term:

- line information: $I = \frac{S^2}{S + B + U}$

Where:

- $S$ is expected signal counts
- $B$ is expected background counts
- $U$ is expected interference counts

Window score aggregates weighted line information with a diminishing-return factor per nuclide. In full-spectrum mode, close-energy overlaps introduce a configurable subtraction term.

Total MWDCS score is the sum of per-window marginal scores.

## CLI usage

Example run:

```bash
PYTHONPATH=src python -m fluxforge.cli.app optimization-sweep \
  --input examples/RAFM_irradiation/results/analysis_json/optimization_candidates.json \
  --objective mwdcs \
  --mwdcs-window-offsets-s 0,3600,21600 \
  --mwdcs-window-count-time-s 900 \
  --mwdcs-full-spectrum-mode \
  --mwdcs-overlap-penalty 0.1 \
  --output examples/RAFM_irradiation/results/method_benchmark/m3_mwdcs_sweep.json \
  --csv-output examples/RAFM_irradiation/results/method_benchmark/m3_mwdcs_sweep.csv
```

Notes:

- If a candidate provides `windows`, those are used directly.
- If a candidate only provides `lines`, windows are generated from `--mwdcs-window-offsets-s`.
- If line items provide `half_life_s`, generated windows decay the line signal by cooldown time.

## GUI usage

1. Run activity review so inventory has activity results.
2. Open Inventory Timeline panel.
3. Set `MWDCS windows` count and optionally enable `MWDCS full-spectrum mode`.
4. Click `Preview MWDCS`.

The summary card reports score and number of windows included in the preview objective.

## Method comparison gate (M1/M2/M3 + legacy)

Run the benchmark script:

```bash
PYTHONPATH=src python examples/RAFM_irradiation/compare_m1_m2_m3_legacy_schedule_objectives.py
```

Generated artifacts:

- `examples/RAFM_irradiation/results/method_benchmark/m1_m2_m3_legacy_schedule_comparison.csv`
- `examples/RAFM_irradiation/results/method_benchmark/m1_m2_m3_legacy_schedule_comparison.md`

The legacy baseline is imported from:

- `../rafm_irradiation_ldrd_copy/scripts/schedule_optimizer.py`

## Known limits

- Window generation from `lines` is currently parameterized (offset list/count time), not optimizer-discovered.
- Full-spectrum mode uses an overlap penalty heuristic rather than a full deconvolution fit.
- Candidate quality still depends on upstream line term quality in RAFM artifact payloads.
