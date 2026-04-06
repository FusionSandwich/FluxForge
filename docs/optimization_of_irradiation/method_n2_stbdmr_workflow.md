# Method N2 STBD-MR Workflow (Implemented Baseline)

Method N2 introduces a spectro-temporal Bayesian objective with masking regularization and an optional differentiable interference-graph mode. The baseline implementation is exposed as `stbd-mr` in CLI and as `Preview STBD-MR` in the Inventory Timeline GUI panel.

## Scope of this baseline

- Multi-window spectro-temporal score accumulation
- Interference graph construction from energy overlap and overlap-group labels
- Masking regularization penalty from graph-adjacent line terms
- Optional differentiable graph affinity bonus for advanced optimization mode
- Advanced-mode guard in CLI and GUI

## Objective summary

Each line contributes an information term:

- $I = \frac{S^2}{S + B + C}$

Where:

- $S$ is signal counts
- $B$ is background counts
- $C$ is continuum burden counts

Each window score is:

- $J_w = I_w - \lambda_m P_w + G_w$

Where:

- $I_w$ is weighted base information
- $\lambda_m$ is masking regularization weight
- $P_w$ is graph-based masking penalty
- $G_w$ is optional differentiable graph bonus

Total schedule score is the sum of window scores.

## CLI usage

The STBD-MR objective is guarded and requires `--enable-advanced-objectives`.

```bash
PYTHONPATH=src python -m fluxforge.cli.app optimization-sweep \
  --input examples/RAFM_irradiation/results/analysis_json/optimization_candidates.json \
  --objective stbd-mr \
  --enable-advanced-objectives \
  --stbdmr-window-offsets-s 0,3600,21600 \
  --stbdmr-window-count-time-s 900 \
  --stbdmr-masking-regularization 0.2 \
  --stbdmr-differentiable-graph \
  --stbdmr-graph-temperature 2.0 \
  --output examples/RAFM_irradiation/results/method_benchmark/n2_stbdmr_sweep.json \
  --csv-output examples/RAFM_irradiation/results/method_benchmark/n2_stbdmr_sweep.csv
```

Payload notes:

- Candidates can provide explicit `windows` with `lines`.
- Candidates with top-level `lines` are expanded into windows by the offset arguments.
- Optional line fields include `continuum_counts` and `overlap_group`.

## GUI usage

1. Run activity review so inventory has line estimates.
2. Open Inventory Timeline panel.
3. Enable `Enable advanced objectives`.
4. Optionally enable `STBD-MR differentiable graph mode`.
5. Click `Preview STBD-MR`.

The summary reports STBD-MR score, graph density, and masking penalty.

## N2 comparison gate artifact

Run:

```bash
PYTHONPATH=src python examples/RAFM_irradiation/compare_m1_m2_m3_n1_legacy_schedule_objectives.py
```

Outputs:

- `examples/RAFM_irradiation/results/method_benchmark/m1_m2_m3_n1_n2_legacy_schedule_comparison.csv`
- `examples/RAFM_irradiation/results/method_benchmark/m1_m2_m3_n1_n2_legacy_schedule_comparison.md`

## Verification pointers

- Core implementation: `src/fluxforge/analysis/optimization_stbdmr.py`
- CLI routing and guard: `src/fluxforge/cli/app.py`
- GUI preview and guard toggle: `src/fluxforge/gui/panels/modern_shell.py`
- Tests: `tests/test_optimization_stbdmr.py`, `tests/test_cli_app.py`, `tests/test_analysis_workspace_qt.py`

## Known limits

- Baseline graph affinity uses energy-distance kernels and does not include full response-matrix coupling.
- Differentiable graph mode is heuristic and intended as an initial advanced optimization path.
- Full publication diagnostics and graph explorer UI remain future N2 enhancements.
