# Method N1 BASS-D Workflow (Implemented Baseline)

Method N1 introduces a Bayesian adaptive objective with dose-aware utility. The baseline implementation is exposed as the `bass-d` objective in CLI and as `Preview BASS-D` in the Inventory Timeline GUI panel.

## Scope of this baseline

- Bayesian posterior variance update per action and line
- Dose-weighted value-of-information utility
- Action-sequence utility accumulation with optional exploration noise
- Reproducibility through deterministic seed control
- Advanced-mode guard for CLI and GUI

## Utility summary

For each line in each action:

- Posterior update uses a precision sum between prior variance and observation variance.
- Value of information is computed as variance reduction in log-space.
- Line utility is weighted by detection probability and optional isotope weight.

Per-action utility:

- $U_a = \mathrm{VoI}_a - w_d \cdot D_a + \epsilon_a$

Where:

- $\mathrm{VoI}_a$ is weighted value of information
- $w_d$ is dose penalty weight
- $D_a$ is expected dose during the action
- $\epsilon_a$ is optional exploration noise controlled by temperature and seed

Total schedule utility is the sum of action utilities.

## CLI usage

The BASS-D objective is guarded and requires `--enable-advanced-objectives`.

```bash
PYTHONPATH=src python -m fluxforge.cli.app optimization-sweep \
  --input examples/RAFM_irradiation/results/analysis_json/optimization_candidates.json \
  --objective bass-d \
  --enable-advanced-objectives \
  --bassd-dose-weight 0.03 \
  --bassd-exploration-temperature 0.0 \
  --bassd-seed 17 \
  --output examples/RAFM_irradiation/results/method_benchmark/n1_bassd_sweep.json \
  --csv-output examples/RAFM_irradiation/results/method_benchmark/n1_bassd_sweep.csv
```

Payload notes:

- Candidates may provide explicit `actions`, each with `lines`.
- For baseline compatibility, a candidate may provide top-level `lines` and an implicit single action is built.

## GUI usage

1. Run activity review so inventory has line estimates.
2. Open Inventory Timeline panel.
3. Enable `Enable advanced objectives`.
4. Click `Preview BASS-D`.

The panel reports total adaptive utility and action count for the preview candidate.

## Verification pointers

- Core implementation: `src/fluxforge/analysis/optimization_bassd.py`
- CLI routing and guard: `src/fluxforge/cli/app.py`
- GUI preview and guard toggle: `src/fluxforge/gui/panels/modern_shell.py`
- Tests: `tests/test_optimization_bassd.py`, `tests/test_cli_app.py`, `tests/test_analysis_workspace_qt.py`

## N1 comparison gate artifact

Run:

```bash
PYTHONPATH=src python examples/RAFM_irradiation/compare_m1_m2_m3_n1_legacy_schedule_objectives.py
```

Outputs:

- `examples/RAFM_irradiation/results/method_benchmark/m1_m2_m3_n1_legacy_schedule_comparison.csv`
- `examples/RAFM_irradiation/results/method_benchmark/m1_m2_m3_n1_legacy_schedule_comparison.md`

## Known limits

- Baseline uses scalar line-wise posterior updates, not full covariance coupling.
- Expected dose is derived from per-line dose-rate placeholders in payload/preview models.
- Action design is currently input-driven; adaptive action proposal generation is deferred to later N1 iterations.
