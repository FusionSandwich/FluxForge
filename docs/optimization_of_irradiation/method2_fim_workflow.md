# Method 2 FIM Workflow (Implemented Baseline)

This note documents the second implemented optimization method in FluxForge:
Fisher Information Matrix (FIM) schedule ranking.

## Scope

Implemented surfaces:

- Analysis core: Fisher matrix assembly and FIM objectives
- CLI workflow: optimization-sweep objective routing for fim-d, fim-a, fim-c
- GUI hook: Inventory panel FIM preview with matrix diagnostics

## Objectives

Supported objective modes:

- fim-d: maximize log-determinant of the regularized Fisher matrix
- fim-a: minimize trace of inverse Fisher matrix (reported as negative trace for ranking)
- fim-c: minimize one target-parameter variance (reported as negative variance for ranking)

All objectives rank higher scores as better.

## CLI Usage

Example FIM-D run:

```bash
fluxforge optimization-sweep \
  --input optimization_candidates.json \
  --objective fim-d \
  --nuisance-variance-fraction 0.05 \
  --fim-regularization 1e-6 \
  --output optimization_sweep_fim_d.json \
  --csv-output optimization_sweep_fim_d.csv
```

Example FIM-C run:

```bash
fluxforge optimization-sweep \
  --input optimization_candidates.json \
  --objective fim-c \
  --target-nuclide Mo-99 \
  --output optimization_sweep_fim_c.json
```

JSON output schema:

- fluxforge.optimization_sweep.fim.v1

Key output fields:

- ranked_candidates[].objective_score
- ranked_candidates[].matrix_diagnostics.condition_number
- ranked_candidates[].matrix_diagnostics.effective_rank
- ranked_candidates[].matrix_diagnostics.target_nuclide

## Formulae

Line variance model:

$$
\sigma_i^2 = s_i + b_i + i_i + (\alpha s_i)^2
$$

where $\alpha$ is nuisance_variance_fraction.

Fisher matrix:

$$
F = \sum_i w_i J_i^T J_i,\quad w_i = \frac{1}{\sigma_i^2}
$$

with regularized matrix $F_\lambda = F + \lambda I$.

Objective definitions:

$$
\text{D-opt}:\; \max \log \det(F_\lambda)
$$

$$
\text{A-opt}:\; \min \operatorname{tr}(F_\lambda^{-1})
$$

$$
\text{C-opt}:\; \min c^T F_\lambda^{-1} c
$$

## GUI Preview

Inventory / Time Evolution panel includes:

- FIM objective selector (D/A/C)
- Preview FIM button
- Summary with objective score, condition number, and effective rank

This preview uses current activity-review line proxies and is intended for quick triage,
not full transport-coupled sensitivity inversion.

## M1 vs M2 RAFM Gate

Comparison artifacts are generated with:

```bash
PYTHONPATH=src python examples/RAFM_irradiation/compare_m1_m2_schedule_objectives.py
```

Outputs:

- examples/RAFM_irradiation/results/method_benchmark/m1_m2_schedule_comparison.csv
- examples/RAFM_irradiation/results/method_benchmark/m1_m2_schedule_comparison.md

## Known Limits

- Current Fisher sensitivities are built from line proxy terms keyed by nuclide labels.
- Cross-nuclide coupling terms are limited in this baseline and will be extended in later methods.
- FIM candidate generation remains external and must be supplied in input payloads.
