# Phase 6 LDRD Worked Example (RAFM Corpus)

This worked example runs the full Phase 6 optimization workflow on the real RAFM irradiation dataset that backs the LDRD campaign artifacts in this repository.

## Input Data

- Analysis JSON corpus: `examples/RAFM_irradiation/results/analysis_json`
- Default sample: `RAFM4-C_15dEOI`
- Schedule metadata: `examples/RAFM_irradiation/metadata/sample_schedules.json`
- Unfold result for spectrum-aware scaling: `examples/RAFM_irradiation/results/unfolding/mlem.json`

## Command

From the `FluxForge` repository root:

```bash
PYTHONPATH=src python examples/RAFM_irradiation/run_phase6_ldrd_worked_example.py \
  --sample-id RAFM4-C_15dEOI
```

Optional output location override:

```bash
PYTHONPATH=src python examples/RAFM_irradiation/run_phase6_ldrd_worked_example.py \
  --sample-id RAFM4-C_15dEOI \
  --output-root examples/RAFM_irradiation/results/phase6_ldrd_worked_example/RAFM4-C_15dEOI
```

## What It Produces

The script writes a complete bundle under:

`examples/RAFM_irradiation/results/phase6_ldrd_worked_example/<sample_id>/`

Expected key artifacts:

- `activity_review.json`
- `inventory_review.json`
- `masking_review.json`
- `optimization_di_fom.json`
- `optimization_fim_d.json`
- `optimization_mwdcs.json`
- `optimization_bass_d.json`
- `optimization_stbd_mr.json`
- `second_irradiation_schedule.json`
- `second_irradiation_candidates.json`
- `second_irradiation_plan.json`
- `benchmark_experimental_bundle.ffexp`
- `WORKED_EXAMPLE_SUMMARY.md`

## Workflow Coverage

This worked example explicitly exercises these Phase 6 CLI commands end-to-end:

- `inventory-review`
- `masking-review`
- `optimization-sweep` for objectives `di-fom`, `fim-d`, `mwdcs`, `bass-d`, and `stbd-mr`
- `second-irradiation-plan`
- `ffexp-export`

## Notes

- Advanced objectives are enabled where required (`bass-d`, `stbd-mr`).
- The script builds an activity-review payload directly from the real analysis JSON and keeps all generated artifacts in one reproducible output directory.
