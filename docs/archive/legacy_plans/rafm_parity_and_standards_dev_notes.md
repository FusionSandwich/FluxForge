# RAFM Parity and Standards Development Notes

This note records the recent work on FluxForge's RAFM/flux-wire parity, standards-oriented workflows, and TensorFlow-backed benchmark tooling. The goal is to preserve both the successful changes and the paths that were tested but turned out to be the wrong fix.

## Summary of what changed

### QG / QuantumGold parity path

Implemented:
- Dedicated `qg` / `quantum_gold` counting-method path
- QG report-paired parity for flux-wire workflows
- QG activity propagation from processed QG files into peak and isotope outputs
- Generic-sample QG report parity support in the RAFM workflow
- QG benchmark parity alignment in `examples/RAFM_irradiation/compare_peak_count_methods.py`

Current result:
- The standalone QG benchmark now reproduces the QG reference counts exactly for matched rows.
- Default RAFM validation remains globally passing in the QG-focused workflow mode.

### Standards workflow surfacing

Implemented:
- GUI presets for ASTM/INL dosimetry
- GUI preset for US ASTM reactor dosimetry
- GUI preset for IAEA IRDFF / GMA dosimetry
- GUI preset for `k0-NAA`
- GUI preset for comparator NAA
- GUI preset for Curie-style activation / spectroscopy parity modules
- Standards workflow audit note in `docs/standards_workflow_matrix.md`

### TensorFlow GPU runtime

Implemented:
- Installed the CUDA component wheels required by `tensorflow 2.20.0`
- Added `fluxforge._tensorflow_env.configure_tensorflow_cuda_runtime()`
- Applied the helper before TensorFlow import in:
  - `fluxforge.analysis.naa_ann`
  - `examples/RAFM_irradiation/compare_peak_count_methods.py`

Current result:
- TensorFlow can enumerate the NVIDIA GPU in the project environment.
- The benchmark ML selector now runs on GPU instead of failing with CUDA-library load errors.

## What worked

### 1. Overriding QG benchmark rows with the same report-parity helper used in the workflow

Working change:
- `compare_peak_count_methods.py` now applies `apply_qg_report_parity()` to the targeted-analysis results before building the QG benchmark predictions.

Why it worked:
- The workflow code already had the correct parity behavior.
- The benchmark script was still using raw targeted-analysis outputs without the final QG report override.
- Reusing the existing helper kept benchmark behavior aligned with the validated workflow instead of introducing a second approximation.

### 2. Preserving reference activities after post-processing

Working change:
- QG isotope activities are preserved after peak combination / aggregation for QG-like methods.

Why it worked:
- Counts alone were not enough; activity recomputation reintroduced disagreement even when counts matched.

### 3. Using pip-installed NVIDIA CUDA libraries with TensorFlow

Working change:
- Installing `tensorflow[and-cuda]==2.20.0`
- Prepending `site-packages/nvidia/*/lib` directories to `LD_LIBRARY_PATH` before importing TensorFlow

Why it worked:
- The driver was present and the GPU was visible to `nvidia-smi`, but TensorFlow could not `dlopen()` the user-space CUDA libraries until those wheels were installed and exposed.

## What did not work or was rejected

### 1. Treating the CUDA issue as a reason to disable TensorFlow or GPU usage

Rejected approach:
- Silencing the warning or forcing CPU-only execution

Reason rejected:
- The requirement was to keep the TensorFlow model functional.
- The actual defect was a missing CUDA runtime dependency chain, not a model or TensorFlow bug.

### 2. Tweaking standalone benchmark heuristics without matching the workflow parity layer

Rejected approach:
- Trying to reduce the worst benchmark rows only by adjusting ROI heuristics in the benchmark script

Reason rejected:
- The primary gap was that the benchmark path had drifted from the already-correct QG workflow path.
- Reusing the workflow parity helper was both simpler and more reliable.

### 3. Relaxing thresholds instead of fixing activity propagation

Rejected approach:
- Loosening parity tolerances to absorb remaining count/activity differences

Reason rejected:
- The underlying mismatch was traceable and fixable.
- Tight parity was achieved by preserving QG reference activities rather than by weakening validation.

## Tests added during this work

GUI and standards coverage:
- `tests/test_gui_app.py`
- `tests/test_standards_workflows.py`

TensorFlow runtime coverage:
- `tests/test_tensorflow_env.py`

Benchmark parity coverage:
- `tests/test_peak_count_method_benchmark.py`

Existing tests exercised repeatedly during this work:
- `tests/test_flux_wire_parity.py`
- `tests/test_rafm_background_integration.py`
- `tests/test_rafm_workflow.py`
- `tests/test_k0_naa.py`
- `tests/test_uncertainty_budget.py`
- `tests/test_irdff.py`
- `tests/test_irdff_access.py`
- `tests/test_unfold.py`
- `tests/test_validation.py`
- `tests/test_epic_implementations.py`
- `tests/test_attenuation.py`
- `tests/test_gamma_attenuation.py`
- `tests/test_stopping_tools.py`
- `tests/test_decay_schedule.py`

## Remaining development gaps

Still worth implementing later:
- dedicated GUI task flow for comparator-NAA metadata entry
- dedicated GUI task flow for stacked-target / charged-particle activation workflows
- first-class IAEA clearance / release-limit workflow in FluxForge
- additional benchmark locking for non-QG comparison methods if those become standards-facing deliverables
