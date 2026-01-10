# Unfolding Benchmark

This directory contains benchmark data and code for validating FluxForge's GRAVEL and MLEM spectrum unfolding implementations against reference implementations from the [Neutron-Unfolding](../../testing/Neutron-Unfolding) repository.

## Data Files

| File | Description |
|------|-------------|
| `response-matrix.txt` | Scintillator detector response matrix (201×1024, CSV format) |
| `reduced_data.csv` | Pulse-height spectrum measurements (1024 channels) |
| `energy-spectrum.txt` | Time-of-Flight reference spectrum (ground truth) |

## Reference Implementations

| File | Description |
|------|-------------|
| `reference_gravel.py` | Original GRAVEL implementation |
| `reference_mlem.py` | Original MLEM implementation |

## Running the Benchmark

```bash
cd FluxForge
python examples/unfolding_benchmark/run_benchmark.py
```

## Results

The benchmark validates that FluxForge's implementations produce results consistent with the reference:

| Metric | Reference GRAVEL | FluxForge GRAVEL | Reference MLEM | FluxForge MLEM |
|--------|------------------|------------------|----------------|----------------|
| Error vs ToF | 8.23 | 8.47 | 8.10 | 8.06 |
| Iterations | 48 | 47 | 42 | 41 |
| RMSE Agreement | - | 0.000158 | - | 0.000114 |

## Algorithm Details

### GRAVEL (SAND-II Variant)
Log-space multiplicative update using data-weighted response:
```
W[i,j] = data[i] * R[i,j] * x[j] / (R @ x)[i]
x[j] *= exp(sum_i(W[i,j] * log(data[i] / predicted[i])) / sum_i(W[i,j]))
```

### MLEM (Maximum-Likelihood Expectation Maximization)
```
x[j] *= (1/sum_i(R[i,j])) * sum_i(R[i,j] * data[i] / predicted[i])
```

## Output Files

Generated in `examples_output/`:
- `unfolding_benchmark_comparison.png` - Comparison of all methods vs ground truth
- `unfolding_implementation_diff.png` - Difference between FluxForge and reference
- `unfolding_benchmark_results.json` - Numerical results
