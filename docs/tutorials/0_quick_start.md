# FluxForge Quick Start Guide

## Running Flux Spectrum Analysis

### RAFM Background-Subtracted Ingest

For RAFM irradiation campaigns, use the explicit `rafm_25cm` profile. It supplies the shared background file plus the shared 25 cm detector efficiency and resolution defaults, while each raw `.ASC` file keeps its own header energy calibration (`A + B*Ch + C*Ch^2`) unless you override it.

```bash
python -m fluxforge.cli.app ingest-batch \
  --input-dir examples/RAFM_irradiation/raw_gamma_spec \
  --profile rafm_25cm \
  --background-scale-mode live \
  --output-dir examples/RAFM_irradiation/results/spectrum_artifacts \
  --background-adjusted-dir examples/RAFM_irradiation/results/background_adjusted \
  --final-corrected-dir examples/RAFM_irradiation/results/final_corrected
```

Notes:
- Background subtraction is enabled by default in `ingest`.
- `ingest-batch --profile rafm_25cm` applies the same measured background file and shared detector efficiency defaults to every spectrum in the input directory tree.
- If no background file is provided, FluxForge emits a warning and continues.
- Negative bins from subtraction are retained in stored spectra and in uncertainty propagation.
- SNIP now uses an internal offset working copy instead of clipping RAFM background-subtracted counts before background estimation.
- Background-adjusted outputs are written as channel tables under the directory passed to `--background-adjusted-dir`.
- Final corrected channel tables can also be written with `--save-final-corrected` or `--final-corrected-dir` when efficiency coefficients are available.

### RAFM Gold-Standard Validation Workflow

Run the full self-contained RAFM example workflow:

```bash
cd FluxForge
PYTHONPATH=src python examples/RAFM_irradiation/run_validation.py
```

This workflow:
- analyzes every committed raw RAFM and flux-wire spectrum with the shared `background.ASC`
- uses the bundled `rafm_25cm` detector profile while preserving each raw `.ASC` file's own `A + B*Ch + C*Ch^2` calibration
- compares matched raw/QG pairs against the committed QG gold-standard dataset
- writes per-spectrum JSON artifacts, count tables, comparison CSVs, and plots under `examples/RAFM_irradiation/results/`
- computes flux-wire Cd ratios and unfolds the neutron spectrum with `DISCRETE`, `GLS`, `GRAVEL`, and `MLEM`

If you want the workflow to complete and write the bundle even when validation thresholds are violated:

```bash
cd FluxForge
PYTHONPATH=src python examples/RAFM_irradiation/run_validation.py --no-fail
```

Single-spectrum example with both exports:

```bash
python -m fluxforge.cli.app ingest \
  --input examples/RAFM_irradiation/raw_gamma_spec/RAFM4/RAFM4-B_15dEOI.ASC \
  --profile rafm_25cm \
  --output examples/RAFM_irradiation/results/spectrum_artifacts/RAFM4/RAFM4-B_15dEOI.json \
  --save-background-adjusted examples/RAFM_irradiation/results/background_adjusted/RAFM4/RAFM4-B_15dEOI_background_adjusted.csv \
  --save-final-corrected examples/RAFM_irradiation/results/final_corrected/RAFM4/RAFM4-B_15dEOI_final_corrected.csv
```

### Separate Manual Inspection Example

The main RAFM and flux-wire validation workflow should remain automatic. If you want a user-driven inspection path over SSH before the GUI is ready, use the separate example under [manual_peak_inspection](/filespace/s/smandych/CAE/projects/ALARA/FluxForge/examples/manual_peak_inspection/README.md).

### Basic Example: Fe-Cd-RAFM-1 Single Foil

Generate flux spectrum from experimental data:

```bash
cd FluxForge
python examples/generate_flux_spectrum.py
```

**Output:**
- `output/fe_cd_rafm_1_gls.json` - GLS unfolded spectrum with covariance
- `output/fe_cd_rafm_1_gravel.json` - GRAVEL iterative result
- `output/fe_cd_rafm_1_mlem.json` - MLEM iterative result

### Running Tests

Validate all functionality:

```bash
# Complete test suite
python -m pytest tests/ -v

# Just master plan goal validation
python -m pytest tests/test_master_plan_goals.py -v

# Just pipeline validation with experimental data
python -m pytest tests/test_pipeline_validation.py -v
```

Run these from the `FluxForge/` repo root. In the full environment from [environment.yml](/filespace/s/smandych/CAE/projects/ALARA/FluxForge/environment.yml), TensorFlow and `h5py` are installed so optional-library tests do not skip for missing dependencies.

### Example Workflow in Python

```python
from pathlib import Path
import json

# Import FluxForge modules
from fluxforge.physics.activation import (
    GammaLineMeasurement,
    weighted_activity,
    reaction_rate_from_activity,
    IrradiationSegment,
)
from fluxforge.core.response import (
    EnergyGroupStructure,
    ReactionCrossSection,
    build_response_matrix,
)
from fluxforge.solvers.gls import gls_adjust

# 1. Load experimental data
data_dir = Path("src/fluxforge/examples/fe_cd_rafm_1")
measurements = json.loads((data_dir / "measurements.json").read_text())
cross_sections = json.loads((data_dir / "cross_sections.json").read_text())
number_densities = json.loads((data_dir / "number_densities.json").read_text())
boundaries = json.loads((data_dir / "boundaries.json").read_text())
prior_flux = json.loads((data_dir / "prior_flux.json").read_text())

# 2. Calculate reaction rates from measured activities
reaction = measurements["reactions"][0]
gamma_lines = [GammaLineMeasurement(**gl) for gl in reaction["gamma_lines"]]
activity, activity_unc = weighted_activity(gamma_lines)

segments = [IrradiationSegment(**seg) for seg in measurements["segments"]]
rate_result = reaction_rate_from_activity(
    activity, segments, reaction["half_life_s"]
)

print(f"Reaction rate: {rate_result.rate:.3e} ± {rate_result.uncertainty:.3e} /s")

# 3. Build response matrix
groups = EnergyGroupStructure(boundaries)
reactions = [
    ReactionCrossSection(reaction_id=r_id, sigma_g=sigma)
    for r_id, sigma in cross_sections.items()
]
nd_values = [number_densities[rx.reaction_id] for rx in reactions]
response = build_response_matrix(reactions, groups, nd_values)

# 4. Unfold spectrum using GLS
measurement_cov = [[(rate_result.uncertainty ** 2)]]
prior_cov = [
    [(0.25 * val) ** 2 if i == j else 0.0 for j, val in enumerate(prior_flux)]
    for i, _ in enumerate(prior_flux)
]

solution = gls_adjust(
    response.matrix,
    [rate_result.rate],
    measurement_cov,
    prior_flux,
    prior_cov,
)

print(f"χ² = {solution.chi2:.4f}")
print(f"Unfolded flux: {solution.flux}")
```

## Understanding the Outputs

### Output JSON Structure

```json
{
  "schema": "fluxforge.unfold_result.v1",
  "boundaries_eV": [0.0305, 100.0, 10000.0, 636000.0],
  "reactions": ["fe58_ng_fe59"],
  "flux": [0.0, 46149.28, 19931.54],
  "covariance": [[...], [...], [...]],
  "chi2": 18.40,
  "method": "gls",
  "provenance": {
    "created_at": "2026-01-03T04:13:26.661725+00:00",
    "versions": {"fluxforge": "0.1.0"},
    "units": {
      "flux": "a.u.",
      "covariance": "a.u.^2",
      "boundaries_eV": "eV"
    }
  }
}
```

### Reading Results Back

```python
from fluxforge.io.artifacts import read_unfold_result

result = read_unfold_result("output/fe_cd_rafm_1_gls.json")

print(f"Method: {result['method']}")
print(f"Energy groups: {len(result['flux'])}")
print(f"Group boundaries: {result['boundaries_eV']}")
print(f"Flux values: {result['flux']}")
print(f"χ²: {result['chi2']}")
```

## Data Files Reference

### Input Data Location

```
src/fluxforge/examples/
├── fe_cd_rafm_1/           # Single-foil example
│   ├── measurements.json    # Gamma line measurements
│   ├── cross_sections.json  # IRDFF cross sections
│   ├── number_densities.json
│   ├── boundaries.json      # Energy group boundaries
│   └── prior_flux.json      # Prior spectrum
│
├── flux_wire/              # Multi-wire campaign
│   ├── flux_wire_timing.csv
│   ├── Co-Cd-RAFM-1_25cm.txt
│   └── Co-Cd-RAFM-1_25cm.ASC
│
└── eff.csv                 # Detector efficiency calibration
```

### Measurement JSON Format

```json
{
  "reactions": [
    {
      "reaction_id": "fe58_ng_fe59",
      "half_life_s": 3843936.0,
      "gamma_lines": [
        {
          "energy_kev": 1099.245,
          "intensity": 0.565,
          "net_counts": 8743.0,
          "counts_uncertainty": 94.82,
          "efficiency": 0.001085,
          "efficiency_uncertainty": 5.425e-05
        }
      ]
    }
  ],
  "segments": [
    {
      "flux_level": 1.0,
      "duration_s": 7200.0
    }
  ],
  "live_time_s": 43200.0,
  "cooling_time_s": 172800.0
}
```

## Common Tasks

### Adding a New Reaction

1. Get cross sections from IRDFF database:
```python
from fluxforge.data.irdff import IRDFFDatabase

db = IRDFFDatabase()
sigma = db.get_group_xs("ni58_np_co58", boundaries)
```

2. Add to measurements JSON with gamma lines

3. Update cross_sections.json and number_densities.json

4. Rerun analysis

### Changing Energy Group Structure

Edit `boundaries.json`:
```json
[0.0305, 0.5, 1.0, 100.0, 10000.0, 636000.0]
```

This creates 5 groups instead of 3.

### Using Different Solvers

```python
from fluxforge.solvers.iterative import gravel, mlem

# GRAVEL
gravel_result = gravel(
    response.matrix,
    measurements,
    initial_flux=prior_flux,
    max_iters=500,
)

# MLEM  
mlem_result = mlem(
    response.matrix,
    measurements,
    initial_flux=prior_flux,
    max_iters=200,
)
```

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'fluxforge'"

**Solution:** Install package in development mode:
```bash
cd FluxForge
pip install -e .
```

### Issue: Iterative solvers give zero flux

**Cause:** Problem is underdetermined (too few measurements)

**Solution:** Either:
- Use GLS with informative prior
- Add more foil reactions
- Reduce number of energy groups

### Issue: χ² is very large

**Possible causes:**
- Prior is far from truth → Adjust prior or increase prior uncertainty
- Measurement error underestimated → Check detector calibration
- Energy group structure mismatch → Refine boundaries

### Issue: Negative covariance diagonal

**Cause:** Numerical precision in matrix inversion

**Solution:** Check condition number of matrices, increase measurement precision

## Next Steps

1. **Process full flux wire campaign:** 10 wires with multiple reactions
2. **Compare with transport:** Run OpenMC model of TRIGA reactor
3. **Sensitivity analysis:** Vary cross sections, efficiencies within uncertainties
4. **Publication plots:** Use matplotlib to generate figures with error bars

## Getting Help

- **Documentation:** See `docs/master_plan.md` for complete methodology
- **Issues:** Check `docs/issue_map.md` for known limitations  
- **Tests:** Run test suite to verify installation
- **Examples:** Review `examples/` directory for more workflows

---

**Version:** FluxForge 0.1.0  
**Updated:** January 2, 2026
