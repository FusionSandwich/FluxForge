# Method 1 DI-FOM Workflow (Implemented Baseline)

This note documents the first implemented optimization method in FluxForge:
Dose-Importance Figure of Merit (DI-FOM) schedule ranking.

## Scope

Implemented surfaces:

- Analysis core: line-level and schedule-level DI-FOM scoring
- CLI workflow: `optimization-sweep` for ranking candidate schedules
- GUI hook: Inventory panel DI-FOM preview from current activity-review lines

Current objective:

- `di-fom` (the first baseline gate objective)

## CLI Input Bundle

`optimization-sweep` consumes one JSON payload with two top-level fields:

- `candidates` (required): list of candidate schedules
- `isotope_weights` (optional): per-nuclide weighting map

Example:

```json
{
  "isotope_weights": {
    "Mo-99": 1.2,
    "Tc-99m": 0.8
  },
  "candidates": [
    {
      "label": "candidate_a",
      "irradiation_time_s": 3600,
      "cooldown_time_s": 7200,
      "count_time_s": 900,
      "lines": [
        {
          "nuclide": "Mo-99",
          "line_energy_keV": 140.5,
          "signal_counts": 120,
          "background_counts": 30,
          "interference_counts": 10
        }
      ]
    }
  ]
}
```

## CLI Usage

```bash
fluxforge optimization-sweep \
  --input optimization_candidates.json \
  --objective di-fom \
  --output optimization_sweep.json \
  --csv-output optimization_sweep.csv
```

JSON output schema:

- `fluxforge.optimization_sweep.difom.v1`

Primary output fields:

- `ranked_candidates[].rank`
- `ranked_candidates[].difom_score`
- `ranked_candidates[].line_scores[]`

## GUI Preview

The Inventory / Time Evolution panel includes a **Preview DI-FOM** control.

Preview behavior:

- Uses currently loaded activity-review rows
- Uses age-corrected activity as proxy signal
- Uses activity uncertainty as proxy background
- Returns a single preview score and line count summary

This preview is intentionally lightweight and is not yet a full transport-aware masking calculation.

## Formula

Line-level contribution:

$$
\mathrm{DI\text{-}FOM}_{line} = \frac{s^2}{s + b + i}
$$

where:

- $s$: expected signal counts
- $b$: expected background counts
- $i$: expected interference counts

Schedule score:

$$
\mathrm{DI\text{-}FOM}_{schedule} = \sum_{l \in L} w_{\nu(l)} \cdot \mathrm{DI\text{-}FOM}_{line}(l)
$$

with nuclide weight $w_{\nu(l)}$.

## Known Limits

- Objective support is currently DI-FOM only.
- Candidate generation is external to this command and must be provided in the input JSON.
- Full masking graph and detector-response coupling are planned for later method gates.
