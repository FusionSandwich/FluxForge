# RAFM Count-Parity Handoff

## Purpose
This document is a handoff for another LLM or engineer to continue the RAFM count-parity work in FluxForge without re-deriving the problem from scratch.

The current blocker is **step 1**:
- make FluxForge automatic peak identification and counting match the QG processed results closely enough for:
  - flux wires
  - RAFM activation samples
- confirm peak IDs are correct
- confirm gross and net counts are trustworthy
- only after that, audit uncertainty and then proceed to activity/efficiency work

Do **not** move to step 2 until step 1 is actually defensible.

## What we are trying to do
FluxForge must analyze the raw gamma spectra in:
- `examples/RAFM_irradiation/raw_gamma_spec/`

and compare them against the QG gold-standard processed files in:
- `examples/RAFM_irradiation/QG_processed_gamma_data/`

The main workflow must stay:
- automatic
- self-contained inside FluxForge
- independent of external repos at runtime

The QG processed files are:
- allowed as gold-standard comparison outputs
- not allowed to steer runtime analysis logic directly

## Why this matters
The RAFM example is the main real-data validation case for FluxForge.

If gross/net counts and peak IDs are wrong:
- activities will also be wrong
- uncertainty analysis will be meaningless
- later GUI work will be built on unstable analysis behavior

## Hard constraints
- Keep the main RAFM workflow automatic.
- Do not use manual ROI selection for the main RAFM example.
- Do not use QG outputs to choose runtime isotopes or peak windows.
- Do not overfit to one spectrum or one wire.
- Peaks below `80 keV` are ignored for RAFM validation.
- Use the shared background:
  - `examples/RAFM_irradiation/background.ASC`
- Use the shared RAFM profile:
  - `rafm_25cm`

## Core data and code

### Data
- Raw RAFM and wire spectra:
  - `examples/RAFM_irradiation/raw_gamma_spec/`
- QG processed reference:
  - `examples/RAFM_irradiation/QG_processed_gamma_data/`
- Shared RAFM background:
  - `examples/RAFM_irradiation/background.ASC`
- Workflow config:
  - `examples/RAFM_irradiation/metadata/workflow_config.json`

### Main code
- Flux-wire analysis:
  - `src/fluxforge/analysis/flux_wire_analysis.py`
- RAFM workflow:
  - `src/fluxforge/examples/rafm_workflow.py`
- Peak fitting:
  - `src/fluxforge/analysis/peakfit.py`
- Background subtraction:
  - `src/fluxforge/analysis/spectrum_math.py`

### Main result artifacts
- Flux-wire count disagreement table:
  - `examples/RAFM_irradiation/results/tables/flux_wire_count_disagreement.csv`
- Flux-wire count summary:
  - `examples/RAFM_irradiation/results/tables/flux_wire_count_disagreement_summary.md`
- Per-sample line diagnostics:
  - `examples/RAFM_irradiation/results/tables/line_diagnostics/`
- Per-sample reports:
  - `examples/RAFM_irradiation/results/reports/`

## Current status

### Flux wires
The automatic flux-wire count path is improved but not finished.

Representative current lines:
- `Co-Cd-RAFM-1_25cm`, `Co60 @ 1173.13 keV`: gross `-13.25%`, net `-0.92%`
- `Co-Cd-RAFM-1_25cm`, `Co60 @ 1332.44 keV`: gross `-14.45%`, net `-7.44%`
- `Ti-RAFM-1a_25cm`, `Sc47 @ 159.30 keV`: gross `+3.24%`, net `+0.35%`
- `Ti-RAFM-1a_25cm`, `Sc48 @ 175.28 keV`: gross `+11.09%`, net `-3.84%`
- `Ti-RAFM-1a_25cm`, `Sc46 @ 889.11 keV`: gross `-2.53%`, net `+7.73%`
- `Sc-RAFM-1_25cm`, `Sc46 @ 889.36 keV`: gross `+1.27%`, net `-0.44%`
- `Sc-RAFM-1_25cm`, `Sc46 @ 1120.41 keV`: gross `-1.61%`, net `-1.94%`

Main remaining flux-wire count issues:
- `Co-Cd 1332 keV`
- `Ti-RAFM-1 / 1a / 1b` `Sc48`
- some `Ti Sc46`

### RAFM generic samples
This path is still not good enough.

Main unresolved RAFM sample issues:
- RAFM3 `W187` count parity around:
  - `133 keV`
  - `551 keV`
  - `618 keV`
  - `773 keV`
- RAFM4 crowded low-energy `Ta182` count parity around:
  - `99 keV`
  - `113 keV`
  - `152 keV`
  - `156 keV`
  - `222 keV`
- RAFM4 missing QG peaks currently reported:
  - `Ta182 @ 229.55 keV`
  - `Ta182 @ 1231.00 keV`
  - `Fe59 @ 1099.25 keV`
  - `Co60 @ 1173.17 keV`

Latest focused result:
- a new sample-driven generic targeted-library selector is now implemented
- focused RAFM4-A and RAFM3-B checks no longer miss:
  - `Ta182 @ 229.55 keV`
  - `Ta182 @ 1231.00 keV`
  - `Fe59 @ 1099.25 keV`
  - `Co60 @ 1173.17 keV`
  - RAFM3-B `W187` targeted peaks
- the generic sample path is still open because count parity in crowded `Ta182` and `W187` regions is not fixed yet

## What has already been tried

### Background and calibration
- Added measured-background subtraction as a first-class path.
- Background subtraction now propagates channel uncertainty.
- Background is energy-aligned before subtraction if sample and background calibrations differ.
- Shared `rafm_25cm` profile is used for RAFM example work.
- The RAFM workflow explicitly uses the shared QG `@25 cm` energy calibration override for the example workflow.

### Negative bins / SNIP
- Signed counts are preserved after measured-background subtraction.
- SNIP now uses an internal offset copy instead of clip-before-SNIP.

### Flux-wire count path
- Split QG `GROSS/NET` comparison from the activity path.
- Count parity now uses raw-spectrum comparison windows.
- Added compact comparison-window scoring inside the `32 channel` capture range.
- Added a broad-window override using smoothed local minima.
- Broad-window override is now only allowed when it preserves compact-window net area.

### Generic RAFM path
- Generic targeted recovery is combined with exploratory detection.
- Generic gamma library already dedupes near-identical cross-isotope lines at very small energy separation.
- Added a linear/trapezoid comparison-background option for generic sample parity.
- Added a broad-window cap based on widened gross counts vs the primary raw ROI gross counts.
- Added a sample-driven targeted-library selector:
  - supported isotopes keep their full line sets
  - unsupported isotopes keep only their strongest fallback lines
  - weak nearby nuisance lines are then pruned from that reduced library

### Documentation / forensic outputs
- Per-sample text comparison reports
- Per-sample line diagnostics CSVs
- Flux-wire count disagreement CSV and summary
- Multiple issue notes under:
  - `examples/RAFM_irradiation/issues/`

## Important findings from debugging

### 1. Flux-wire and RAFM generic problems are different
Do not assume one fix applies to both paths.

- Flux wires:
  - main problem was automatic support-window selection
- RAFM generic samples:
  - still have crowded-region and local-background-model problems

### 2. Some RAFM4 “missing peaks” were a targeted-library interaction problem
This is no longer just a hypothesis. Focused regressions now show the missing-line problem is largely caused by the generic targeted library being too dense.

Observed example:
- `Fe59 @ 1099.25 keV`
- `Co60 @ 1173.17 keV`

Likely cause:
- they are being grouped or destabilized by weak nearby nuisance lines in the generic targeted library

Concrete example from the current generic library:
- `Fe59 1099.25` is near weak `Tb154m 1102.43`
- `Co60 1173.23` is near weak `Tb154m 1177.71`

When those weak nearby lines are pruned in a narrow test, `Fe59 1099` and `Co60 1173` reappear.

Current implemented fix:
- build the targeted library from exploratory support first
- keep full line sets for supported isotopes
- keep only strongest fallback lines for unsupported isotopes
- then apply weak-neighbor pruning

Focused regression status after the fix:
- RAFM4-A recovers `Ta182 229.55`, `Ta182 1231.00`, `Fe59 1099.25`, and `Co60 1173.17`
- RAFM3-B no longer has missing `W187` targeted peaks

### 3. Low-energy Ta182 crowding is still the hardest generic RAFM problem
The `99-229 keV` region remains the biggest source of bad gross/net mismatch for RAFM4.

### 4. W187 in RAFM3 looks like a background-model issue more than a support-window issue
For some `W187` lines:
- gross counts are already close
- net counts are still far off

That points to local background estimation, not just wrong support bounds.

## Promising next options

### Option A: Improve crowded-region support selection for low-energy Ta182
Needed after the new targeted-library selector.

Possible directions:
- tighter support-window cap in crowded low-energy regions
- stronger local shape penalty in the generic comparison-window scorer
- separate crowded-region support rule for generic samples, still fully automatic and not sample-specific

### Option B: Improve the generic local background estimator for RAFM sample parity
Needed for RAFM3 `W187`.

Possible directions:
- use more robust trapezoid/linear sideband estimation
- allow wider sidebands for RAFM sample parity only
- compare direct ROI/trapezoid results with the current helper on a few `W187` lines

### Option C: Audit merge/match behavior directly if missing-line regressions reappear
This is lower priority now because the focused RAFM4/RAFM3 missing-line regressions are green.

## What should not be done
- Do not use manual ROIs in the main RAFM example.
- Do not use QG to choose runtime isotopes, seeds, or windows.
- Do not move on to activity/efficiency fixes until step 1 is actually good enough.
- Do not “fix” the tests by loosening all thresholds globally.

## Recommended immediate next implementation
1. Re-run focused generic-sample count diagnostics after the new targeted-library selector:
   - RAFM4 low-energy `Ta182`
   - RAFM3 `W187`
2. Tighten the generic local background / sideband model for those regions without changing the flux-wire path.
3. Once generic-sample gross/net parity is defensible, audit the propagated uncertainty on the stabilized count path.
4. Only then return to the activity/efficiency path.

## Tests and commands

### Fast compile check
```bash
cd /filespace/s/smandych/CAE/projects/ALARA/FluxForge
PYTHONPATH=src python -m py_compile \
  src/fluxforge/analysis/flux_wire_analysis.py \
  src/fluxforge/examples/rafm_workflow.py
```

### Current focused count-path regression
```bash
cd /filespace/s/smandych/CAE/projects/ALARA/FluxForge
PYTHONPATH=src python -m pytest \
  tests/test_flux_wire_analysis.py \
  tests/test_rafm_workflow.py::test_flux_wire_count_parity_representative_lines \
  -q
```

### Focused generic-targeted-library regressions
```bash
cd /filespace/s/smandych/CAE/projects/ALARA/FluxForge
PYTHONPATH=src python -m pytest \
  tests/test_rafm_workflow.py::test_select_generic_targeted_lines_keeps_supported_sets_and_limits_dense_unsupported_isotopes \
  tests/test_rafm_workflow.py::test_generic_targeted_selection_recovers_known_rafm4_missing_lines \
  -q
```

### Honest flux-wire parity test
This still fails and should keep failing until the analysis is actually fixed:
```bash
cd /filespace/s/smandych/CAE/projects/ALARA/FluxForge
PYTHONPATH=src python -m pytest \
  tests/test_flux_wire_parity.py::test_raw_flux_wire_parity \
  -q
```

### Full RAFM validation
Heavy and slow:
```bash
cd /filespace/s/smandych/CAE/projects/ALARA/FluxForge
PYTHONPATH=src python examples/RAFM_irradiation/run_validation.py --no-fail
```

### Regenerate only the flux-wire count disagreement table
Much faster than the full bundle:
```bash
cd /filespace/s/smandych/CAE/projects/ALARA/FluxForge
PYTHONPATH=src python - <<'PY'
from pathlib import Path
from fluxforge.examples.rafm_workflow import (
    analyze_flux_wire_sample,
    build_flux_wire_count_disagreement_rows,
    default_paths,
    ensure_results_tree,
    load_rafm_example_metadata,
    normalize_pairing_key,
    write_flux_wire_count_disagreement_summary,
    write_rows_csv,
)
from fluxforge.io.flux_wire import read_raw_asc

example_root = Path('examples/RAFM_irradiation')
metadata = load_rafm_example_metadata(example_root)
paths = default_paths(example_root)
tree = ensure_results_tree(paths.results_root)
background = read_raw_asc(
    paths.background_path,
    energy_calibration_override=[-1.694, 0.4996, 6.710e-08],
    profile_name=metadata.config['profile_name'],
).spectrum
line_rows = []
for raw_path in sorted((paths.raw_root / 'flux_wires').glob('*.ASC')):
    qg_path = paths.qg_root / 'flux_wires' / f'{raw_path.stem}.txt'
    if not qg_path.exists():
        qg_path = None
    sample_key = normalize_pairing_key(raw_path.stem, metadata.pairing_aliases)
    artifact = analyze_flux_wire_sample(raw_path, metadata, paths, tree, background, qg_path, sample_key)
    line_rows.extend(artifact['line_diagnostics'])
rows = build_flux_wire_count_disagreement_rows(line_rows)
write_rows_csv(rows, tree['tables'] / 'flux_wire_count_disagreement.csv')
write_flux_wire_count_disagreement_summary(rows, tree['tables'] / 'flux_wire_count_disagreement_summary.md')
print(f'updated {len(rows)} count rows')
PY
```

## Current acceptance bar for step 1
Do not treat step 1 as complete until:
- flux-wire gross/net parity is close across the full set, not just a few examples
- RAFM generic-sample line diagnostics stop showing the large `Ta182` and `W187` count failures
- the “missing” RAFM4 peaks are either recovered correctly or explained by a defensible automatic rule
- peak IDs look stable in the per-sample reports and annotated plots
- then, and only then, do the uncertainty audit

## Last confirmed state
- flux-wire count path improved and regression still passes
- generic RAFM missing-line regressions improved with the new targeted-library selector
- RAFM generic-sample count parity is still open
- step 2 should not start yet
