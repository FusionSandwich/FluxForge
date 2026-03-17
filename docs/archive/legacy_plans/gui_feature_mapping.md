# FluxForge GUI Feature Mapping (from `testing/`)

This document translates `testing/` reference programs into concrete GUI
requirements and implementation targets for FluxForge.

## Priority Reference Repos

### Tier 1 (Direct GUI behavior references)

1. `SpecKit`
- Why: closest match to FluxForge's workflow style (tabbed desktop UI + solver controls + plots).
- Borrowed patterns:
  - tabbed workflow navigation,
  - file-picker driven analysis steps,
  - in-app solver execution with visible diagnostics.

2. `hdtv`
- Why: strongest spectrum-interaction ergonomics.
- Borrowed patterns:
  - keyboard-first optional controls,
  - fast ROI/peak/cursor interactions,
  - interactive calibration workflows.

3. `Gamma-MCA`
- Why: strong usability model for spectrum UX and metadata/export.
- Borrowed patterns:
  - multiple spectra overlays/buffers,
  - peak assist tools and annotations,
  - robust import/export with reproducible metadata.

### Tier 2 (Algorithm and analysis capability references)

4. `peakingduck` & QuantumGold Parity
- Why: classical and ML-style peak detection and deconvolution parity.
- Borrowed patterns:
  - first/second derivative and top-hat peak finding,
  - weighted NLLS Gaussian multiplet deconvolution,
  - alternative peak-finding backends (AI/ML),
  - dynamic ROI sizing and zero-count MDA.

5. `gamma_spec_analysis`, `PyGammaSpec`, `curie`, `becquerel`, `npat`, `irrad_spectroscopy`
- Why: feature depth for spectroscopy, isotope ID, activity, calibration utilities.
- Borrowed patterns:
  - configurable peak/smoothing routines,
  - isotope/line-centric analysis views,
  - analysis result tables and exports.

### Tier 3 (Neutron unfolding + validation workflow references)

6. `Neutron-Unfolding`, `pyunfold`, `Neutron-Spectrometry`, `gmapy`
- Why: unfolding and covariance-oriented diagnostics.
- Borrowed patterns:
  - solver selection and convergence diagnostics,
  - config-driven reproducible runs,
  - uncertainty and comparison views.

## GUI Feature Mapping to FluxForge Modules

| GUI Capability | FluxForge Core Modules | Reference Repos |
|---|---|---|
| Spectrum ingest + metadata | `fluxforge.io.*` | Gamma-MCA, SpecKit |
| Interactive ROI/peaks | `analysis.peak_finders`, `analysis.peakfit` | hdtv, SpecKit, peakingduck |
| Calibration/efficiency tools | `analysis.detector_calibration`, `analysis.auto_calibration` | hdtv, gamma_spec_analysis |
| Activities and corrections | `physics.activation`, `corrections.*` | curie, irrad_spectroscopy |
| Reaction-rate build | `physics.sigphi`, `physics.activation` | STAYSL-style parity in FluxForge |
| Unfolding controls | `solvers.*`, `unfold.*` | Neutron-Unfolding, pyunfold, Neutron-Spectrometry |
| Validation/compare views | `validation.*`, `workflows.*` | gmapy, SpecKit |
| Reproducible artifacts/report | `io.artifacts`, `reporting` | Gamma-MCA style exports |

## Implementation Strategy

### Phase A (current start)
- Build a desktop tabbed GUI shell that is fully CLI-mapped.
- Each GUI action emits and can copy an equivalent `fluxforge ...` command.
- Include a persistent run log panel for reproducibility.

### Phase B
- Add interactive spectrum canvas with ROI tools and peak table.
- Add calibration and efficiency editing panels.

### Phase C
- Add full unfolding + model-comparison tabs and uncertainty diagnostics.
- Add optional keyboard-first controls inspired by `hdtv`.

## Non-Negotiables

- GUI is optional: core workflows remain headless/scriptable.
- No mandatory network services.
- Reproducible outputs: config + command transcript + artifacts.
