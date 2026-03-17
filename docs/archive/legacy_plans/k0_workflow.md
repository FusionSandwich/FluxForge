# FluxForge k0-NAA Workflow Status

## Current implementation scope

FluxForge now contains a first-pass standards-oriented k0 workflow that is aligned with the repository's CLI-first architecture.

Implemented now:
- normalized `PeakObservation` handling that separates peak analysis from k0 interpretation,
- explicit line-eligibility classification and default rejection of escape/sum/interference-sensitive lines,
- governed starter k0 library metadata plus a separate auxiliary threshold / fast-flux library,
- selectable external governed library loading so the current bundled starter library can remain the default while fuller TECDOC-aligned JSON libraries are added as an option,
- reusable detector-characterization artifacts with a reference position and explicit geometry-conversion provenance,
- reusable facility-characterization artifacts for bare triple-monitor, Cd-ratio multi-monitor, and single-monitor-with-known-f/α workflows,
- first-pass k0 analysis bundles with applied-correction reporting, recognized-but-not-applied corrections, capability flags, library provenance, and project/sample/irradiation/measurement identifiers,
- cross-bundle aggregation artifacts for deeper multi-measurement / multi-irradiation combination,
- blank / CRM QA-QC artifacts,
- richer k0 report generation with text + CSV table outputs,
- CLI commands:
  - `k0-normalize`
  - `k0-detector`
  - `k0-facility`
  - `k0-analyze`
  - `k0-aggregate`
  - `k0-qaqc`
  - `k0-report`
  - `k0-import-kayzero`

## Supported workflow in this pass

Validated target:
- thermal INAA,
- bare triple-monitor facility characterization,
- single-monitor analysis when `f` and `α` are already known from prior characterization,
- relative k0 analysis with an Au reference observation.

Partially supported / scaffolded:
- Cd-covered workflows via multi-monitor characterization artifacts,
- fast-flux / threshold-interference bookkeeping,
- detector peak-to-total and coincidence characterization metadata,
- geometry conversion using empirical reference-position transfer or explicit fallback provenance,
- external full-library ingestion subject to user-supplied governed JSON and still awaiting an in-tree TECDOC benchmark library payload.

Not validated in this pass:
- low-energy photon mode,
- prompt-gamma mode,
- full TECDOC benchmark dataset ingestion,
- full CRM/blank trend tracking UI.

## Input expectations

### Detector characterization
Input rows for `k0-detector` should provide JSON or CSV fields such as:
- `position_mm`
- `reference_energy_keV`
- `net_counts`
- `live_time_s`
- `activity_bq`
- `emission_probability`
- optional uncertainty fields

### Facility characterization
Input JSON for `k0-facility` should include:
- `facility_id`
- `irradiation`
- `monitors` with `monitor_id` and `activity`
- optional temperature / gradient / fast-flux notes

### Sample analysis
The current first-pass analysis expects:
- a `k0-normalize` observation bundle,
- a `k0-facility` artifact,
- a valid Au reference observation,
- sample mass and reference mass.

## Example CLI flow

1. Normalize identified peaks into governed observations.
2. Build detector characterization.
3. Build facility characterization.
4. Run k0 analysis.

Example command sequence:
- `fluxforge k0-import-kayzero --input KayWinV4.zip --output kayzero_k0_library.json --report-output kayzero_k0_library_import_report.json`
- `fluxforge k0-detector --points-file detector_points.json --detector-id HPGe-01 --reference-position-mm 200 --output detector_characterization.json`
- `fluxforge k0-normalize --peaks-file peaks.json --spectrum-file spectrum.json --detector-characterization-file detector_characterization.json --irradiation-time-s 600 --decay-time-s 3600 --output k0_observations.json`
- `fluxforge k0-facility --input facility_input.json --output facility_characterization.json`
- `fluxforge k0-analyze --observations-file k0_observations.json --facility-file facility_characterization.json --sample-mass-g 0.1 --reference-mass-g 0.001 --output k0_analysis.json`

## Kayzero import path

FluxForge can now ingest a user-supplied Kayzero folder or zip and write:
- a governed FluxForge k0 library JSON,
- a companion import report listing unresolved fields.

Current importer behavior:
- parses the text-discoverable Kayzero files such as `uk0`, `uQ0`, `uT12`, `MDcode`, and `FCd`,
- infers same-element `(n,γ)` parent target isotopes for the imported product isotopes,
- uses transparent FluxForge fallbacks where possible,
- does **not** decode opaque `LB1` / `LB2` payloads yet,
- explicitly reports which fields remain unresolved because of that limitation.

## Scientific honesty notes

This is not full TECDOC-grade compliance yet.

Current gaps include:
- broader governed k0 library coverage beyond the starter bundle,
- benchmark ingestion of the official TECDOC supplementary dataset,
- full threshold-correction solvers with validated numerical fast-flux corrections,
- trend / control-chart QA-QC over time rather than one-shot bundle evaluation,
- covariance-aware / Monte Carlo uncertainty propagation,
- broader GUI integration tests for aggregation / QA-QC / report flows.

## Benchmark note

For now, the verification path should use local repository datasets and deterministic synthetic cases.
A future master-plan update should add the official TECDOC benchmark dataset into the repository's verification harness.
