# FluxForge CLI Reference

This reference is generated from the FluxForge CLI parser metadata.
Refresh it with `python tools/generate_cli_reference.py` after updating CLI commands or catalog metadata.

Use `fluxforge commands` for the terminal view and `fluxforge <command> --help` for full flag details.

## Setup and Discovery

Discover installed entrypoints, grouped command families, and next-step help.

### `commands`

- Purpose: List FluxForge CLI commands grouped by workflow family
- Common use case: Browse the installed FluxForge command surface by family before choosing a workflow.
- Detailed help: `fluxforge commands --help`

```bash
fluxforge commands --family spectrum
```

## Spectrum Analysis

Ingest spectra, generate plots, and perform peak and ROI analysis workflows.

### `ingest`

- Purpose: Ingest spectrum files into schema artifacts
- Common use case: Convert one measured spectrum into a normalized FluxForge artifact before downstream analysis.
- Detailed help: `fluxforge ingest --help`

```bash
fluxforge ingest --input examples/RAFM_irradiation/raw_gamma_spec/RAFM4/RAFM4-B_15dEOI.ASC --profile rafm_25cm --output rafm4_b_ingest.json
```

### `ingest-batch`

- Purpose: Ingest a directory tree of spectra with one shared background file
- Common use case: Ingest a directory tree of measured spectra using one shared detector/background profile.
- Detailed help: `fluxforge ingest-batch --help`

```bash
fluxforge ingest-batch --input-dir examples/RAFM_irradiation/raw_gamma_spec/RAFM4 --profile rafm_25cm --output-dir artifacts/rafm4_ingest
```

### `spectrum-plot`

- Purpose: Save a calibrated gamma-spectrum plot with optional background subtraction and manual ROI overlays
- Common use case: Save a calibrated spectrum plot, optionally with background subtraction and manual ROI overlays.
- Detailed help: `fluxforge spectrum-plot --help`

```bash
fluxforge spectrum-plot --input examples/RAFM_irradiation/raw_gamma_spec/flux_wires/Ti-RAFM-1a_25cm.ASC --profile rafm_25cm --background-subtracted --manual-peaks-file examples/manual_peak_inspection/manual_flux_wire_ti_rafm_1a.csv --output Ti-RAFM-1a_25cm_manual.png
```

### `peaks`

- Purpose: Detect peaks from a spectrum artifact
- Common use case: Detect peaks from an ingested spectrum artifact or compute manual ROI counts from an overlay file.
- Detailed help: `fluxforge peaks --help`

```bash
fluxforge peaks --spectrum-file rafm4_b_ingest.json --output rafm4_b_peaks.json
```

### `roi-analyze`

- Purpose: Analyze one explicit ROI with sideband/SNIP background handling and optional overlap decomposition
- Common use case: Analyze one explicit ROI with sideband or SNIP background handling.
- Detailed help: `fluxforge roi-analyze --help`

```bash
fluxforge roi-analyze --input rafm4_b_ingest.json --left-keV 1170 --right-keV 1180 --output roi_analysis.json
```

### `roi-statistics`

- Purpose: Run the same ROI across many spectra and summarize detector-consistency statistics
- Common use case: Apply the same ROI definition across many spectra and summarize detector-consistency statistics.
- Detailed help: `fluxforge roi-statistics --help`

```bash
fluxforge roi-statistics --inputs rafm4_a_ingest.json rafm4_b_ingest.json --left-keV 1170 --right-keV 1180 --output roi_statistics.json
```

## Artifact Review and Comparison

Query artifact trees and compare structured outputs during review work.

### `file-query`

- Purpose: Query spectrum/artifact files in a workspace tree for archive-style review
- Common use case: Search an artifact tree for reviewable files during archive or result audits.
- Detailed help: `fluxforge file-query --help`

```bash
fluxforge file-query --root examples/RAFM_irradiation/results --contains validation --format json --output file_query.json
```

### `batch-compare`

- Purpose: Compare two tabular JSON/CSV artifacts by key fields and numeric deltas
- Common use case: Compare baseline and candidate JSON/CSV tables by shared key columns and numeric deltas.
- Detailed help: `fluxforge batch-compare --help`

```bash
fluxforge batch-compare --baseline baseline.csv --candidate candidate.csv --keys sample_id,line_energy_keV --output batch_compare.json
```

## Validation and Governance

Run parity, crosswalk, GUI acceptance, and release-gate checks.

### `parity-check`

- Purpose: Run algorithm/workflow parity fixtures and report pass/fail summaries
- Common use case: Run algorithm and workflow parity fixtures against committed reference cases.
- Detailed help: `fluxforge parity-check --help`

```bash
fluxforge parity-check --scope all --output parity_check.json
```

### `phase5-crosswalk-report`

- Purpose: Validate and summarize the machine-readable Phase 5 writeup crosswalk tracker
- Common use case: Validate and summarize the Phase 5 writeup crosswalk with optional parity coverage.
- Detailed help: `fluxforge phase5-crosswalk-report --help`

```bash
fluxforge phase5-crosswalk-report --include-parity-summary --output phase5_crosswalk_report.json --markdown-output phase5_crosswalk_report.md
```

### `gui-acceptance-check`

- Purpose: Validate 3.27 GUI release-checklist and probe artifact readiness
- Common use case: Check that the GUI release checklist and probe artifacts are present and consistent.
- Detailed help: `fluxforge gui-acceptance-check --help`

```bash
fluxforge gui-acceptance-check --output gui_acceptance_check.json
```

### `phase5-release-gate`

- Purpose: Run the Phase 5.6 release-gate checklist across parity, fixtures, tests, GUI evidence, and status-doc synchronization
- Common use case: Run the strict Phase 5 release-gate bundle across parity, tests, GUI evidence, and docs.
- Detailed help: `fluxforge phase5-release-gate --help`

```bash
fluxforge phase5-release-gate --output phase5_release_gate.json
```

## Activation, Inventory, and Libraries

Manage libraries and convert peak outputs into activity, rate, and inventory artifacts.

### `library-list`

- Purpose: List bundled and user-registered nuclear-data sources
- Common use case: Inspect bundled and user-registered nuclear-data sources before choosing a library for analysis.
- Detailed help: `fluxforge library-list --help`

```bash
fluxforge library-list --capability peak-identification
```

### `library-register`

- Purpose: Register a user gamma-line library by alias and locator
- Common use case: Register a user-supplied gamma-line library under a governed alias.
- Detailed help: `fluxforge library-register --help`

```bash
fluxforge library-register --alias my_lines --locator /path/to/library.csv --description "User review library"
```

### `library-remove`

- Purpose: Remove a registered user gamma-line library
- Common use case: Remove a previously registered governed library entry.
- Detailed help: `fluxforge library-remove --help`

```bash
fluxforge library-remove --source-id user:my_lines
```

### `activity`

- Purpose: Compute line activities from peak report
- Common use case: Convert a peak report into line-level activities for one isotope or reaction.
- Detailed help: `fluxforge activity --help`

```bash
fluxforge activity --peaks-file rafm4_b_peaks.json --live-time-s 3600 --output activities.json
```

### `activity-review`

- Purpose: Review all matched isotope activities for one spectrum and export EOI tables/plots
- Common use case: Review matched isotopes and gamma lines from one peak report and export line/isotope tables.
- Detailed help: `fluxforge activity-review --help`

```bash
fluxforge activity-review --peaks-file rafm4_b_peaks.json --output activity_review.json --line-csv-output activity_lines.csv --isotope-csv-output activity_isotopes.csv
```

### `inventory-review`

- Purpose: Propagate an activity-review inventory to arbitrary times and export time-series tables/plots
- Common use case: Propagate an activity-review inventory to arbitrary times for decay, atoms, mass, or dose review.
- Detailed help: `fluxforge inventory-review --help`

```bash
fluxforge inventory-review --activity-review-file activity_review.json --time-stop-s 86400 --plot-output inventory_plot.png --output inventory_review.json
```

### `rates`

- Purpose: Compute reaction rates from line activities
- Common use case: Convert line activities into reaction-rate estimates using irradiation history information.
- Detailed help: `fluxforge rates --help`

```bash
fluxforge rates --lines-file activity_lines.json --duration-s 3600 --output rates.json
```

## Planning and Optimization

Rank isotopes, review masking, optimize schedules, and package planning bundles.

### `second-irradiation-plan`

- Purpose: Rank second-irradiation candidates from an inventory seed and schedule definition
- Common use case: Rank second-irradiation candidates from an inventory seed, schedule, and candidate definition.
- Detailed help: `fluxforge second-irradiation-plan --help`

```bash
fluxforge second-irradiation-plan --inventory-file inventory_review.json --schedule-file schedule.json --candidates-file candidates.json --output second_irradiation_plan.json
```

### `ffexp-export`

- Purpose: Package activity, inventory, masking, optimization, and second-irradiation products into a benchmark .ffexp bundle
- Common use case: Package Phase 6 review artifacts into a portable `.ffexp` benchmark bundle.
- Detailed help: `fluxforge ffexp-export --help`

```bash
fluxforge ffexp-export --activity-review-file activity_review.json --inventory-review-file inventory_review.json --output benchmark_bundle.ffexp
```

### `isotope-priority`

- Purpose: Rank important isotopes from activity-review gamma-spectrum outputs
- Common use case: Rank isotopes of interest from an activity-review bundle before planning follow-on irradiations.
- Detailed help: `fluxforge isotope-priority --help`

```bash
fluxforge isotope-priority --activity-review-file activity_review.json --csv-output isotope_priority.csv --output isotope_priority.json
```

### `masking-review`

- Purpose: Rank line-level masking interactions and alternate-line guidance from activity-review outputs
- Common use case: Rank masking interactions and alternate-line guidance from activity-review outputs.
- Detailed help: `fluxforge masking-review --help`

```bash
fluxforge masking-review --activity-review-file activity_review.json --csv-output masking_lines.csv --output masking_review.json
```

### `optimization-sweep`

- Purpose: Rank schedule candidates using DI-FOM, FIM, MWDCS, or advanced N1/N2 objectives
- Common use case: Score candidate irradiation schedules using DI-FOM, FIM, MWDCS, or advanced objectives.
- Detailed help: `fluxforge optimization-sweep --help`

```bash
fluxforge optimization-sweep --activity-review-file activity_review.json --objective di-fom --output optimization_sweep.json
```

## Dosimetry and Standards

Run ASTM-style dosimetry workflows and browse governed dosimetry reactions.

### `astm-e2005`

- Purpose: Run the ASTM E2005 reactor dosimetry workflow
- Common use case: Run the ASTM E2005 reactor dosimetry workflow from a structured plan file.
- Detailed help: `fluxforge astm-e2005 --help`

```bash
fluxforge astm-e2005 --plan-file my_astm_e2005_plan.json --output astm_e2005.json
```

### `astm-e261`

- Purpose: Run the ASTM E261 reactor dosimetry workflow
- Common use case: Run the ASTM E261 reactor dosimetry workflow from the shipped plan format.
- Detailed help: `fluxforge astm-e261 --help`

```bash
fluxforge astm-e261 --plan-file examples/astm_e261_plan.json --output astm_e261.json
```

### `astm-e262`

- Purpose: Run the ASTM E262 thermal neutron fluence workflow
- Common use case: Run the ASTM E262 thermal neutron fluence workflow from a plan file.
- Detailed help: `fluxforge astm-e262 --help`

```bash
fluxforge astm-e262 --plan-file my_astm_e262_plan.json --output astm_e262.json
```

### `astm-e3376`

- Purpose: Run the ASTM E3376 high-purity germanium detection workflow
- Common use case: Run the ASTM E3376 HPGe detection workflow from a plan file.
- Detailed help: `fluxforge astm-e3376 --help`

```bash
fluxforge astm-e3376 --plan-file my_astm_e3376_plan.json --output astm_e3376.json
```

### `reactions`

- Purpose: Browse IRDFF-II dosimetry reactions
- Common use case: Browse bundled IRDFF-II dosimetry reactions by category or target nuclide.
- Detailed help: `fluxforge reactions --help`

```bash
fluxforge reactions --category thermal --format table
```

## Reference Workflows

Replay bundled RAFM validation, benchmark, and planning-oriented worked examples.

### `rafm-validate`

- Purpose: Run the committed RAFM raw-spectrum validation workflow against QG reference data
- Common use case: Replay the committed RAFM raw-spectrum validation workflow against bundled reference assets.
- Detailed help: `fluxforge rafm-validate --help`

```bash
fluxforge rafm-validate --example-root examples/RAFM_irradiation --results-root /tmp/rafm_validation --no-fail
```

### `rafm-qg-benchmark`

- Purpose: Process committed QG RAFM flux-wire data through reaction-rate and unfolding outputs
- Common use case: Process committed Quantum Gold RAFM data into reaction-rate and unfolding outputs.
- Detailed help: `fluxforge rafm-qg-benchmark --help`

```bash
fluxforge rafm-qg-benchmark --example-root examples/RAFM_irradiation --results-root /tmp/rafm_qg
```

### `rafm-compare-branches`

- Purpose: Compare completed raw-branch and QG-branch RAFM results
- Common use case: Compare completed raw-branch and Quantum Gold RAFM result trees.
- Detailed help: `fluxforge rafm-compare-branches --help`

```bash
fluxforge rafm-compare-branches --raw-results-root raw_results --qg-results-root qg_results --output-root branch_compare
```

### `phase6-ldrd-worked-example`

- Purpose: Run the Phase 6 worked example on RAFM LDRD-backed irradiation data
- Common use case: Generate the bundled Phase 6 worked example outputs for one committed RAFM sample.
- Detailed help: `fluxforge phase6-ldrd-worked-example --help`

```bash
fluxforge phase6-ldrd-worked-example --sample-id RAFM4-C_15dEOI --output-root /tmp/phase6_ldrd_worked_example
```

### `phase6-ldrd-second-irradiation-repo`

- Purpose: Build a reproducible RAFM LDRD second-irradiation decision repository bundle
- Common use case: Build the Phase 6 second-irradiation decision repository bundle for one RAFM sample.
- Detailed help: `fluxforge phase6-ldrd-second-irradiation-repo --help`

```bash
fluxforge phase6-ldrd-second-irradiation-repo --sample-id RAFM4-C_15dEOI --output-root /tmp/phase6_second_irradiation_repo
```

## Unfolding and Reporting

Build response matrices, unfold spectra, compare results, and generate plots or reports.

### `response`

- Purpose: Build response matrix from cross sections
- Common use case: Build a response matrix from cross sections, number densities, and group boundaries.
- Detailed help: `fluxforge response --help`

```bash
fluxforge response --cross-section-file cross_sections.json --number-densities-file number_densities.json --boundaries-file boundaries.json --output response.json
```

### `unfold`

- Purpose: Infer spectrum using GLS, GRAVEL, MLEM, MAXED, RMLE, or ML Seed
- Common use case: Infer a neutron spectrum using GLS, GRAVEL, MLEM, MAXED, RMLE, or ML-seeded workflows.
- Detailed help: `fluxforge unfold --help`

```bash
fluxforge unfold --rates-file rates.json --response-file response.json --method gravel --output spectrum.json
```

### `compare`

- Purpose: Compare unfolded spectrum with reference
- Common use case: Compare an unfolded spectrum against a trusted reference spectrum.
- Detailed help: `fluxforge compare --help`

```bash
fluxforge compare --unfold-file spectrum.json --truth-flux-file truth_flux.json --output validation.json
```

### `report`

- Purpose: Compile a report bundle from artifacts
- Common use case: Assemble a report bundle from one or more analysis artifacts.
- Detailed help: `fluxforge report --help`

```bash
fluxforge report --spectrum-file spectrum.json --peaks-file peaks.json --output report.json
```

## k0-NAA

Normalize observations, characterize detector and facility state, analyze, aggregate, and report.

### `k0-normalize`

- Purpose: Normalize a peak report into standards-oriented k0 peak observations
- Common use case: Convert a peak report into normalized k0 peak observations.
- Detailed help: `fluxforge k0-normalize --help`

```bash
fluxforge k0-normalize --peaks-file rafm4_b_peaks.json --output k0_observations.json
```

### `k0-detector`

- Purpose: Build a reusable detector-characterization artifact for k0 workflows
- Common use case: Fit a reusable detector-characterization artifact from calibration points.
- Detailed help: `fluxforge k0-detector --help`

```bash
fluxforge k0-detector --points-file detector_points.csv --detector-id hpge_demo --reference-position-mm 250 --output detector_characterization.json
```

### `k0-facility`

- Purpose: Characterize a thermal irradiation facility using a bare triple-monitor workflow
- Common use case: Characterize a thermal irradiation facility from a bare triple-monitor dataset.
- Detailed help: `fluxforge k0-facility --help`

```bash
fluxforge k0-facility --input facility_input.json --output facility_characterization.json
```

### `k0-analyze`

- Purpose: Run a first-pass k0 analysis from normalized observations and facility characterization
- Common use case: Run a first-pass k0 analysis from normalized observations and a characterized facility.
- Detailed help: `fluxforge k0-analyze --help`

```bash
fluxforge k0-analyze --observations-file k0_observations.json --facility-file facility_characterization.json --sample-mass-g 0.5 --output k0_analysis.json
```

### `k0-aggregate`

- Purpose: Aggregate k0 analysis bundles across measurements and irradiations
- Common use case: Aggregate multiple k0 analysis bundles across measurements or irradiations.
- Detailed help: `fluxforge k0-aggregate --help`

```bash
fluxforge k0-aggregate --analysis-files sample_a.json sample_b.json --output k0_aggregation.json
```

### `k0-qaqc`

- Purpose: Evaluate blank and CRM QA/QC from k0 analysis bundles
- Common use case: Evaluate blank and CRM QA/QC performance for a k0 analysis plan.
- Detailed help: `fluxforge k0-qaqc --help`

```bash
fluxforge k0-qaqc --plan-file k0_qaqc_plan.json --output k0_qaqc.json
```

### `k0-report`

- Purpose: Write a richer k0 report bundle with optional aggregation and QA/QC summaries
- Common use case: Build a richer k0 report bundle with optional aggregation and QA/QC context.
- Detailed help: `fluxforge k0-report --help`

```bash
fluxforge k0-report --analysis-file k0_analysis.json --output k0_report.json
```

### `k0-import-kayzero`

- Purpose: Import a user-supplied Kayzero library folder or zip into governed FluxForge k0 JSON
- Common use case: Import a user-supplied Kayzero folder or archive into governed FluxForge JSON.
- Detailed help: `fluxforge k0-import-kayzero --help`

```bash
fluxforge k0-import-kayzero --input /path/to/kayzero_library --output kayzero_k0_library.json
```

## GUI and Visualization

Launch the desktop GUI and generate headless plot bundles.

### `gui`

- Purpose: Launch FluxForge desktop GUI
- Common use case: Launch the desktop GUI for interactive spectrum review and ROI editing.
- Detailed help: `fluxforge gui --help`

```bash
fluxforge gui --project-dir .
```

### `plots`

- Purpose: Generate master-plan plots (headless; SSH-safe)
- Common use case: Generate headless plot bundles from analysis artifacts or bundled example inputs.
- Detailed help: `fluxforge plots --help`

```bash
fluxforge plots --example --output-dir output/plots
```
