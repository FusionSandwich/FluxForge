# FLUXFORGE Consolidated Master

**Status:** active feature master  
**Last Updated:** 2026-04-19  
**Purpose:** consolidated source of truth for product scope, feature commitments, and
roadmap-level capability planning.

This file supersedes the older top-level planning inputs that were merged into
`docs/archive/planning_snapshot_2026-04-06/`.

## 1. Source of Truth Order

Use the docs set in this order when there is overlap:

1. `docs/ROADMAP_EXECUTION_STATUS.md` for live implementation and sequence state
2. `docs/FLUXFORGE_CONSOLIDATED_MASTER.md` for feature scope and roadmap intent
3. `docs/GUI_PLAN.md` for GUI architecture, interaction rules, and workspace direction
4. `docs/FluxForge_Testing_Master.md` for testing, parity, and acceptance rules
5. `../testing/writeup.md` for repo-by-repo capability catalog, replay targets, and feature-to-file traceability
6. `docs/PHASE3_EXECUTION_HANDOFF.md` for implementation lifecycle, mandatory test/probe gates, and definition-of-done discipline
7. `docs/archive/planning_snapshot_2026-04-06/` for historical rationale and source-detail recovery

Conflict rules:

- Newer consolidated docs win over archived inputs.
- Additive delta documents add missing capabilities; they do not silently delete
  older valid requirements.
- The additive-capability rule remains in force: adding a method never justifies
  removing an existing valid one without an explicit correctness or safety reason.

## 2. Product Scope

FluxForge remains an offline-first native desktop toolkit for:

- HPGe gamma-spectroscopy analysis
- activity, reaction-rate, and standards-driven reduction workflows
- neutron dosimetry, response construction, and spectrum adjustment/unfolding
- reproducible experimental handoff bundles for downstream model-comparison tools

FluxForge owns the experimental side of the workflow. The separate comparison tool
continues to own modeled transport, modeled activation/photon sources, and final
experiment-vs-code comparison logic.

## 3. Non-Negotiable Rules

- Issue-first execution: significant work is tracked through GitHub issues and the ordered roadmap.
- Additive capability policy: new methods do not replace valid old ones by default.
- User-choice policy: scientifically defensible alternatives remain selectable and provenance-recorded.
- Standards-locked workflows: Standards mode constrains equations, defaults, and reports where required.
- Offline-first execution: supported workflows must not require network access.
- Provenance-first outputs: sessions, exports, and reports must record method, library, and parameter choices.
- Library-governance policy: optional downloadable libraries remain user-selected, external library locations must be registerable from both GUI and CLI, built-in FluxForge library IDs are reserved, and user-supplied libraries must never silently shadow or overwrite bundled libraries.
- Configuration-first policy: remove avoidable hardcoded workflows and values where practical, except where deliberate locking is required for QuantumGold or PeakEasy parity, governed standards workflows such as ASTM and k0-NAA, or RAFM irradiation-analysis paths.

## 4. Implemented Baseline

The repository is currently implemented through:

- Stage 0 governance and project-tracker scaffolding
- Phase 1 foundation in the modern Qt shell and shared package layout
- Phase 2 core analysis in the redesigned Qt path
- Phase 3.1 through 3.17, including unfolding, standards, reporting, batch, ROI parity,
  and calibration/efficiency parity
- the first landed Phase 6 irradiation-optimization slice, including deterministic
  masking/inventory/optimization/second-irradiation workflows, `.ffexp` bundle export,
  and saved GUI workflow/workspace presets for `quantumgold-workflow` and
  `astm-ldrd-irradiation`
- User-directed predictive items `4P.1` through `4P.7`

Live acquisition, HAL transport, and true MCA device surfaces remain deferred until
the offline-parity module closes.

## 5. Active Roadmap

| Step | Capability Family | Status | Required Deliverables |
|---|---|---|---|
| 3.18 | Identification / activity / reference parity | In Progress | Editable text/reference libraries; richer energy-window search; reference overlays; relative activity; source-age and decay-chain views; pinned mixtures; common-lab, natural, capture, inelastic, and delayed-activation library families; bundled GSA-v4 edited and natural gamma libraries; bundled NASA-gamma common-lab, natural-radiation, CapGam capture, IAEA capture, IAEA delayed-activation, Baghdad inelastic, and TALYS 14 MeV reaction-gamma libraries; bundled ICRP-107 decay-network dataset plus Kayzero 2020/2023 half-life-uncertainty overlays; bundled ENDF/B-VIII supplemental decay-line library; library install/register surfaces with opt-in downloads, GUI/CLI custom library-location entry, reserved built-in names and IDs, and collision-safe user aliases; arbitrary-time and EOI activity solving; GUI activity-unit selectors for activity-review and inventory plots/tables; `IrradiationSchedule`, `InventoryState`, `ObservableTimeSeries`, `LineMaskingResult`, and `OptimizationScenario` core models; GUI/CLI spectrum activation review; isotope-at-irradiation-time CSV export; uncertainty-bearing half-life decay plots and Bateman parent/daughter review plots; direct/comparator/relative-comparator/k0/activation-only workflows; experimental inventory object; isotopics-by-peaks and isotopics-by-nuclides workspaces |
| 3.19 | Operational calculator parity | Planned | Detection-limit workspaces; dose and attenuation calculators; shielding/source-fit tools; units conversion; FISPACT-style activity/heat/dose observables; dominant-contributor curves; irradiation/cooldown/count optimization; masking-isotope ranking; alternate-line and de-masking planning; second-irradiation planning; shutdown-through-100-year dose studies; in-GUI math/command terminal |
| 3.20 | Archive / batch / k0 parity | Planned | File-query and archive workbench; compact role-aware file management; session auto-store/resume; exemplar-driven batch analysis; detector-response lifecycle manager; multi-spectrum ROI/detector-consistency review; KayWin-style detector/facility characterization and governed k0 reporting; benchmark experimental bundle (`.ffexp`) export; library cache/install management and provenance-backed external-library registration |
| 3.21 | Repo-backed parity fixtures | Planned | Curated manifests under `tests/spectra/reference_parity/` plus `tests/activation_inventory/` fixtures with source repo, workflow, golden outputs, tolerances, provenance notes, and plot smoke inputs |
| 3.22 | Algorithm-level parity tests | Planned | Parser, calibration, peak search, fit, activity, detector-response, detection-limit, dose/shielding, masking, optimization, activation-inventory, and k0 parity checks against curated fixtures |
| 3.23 | Workflow-level parity tests | Planned | End-to-end parity suites for source families, tutorial/example datasets, NAA/activation inventory workflows, second-irradiation planners, and benchmark export bundles |
| 3.24 | Direct-manipulation canvas parity | Planned | Right-click peak actions, ROI/background drag handles, plot-driven edits, explicit foreground/background/secondary actions, overlap Gaussian insertion, and peak-label toggles |
| 3.25 | New parity workspaces in the Qt shell | Planned | ROI Statistics, Detection Limit, Relative Activity, Dose/Shielding, File Query/Batch Compare, Reference/Library Workbench, detector-response tools, and k0 characterization/report views |
| 3.26 | GUI polish and theme parity | Implemented (repo) | Shipped dark mode, saved theme profiles, stronger graph-table synchronization, clearer launch points, and role-aware overlay polish |
| 3.27 | GUI verification and release acceptance | Implemented (repo) | Qt workflow tests, native probes, artifact-gallery review states, and release-blocking checklists for new parity surfaces |
| 4.1 | HAL drivers | Deferred | First real MCA hardware drivers after offline-parity closure |
| 4.2 | Device discovery surfaces | Deferred | Device list, thumbnails, and discovery dialogs after HAL transport exists |
| 4.3 | Live Digital Twin dashboard | Deferred | Telemetry-driven dashboard once live acquisition is present |
| 4.4 | Live spectrogram panel | Deferred | Time-energy spectrogram after live acquisition lands |
| 5.1 | testing/ catalog crosswalk closure | In Progress (repo) | Build and maintain a complete crosswalk from every `testing/writeup.md` repo section to FluxForge backend, CLI, modern Qt GUI, fixtures, parity tests, and probe evidence; include exact source script/data paths and replay classification (`replay-now`, `adapter-required`, `reference-only`). Initial implementation baseline now includes `.github/project-management/phase5_crosswalk.json`, backend report helpers in `src/fluxforge/validation/phase5_crosswalk.py`, CLI command `phase5-crosswalk-report`, and a modern-shell `Phase 5 Parity` tab in `src/fluxforge/gui/panels/phase5.py`. |
| 5.2 | Spectrum IO and analysis parity closure | Planned | Close parser/calibration/background/peak-search/fit parity against `actigamma`, `becquerel`, `curie`, `gamma_spec_analysis`, `NASA-gamma`, `GSA-v2`, `GSA-v4`, `InterSpec`, `PyGammaSpec`, `peakingduck`, and `py-findpeaks`, with source-linked fixtures and algorithm/workflow golden checks |
| 5.3 | GUI and workflow parity closure | Planned | Close direct-manipulation and analyst-workflow parity (role-aware overlays, marker editing, ROI/statistics, detection-limit, shielding/source-fit, archive/file-query, multi-spectrum diagnostics, saved context, and report/export parity) using `InterSpec`, `GSA-v4`, `Gamma-MCA`, `hdtv`, `SpecKit`, and `NASA-gamma` behavior baselines |
| 5.4 | Inventory, NAA, and activation parity closure | Planned | Close activity/inventory/time-evolution/k0/INAA/activation parity against `irrad_spectroscopy`, `radioactivedecay`, `activation`, `KayWinV410`, `INAA-INRIM 3.1`, `NAA-ANN-1`, and `npat`; include uncertainty-bearing exports and provenance-complete bundle outputs |
| 5.5 | Inverse and covariance parity closure | Planned | Close unfolding and covariance-aware inverse-analysis parity against `Neutron-Unfolding`, `Neutron-Spectrometry`, `pyunfold`, `gmapy`, and `SpecKit`; include algorithm-level and workflow-level parity fixtures with explicit tolerances and divergence rationale |
| 5.6 | Phase 5 acceptance and release gate | Planned | Require backend+CLI+GUI completion, source-linked fixture manifests, targeted and full-suite test pass, native Qt probes, browser-lane artifact review, manual GUI sizing check, and synchronized status docs before any Phase 5 step is marked complete |

The sequence map above remains the controlling roadmap map. The activation /
FISPACT-style, library-governance, and de-hardcoding additions below are
explicit additive overlays on that map, not replacements for it.

Phase 5 is an additive closure phase for the audited `testing/` capability
catalog. It does not remove or downgrade existing FluxForge-native capability
families already committed in Phases 1 through 4 and the additive `3N` / Phase 6
workstream.

## 6. Feature Workstreams That Must Be Preserved

These workstreams came from the merged planning inputs and remain mandatory even when
they span multiple roadmap steps.

### 6.1 Experimental Analysis and NAA

- Support the full quantity ladder from spectrum to peak areas, net count rates,
  line activity, nuclide activity, arbitrary-time activity, production estimate,
  mass estimate, and concentration result.
- Keep direct activity, comparator NAA, relative comparator, k0-NAA, and
  activation-only workflows as explicit user-selectable modes.
- Maintain a first-class experimental inventory object containing activities,
  uncertainties, timestamps, daughter links, inferred masses/concentrations,
  sample metadata, irradiation metadata, geometry assumptions, and workflow mode.

### 6.2 Time-State, Decay, and Daughter Analysis

- Provide arbitrary-time activity solving at count start, midpoint, end, EOI, and future times.
- Support back-decay and forward-decay calculations with optional parent-daughter handling.
- Add daughter/ancestor exploration, line-emergence forecasting, and time-history visualization.
- Preserve spectrum-level activation review in both GUI and CLI, including isotope CSV export at EOI/irradiation time plus uncertainty-bearing decay and Bateman review plots.
- Keep the bundled `radioactivedecay` ICRP-107 decay network and the Kayzero `uT12` half-life-uncertainty overlays as first-class provenance-tracked sources for future source-age, daughter, and inventory workflows.
- Standardize time-evolution work around explicit `IrradiationSchedule`, `InventoryState`, and `ObservableTimeSeries` analysis objects rather than ad hoc per-workflow payloads.

### 6.3 Planning, Optimization, and Interference Analysis

- Add irradiation, cooldown, and count-time optimization against uncertainty, MDA,
  dead-time, and interference objectives.
- Add masking-isotope ranking, overlap provenance, alternate-line recommendations,
  and time-dependent de-masking guidance.
- Add staged and second-irradiation scenario planning, objective-based schedule comparison,
  and recommended schedule summaries with dose, detectability, and masking tradeoffs.
- Support FISPACT-style observable reviews for total activity, decay heat, gamma dose rate,
  beta heat, gamma heat, ingestion metric, inhalation metric, clearance index, and
  dominant-contributor timelines where source data support them.

### 6.4 InterSpec and GSA Capability Families

- Preserve all four registered efficiency-model families as first-class options.
- Keep detector slots, preserve/fine-tune calibration flows, and detector-response lifecycle management.
- Add file query, archive search, compact role-aware file staging, exemplar-driven batch reuse,
  dedicated detection-limit workspaces, relative-activity/isotopics workflows, and dose/shielding tools.
- Keep manual-first analyst control available after automation.
- Preserve the bundled GSA-v4 edited gamma library, NASA-gamma common-lab/natural/capture libraries, and the ENDF/B-VIII decay-line supplement as selectable provenance-bearing reference sources rather than collapsing back to one hardcoded line table.
- Let users choose which optional reference libraries are downloaded locally, and let them register external library locations from either the GUI or CLI without losing provenance or bundled defaults.
- Protect bundled library names and IDs from accidental user override; user-supplied libraries must be assigned distinct aliases when a collision would occur.

### 6.5 STAYSL-Grade Experimental Inverse Analysis

- Add corrected activation-rate workflows with explicit correction breakdowns.
- Provide a SigPhi-style workspace for corrected reaction observables.
- Keep generalized least-squares spectral adjustment as a governed, covariance-aware workflow.
- Plan for covariance outputs, run-family variants, reaction/covariance-library management,
  SHIELD-style self-shielding, BCF/flux-history correction handling, and monitor diagnostics.

### 6.6 Activation-Inventory and Benchmark Export Extension

- Adopt the additive `3N` workstream from the activation / FISPACT-style supplement as a preserved cross-cutting scope across `3.18` through `3.23`.
- Preserve the planned `Inventory / Time Evolution`, masking/interference, optimization, and long-term dose-study surfaces together with their machine-readable outputs.
- Add a benchmark experimental bundle export contract centered on `.ffexp` packaging for downstream comparison-tool ingestion.
- Repo status note: the first deterministic landing now exists in
  `src/fluxforge/workflows/irradiation_optimization.py`,
  `src/fluxforge/io/artifacts.py`, and `src/fluxforge/gui/panels/phase6.py`,
  with workflow-preset persistence in `src/fluxforge/gui/workflow_presets.py`
  and built-in `quantumgold-workflow` / `astm-ldrd-irradiation` presets.

### 6.7 De-Hardcoding and Configuration Cleanup

- Audit workflows and defaults for hardcoded values, special-case branches, and hidden source assumptions, then move removable cases behind registries, governed presets, or explicit user configuration.
- Keep explicit carve-outs only where deliberate hardcoding is part of required parity or governance: QuantumGold parity, PeakEasy parity, standards workflows including ASTM and k0-NAA, and RAFM irradiation-analysis paths.

## 7. Capabilities That Must Remain First-Class

- Simple, Expert, and Standards GUI modes
- Manual, Bayesian, and ML-assisted identification workflows
- Registry-backed peak-search families rather than one hardcoded algorithm
- GRAVEL, MAXED, RMLE, and ML-seed unfolding options
- Multiple background and counting methods where scientifically valid
- All registered efficiency models and calibration workflows
- Foreground, background, and secondary spectrum roles with explicit provenance
- Bundled, downloadable, and user-supplied nuclear-data libraries with explicit selection,
  provenance capture, collision-safe naming, and GUI/CLI parity

## 8. Experimental Handoff Contract

FluxForge must emit machine-readable handoff bundles that can include:

- sample, detector, and geometry metadata
- calibration and detector-response versions
- fitted peaks, line activities, nuclide activities, and EOI activities
- corrected activation rates
- inferred masses and concentrations
- uncertainty decomposition
- daughter-chain assumptions
- selected NAA and inverse-analysis modes
- nuclear-data provenance
- masking/interference outputs
- inventory timeseries, dominant-contributor tables, and chosen schedule definitions
- dose endpoints, optimization grids, recommended schedules, and staged-irradiation comparisons
- planned-vs-actual irradiation/cool/count schedules
- `.ffexp` benchmark experimental bundles and plot-manifest sidecars when the workflow requires them
- adjusted flux spectra and covariance when available
- readiness flags such as `peak_fit_complete`, `activity_complete`,
  `naa_quant_complete`, `inverse_adjustment_complete`, and `comparison_ready`

## 9. Archived Planning Inputs

The following superseded inputs were preserved in
`docs/archive/planning_snapshot_2026-04-06/`:

- `FLUXFORGE_CONSOLIDATED_MASTER.md`
- `FluxForge_Final_Additions.md`
- `FluxForge_Additions_v3_Final.md`
- `FluxForge_NAA_InterSpec_STAYSL_Addition.md`
- `GUI_PLAN.md`
- `GUI_PLAN_old.md`
- `GUI_CAPABILITY_PROGRAM.md`
- `GUI_CAPABILITY_PROGRAM_old.md`

## 10. Phase 5 Capability Closure Program (`testing/writeup.md` alignment)

### 10.1 Mission

Phase 5 is the explicit convergence phase that ensures FluxForge planning
contains all capability families audited in `../testing/writeup.md`, while
preserving FluxForge-specific capability additions already landed or planned
(standards-lock workflows, predictive surfaces, benchmark `.ffexp` handoff,
and additive Phase 6 optimization/inventory tooling).

Phase 5 outputs must be implementation-ready. Every capability entry must say:

- where the behavior is described in `../testing/writeup.md`
- which local source scripts, tests, notebooks, or sample data define replay
  expectations
- where FluxForge backend/CLI/GUI work lands
- which tests/probes/documentation updates close the slice

### 10.2 Mandatory Execution Method (adopted from `docs/PHASE3_EXECUTION_HANDOFF.md`)

For every Phase 5 capability, execute in this order:

1. Reference extraction from `../testing/writeup.md` and source paths.
2. Native adaptation in FluxForge backend modules.
3. Mathematical and physics verification via pytest fixtures/tolerances.
4. CLI integration with command-level tests.
5. Modern Qt integration in `src/fluxforge/gui/`.
6. GUI automation plus native probe artifact generation and browser-lane review.
7. Manual modern GUI launch and sizing/usability validation.
8. Status-document updates with evidence and residual-risk notes.

Per-slice exit gate remains strict: backend + CLI + GUI + tests + probe
evidence + manual GUI check + doc sync are all required before completion.

### 10.3 Full Writeup Crosswalk (all audited `testing/` repos)

The following map is the minimum traceability baseline for Phase 5 issue
seeding and implementation planning.

| writeup section | Capability families to preserve in FluxForge | Primary local replay evidence to cite in implementation work |
|---|---|---|
| `writeup.md` section 1 `actigamma` | deterministic line-engine generation, mixed-radiation binning, nuclide-ID heuristics, database query parity | `testing/actigamma/examples/plotlines.py`, `plotmultilines.py`, `identify.py`, `getenergies.py`, `classifier.py` |
| `writeup.md` section 2 `becquerel` | multi-format parser normalization, calibration/auto-calibration, fit-model parity, rebinning parity | `testing/becquerel/tests/samples/`, `tests/samples/INVENTORY.md`, `examples/fitting.ipynb`, `tests/fitting_test.py`, `examples/rebinning.ipynb`, `tests/rebin_test.py` |
| `writeup.md` section 3 `curie` | spectroscopy + efficiency, stacked-target workflows, attenuation/material workflows, reaction/decay API parity | `testing/curie/examples/spectroscopy_examples.py`, `eu_calib_7cm.Spe`, `stack_examples.py`, `test_stack.csv`, `reaction_examples.py`, `isotope_decay_examples.py` |
| `writeup.md` section 4 `gamma_spec_analysis` | ORTEC `.spe` parsing, smoothing and peak search, trapezoid background, Gaussian fit parity | `testing/gamma_spec_analysis/test_data/`, `test_spec_analysis.ipynb`, `gs_analysis.py` |
| `writeup.md` section 5 `Gamma-MCA` | role-aware import UX, chronological stream histogramization, calibration workflows, Gaussian-correlation peak finder references, isotope lookup UX, acquisition-oriented interaction model | `testing/Gamma-MCA/README.md`, `source/` workflow files and import/calibration descriptions |
| `writeup.md` section 6 `gmapy` | covariance-aware evaluation workflows, legacy-data ingest, simplified-vs-full equivalence and controlled-divergence methodology | `testing/gmapy/tests/testdata/`, `tests/test_gmap_simplified.py`, `tests/test_legacy_divergence.py`, `examples/example-005-compute-sacs-results.ipynb`, `legacy-tests/` |
| `writeup.md` section 7 `hdtv` | marker-centric fit workflows, fit XML import/export, calibrated-bin conversion, integral workflows, matrix cut/projection interaction model | `testing/hdtv/tests/share/osiris_bg.spc`, `tests/share/binning.root`, `tests/plugins/test_calbin.py`, `tests/integral/test.asc`, `tests/mat/mat.prx`, `tests/mat/cut.spc` |
| `writeup.md` section 8 `irrad_spectroscopy` | energy and efficiency calibration from measured standards, activity reconstruction, dose and fluence helper workflows | `testing/irrad_spectroscopy/tests/test_data/Eu152_point.txt`, `152Eu_peaks.yaml`, `152Eu_point_source_2.yaml`, `example_sample.txt`, `test_spectroscopy.py` |
| `writeup.md` section 9 `NAA-ANN-1` | ANN-assisted NAA preprocessing and result-surface parity as optional plugin path | `testing/NAA-ANN-1/RID_extracted/`, `NAA2 data augmentation output 4e.zip`, `NAA1 2022-05-09 4e results.csv` |
| `writeup.md` section 10 `Neutron-Spectrometry` | config-driven unfolding, objective/solver comparators, trend-analysis and dose/fluence report workflows | `testing/Neutron-Spectrometry/unfolding/input/template_measurements.txt`, `template_unfold_spectrum.cfg`, `template_unfold_trend.cfg`, `instructions_plot_spectra.md` |
| `writeup.md` section 11 `Neutron-Unfolding` | compact GRAVEL/MLEM parity workflows with committed result artifacts | `testing/Neutron-Unfolding/unfolding_inputs/reduced_data.csv`, `response-matrix.txt`, `energy-spectrum.txt`, `final-results/`, `gravel-results/` |
| `writeup.md` section 12 `npat` | spectroscopy + activation + decay-chain + stacked-target multi-domain integration, listfile parsing | `testing/npat/examples/eu_calib_7cm.Spe`, `mvmelst_007.zip`, `test_stack.csv`, `dbmgr.py`, `Reaction`/`Isotope`/`DecayChain` APIs |
| `writeup.md` section 13 `peakingduck` | SNIP/background estimation, process-chain peak candidate generation, pluggable low-level peak-processing architecture | `testing/peakingduck/reference/spectrum0.csv`, `examples/py/snip.py`, `examples/py/realspectrum.py`, `include/io/spectralio.hpp` |
| `writeup.md` section 14 `py-findpeaks` | algorithm-comparison micro-benchmarks for peak and valley index detection | `testing/py-findpeaks/tests/vector.py`, `scipy_signal_find_peaks.py`, `peakutils_indexes.py`, `lows_and_highs.py` |
| `writeup.md` section 15 `PyGammaSpec` | low-ceremony spectrum arithmetic, polynomial calibration, bounded Gaussian fitting, gamma-line and daughter overlay lookup | `testing/PyGammaSpec/docs/utils/calibration.txt`, `background.txt`, `weak_radium.txt`, `src/pygammaspec/data/gamma_data.csv` |
| `writeup.md` section 16 `pyunfold` | covariance-aware iterative unfolding with prior sensitivity and regularization paths | `testing/pyunfold/pyunfold/tests/test_data/example1_python3.hdf`, `docs/source/notebooks/user_prior.ipynb`, `regularization.ipynb`, `tests/test_teststat.py` |
| `writeup.md` section 17 `radioactivedecay` | analytic inventory decay, daughter/progeny logic, inventory file I/O parity, nuclide parser normalization | `testing/radioactivedecay/radioactivedecay/icrp107_ame2020_nubase2020/`, `tests/test_inventory.py`, `tests/test_fileio.py`, `tests/test_nuclide.py` |
| `writeup.md` section 18 `SpecKit` | response-matrix construction, neutron-spectrum solving, uncertainty/error-band and benchmark comparison workflows | `testing/SpecKit/Example/Au-197_Au-198.txt`, `Example/prior.txt`, `benchmark/double_peak/`, `benchmark/quasi_single_peak/` |
| `writeup.md` section 19 `GSA-v4` | mature desktop workflow behavior, detector calibration/efficiency state handling, line-library workflows, multi-spectrum ROI statistics | `testing/GSA-v4/Spectrum/`, `Calibration/Detector1.txt`, `Calibration/Efficiency1.txt`, `Lib/libEdit.dat`, `Lib/Lib-gamma-natur.dat`, `ROI_Statistic/` |
| `writeup.md` section 20 `InterSpec` | role-aware N42 ingest/overlay, activity/detection-limit/shielding calculators, file-query/archive workbench behavior, persisted analyst context and themes | `testing/InterSpec/example_spectra/*.n42`, `target/testing/test_data/SimpleActivityCalc/`, `target/testing/test_data/det_eff/`, `target/testing/analysis_tests/` |
| `writeup.md` section 21 `KayWinV410` | detector/facility characterization, k0-NAA workspace structure, short/long irradiation campaign organization, results packaging | `testing/KayWinV410/KayWinV4/Calibration CA6C final/`, `PTIC40/`, `FAST/`, `LONG/`, `order/results/PT2020.RES` |
| `writeup.md` section 22 `NASA-gamma` | parser breadth (`MCA/CNF/CSV/SPE/TXT`), calibration and advanced peak-fit helpers, repeated-run diagnostics, extended capture/reaction reference workflows | `testing/NASA-gamma/examples/data/`, `gui_test_data_cebr_cal.csv`, `gui_test_data_hpge_Cu.Spe`, `test_folder_diag/RUN*.Spe` |
| `writeup.md` section 23 `prospect_trial_installation` | commercial-workstation workflow shape for ROI, calibration, overlay, preferences, reporting, and live-session assumptions | `testing/prospect_trial_installation/ProSpect User's Manual.pdf`, `readme first.txt` |
| `writeup.md` section 24 `GSA-v2` | source-visible GSA algorithms for parser parity, derivative peak search, overlap area extraction, identification/activity workflows | `testing/GSA-v2/GSA.v2/example/`, `src/mariscoti.java`, `src/PeakSearch.java`, `src/treatment1.java`, `Lib/Lib.dat`, `Detectors/*.txt` |
| `writeup.md` section 25 `activation` | request-schema and activation scenario design, resonance/cross-section helpers, web-form input models | `testing/activation/README.md`, `cgi-bin/nact.py`, `endf/isotopes_ENDF-B-VIII.1.txt`, `endf/endf.py`, `activation/index_template.html` |
| `writeup.md` section 26 `gammaspectroscopy` | large measured-spectrum replay corpus, streamlined process-and-identify pipelines, notebook batch workflows | `testing/gammaspectroscopy/Daten.zip`, `databasegamma.txt`, `databasegamma.parquet`, `notebooks/` |
| `writeup.md` section 27 `INAA-INRIM 3.1` | k0 data/workspace model parity and correction-chain coverage (decay/efficiency/blank/mass/fission) | `testing/INAA-INRIM 3.1/data/k0data/from_k0data.k0d`, `data/nuclear_data/nndc_nudat_data_export.nds`, `data/eqs/*.png` |

### 10.4 Capability-Bundle Delivery Requirements for Phase 5

Each implementation slice should map one or more crosswalk rows above into a
single FluxForge capability bundle with all required surfaces:

- Backend: native implementation under `src/fluxforge/` (no runtime dependency
  on `../testing` code).
- CLI: command and output support in `src/fluxforge/cli/app.py`.
- GUI: modern Qt workspace in `src/fluxforge/gui/` and related panels/dialogs.
- Data provenance: explicit source library/version/selection captured in
  artifacts and reports.
- Testing: algorithm-level and workflow-level parity tests, plus fixture
  manifests with tolerances and provenance notes.
- GUI evidence: pytest-qt interaction tests, native probe screenshots/artifacts,
  browser-lane artifact review, and manual GUI sizing validation.

The writeup-driven classification rule also applies throughout Phase 5:

- decay/activity libraries stay in `peak-identification` and activity workflows
- prompt capture / reaction-gamma datasets remain separate capability buckets
  (`activation-reference`, `capture-gamma`, `reaction-gamma`,
  `delayed-activation`)

### 10.5 Testing-Data and Traceability Contract (required in every Phase 5 PR)

For every added fixture or replay path, document all of the following:

- source repo and writeup section (for example, `writeup.md` section 20 `InterSpec`)
- exact local source path(s) used for replay input and expected output
- data-class label: `bundled locally`, `downloaded dynamically`,
  `generated during runtime`, or `docs-only / implied`
- replay status target: `replay-now`, `adapter-required`, or `reference-only`
- expected output contract (tables, files, figures, tolerances)
- FluxForge landing paths (backend module, CLI command, GUI panel, tests)

This contract is mandatory for issue descriptions, implementation PRs,
parity-fixture manifests, and status-doc updates.

### 10.6 Phase 5 Definition of Done

A Phase 5 capability is complete only when all conditions are true:

- The targeted writeup capability is implemented in native FluxForge backend,
  CLI, and modern Qt GUI surfaces.
- Source-linked replay fixtures are added with provenance and tolerances.
- Algorithm-level and workflow-level tests pass for the added capability.
- Native GUI probe artifacts and browser-lane review evidence are generated.
- Manual modern GUI launch confirms sizing/usability for the new surface.
- `docs/ROADMAP_EXECUTION_STATUS.md`, this master plan, and related testing/GUI
  docs are updated with evidence and residual risks.

If any requirement above is missing, the Phase 5 slice remains in progress.

### 10.7 Evidence and Provenance Rules (from `testing/writeup.md` methodology)

Phase 5 implementation and review work must apply the writeup evidence rules
explicitly, not implicitly.

- Every claim about tests, examples, tutorials, notebooks, demos, or sample
  workflows must cite at least one concrete local path when such a path exists.
- Every dataset reference must carry one data-class label:
  `bundled locally`, `downloaded dynamically`, `generated during runtime`, or
  `docs-only / implied`.
- Every implementation claim must carry one provenance label when needed:
  `source-verified`, `binary/workspace-derived`, or `docs-derived`.
- Every replay target must state input artifact, processing step, and expected
  output/result to compare.
- Missing local assets are never implied by omission; absence must be stated
  explicitly in parity notes and manifests.
- For compiled or packaged reference apps, workspace-file evidence and
  installer/manual-only claims must be reported separately.

### 10.8 Code-Level Traceability Packet (required per Phase 5 capability)

Every Phase 5 issue, implementation PR, and parity-fixture manifest must
include this traceability packet:

1. Parser/importer implementation file(s) for the capability.
2. Primary numerical or physics algorithm file(s).
3. User-facing workflow entry file(s): CLI command path and modern Qt
   panel/dialog path.
4. Validation-asset path(s) used for checks.
5. Sample input path(s) and expected output artifact contract.
6. Replay status (`replay-now`, `adapter-required`, or `reference-only`).
7. If code is split across multiple files, one primary file plus supporting
   files listed in order.
8. If implementation is docs-only for a source capability, the claim must be
   marked `docs-derived` until source-verified evidence is added.

### 10.9 GUI Behavior-Extraction Checklist (required for GUI-focused sources)

When preserving behaviors from GUI-oriented sources (`InterSpec`, `GSA-v4`,
`Gamma-MCA`, `hdtv`, `SpecKit`, `NASA-gamma`, and related references), parity
planning must explicitly record:

- Whether the plot is a primary input surface or only a passive display.
- Graph/table synchronization behavior, including click-table-to-zoom and
  plot-edit-to-update-table loops.
- Foreground/background/secondary spectrum-role preservation across load,
  legend, drag-drop, and saved-state workflows.
- Whether automated workflows are manually overridable by analysts.
- Whether saved themes, saved workspaces, or saved workflow context are
  first-class and reproducible.
- Whether specialized workspaces exist for ROI statistics, detection limit,
  relative activity, shielding/source-fit, and file-query/archive review.

If a source lacks one or more behaviors above, that absence must be stated
explicitly before marking parity scope complete.

### 10.10 Writeup Addendum Coverage Deltas (section 0 alignment)

The writeup addendum includes high-value capability families that are not all
expected to land at once. Phase 5 must keep these visible as explicit planning
targets with provenance-aware staging:

- `testing/InterSpec/data/sandia.reactiongamma.xml` compatibility bridge,
  staged under `reaction-gamma` / `activation-reference` capability buckets.
- ENSDF archival ingestion path for richer daughter/cascade traceability.
- IAEA LiveChart cache/sync layer for optional provenance-tracked updates.
- DDEP/LNHB recommended-decay overlay support.
- IAEA X-ray and gamma-ray standards subset support.
- SandiaDecay-compatible XML import/export path.
- Coincidence/cascade JSON derivations from ENSDF-style sources.

Classification guardrail remains mandatory: prompt capture or reaction-gamma
references are not interchangeable with decay emission-probability libraries
used for activity calculations.

### 10.11 Replay-State Matrix for Audited `testing/` Sources

The table below records the current planning-state classification for each
audited source family based on `testing/writeup.md` replay guidance.

| Source family | Initial replay-state target | Notes for Phase 5 planning |
|---|---|---|
| `actigamma` | `replay-now` | Deterministic synthetic line-engine parity and inventory-to-spectrum checks. |
| `becquerel` | `replay-now` | Strong parser/calibration/fitting parity with bundled sample assets. |
| `curie` | `replay-now` + `adapter-required` | Local spectroscopy/stack workflows are replayable; downloaded DB paths need adapter staging. |
| `gamma_spec_analysis` | `replay-now` | Local `.spe` parser/smoothing/peak/fit parity target. |
| `Gamma-MCA` | `adapter-required` | UX and workflow parity high value; fixture construction needed for full numerical replay. |
| `gmapy` | `replay-now` | Covariance-aware methodology and legacy-divergence checks are locally replayable. |
| `hdtv` | `replay-now` | Strong local fit/cut/XML/matrix fixtures; environment/runtime constraints apply. |
| `irrad_spectroscopy` | `replay-now` | Local calibration/activity/dose examples and tests are available. |
| `NAA-ANN-1` | `adapter-required` | Real corpus exists, but zip/path-sensitive ANN preprocessing must be reconstructed first. |
| `Neutron-Spectrometry` | `replay-now` | Config-driven unfolding and trend workflows are replayable with local templates/inputs. |
| `Neutron-Unfolding` | `replay-now` | Local compact GRAVEL/MLEM datasets and committed outputs. |
| `npat` | `replay-now` + `adapter-required` | Local spectroscopy/listfile/stack inputs replay now; downloaded DB-backed workflows need staging. |
| `peakingduck` | `replay-now` | Local SNIP/background/process-chain replay assets available. |
| `py-findpeaks` | `replay-now` | Vector-level algorithm benchmark parity. |
| `PyGammaSpec` | `replay-now` | Local calibration/background/fit/line-lookup tutorial assets. |
| `pyunfold` | `replay-now` | Local HDF/notebook/test-stat assets for unfolding and uncertainty parity. |
| `radioactivedecay` | `replay-now` | Local decay-network packaging and inventory regression surfaces. |
| `SpecKit` | `replay-now` | Local response/benchmark datasets and uncertainty viewer workflows. |
| `GSA-v4` | `replay-now` | Local spectra/calibration/library/ROI-stat artifacts support parity comparisons. |
| `InterSpec` | `replay-now` | Rich local N42/activity/det-eff/analysis test assets. |
| `KayWinV410` | `adapter-required` | Strong data-model/workspace reference; executable/code-level parity needs adapters. |
| `NASA-gamma` | `replay-now` | High-priority local parser/calibration/fit/diagnostics assets. |
| `prospect_trial_installation` | `reference-only` | Workflow-design reference; limited local numerical replay corpus. |
| `GSA-v2` | `replay-now` | Source-visible algorithm parity with local example corpus. |
| `activation` | `adapter-required` | Request-schema/API parity useful; core physics coupling is external. |
| `gammaspectroscopy` | `replay-now` | Large local measured-spectrum corpus and notebook workflows. |
| `INAA-INRIM 3.1` | `adapter-required` | Strong k0 data/workspace reference; code-level replay needs importer/model adapters. |

Status rule: this table tracks planning-state targets, not implementation
completion. Completion state is tracked in roadmap and parity fixture evidence.

### 10.12 Current Phase 5.1 Baseline (2026-04-19)

The first executable Phase 5 slice is now landed in-repo and should be treated
as the baseline for `5.1` closure work:

- Machine-readable crosswalk tracker: `.github/project-management/phase5_crosswalk.json`
  with all 27 audited writeup source families and explicit replay-state labels.
- Backend crosswalk loading/validation/reporting utilities:
  `src/fluxforge/validation/phase5_crosswalk.py`.
- CLI reporting surface:
  `src/fluxforge/cli/app.py` command `phase5-crosswalk-report` supporting JSON,
  markdown, and optional parity-summary integration.
- Modern Qt review surface:
  `src/fluxforge/gui/panels/phase5.py` `Phase5ParityPanel` wired into
  `src/fluxforge/gui/panels/modern_shell.py` for crosswalk inspection and
  parity-suite execution.
- Verification and evidence assets:
  `tests/test_phase5_crosswalk.py`, `tests/test_cli_app.py`,
  `tests/test_modern_gui_shell.py`, `tests/gui_phase5_parity_probe.py`, and
  `artifacts/gui_review/phase5_parity/` including Playwright audit reports.

## 11. Online-Informed Future Feature Candidates (Do Not Implement Yet)

This section captures future feature ideas from online references relevant to
FluxForge goals and planned superconducting-material irradiation campaigns.
These are planning candidates only.

Reference pages reviewed for this update:

- IAEA LiveChart and API entry points:
  `https://www-nds.iaea.org/relnsd/vcharthtml/VChartHTML.html`
- NNDC NuDat 3 data/navigation surfaces:
  `https://www.nndc.bnl.gov/nudat3/`
- OpenMC depletion/transmutation user guidance:
  `https://docs.openmc.org/en/stable/usersguide/depletion.html`
- UKAEA FISPACT-II capability overview:
  `https://www.ukaea.org/service/fispact/`
- ITER machine references for magnet/blanket/divertor constraints:
  `https://www.iter.org/mach/Magnets`
  `https://www.iter.org/mach/Blanket`
  `https://www.iter.org/mach/Divertor`

### 11.1 Nuclear Data and Provenance Expansion

- Add a multi-source nuclear-data resolver that can blend IAEA LiveChart,
  NuDat, and local bundled libraries with ranked provenance and explicit
  conflict reporting.
- Add offline snapshot/cache packs for remote datasets so analysis remains
  reproducible in air-gapped labs.
- Add service-health and deprecation tracking in provenance metadata
  (important when upstream web services are retired or changed).

### 11.2 Activation and Depletion Workflow Expansion

- Add explicit source-rate vs power-normalization controls in depletion-like
  workflows, with warnings about model assumptions and normalization caveats.
- Add local-spectrum handling for repeated materials so irradiation estimates do
  not collapse distinct local spectra into one averaged state.
- Add material transfer-rate modeling for feed/removal scenarios in long
  irradiation campaigns (including units, sign conventions, and audit trails).
- Add optional transport-independent mode hooks for pre-tabulated microscopic
  cross sections and external flux inputs.

### 11.3 Radiation-Damage and Fusion-Materials Metrics

- Add first-class radiation-damage outputs aligned with fusion materials work:
  dpa, kerma, PKA proxies, gas production (He/H), and nuclide-production chains.
- Add uncertainty-aware trend views for damage and gas-production endpoints over
  irradiation, cooldown, and post-irradiation windows.
- Add dominant-contributor analysis specifically for damage and gas channels, in
  parallel with existing activity and dose contributor plots.

### 11.4 HTS Irradiation Campaign Support (Future Research Mode)

- Add campaign objects for superconducting sample metadata:
  conductor type, geometry, cryogenic test conditions, magnetic-field setpoints,
  and pre/post irradiation measurement records.
- Add derived-observable scaffolding for HTS-oriented endpoints:
  critical current retention, transition-temperature shifts, resistivity change,
  and quench-margin proxies linked to irradiation state.
- Add coupled optimization objectives for fusion-material studies:
  target activation observables + damage/gas limits + cooldown handling windows.
- Add post-irradiation exam (PIE) planning outputs (measurement queue templates,
  cooldown gates, transport safety metadata, and sample lineage tracking).

### 11.5 Fusion-Device Context Constraints

- Add high-heat-flux and tungsten-facing-surface constraint templates inspired by
  divertor/blanket operating envelopes so schedule optimizers can respect
  realistic fusion-system boundaries.
- Add shielding-alignment and tolerance-aware review cards for campaigns that
  depend on narrow geometric windows or strict positional tolerances.
- Add remote-handling-aware replacement/inspection planning placeholders for
  long-horizon irradiation facility operations.

### 11.6 Planning Rules for Future Candidates

- These ideas are roadmap candidates only and must not be treated as implemented
  until they pass the normal FluxForge lifecycle gates.
- Every future feature must preserve offline-first execution, additive method
  selection, and provenance-complete outputs.
- Every future feature should be mapped to explicit backend, CLI, modern Qt GUI,
  fixture, parity-test, and probe-evidence requirements before implementation.
