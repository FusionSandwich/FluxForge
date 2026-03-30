# FluxForge GUI Plan (Archive Copy)

**Status:** Historical MVP prototype plan with active redesign notes  
**File:** `docs/GUI_PLAN.md`  
**Scope:** Cross-platform, lightweight desktop GUI that remains **CLI-first** and **scriptable**.

---

**Supersession Note (2026-03-30):** The active redesign roadmap is now governed by
`docs/FluxForge_Final_Additions.md`, `docs/FluxForge_Additions_v3_Final.md`,
`docs/ROADMAP_EXECUTION_STATUS.md`, and `docs/adr/ADR-001-gui-stack.md`. This file
still documents the original Tk prototype and MVP rationale, but the primary GUI path
has moved to the Qt shell under `src/fluxforge/gui/`. The Tk application under
`src/fluxforge_gui/` is retained as a legacy/archive fallback during migration.

## Goals and Constraints

- **CLI-first and fully scriptable:** the GUI is an optional front-end; every GUI action maps to a CLI command or core API call.
- **Cross-platform desktop:** **Windows + Linux required**; macOS optional.
- **Globally usable + free + open source:** no paywalls, no account gates, no region restrictions; permissive licensing preferred (MIT/BSD/Apache/LGPL).
- **Project license target:** keep FluxForge itself under a permissive open-source license, with `MIT` preferred for the project root and packaging metadata.
- **Offline-first:** no license servers, no mandatory network access.
- **No external APIs:** all GUI features must function without network calls or cloud services.
- **Low overhead:** avoid heavyweight runtimes (no embedded browser engine if it can be avoided).
- **No sibling-repo dependency:** FluxForge must install, run, and package correctly even when the top-level `testing/` tree is absent.
- **Dependency policy:** keep shipped GUI/runtime dependencies permissive and cross-platform; external GPL projects may inform behavior but are not copied into FluxForge runtime code or packaging.
- **Simple setup:** "one executable" for students/labs + standard Python install for developers.
- **Feature parity target:** provide the *interactive analysis ergonomics* users expect from tools like **PeakEasy** and **QuantumGold** (spectrum interaction, ROI/peak workflows, calibration UIs, batch operations, exports), while supporting FluxForge's neutron dosimetry / unfolding / validation pipeline.
- **Replacement targets:** FluxForge should replace PeakEasy, QuantumGold, and STAYSL while remaining usable with PyNE, OpenMC, and ALARA.

## Current Implemented Prototype Status (2026-03-17)

The legacy GUI prototype is implemented with **Tkinter + ttk + Matplotlib**.

The active redesign now targets **PySide6 + PyQtGraph** for the primary shell while
preserving the Tk codebase as an archive/fallback path.

Why this is acceptable right now:
- fully open source and Python-native,
- works on **Linux and Windows**,
- no proprietary runtimes or license restrictions,
- no browser requirement for the shipped GUI,
- low-friction for local labs and remote X/virtual-display testing,
- already integrated with FluxForge's CLI-first workflow model.

What is already implemented and working in the current prototype:
- tabbed workflow shell for ingest, spectrum, peaks, activity, rates, unfold, compare, standards, and report,
- native desktop runtime with explicit `FLUXFORGE_OFFLINE=1` gating for remote HTTP(S) sources and runtime downloads,
- ingest tab now includes both **single-spectrum** and **batch ingest** workflows,
- standards-oriented presets for ASTM/INL, US ASTM, IAEA IRDFF/GMA, `k0-NAA`, comparator NAA, and Curie-style workflows,
- standards tab now includes a first-pass stepwise `k0-NAA` workflow section for detector characterization, facility characterization, peak normalization, selectable governed library inputs, multi-measurement aggregation, blank/CRM QA-QC, and k0 report preview/loading,
- standards tab now also surfaces **Kayzero import**, **ASTM E3376**, and **RAFM validation / benchmark** workflows that previously existed only as orphan handlers,
- spectrum viewer with pan/zoom toolbar, default **log counts**, x/y scale controls, isotope-colored peak markers, and PNG export,
- spectrum tab now includes a direct **CLI spectrum-plot export** surface using the same desktop workflow state,
- **multi-buffer manager** for keeping multiple spectra in memory and using them as primary/overlay buffers,
- **buffer arithmetic** for sum/subtract/average/ratio combination of selected spectra into a reusable artifact,
- **manual ROI draw/edit** support via plot clicks plus ROI load/save JSON helpers,
- **drag-resize ROI handles on-canvas** for the selected manual ROI,
- **calibration editor** for previewing updated polynomial energy coefficients in the viewer,
- **scrollable control rail** in the Spectrum workspace so ROI, calibration, efficiency, and fit tooling remain reachable on standard desktop displays,
- **calibration-point picking + polynomial fit + residual summary** for manual calibration workflows,
- **residual / diagnostic subplot rendering** for calibration, counting, multiplet, Hypermet, and efficiency-fit review,
- **peak-fit / deconvolution inspector** for selected peaks with local multiplet-neighbor context,
- selectable **automatic peak finding**, **peak identification**, and **peak counting** methods with uncertainty-aware summaries, including ASTM/INL-oriented IEC-tiered counting defaults when the dosimetry preset is applied,
- selectable **nuclear-data source UI** for peak identification plus downstream Peaks / Activity / Standards panels with built-in actigamma, FluxForge bundled, NNDC offline, IRDFF, k0 monitor, calibration-source, flux-wire, and user custom-source connectors,
- **constrained multiplet / Hypermet fit widgets** for selected peak groups, including free-form constraint-matrix editing and template fills,
- **activity** and **reaction-rate** tabs now include live embedded Matplotlib panels with zoom/pan toolbars and y-scale controls in addition to their post-run uncertainty summaries,
- **compare/validation** tab now loads post-run summaries that surface final agreement metrics, while unfold retains the embedded live diagnostics panel,
- **unfolded spectrum display** in the Unfold tab with embedded plotting and solver-summary text,
- selectable **GLS / GRAVEL / MLEM** unfolding workflows with GUI-exposed covariance and iterative solver parameters, plus uncertainty-aware measured/predicted, residual/pull, and flux-correlation diagnostics,
- unfold tab now includes a visible **response-matrix builder** surface for the CLI `response` command,
- **efficiency calibration workflow** with reference-source line selection, counted-point capture, curve fitting, residual diagnostics, and JSON export,
- **Curie-style stacked-target and decay-chain tabs** backed by existing FluxForge physics APIs,
- reproducible "copy as CLI" workflow behavior,
- report tab now includes a visible **master plot suite** surface for the CLI `plots` command,
- native desktop acceptance coverage that opens the real GUI, drives ROI/calibration/report workflows, captures screenshots, saves artifacts, and verifies copied CLI commands,
- native desktop evidence now also captures a standards/k0 preset interaction and unfold/compare summary flow,
- native desktop evidence now also captures RAFM-backed peak picking plus Activity and Rates tab screenshots from the real GUI,
- native desktop review artifacts are now saved directly as raw screenshots and `run.json` bundles under `artifacts/gui_review/...`.

What is only partially implemented today:
- ROI drag-resize is now working for the selected ROI, but overlap handles and full click-drag ROI creation still need more polish,
- multi-buffer support now includes arithmetic, but still does not provide waterfall plotting or named workspace/session presets,
- calibration fitting now supports manual points and residual summaries, but still needs a full efficiency-fit wizard and residual plot panel,
- calibration fitting now supports manual points and residual summaries, and the efficiency workflow now fits/saves curves, but detector-profile import and resolution-fit coupling still remain,
- peak inspection now triggers counting and constrained fit helpers with plotted diagnostics and free-form constraint matrices, but richer tied-parameter editors and more guided multiplet UX still remain,
- Curie tabs are functional MVP panels, but they currently summarize backend results in text instead of dedicated diagnostic plots/tables,
- downstream non-gamma source selectors are now exposed in Peaks / Activity / Standards, but Rates / Unfold / Compare still need broader provenance-aware source plumbing where applicable,
- unfolding plots now show solved spectra, measured/predicted rate agreement, residual/pull behavior, and correlation heatmaps, but they still do not yet expose the full standalone master-plot suite or report packaging for every diagnostic figure.
- standards/manual coverage is now explicitly tracked: ASTM E261/E262/E3376, response/unfold/compare/report, and current k0 GUI/CLI surfaces are implemented; advanced coincidence, pile-up, f/alpha facility-characterization, MDA, and QA-trending helpers remain open roadmap items.

Known issues / not yet correct:
- ROI dragging is intentionally limited to the selected ROI row to keep the interaction deterministic,
- non-gamma source selection is now wired into the main Peaks / Activity / Standards workflows, but broader downstream use in Rates / Unfold / Compare remains incomplete,
- custom user data sources now support local JSON/CSV/YAML, HTTP(S) JSON/CSV/YAML, SQLite URIs, and Python plugin connectors, but connector validation/security hardening still needs more polish.

Tests already added for this phase:
- helper tests for buffer arithmetic,
- manual calibration-fit coefficient/residual regression tests,
- auto peak-detection + uncertainty-aware counting regression tests,
- diagnostic subplot rendering tests,
- nuclear data source registry and NNDC-source tests,
- free-form constraint-matrix parser tests,
- SQLite and Python-plugin connector tests,
- GUI unfold rendering helper tests,
- CLI unfold-method selection tests for iterative solvers,
- activity/rate/validation uncertainty-summary helper tests,
- prior GUI spectrum preview rendering tests and CLI peak-command regression remain in place.

Tests still needed:
- GUI-event coverage for ROI drag-resize edge cases,
- broader native desktop acceptance coverage for the new calibration, standards/k0, unfold/compare, and Curie-style panels,
- integration tests for multiplet fit and Hypermet constraint controls,
- broader end-to-end tests for buffer arithmetic feeding downstream GUI workflows,
- native desktop coverage that exercises the newly added downstream source selectors and custom connector registration flows.
- deeper end-to-end native desktop coverage for the expanded `k0-NAA` section, especially external-library selection, aggregation, QA/QC, and report-generation actions.

Latest validation snapshot for this phase:
- warning-regression subset: `28 passed` (`tests/test_gui_app.py`, `tests/test_gui_desktop_native.py`) with `MatplotlibDeprecationWarning` treated as an error,
- expanded GUI/offline/native-desktop regression suite: `102 passed` (`tests/test_gui_app.py`, `tests/test_astm_e261.py`, `tests/test_astm_e262.py`, `tests/test_cli_app.py`, `tests/test_gui_parity_registry.py`, `tests/test_gui_native_app.py`, `tests/test_gui_desktop_native.py`, `tests/test_nuclear_data_sources.py`, `tests/test_irdff.py`, `tests/test_no_external_repo_paths.py`),
- native desktop acceptance: passes with screenshots, ROI/calibration interaction, standards/k0 preset application, unfold/compare summary loading, report plot-suite generation, and copied CLI verification,
- review artifacts: `tests/gui_desktop_driver.py` now produces `run.json` plus raw screenshots for manual review,
- current Linux review bundle: `artifacts/gui_review/current_linux/` with RAFM-backed screenshots for launch, spectrum, peaks, activity, rates, standards, unfold/compare, and report tabs,
- ASTM/INL preset regression still applies `iec_tiered` counting, IRDFF source selection, and background-subtracted workflow defaults.

Conda environment note:
- [environment.yml](environment.yml) now explicitly includes `tk` and `pillow` alongside `matplotlib`/`scipy` for the desktop GUI runtime.

## GUI QA Tooling

- The archived FluxForge prototype GUI is a native desktop application built with **Tkinter + ttk + Matplotlib**.
- The shipped FluxForge GUI must remain usable **offline and without a browser** on Windows and Linux.
- Because of that, **Playwright is not the primary automation tool for the legacy shipping GUI path**.
- Maintain two explicit QA lanes:
  - **Reference / web lane:** use Playwright only for browser-based reference GUIs and any future FluxForge web/Electron prototype.
  - **Native desktop lane:** use real desktop-driven acceptance runs, helper-level Tk regressions, artifact-comparison tests, and Windows/Linux packaging checks for the current FluxForge GUI.
- A `js_repl`-enabled Codex session may be useful for reference-GUI inspection, but that is a developer-tooling concern rather than a FluxForge runtime dependency.
- Do not claim that Playwright directly covers the current Tk desktop GUI until FluxForge has a web or Electron surface that Playwright can actually drive.
- Reference GUI work under `testing/` is inspiration-only and optional; it is not part of FluxForge's build, install, packaging, or runtime contract.
- The standing product contract for adopted GUI behaviors now lives in [GUI_CAPABILITY_PROGRAM.md](GUI_CAPABILITY_PROGRAM.md).
- The screenshot review path now uses:
  - `tests/gui_desktop_driver.py` for native evidence capture,
  - raw screenshot artifacts under `artifacts/gui_review/...`,
  - committed Linux baseline screenshots under `tests/data/gui_review_baselines/linux/`.

## Out of Scope (v1)

- Full instrument control / MCA acquisition "turnkey DAQ" (plugin later).
- Vendor-only formats that require proprietary SDKs.

---

## Reference Insights

### From internal inspiration audits in `testing/` (non-dependency)
- **SpecKit:** tabbed workflow UI; file pickers; CSV interchange; real-time plots.
- **HDTV:** interactive spectrum viewer; keyboard-driven workflows; calibration focus; ROOT matrix workflows.
- **Gamma-MCA:** PWA spectrum viewer; live serial plotting; import/export; auto peak detection; calibration; offline install.
- **peakingduck:** AI/ML peak finding and optional C++ acceleration.
- **gamma_spec_analysis:** smoothing + peak plotting helpers in notebooks.
- **curie:** spectrum fitting, calibration, stacked-target modeling, decay-chain solving, and nuclear-library search that should be surfaced as first-class GUI workflows.
- **npat/becquerel:** spectrum classes, calibration utilities, decay-chain tools, and nuclear data access.
- **irrad_spectroscopy:** isotope identification, activity determination, dose rate calculations.
- **actigamma:** gamma line synthesis from activities.
- **PyGammaSpec:** scintillation detector analysis and visualization utilities.
- **Neutron-Unfolding/pyunfold:** GRAVEL/MLEM iterative unfolding examples.
- **Neutron-Spectrometry:** MLEM-STOP + config-driven CLI tools with ROOT plotting.
- **gmapy:** cross section evaluation with uncertainties.
- **py-findpeaks:** catalog of peak-detection filters and algorithm interfaces.
- **NAA-ANN-1:** ANN data augmentation + spectrum patches workflow.

### From publicly available documentation (capability requirements)

#### PeakEasy (LANL)
PeakEasy is documented as:
- reading/displaying **200+ file formats** and converting to common formats,
- providing a **multiple-gaussian peak analysis** with real-time fit results,
- **rapid energy calibration adjustment** via sliders,
- **batch mode** (summing, file converting/appending, ROI analysis),
- **GPS extraction/display** (when present in files),
- large nuclide library / search tools (ENDF/B-VIII.0 based), mixtures, "find all peaks" hotkeys, and count-rate chart interactions.

PeakEasy is **Windows-only**, so FluxForge must reproduce the capability set on Linux/Windows without inheriting that platform limitation.

#### QuantumGold / QuantumMCA family (PGT/BNC brochure)
QuantumGold/QuantumMCA is described as providing:
- interactive spectrum analysis with a **Tools Setup** panel controlling peak search and ROI definitions,
- **manual ROI drawing/editing** with mouse + modifier keys and support for **overlapped ROIs**,
- display of **multiple spectra simultaneously** (up to 8 buffers),
- **peak deconvolution** for multiplets,
- quantitative workflows including **efficiency calibration** (select lines -> generate curve),
- convolution/smoothing utilities (e.g., quadratic smooth, top-hat, derivatives, peak finder),
- spectrum arithmetic (add/subtract spectra) and multiple report types (ROI data, channel data, peak search/identify),
- an automation/scripting concept (Q-Script).

#### STAYSL PNNL
The STAYSL PNNL suite is documented as a set of tools for neutron spectral adjustment:
- generalized least squares adjustment using measured reaction rates + covariances,
- adjacent tools for flux-history correction factors and self-shielding corrections,
- outputs including adjusted spectrum and covariance matrices.
STAYSL is described as a **command-line executable** workflow (historically Windows-focused), which matches FluxForge's philosophy: **core compute headless**, GUI as a runner/inspector.

#### Curie
Curie is not a desktop GUI reference, but it is a critical **workflow-capability reference** for FluxForge's GUI planning:
- HPGe **spectrum** fitting,
- energy/efficiency **calibration** workflows,
- **stacked-target** charged-particle activation analysis,
- **Bateman decay-chain** solving,
- **reaction-library** search across IRDFF/TENDL/ENDF-like data,
- attenuation and stopping-power tools.

FluxForge's GUI plan therefore needs Curie-inspired tabs/panels even where Curie itself is API/notebook-driven rather than desktop-UI-driven.

---

## UX Scope (MVP -> Beta -> Stretch)

### MVP (Release 0): "Spectrum + Peaks + Exports + Scriptability"
**Primary promise:** the GUI can replace common "PeakEasy/QuantumGold plotting ergonomics" for day-to-day spectrum inspection, peak/ROI workflows, and export - while writing reproducible artifacts.

1. **Project browser + wizard**
   - Create/open project directory (FluxForge schema).
   - Import spectra from common formats (N42/SPE/CNF/CHN/IEC ASCII/etc.).
   - Track detector metadata (live/real time, calibration, efficiency file references).

2. **Spectrum viewer (interactive)**
   - Pan/zoom, log/linear y-scale.
   - Multi-spectrum overlays (stacked or overlaid).
   - Cursor readout (energy/channel, counts, uncertainties if available).
   - **Select points and label** (markers with editable labels/notes).
   - **ROI tools**:
     - click-drag to create ROI,
     - manual resize/move,
     - support overlapped ROIs and multiplets.

   **Implemented in current prototype:**
   - pan/zoom toolbar,
   - log/linear y-scale,
   - multiple overlay buffers,
   - manual ROI creation from plot clicks,
  - selected-ROI handle dragging,
   - ROI table and JSON save/load,
  - selected-peak highlight and local inspector text,
  - selectable auto peak finding / identification / counting controls,
  - arithmetic buffer combination.

3. **Peak finding + fitting panel**
   - "Find peaks" with configurable thresholds.
   - Fit model selection (Gaussian + background; multiplet fitting).
   - Live re-fit (run in background; results update without freezing UI).
   - Peak table: centroid, FWHM, net area, uncertainty, significance.

4. **Calibration tools**
   - Energy calibration editor (interactive sliders + regression view).
   - Resolution curve view (optional).
   - Efficiency calibration screen (select lines, generate curve, residuals).
   - Attenuation helper (material/mixture transmission + HVL readouts).
   - Semi-empirical HPGe efficiency model (window/dead-layer terms + covariance).

  **Implemented in current prototype:**
  - preview-oriented polynomial coefficient editor for energy calibration.

  **Still required:**
  - calibration-point picking,
  - regression residual view,
  - efficiency-fit workflow,
  - resolution-curve model editor.

5. **Batch operations (MVP subset)**
   - Convert formats, append/sum spectra, run ROI integrations in batch.

6. **Export artifacts**
   - Export peaks/ROIs (CSV/JSON).
   - Export annotated spectrum figure.
   - Export a "run bundle" containing:
     - inputs snapshot,
     - config hash,
     - CLI command transcript ("copy as CLI").

---

### Beta (Release 1): "End-to-end activation + unfolding + validation tabs"
Add pipeline orchestration and neutron-dosimetry views.

- Pipeline tabs:
  - Data preparation (sample geometry, irradiation history, detector model).
  - Activity -> reaction rates (including decay corrections, monitor metadata).
  - k0-NAA panel (f/α from bare+Cd, Cd correction factors, G_th/G_ep self-shielding inputs, k0 library selection).
  - NAA-ANN panel (dataset selection, training config, prediction review, parity metrics).
  - Response matrix assembly (nuclear data selection + uncertainty).
  - Unfolding solver selection + diagnostics (RMLE/least-squares parity modes).
  - Model comparison (C/E tables; parity plots; residuals; contribution breakdown).
- Batch runner for saved workflows (headless execution with progress + logs).
- Uncertainty summaries (error bands, covariance heatmaps, correlation views).
- "STAYSL parity mode" views:
  - reaction list management,
  - cover corrections visualization (e.g., Cd transmission factors),
  - adjusted spectrum + covariance display and export.

---

### Stretch (Release 2): "Power-user spectroscopy ergonomics"
- Keyboard-first operations (HDTV-style) as optional bindings.
- Waterfall / time-series plotting for sequences of spectra (PeakEasy-style).
- Coincidence matrix/histogram imports (if required).
- Optional ML-assisted peak assistance (pluggable; off by default).
- Optional MCA acquisition plugin (serial/WebUSB-class device support).
- Curie-inspired stacked-target activation workflow panel and decay-chain orchestration screen.

---

## Online GUI (Future / PWA)

- Build a browser-based GUI similar to Gamma-MCA for lightweight access.
- Same project file + artifact formats as desktop GUI (offline-first).
- Read-only by default; enable editing where performance permits.
- No license servers or cloud dependencies required; runs locally or hosted.

---

## CLI Parity Targets (HDTV + Neutron-Spectrometry)

- **HDTV parity:** interactive command shell, keybindings, batch scripts, and
  command prefixes (Python, shell, batch execution) for spectrum workflows,
  plus calibration and peak-search commands.
- **Neutron-Spectrometry parity:** config-driven CLI tools for unfold/plot/trend
  workflows with reproducible settings files and standard output artifacts
  (unfold_spectrum, plot_spectra, unfold_trend, plot_lines).

---

## UI Concept

- **Left:** workflow navigation (Ingest -> Spectrum -> Peaks/ROIs -> Calibrate -> Activities -> Unfold -> Compare -> Report).
- **Center:** interactive plots (spectrum + overlays; residuals; calibration curves; covariance heatmaps).
- **Right:** inspector panel (parameters; library selection; ROI/peak properties; uncertainty settings).
- **Bottom:** run log + task status (queued/running/completed) + "copy CLI command".

---

## Required GUI Capabilities (derived from PeakEasy + QuantumGold)

### A. Data ingestion & interchange
- Read common gamma spectrum formats and convert to standardized FluxForge internal representation.
- Include IEC 62755 ASCII (.iec) import alongside N42 XML and legacy formats.
- Maintain multiple "buffers" (multiple spectra in memory) with consistent calibration handling.

### B. Interactive spectrum manipulation
- High-performance pan/zoom and ROI edits for 16k-channel spectra with
  multiple overlays.
- Manual ROI drawing/editing; overlapped ROIs; multiplet support.
- Marker/annotation system: select points, label, export annotations.
- Convolution/smoothing for *display and peak assistance* (never mutate raw data):
  - quadratic smooth,
  - top-hat,
  - derivatives,
  - peak-finder filter.

**Implemented now:**
- overlay plotting,
- ROI table + plot-click ROI creation,
- buffer list management,
- selected peak highlighting.

**Still required:**
- on-canvas drag/edit handles,
- annotation persistence and editable labels on-canvas,
- smoothing/filter overlays as view layers.

### C. Peak workflows
- Peak search (configurable filtering: first/second derivative, top-hat, optional AI/ML).
- Dynamic ROI sizing linked to theoretical FWHM resolution calibration.
- Multi-peak fitting + weighted NLLS deconvolution for overlapped peaks.
- Library-directed (auto-seed from lines) and ROI-directed (manual ROI) operation modes.
- Linear continuum background subtraction between ROI edges.
- Negative channel counts display support for background subtraction scenarios (where relevant).
- Robust MDA reporting capabilities (even for zero-count expectations).

**Implemented now:**
- peak list display,
- selected peak inspector,
- local multiplet-neighbor summary for deconvolution triage.

**Still required:**
- live refit panel,
- explicit constrained multiplet solver UI,
- fit residual plots,
- direct MDA workflow widgets.

### D. Calibration workflows
- Energy calibration editor (interactive).
- Efficiency calibration workflow with selectable reference lines and curve display.
- Calibration "standards" database (lightweight local SQLite) to emulate "certificate database" workflows without heavy dependencies.
- Resolution curve fitting (FWHM vs energy) with editable model selection.

**Implemented now:**
- editable energy-calibration coefficients in the spectrum viewer.

**Still required:**
- calibration-point manager,
- line-selection workflow,
- efficiency and resolution fit dashboards,
- calibration artifact browser.

### E. Batch workflows
- Summing/appending/format conversion.
- ROI integration batches.
- Optional "waterfall" batch visualization.

**Implemented now:**
- reusable CLI-mapped task tabs and copyable commands.

**Still required:**
- explicit GUI batch queue,
- waterfall/time-series view,
- spectrum arithmetic buffers.

### F. Reporting
- "Channel data", "ROI data/detail", "peak search", "peak identify" style exports.
- PDF/HTML report bundle generation driven by the same CLI steps.
 - Attenuation correction summary (material, thickness, transmission) when applied.

### G. Automation ergonomics
- "Copy as CLI" for every action.
- Optional "macro recorder" that stores a sequence of GUI actions as a FluxForge YAML workflow.

---

## Tech Stack (Open Source, Low Overhead)

### Recommendation (GUI v1+): **Dear PyGui + Click + FluxForge core**
**Why:**
- Permissive license and globally usable.
- Lightweight rendering (no browser runtime).
- Strong interactive plotting primitives appropriate for spectrum inspection and ROI/marker workflows.
- Can ship as a single executable via standard Python freezing/compilation tools.

**Stack:**
- **GUI:** Dear PyGui
- **CLI:** Click
- **Core:** `fluxforge` Python package (headless)
- **Optional kernels:** C++ + nanobind (+ Eigen) for profiled hotspots only

### Alternatives (historical evaluation)
1. **Qt (PySide6) + pyqtgraph**
   - **Selected by ADR-001 for the active redesign.**
   - Best for very complex widget-heavy apps.
2. **Tkinter + Matplotlib**
  - Minimal deps and fully open-source on Linux/Windows.
  - This is the **legacy prototype stack** retained during migration.
  - Interaction ergonomics were good enough for the MVP, but are weaker than the active roadmap target.
3. **Web UI (Tauri + Svelte/React)**
   - Excellent UI flexibility; higher build complexity and more moving parts.

---

## Architecture

- **Core-first:** keep `fluxforge` core modules stable and UI-agnostic.
- **Primary package:** `fluxforge/gui/` for the Qt redesign selected by ADR-001.
- **Legacy package:** `fluxforge_gui/` that continues to expose the archived Tk shell.
- **Current implementation split:** modern Qt shell in `fluxforge/gui/`; legacy Tk prototype in `fluxforge_gui/`.
- **Superseded note:** earlier Dear PyGui exploration is no longer the controlling direction.
- **Workflow runner:** GUI actions create a task graph of pipeline steps (inputs -> outputs -> artifacts).
- **Artifact I/O:** JSON/CSV/YAML as primary interchange so GUI and CLI share outputs.
- **Long tasks:** execute in a worker process (preferred) or thread; stream logs back to UI.

### Reproducibility contracts
- Every run produces:
  - `run_summary.json` (inputs, library selections, hashes)
  - `qc_summary.json` (guards and warnings)
  - exported figures/tables and optional report bundle
- GUI never stores "secret state" that can't be reproduced from project files.

---

## STAYSL Parity UI Requirements (from PNNL-22253)

- SigPhi Calculator parity:
  - Apply irradiation history renormalization (BCF output).
  - Correct for decay during irradiation, gamma self-absorption, and burnup.
  - Wire/foil selection with stored absorption parameters.
  - Produce saturated reaction rates (sig-phi) for STAYSL input.
  - Spreadsheet-style workflow with reaction data tables and copy/export for
    STAYSL input files.
- SHIELD parity:
  - Compute self-shielding factors for foils/wires and flux type (beam/isotropic).
  - Import/export self-shielding libraries for STAYSL.
- STAYSL output parity:
  - Adjusted spectrum + covariance matrix.
  - Broad-group flux/fluence summaries and reaction rates for library reactions.
  - Plot-friendly output files and run summaries.

---

## MCA Metadata Parsing (Future-Proofing)

Parse spectrum headers into structured metadata for reproducibility, e.g.:
- File/ID, measurement date/time, LT/RT/DT.
- Energy calibration polynomial (channel -> keV).
- Efficiency model parameters and detector geometry inputs.
- Resolution polynomial and FWHM reference.
- Detector geometry/material, source distance, well dimensions.
- Hardware settings: HV, gains, shaping, ADC config, dead-time settings.
- Stabilizer states, library used, and notes (e.g., efficiencies ignored).
- Analysis tables: nuclide list, ROI centroid/energy, gross/net counts,
  line assignments, and per-line activity estimates.

Example header fields captured from `Co-Cd-RAFM-1_25cm.txt`:
- ID, file name, measurement date/time, LT/RT/DT.
- Energy calibration: E = -1.694E+00 + 4.996E-01*Ch + 6.710E-08*Ch^2 (keV).
- Efficiency model coefficients: C1 -2.026E+01, C2 1.029E+01, C3 -1.655E+00,
  C4 8.666E-02.
- Geometry/detector model: A 3.48E-03, Al window T1 1000 um, DI 6.450 cm,
  DL 700 um, AI 0.000 deg.
- Detector: HpGe(P) coaxial, 6.000 cm diameter, 25.000 cm source distance,
  well depth 5.060 cm, well diameter 0.920 cm.
- Resolution: 1.389E+00 + 7.800E-04*E + 4.072E-08*E^2; FWHM at 661.66 keV
  reported as 1.89 keV.
- Hardware: HV 1800 (on), gains, shaping time 4.00 usec, ADC group size 8192,
  dead-time correction mode, gate mode, sync mode, and stabilizer state table.

---

## C/C++ Acceleration (Optional, Profiling-Driven)

FluxForge is primarily analysis-scale, but interactive UX benefits from fast kernels.

**Only migrate if profiling shows real pain**:
- iterative unfolding kernels (RMLE/MLEM-like loops),
- covariance propagation / large linear algebra,
- peak fitting/deconvolution hot paths,
- background estimation kernels used interactively.

**Recommended approach:**
- First: NumPy/SciPy vectorization + sparse methods
- Then: optional compiled kernels with **nanobind** (small, modern bindings) and **Eigen** for linear algebra
- Always keep Python fallbacks for developer installs.

---

## Packaging and Distribution

- **Windows (required):**
  - Standalone installer or one-folder distribution; goal: "download and run".
- **Linux (required):**
  - AppImage or standalone folder build; optional pip/conda for power users.
- Keep CLI intact; GUI is a separate entry point: `fluxforge gui`.
- Offline-first: no mandatory network calls or license checks.

---

## Implementation Phases

0. **Headless plotting baseline (implemented)**:
   - `fluxforge plots --example` generates the core master-plan plots (G1.1-G1.5)
     without opening windows (Agg backend), so SSH sessions can validate plotting
     capability before interactive GUI work.
1. **UI skeleton + spectrum viewer (read-only)** + artifact export.  implemented
2. **Annotations + ROI + peak tools** (interactive fit panel). ◑ partially implemented
3. **Calibration UI** (energy + efficiency) + exportable calibration artifacts. ◑ partially implemented
4. **Pipeline tabs** (activities -> reaction rates -> unfolding) + run dashboard.
5. **Validation dashboards** (C/E plots, parity, uncertainty/covariance views).
6. **Packaging automation** for Windows/Linux with CI native desktop acceptance checks.

### Immediate next GUI tasks
- extend native desktop acceptance beyond Spectrum/Report into Peaks, Standards/k0, and Unfold/Compare,
- add explicit buffer arithmetic (`A+B`, `A-B`, normalize, sum live times),
- expand the new scrollable-control treatment to other dense tabs, especially Report and Standards,
- add true multi-peak deconvolution widgets and residual panels,
- add Curie-inspired stacked-target and decay-chain workflow tabs,
- continue polishing the visual style while staying on open-source, Linux/Windows-safe toolchains.

---

## Open Questions

- Do we need full coincidence matrix workflows in v1, or can those be deferred?
- Which group structures and monitor libraries should ship as GUI "presets"?
- Should the GUI support "portable mode" (no installer, self-contained folder) by default?
- What minimal subset of "PeakEasy-like" non-spectroscopy features (e.g., GPS/metadata/time-series) matter for FluxForge's scientific use cases?

---

## Notes on source availability

- PeakEasy's complete user documentation appears to be gated behind account access; this plan is based on publicly available PeakEasy pages and release notes plus general gamma-spectroscopy UI requirements.
