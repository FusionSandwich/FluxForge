# FluxForge GUI Plan (Updated)

**Status:** Draft (implementation plan)  
**File:** `docs/gui/gui_plan.md`  
**Scope:** Cross-platform, lightweight desktop GUI that remains **CLI-first** and **scriptable**.

---

## Goals and Constraints

- **CLI-first and fully scriptable:** the GUI is an optional front-end; every GUI action maps to a CLI command or core API call.
- **Cross-platform desktop:** **Windows + Linux required**; macOS optional.
- **Globally usable + free + open source:** no paywalls, no account gates, no region restrictions; permissive licensing preferred (MIT/BSD/Apache/LGPL).
- **Offline-first:** no license servers, no mandatory network access.
- **No external APIs:** all GUI features must function without network calls or cloud services.
- **Low overhead:** avoid heavyweight runtimes (no embedded browser engine if it can be avoided).
- **Simple setup:** "one executable" for students/labs + standard Python install for developers.
- **Feature parity target:** provide the *interactive analysis ergonomics* users expect from tools like **PeakEasy** and **QuantumGold** (spectrum interaction, ROI/peak workflows, calibration UIs, batch operations, exports), while supporting FluxForge's neutron dosimetry / unfolding / validation pipeline.
- **Replacement targets:** FluxForge should replace PeakEasy, QuantumGold, and STAYSL while remaining usable with PyNE, OpenMC, and ALARA.

## Out of Scope (v1)

- Full instrument control / MCA acquisition "turnkey DAQ" (plugin later).
- Vendor-only formats that require proprietary SDKs.

---

## Reference Insights

### From `testing/` (internal comparisons)
- **SpecKit:** tabbed workflow UI; file pickers; CSV interchange; real-time plots.
- **HDTV:** interactive spectrum viewer; keyboard-driven workflows; calibration focus; ROOT matrix workflows.
- **Gamma-MCA:** PWA spectrum viewer; live serial plotting; import/export; auto peak detection; calibration; offline install.
- **peakingduck:** AI/ML peak finding and optional C++ acceleration.
- **gamma_spec_analysis:** smoothing + peak plotting helpers in notebooks.
- **curie/npat/becquerel:** spectrum classes, calibration utilities, decay-chain tools, and nuclear data access.
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

### C. Peak workflows
- Peak search (configurable).
- Multi-peak fitting + deconvolution for overlapped peaks.
- Negative channel counts display support for background subtraction scenarios (where relevant).

### D. Calibration workflows
- Energy calibration editor (interactive).
- Efficiency calibration workflow with selectable reference lines and curve display.
- Calibration "standards" database (lightweight local SQLite) to emulate "certificate database" workflows without heavy dependencies.
- Resolution curve fitting (FWHM vs energy) with editable model selection.

### E. Batch workflows
- Summing/appending/format conversion.
- ROI integration batches.
- Optional "waterfall" batch visualization.

### F. Reporting
- "Channel data", "ROI data/detail", "peak search", "peak identify" style exports.
- PDF/HTML report bundle generation driven by the same CLI steps.
 - Attenuation correction summary (material, thickness, transmission) when applied.

### G. Automation ergonomics
- "Copy as CLI" for every action.
- Optional "macro recorder" that stores a sequence of GUI actions as a FluxForge YAML workflow.

---

## Tech Stack (Open Source, Low Overhead)

### Recommendation (GUI v1): **Dear PyGui + Click + FluxForge core**
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

### Alternatives (escape hatches)
1. **Qt (PySide6) + pyqtgraph**
   - Best for very complex widget-heavy apps.
   - Heavier dependency story and packaging; license can be frictionful in some institutions.
2. **Tkinter + Matplotlib**
   - Minimal deps but weaker interaction ergonomics for "PeakEasy/QuantumGold-class" workflows.
3. **Web UI (Tauri + Svelte/React)**
   - Excellent UI flexibility; higher build complexity and more moving parts.

---

## Architecture

- **Core-first:** keep `fluxforge` core modules stable and UI-agnostic.
- **New package:** `fluxforge_gui/` (DPG app) that calls core APIs or runs CLI commands.
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

1. **UI skeleton + spectrum viewer (read-only)** + artifact export.
2. **Annotations + ROI + peak tools** (interactive fit panel).
3. **Calibration UI** (energy + efficiency) + exportable calibration artifacts.
4. **Pipeline tabs** (activities -> reaction rates -> unfolding) + run dashboard.
5. **Validation dashboards** (C/E plots, parity, uncertainty/covariance views).
6. **Packaging automation** for Windows/Linux with CI smoke tests.

---

## Open Questions

- Do we need full coincidence matrix workflows in v1, or can those be deferred?
- Which group structures and monitor libraries should ship as GUI "presets"?
- Should the GUI support "portable mode" (no installer, self-contained folder) by default?
- What minimal subset of "PeakEasy-like" non-spectroscopy features (e.g., GPS/metadata/time-series) matter for FluxForge's scientific use cases?

---

## Notes on source availability

- PeakEasy's complete user documentation appears to be gated behind account access; this plan is based on publicly available PeakEasy pages and release notes plus general gamma-spectroscopy UI requirements.
