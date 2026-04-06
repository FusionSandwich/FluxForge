# QuantumGold GUI Review and Implementation-Oriented Feature Summary for FluxForge

## Purpose

This document reviews and summarizes the features and capabilities of the **QuantumGold / QuantumMCA** software family, based **only** on the two source documents you provided. It is written to support implementation planning for an open-source gamma spectroscopy tool such as **FluxForge**.

Because the sources are a **marketing brochure** and a **basic operations guide**, this document distinguishes between:

- **Confirmed capabilities** explicitly described in the documents.
- **Likely behavior** strongly implied by the GUI and workflow descriptions.
- **Unknown / underdocumented details** that would need reverse engineering, experimentation, or replacement with modern open implementations.

## Source Basis

1. **Quantum brochure** (`Quantum_brochure.pdf`), especially pages 2–8.
2. **Basic Quantum MCA Operations** (`Basic Quantum MCA Operations.pdf`), pages 1–6.

Where useful, this document cites source pages in plain text, for example:

- `(Quantum brochure, p. 5)`
- `(Basic Operations, pp. 3–4)`

---

# 1. Product Architecture: What QuantumGold Appears to Be

The documents show that **QuantumGold** is not just a plotting GUI. It is a layered spectroscopy environment built on top of the broader **Quantum software** stack:

- **QuantumMCA** provides MCA emulation, acquisition control, display, qualitative spectrum analysis, ROI tools, reporting, hardware communication, and calibration/control utilities.
- **QuantumGold** adds the **quantitative analysis** layer, including efficiency calibration, calibration standard database support, nuclide activity calculation, advanced peak deconvolution, library-directed analysis, ROI-directed activity analysis, and sample analysis reports.
- The broader Quantum family also includes domain-specific variants (QuantumNaI, QuantumGe), but the brochure emphasizes QuantumGold as the “premiere” package for qualitative and quantitative gamma analysis.

**Implementation implication for FluxForge:** treat the target not as a monolithic app, but as a stack of subsystems:

1. Acquisition and hardware abstraction
2. Spectrum display and interaction
3. ROI / peak analysis engine
4. Calibration engine
5. Quantitative analysis engine
6. Nuclide library / standard library services
7. Reporting / export
8. Automation / scripting
9. Multi-buffer / multi-spectrum workspace

(Source: Quantum brochure, pp. 2–8)

---

# 2. High-Level Capability Summary

## Confirmed top-level capabilities

The brochure explicitly attributes the following capabilities to QuantumGold:

- Simple, easy-to-understand user interface
- Multiple memory groups for processing spectra
- Peak deconvolution
- Quantitative analysis for spectra acquired with germanium or sodium iodide detectors
- Ethernet network support for PGT multichannel analyzers
- Quantitative analysis workflow driven by efficiency correction data

(Source: Quantum brochure, p. 2)

## Confirmed top-level capabilities shared with QuantumMCA / Quantum software

- Fundamental qualitative gamma spectrum analysis
- Full or near-full MCA hardware control
- Automatic and manual ROI definition
- Peak search
- Isotopic identification
- Spectrum display with multiple concurrent spectra
- Auxiliary display panes
- Manual and automated calibration workflows
- Spectrum filtering and arithmetic
- Reporting
- Automation via Q-Script
- Multi-instrument communication over Ethernet and RS232

(Source: Quantum brochure, pp. 2–7; Basic Operations, pp. 1–6)

---

# 3. GUI and Workspace Model

## 3.1 Main display philosophy

The GUI appears to center on a **primary spectrum display** with a compact but dense toolbar and one or more **auxiliary displays**. The brochure states that the software can:

- Display up to **8 spectra** simultaneously
- Display spectra from up to **8 different spectrometers / MCAs** or from disk
- Provide an **auxiliary display** for:
  - full spectrum context
  - convolution results
  - hardware controls
- Show **ROI info** and **ROI adjustment** controls directly in the interface
- Use toggle-style buttons and tooltips for common operations
- Allow displayed range control via toolbar buttons or mouse interaction

(Source: Quantum brochure, p. 3)

### FluxForge implementation notes

A faithful modern equivalent should probably include:

- A central interactive spectrum plot
- A lower or side “context pane” switchable between:
  - full-spectrum navigator
  - active ROI table
  - peak fit / deconvolution result view
  - acquisition / hardware status
  - calibration controls
- A tabbed or buffer-based multi-spectrum workspace
- Overlay and split-view comparison modes
- Mouse-driven zoom/pan + numeric controls
- Hover tooltips and status-bar readouts for channel, energy, counts, ROI, centroid, FWHM, net area, etc.

## 3.2 Spectrum display controls

The basic operations guide confirms that the display supports:

- **Linear / log** vertical scale switching
- **Auto** vertical sizing
- Manual vertical resizing using toolbar controls
- Horizontal zoom controls with toolbar buttons
- “Max” to show the entire spectrum
- Mouse drag to select a zoomed region

(Source: Basic Operations, p. 1)

### FluxForge requirements

Implement:

- `y_scale = {linear, log}`
- `y_autoscale = bool`
- explicit x-range and y-range state
- mouse drag zoom rectangle
- reset/full-range button
- possibly synchronized zooming when multiple spectra are overlaid

## 3.3 Auxiliary display modes

The basic operations guide states that the auxiliary display can cycle through:

- None
- Spectrum info
- Aux primary
- Aux convolution
- Settings

(Source: Basic Operations, p. 1)

### FluxForge requirements

This is a strong hint that Quantum uses a **mode-switched lower pane** rather than many floating windows. A modern equivalent would be a right-side or bottom dock with switchable panels:

- metadata/info
- secondary plot / navigator
- filter/deconvolution output
- acquisition/settings pane

---

# 4. Acquisition and Run Control

## 4.1 Core acquisition controls

The basic operations manual documents GUI actions for:

- **Start** acquisition
- **Stop** acquisition
- **Erase** spectrum data
- **Refresh** displayed spectrum data

For HPGe with an external high-voltage supply, starting may require acknowledging a warning if HV bias is off.

(Source: Basic Operations, p. 2)

### FluxForge requirements

The acquisition state machine should explicitly support:

- idle
- acquiring
- paused/stopped
- erased/reset
- refresh/requery display

And the hardware layer should expose detector/system interlocks, for example:

- HV off / unsafe
- shaping or ADC not ready
- detector disconnected

## 4.2 Preset counting modes

The manual shows support for configurable presets, including:

- Real time
- Live time
- Peak (counts)
- Integral of all ROIs
- Integral of selected ROI
- Gross ROI statistics
- Net ROI statistics
- ICR counts
- SCA counts
- External counts
- Off
- Enable/disable presets

The guide explains that the user can specify a value and then choose which preset mode governs termination of acquisition.

(Source: Basic Operations, pp. 2–3)

### FluxForge requirements

This is an important capability. FluxForge should separate:

1. **termination criterion definition**
2. **observable used by criterion**

A good design is:

```text
PresetCondition:
  metric: {real_time, live_time, peak_counts, roi_integral_all, roi_integral_selected,
           gross_roi_stat, net_roi_stat, icr_counts, sca_counts, external_counts}
  threshold: float
  enabled: bool
```

The acquisition engine should evaluate these in real time and expose progress.

---

# 5. ROI Management and Interactive Analysis

## 5.1 Manual ROI creation and editing

The software supports manual ROI workflows. Confirmed features include:

- Holding **Ctrl** and click-dragging to define an ROI manually
- Selecting an ROI by clicking it
- Adjusting ROI bounds using lower toolbar controls
- Deleting the current ROI
- Deleting all ROIs
- Clicking an ROI to view its data
- Moving through ROIs using up/down controls
- Handling **overlapped ROIs**

(Source: Quantum brochure, p. 2; Basic Operations, p. 3)

### FluxForge requirements

Implement an ROI object model such as:

```text
ROI:
  id
  left_channel
  right_channel
  centroid_channel
  centroid_energy
  gross_counts
  net_counts
  background_method
  fwhm
  uncertainty
  overlap_group_id
  assigned_transition (optional)
  assigned_nuclide (optional)
```

Behaviorally, FluxForge should support:

- creation by drag
- selection and keyboard nudging
- resizing by handles
- ROI stacking / overlap groups
- ROI metadata panel
- retaining manual ROIs unless an explicit “replace with peak search” option is used

## 5.2 Automatic peak search and auto-ROI generation

Automatic peak search is central to the Quantum workflow. The docs confirm:

- Peak search can automatically find peaks and set ROIs
- Peak search behavior is configured through the **Tools Setup** screen
- Peak search sensitivity depends strongly on **resolution calibration**
- Performing peak search **deletes existing ROIs**

(Source: Quantum brochure, p. 2; Basic Operations, pp. 3–4)

### FluxForge requirements

Peak search should be treated as a reproducible, parameterized operation with:

- configurable detection threshold
- expected resolution model
- background model selection
- overlap mode
- maximum peak count
- option to replace or merge with existing ROIs

Since Quantum deletes ROIs on peak search, FluxForge should probably support both:

- `replace_existing_rois=True`
- `merge_with_existing_rois=True`

with `replace_existing_rois` available for compatibility.

---

# 6. Peak Search Configuration and Background Handling

The basic operations guide provides unusually useful detail about the Tools Setup parameters.

## 6.1 Overlapped mode for HPGe

The guide explicitly states that for HPGe detectors, the **Mode should be Overlapped**.

(Source: Basic Operations, p. 3)

### FluxForge implication

The peak engine should explicitly support an overlap-aware mode in which:

- neighboring peaks are not forced into isolated ROIs
- multiplets can be represented as grouped peaks within a shared ROI extent
- deconvolution/fitting is the default for sufficiently close features

## 6.2 Integral and Power settings

The guide says sensitivity for finding peaks can be increased by decreasing the **Integral** and **Power** settings.

(Source: Basic Operations, p. 3)

### Interpretation

The exact formulas are not documented, but these appear to be peak search tuning parameters for the convolution-based detector. They likely affect smoothing or significance filtering. These should be reinterpreted in FluxForge as more transparent parameters such as:

- smoothing width
- derivative/convolution kernel width
- significance threshold
- local prominence threshold

## 6.3 Peak cap

The maximum number of peaks that can be defined is stated to be **250**.

(Source: Basic Operations, p. 3)

### FluxForge implication

Expose a `max_peaks` parameter but do not hardcode an unnecessarily low ceiling unless compatibility is desired.

## 6.4 Statistical uncertainty thresholding

The guide explains a **Stat Uncertainty** setting that multiplies peak uncertainty to determine whether a peak is valid. Example:

- If the setting is 2.0, net peak area must be > 2× peak uncertainty.
- This corresponds to a 50% error cutoff.
- If the setting is 4.0, cutoff becomes 25% error.
- Larger values reduce the number of accepted peaks.

(Source: Basic Operations, p. 4)

### FluxForge requirements

This should become an explicit quality filter such as:

```text
accept_peak if net_area / sigma_net >= min_signal_to_uncertainty
```

Also surface the equivalent percent uncertainty cutoff in the UI.

## 6.5 ROI width based on FWHM model

The guide states that ROI width is determined by multiplying the **FWHM**, but importantly the FWHM used is **not the measured FWHM of the actual peak**. It is the **theoretical FWHM at that energy from the resolution calibration**.

(Source: Basic Operations, p. 4)

### FluxForge requirements

This is a critical design point. The ROI generator should be able to use a detector resolution function:

```text
FWHM(E) = f(E; calibration parameters)
```

Then define automatic ROI boundaries as something like:

```text
ROI = [centroid - k_left * FWHM(E), centroid + k_right * FWHM(E)]
```

This is far more reproducible than raw hand widths and is likely essential if you want Quantum-like behavior.

## 6.6 Background subtraction modes

The guide describes two background approaches:

1. **Continuum background correction**
   - Draws a line between either edge of the ROI
   - Subtracts everything underneath
2. **Ambient background correction**
   - Channel-by-channel subtraction using an assigned background spectrum
   - Not normally used

It also mentions:

- **Background width** = number of channels on each side averaged to define continuum endpoints
- **Gap** = channels between ROI edge and background-sampling region

(Source: Basic Operations, p. 4)

### FluxForge requirements

Implement a pluggable ROI background model API with at least:

- trapezoidal/linear continuum background
- background spectrum subtraction
- configurable sideband width and gap

Suggested API:

```text
BackgroundModel:
  method: {linear_sidebands, background_spectrum}
  sideband_width_channels: int
  gap_channels: int
  background_spectrum_id: optional
```

---

# 7. Peak Deconvolution and Multiplet Analysis

The brochure explicitly states that QuantumGold performs **gaussian peak deconvolution** to analyze:

- spectra with overlapped peaks
- multiple overlapping ROIs within a multiplet
- repeated deconvolution after manual ROI insertion/removal

(Source: Quantum brochure, p. 4)

## 7.1 Confirmed behavior

- Deconvolution is an analysis operation applied to overlapping peak regions.
- It uses a **Gaussian** model.
- It is integrated with ROI workflows rather than being a completely separate fitting program.
- Users can manually change ROI assignments and rerun the analysis.

## 7.2 What is not documented

The sources do **not** specify:

- whether the Gaussian is pure Gaussian or Gaussian + tail/background terms
- whether centroid and width can float independently
- whether detector response asymmetry is modeled
- what optimizer is used
- whether constraints come from the resolution calibration

## 7.3 FluxForge requirements

To match or exceed QuantumGold, implement a multiplet fitting engine with:

- Gaussian-only compatibility mode
- optional more physical HPGe response model later
- tieable FWHM to calibration model
- optional shared or constrained background
- support for 1..N peaks in one ROI group
- residual visualization
- goodness-of-fit metrics
- uncertainty propagation to net areas

Minimum useful design:

```text
PeakFitRegion:
  roi_bounds
  background_model
  peak_models[]
  fit_result
  covariance
  residuals
```

---

# 8. Calibration Workflows

Calibration is one of the most important areas where QuantumGold appears to be operationally strong.

## 8.1 Automatic hardware adjustment

The brochure says that automatic hardware adjustment is performed by placing a **Cs-137** source near the detector and pressing one button.

- For **NaI systems**, detector high voltage, coarse gain, and fine gain are automatically set.
- For **germanium systems**, high voltage is set and adjusted manually for safety.

(Source: Quantum brochure, p. 6)

### FluxForge implication

This depends on hardware support and may not be fully implementable without vendor protocol access. But architecturally, FluxForge should separate:

- instrument control commands
- analysis/calibration logic
- user approval/safety interlocks

## 8.2 Automatic fine energy calibration

The brochure describes an automatic fine energy calibration workflow:

- Place a multiline source such as **Eu-152** near the detector
- Acquire a spectrum
- Press **Fine Energy Cal**
- A **quadratic energy calibration** is performed on the spectrum in memory

(Source: Quantum brochure, p. 6)

### FluxForge requirements

Implement:

- source-assisted line matching
- automatic calibration line identification
- quadratic fit by default
- display of fitted curve and residuals
- editable list of included lines

## 8.3 Resolution calibration

The basic operations guide says automatic peak search quality depends on having an accurate **resolution calibration**. The workflow includes:

- creating several peak ROIs
- avoiding partially overlapped or poor-statistics peaks
- going to manual calibration → resolution calibration
- clicking **Include** for peaks used in the fit
- selecting **Quadratic** and executing

(Source: Basic Operations, pp. 4–5)

### FluxForge requirements

Implement a resolution calibration object such as:

```text
ResolutionCalibration:
  model_type: {linear, quadratic, sqrt_polynomial, other}
  parameters
  covariance
  included_lines[]
  valid_energy_range
```

and integrate it into automatic peak search, auto-ROI sizing, and deconvolution.

## 8.4 Manual energy calibration methods

The brochure confirms support for:

- quick manual **two-point linear calibration**
- manual **quadratic ROI centroid-based** calibration
- manual resolution calibration

(Source: Quantum brochure, p. 6)

### FluxForge requirements

This implies you need both:

- fully automatic calibration routines
- expert-mode manual calibration workflows with peak/line selection and explicit fit control

## 8.5 Efficiency calibration

The brochure provides a concise but important description:

- Open the detector calibration window
- Select gamma lines to use for the calibration
- Press one button
- The efficiency curve is automatically generated and displayed

(Source: Quantum brochure, p. 3)

### FluxForge requirements

The efficiency calibration module should support:

- standard spectrum ingestion
- library transition selection
- decay-corrected source activity handling
- fitting an efficiency curve
- visualization of points, fit, residuals, and excluded lines
- storing detector-specific efficiency models

## 8.6 Calibration certificate database

QuantumGold manages calibration certificate information in a **Microsoft Access database**. It supports:

- any number of calibration standards
- standards with multiple nuclides
- data entry through the software
- automatic **decay corrections** when a standard is used in calibration

(Source: Quantum brochure, p. 4)

### FluxForge requirements

This is a major feature that should be recreated in a modern format. Instead of Access, use a portable schema such as SQLite / JSON / YAML / Parquet backed by validation.

Suggested data model:

```text
CalibrationStandard:
  id
  name
  reference_date
  geometry
  matrix
  certificate_metadata
  nuclides[]

StandardNuclide:
  nuclide
  certified_activity
  activity_uncertainty
  reference_datetime
  decay_data_source
```

The software should automatically decay-correct source activities to count time or calibration time.

## 8.7 Conversion gain, group size, energy range

The basic operations guide documents:

- **Group Size**: number of channels recorded; typically 8192 for HPGe, but can be 512 or 1024 for efficiency/other usage
- **Conversion Gain**: described as the resolution for the vertical scale, usually set equal to Group Size for convenience
- Ability to change energy range via manual calibration → set energy scale
- Typical HPGe setting noted as **4 MeV**

(Source: Basic Operations, p. 5)

### FluxForge requirements

Support detector/acquisition configuration metadata for:

- number of channels
- energy span
- ADC/binning/grouping mode
- effective channel width
- rebinned views without destroying the original data

---

# 9. Quantitative Analysis Methods

The brochure identifies **two analysis methods** in QuantumGold.

## 9.1 Library-directed analysis

Workflow described:

1. Automatic peak search runs using Tools Setup parameters.
2. Optional deconvolution is applied if selected.
3. Peaks are matched to library lines of selected nuclides.
4. Nuclide identification is made.
5. Nuclide activities are calculated using efficiency correction data.
6. A sample analysis report is generated.
7. Activity information is displayed in order of **level of confidence**.

(Source: Quantum brochure, p. 5)

### FluxForge requirements

This is essentially an integrated identification-and-quantification pipeline. To reproduce it, FluxForge will need:

- peak detection
- candidate transition matching
- nuclide scoring/ranking
- consistency checks across multiple lines
- efficiency-corrected activity inference
- confidence metrics
- formatted report generation

## 9.2 ROI-directed analysis

Workflow described:

- User manually assigns a particular ROI to a specific gamma line of a specific nuclide.
- No peak search or nuclide library search is required.
- Activity is calculated from the net counts in each assigned ROI.
- This is useful when the user knows which nuclides are present and wants to exclude others.
- **MDA values are reported even when no net counts are measured** for a particular ROI.

(Source: Quantum brochure, p. 5)

### FluxForge requirements

This is extremely relevant for activation analysis and flux-wire work. Implement:

- a targeted quantification mode based on an analyst-selected line list
- mandatory transition assignment metadata on ROIs
- activity/MDA computation per assigned line
- exclusion of non-target nuclides from decision logic

This mode is arguably essential for your use case.

## 9.3 Confidence-ordered reporting

The sample analysis report ranks identified nuclides by level of confidence.

(Source: Quantum brochure, p. 5)

### FluxForge requirements

The output should distinguish between:

- matched line evidence
- number of supporting lines
- activity consistency across lines
- residual ambiguities/interferences
- confidence score or classification

---

# 10. Nuclide Libraries and Reference Data

The brochure states the Quantum software includes three main libraries:

- **Gamma library** with more than 100 isotopes, hundreds of gamma emissions, and numerous X-ray emissions associated with some radioactive decays
- **Alpha library** with 63 nuclides and 193 alpha emissions
- **Beta library** with 59 nuclides and 120 beta emissions
- Libraries can be **cloned** and configured for specific applications

(Source: Quantum brochure, p. 4)

## FluxForge requirements

At minimum for gamma spectroscopy, FluxForge should support:

- a gamma emission database
- optional X-ray lines
- application-specific cloned libraries or filtered subsets
- user-editable libraries
- provenance of library source data and revisions

For modern implementation, build library support as a versioned data package rather than hardcoded tables.

Important note: the documents do **not** specify the underlying data source or the full field schema. That would need replacement with an open source / standards-based dataset.

---

# 11. User-Defined Spectrum Processing and Advanced Signal Operations

The brochure describes a **Convolutions** menu that performs spectrum manipulations and stores results in a separate memory location. Confirmed filters/functions include:

- Quadratic smooth
- Top hat filter
- 1st derivative
- 2nd derivative
- Peak finder

In addition, a **Spectrum Calculator** performs spectrum arithmetic such as:

- adding spectra
- subtracting spectra
- other basic calculations

Results are stored/displayed in an available memory buffer.

(Source: Quantum brochure, p. 4)

## FluxForge requirements

This implies a signal-processing workspace model rather than a single immutable spectrum.

Recommended implementation:

```text
Spectrum:
  counts
  metadata
  calibration_refs
  provenance

DerivedSpectrum:
  parent_ids[]
  operation
  parameters
  counts
  provenance
```

Required operations:

- smoothing
- derivative filters
- top-hat / morphological filter
- peak enhancement filters
- arithmetic combination
- background subtraction
- normalization options

A provenance graph is strongly recommended so every derived result can be reproduced.

---

# 12. Reporting and Export

## 12.1 QuantumMCA reports

The brochure lists several concise report types:

- Channel data
- ROI data
- ROI detail
- Peak search
- Peak identify

(Source: Quantum brochure, p. 5)

## 12.2 Save / print functions from the operations guide

The basic operations guide documents:

- Save spectrum via **File → Save As Spectrum**
- Save as native MCA software format or **ASCII**
- Print spectrum via **File → Print Spectrum**
- Print “Primary and ROIs” to include displayed spectrum plus basic ROI information
- Save/print **ROI Data** via Analysis Tools → ROI Data
- Save/print **ROI Detail** via Analysis Tools → ROI Detail
- Load saved spectra via **File → Load to Buffer**
- Switch between multiple spectrum buffers
- Display multiple spectra as separate graphs or overlapped
- Toggle between displaying all spectra or only the primary spectrum

(Source: Basic Operations, pp. 5–6)

## FluxForge requirements

Implement export in layers:

1. **Raw data export**: counts, metadata, calibration state
2. **Analysis export**: ROI tables, peak tables, fit results, nuclide matches, activities, MDA
3. **Presentation export**: report-ready plots and summaries

Formats should include open equivalents such as:

- CSV / TSV
- JSON / YAML
- HDF5 or Arrow/Parquet for richer storage
- PNG / SVG / PDF for plots
- Markdown / HTML / DOCX / PDF for reports

---

# 13. Automation and External Hardware Control

The brochure describes **Q-Script**, a menu-driven automation development tool for constructing automated MCA-based sample counting and analysis routines. It also states that Q-Script can:

- designate logic signal values on the rear-panel auxiliary I/O connector
- control external hardware that accepts those signals
- support workflows such as a **sample changer**

(Source: Quantum brochure, p. 6)

## FluxForge requirements

This is a large capability area and should not be ignored if you want QuantumGold-level operational usefulness.

Recommended modern replacement:

- Python scripting API
- YAML/JSON workflow definitions
- event hooks for acquisition start/stop/error
- hardware abstraction for GPIO / vendor API / serial / TCP control
- batch sample queue runner

Suggested automation primitives:

- connect detector
- load preset
- start count
- wait until preset complete
- save spectrum
- run analysis template
- export report
- toggle external device signal
- move to next sample

---

# 14. Hardware Communication and Multi-Instrument Management

The brochure states that Quantum software supports:

- Ethernet (10/100 base-T)
- RS232
- automatic hardware search to discover connected MCAs
- automatic software configuration based on detected hardware capabilities
- managing data from multiple instruments in a logical way
- up to **8** memory buffers / controlled instruments at a time

It specifically lists MCA products compatible with QuantumMCA and notes that multiple spectrometers or MCAs can be controlled by one software package.

(Source: Quantum brochure, pp. 2, 7–8)

## FluxForge requirements

A clean architecture would use:

```text
DetectorBackend:
  connect()
  disconnect()
  get_status()
  start_acquire()
  stop_acquire()
  erase()
  get_spectrum()
  set_hv()
  set_gain()
  set_presets()
  read_live_time()
  read_real_time()
  ...
```

and then a session layer:

```text
WorkspaceBuffer:
  spectrum_id
  detector_id
  display_state
  primary_flag
```

Even if you do not initially support live hardware, preserve this API boundary so off-line analysis and live acquisition can coexist.

---

# 15. Quadratic Compression Conversion (QCC)

The brochure presents **Quadratic Compression Conversion (QCC)** as a patented NaI-focused technology intended to mitigate:

- non-linear response below about 200 keV
- relatively low resolution
- energy-dependent resolution variation

The brochure claims QCC enhances the ability to identify peaks and make isotopic identifications, and compares:

- 8000-channel linear spectra
- 512-channel linear spectra
- 512-channel QCC spectra with more uniformly defined peaks

(Source: Quantum brochure, p. 7)

## FluxForge interpretation

This is likely hardware/software-specific and may not be reproducible exactly without proprietary details. However, conceptually it suggests support for:

- non-linear channel/energy remapping
- adaptive binning or transformed energy axes
- resolution equalization concepts for scintillation detectors

## FluxForge recommendation

Do **not** block the core project on QCC. Instead, design the data model to allow:

- arbitrary energy calibration transforms
- non-uniform display transforms
- detector-specific preprocessing pipelines

Then implement QCC-like functionality later if its behavior becomes important.

---

# 16. Configuration Management

The basic operations guide states that users can load a **configuration file** via `File → Load Configuration File`, and that the file contains information such as:

- preset settings
- conversion gain information
- detector information
- related setup state

(Source: Basic Operations, p. 6)

## FluxForge requirements

You should support portable analysis and acquisition configuration bundles, for example:

```yaml
instrument:
  detector_type: HPGe
  channels: 8192
  energy_range_keV: 4000
  presets:
    mode: live_time
    threshold: 600
analysis:
  peak_search:
    mode: overlapped
    min_snr: 3.0
    roi_width_fwhm: 1.5
  background:
    method: linear_sidebands
    sideband_width: 1
    gap: 0
libraries:
  gamma_library: default
calibration:
  energy: detector_A_energy_2026-03-01
  resolution: detector_A_resolution_2026-03-01
  efficiency: detector_A_efficiency_2026-03-01
```

This will be much better than opaque legacy configuration files.

---

# 17. What QuantumGold Gives the User Operationally

From a user-experience standpoint, the software seems to provide five important operational modes:

## 17.1 Live acquisition workstation

The analyst can configure the detector, start/stop acquisition, monitor the spectrum, set presets, and manage display settings.

## 17.2 Interactive spectrum interrogation tool

The analyst can zoom, toggle views, define ROIs, inspect ROI statistics, cycle through ROIs, and run manual or automatic peak analysis.

## 17.3 Calibration workstation

The analyst can perform energy, resolution, and efficiency calibration; manage calibration standards; and visualize the resulting curves.

## 17.4 Quantitative nuclide analysis system

The analyst can run library-directed or ROI-directed activity calculations, including deconvolution-aware workflows and MDA output.

## 17.5 Batch / semi-automated counting system

With Q-Script and hardware I/O, the system can support automated routines and external devices such as a sample changer.

(Source: Quantum brochure, pp. 3–6; Basic Operations, pp. 1–6)

---

# 18. Likely Internal Data Structures Implied by the GUI

To reproduce the same capability envelope in FluxForge, you likely need the following persistent entities.

## 18.1 Spectrum entity

- raw counts array
- acquisition times (real/live)
- detector/instrument metadata
- channel count / group size
- energy calibration reference
- resolution calibration reference
- sample metadata
- provenance / file origin

## 18.2 ROI entity

- channel bounds
- energy bounds
- centroid
- FWHM
- gross / net counts
- background parameters
- uncertainty
- overlap/multiplet group
- optional assigned line / nuclide

## 18.3 Peak entity

- fitted centroid
- fitted area
- fitted width
- uncertainty
- local background
- fit quality
- linked ROI / multiplet / line candidates

## 18.4 Nuclide / line entity

- nuclide name
- transition energy
- emission probability
- half-life
- optional X-ray association
- library revision source

## 18.5 Calibration entities

- energy calibration
- resolution calibration
- efficiency calibration
- calibration standard record
- fit residuals and validity ranges

## 18.6 Analysis result entity

- mode: library-directed or ROI-directed
- identified nuclides
- activities
- MDA values
- confidence scores
- supporting line evidence
- report export state

---

# 19. Priority Mapping for FluxForge Development

## Phase 1: Must-have features to match the everyday Quantum workflow

- Spectrum display with zoom, linear/log scaling, autoscale, overlay, multi-buffer support
- Start/stop/erase/load/save for offline and, later, online use
- Manual ROI creation/editing/deletion
- ROI tables with gross/net/centroid/FWHM
- Automatic peak search
- Resolution calibration integrated with peak search
- Energy calibration (two-point + quadratic)
- Linear continuum background subtraction
- Save/export of spectra and ROI data

## Phase 2: Quantitative analysis parity

- Efficiency calibration with standards
- Calibration standard database with decay correction
- Library-directed analysis
- ROI-directed analysis
- MDA reporting
- Nuclide library management
- Confidence-scored reports

## Phase 3: Advanced parity / beyond parity

- Gaussian multiplet deconvolution
- Derived spectra / convolution workspace
- Spectrum calculator
- Automation / scripting
- Multi-instrument support
- External device control hooks
- Detector-specific preprocessing pipelines

## Phase 4: Stretch / hardware-specific parity

- Direct vendor hardware control backends
- Auto hardware adjustment workflows
- QCC-like adaptive transforms for NaI

---

# 20. Important Unknowns and Reverse-Engineering Gaps

The provided documents are useful, but they do **not** fully define several critical technical details.

## Underdocumented items

- Exact native file formats for spectra, configs, reports, and libraries
- Exact deconvolution mathematics and optimizer
- Exact peak search algorithm and meaning of all Tools Setup parameters
- Exact MDA formula used
- Exact confidence scoring formula for nuclide identification
- Exact library schema and source data provenance
- Exact automation language syntax and execution model of Q-Script
- Exact hardware command protocol for MCA control
- Exact report templates and every column definition

## Practical recommendation

For FluxForge, do not attempt a blind one-to-one clone at the algorithm level unless you have validation datasets and can compare outputs. Instead:

1. Reproduce the **capability surface** and analyst workflow.
2. Use modern, transparent algorithms.
3. Build compatibility modes where behavior is known.
4. Validate against real spectra and Quantum output where possible.

---

# 21. Recommended FluxForge Module Breakdown

A strong implementation plan would divide the system into the following modules.

## `fluxforge.spectrum`

- spectrum container
- calibration attachment
- metadata
- provenance
- rebinning and transforms

## `fluxforge.display`

- plotting widgets
- multi-buffer workspace
- overlay / split view
- ROI interactions
- auxiliary panes

## `fluxforge.roi`

- ROI entity
- ROI editing tools
- ROI statistics
- overlap grouping
- background models

## `fluxforge.peaksearch`

- automatic peak detection
- significance filters
- resolution-model-based ROI generation
- replace/merge ROI modes

## `fluxforge.fit`

- Gaussian peak models
- multiplet fitting
- residual visualization
- covariance propagation

## `fluxforge.calibration`

- energy calibration
- resolution calibration
- efficiency calibration
- calibration persistence

## `fluxforge.nuclides`

- gamma/X-ray library
- library cloning/filtering
- transition lookup
- activity inference helpers

## `fluxforge.quant`

- library-directed analysis
- ROI-directed analysis
- activity calculations
- MDA calculations
- confidence scoring

## `fluxforge.processing`

- smoothing
- top-hat filters
- derivatives
- spectrum arithmetic
- derived spectra provenance

## `fluxforge.io`

- import/export
- config files
- report generation
- ASCII / CSV / JSON / YAML / native formats

## `fluxforge.hardware`

- detector abstraction
- MCA communication backends
- presets and control
- status and interlocks

## `fluxforge.automation`

- workflow engine
- Python/YAML automation
- queue/sample changer hooks

---

# 22. Final Takeaway

The QuantumGold GUI, as evidenced by the provided documents, is best understood as a **full spectroscopy workstation**, not just a spectrum viewer. Its defining strengths are:

- a dense but efficient interactive GUI for spectra and ROIs
- a strong calibration workflow
- quantitative gamma analysis with two analysis paradigms
- overlap-aware analysis and Gaussian deconvolution
- multi-buffer / multi-spectrum operation
- reporting and export
- automation and instrument integration

For FluxForge, the most important design lesson is that the software should be built as an **integrated analysis environment** where acquisition, calibration, ROI analysis, nuclide identification, quantification, and reporting all live in one coherent workspace.

If your goal is to replace QuantumGold for real lab work, the most strategically important features to reproduce first are:

1. Interactive ROI-centric spectrum analysis
2. Resolution-aware automatic peak search
3. Energy, resolution, and efficiency calibration workflows
4. ROI-directed activity analysis for known nuclides
5. Clean reporting and reproducible data export

Those features appear to be the core of what made QuantumGold operationally useful in practice.

---

# 23. Source Notes

- Quantum brochure, pp. 2–8
- Basic Quantum MCA Operations, pp. 1–6

