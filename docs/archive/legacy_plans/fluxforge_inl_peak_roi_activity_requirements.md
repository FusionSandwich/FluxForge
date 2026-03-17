# FluxForge requirements for an ASTM/INL-aligned gamma spectrometry and activation-analysis mode

## Purpose

This document defines the **minimum required additions** for FluxForge to support at least one workflow that is credibly aligned with the way recent INL ATR dosimetry publications describe activation/gamma-spectrometry analysis. The focus here is intentionally narrow:

1. **peak finding**,  
2. **ROI / net-peak area determination**, and  
3. **data sources needed to convert net counts to activity**.  

This document is written so that each subsection can be translated into GitHub issues for the `add_gui` branch.

---

## 1. Target operating mode to add

### 1.1 Add an explicit `astm_inl_dosimetry` analysis profile

FluxForge should add a dedicated analysis profile named something like:

- `astm_inl_dosimetry`
- or `us_astm_reactor_dosimetry`

This profile should be treated as **separate** from a generic comparator-NAA profile.

### Why this is needed

Recent INL ATR neutron-dosimetry reports describe a workflow in which **cobalt and nickel wire specific activities** are measured and then used with the ASTM radioactivation methods to infer **thermal- and fast-neutron fluence rates**. Those reports explicitly tie the wire selections to **ASTM E262, ASTM E481, ASTM E264**, under the broader practice **ASTM E261**. For HPGe detector calibration and radionuclide analysis in reactor dosimetry, the closest ASTM metrology standards are **ASTM E3376-23** and **ASTM E181-23**. Recent INL reports also describe reporting wire activities decay-corrected to reactor shutdown and, in at least one recent report, averaging results across **four HPGe detectors**. [R1-R7]

### Required behavior of this profile

The profile shall:

- assume the primary sample classes are **activation foils/wires** and small irradiated samples;
- prefer **ASTM-style dosimetry outputs** after activity calculation;
- use **raw unsmoothed spectra** for final net-area and activity calculations;
- support **Co/Al**, **Ni**, and optionally **Ag** monitors first;
- treat **activity determination** as the immediate objective of gamma analysis;
- allow later conversion of activity to reaction rate / fluence under ASTM E261/E262/E264/E481.

---

## 2. Governing standards and what they imply for implementation

### 2.1 Standards to anchor the implementation

FluxForge should explicitly reference these standards in code documentation and report templates:

1. **ASTM E261-16(2021)**  
   General ASTM practice for determining neutron fluence, fluence rate, and spectra by radioactivation techniques. [R1]

2. **ASTM E262-17(2024)e1**  
   General thermal-neutron reaction-rate / fluence-rate determination by radioactivation. Includes comparator-style thermal-fluence workflows and notes that one advantage of the standard comparison technique is that detector efficiency does not need to be known for that particular method. [R2]

3. **ASTM E481-23**  
   Cobalt/silver activation route for thermal-neutron fluence using the Westcott formalism. [R3]

4. **ASTM E264-25**  
   Nickel activation for fast-neutron reaction-rate measurement, with Practice E261 as the general analysis reference. [R4]

5. **ASTM E3376-23**  
   Calibration and usage of germanium detectors for reactor dosimetry; explicitly covers energy and full-energy peak efficiency calibration for small, approximately point-like samples. [R5]

6. **ASTM E181-23**  
   General detector calibration and radionuclide-analysis guidance for reactor dosimetry; it also makes clear that HPGe-specific calibration/usage content is now separated into E3376. [R6]

7. **ASTM E844-25**  
   Selection, irradiation, post-irradiation handling, and quality control of neutron dosimeters. [R7]

### 2.2 What these standards imply for FluxForge

At the software level, these standards imply that FluxForge needs:

- a **detector-calibration aware** HPGe pipeline;
- a clear separation between **activity determination** and **fluence inference**;
- robust support for **small-source gamma counting**, not only bulk environmental spectra;
- standards-aware metadata for **dosimeter material**, **mass**, **geometry**, **count time**, **reference time**, and **reaction product**;
- reproducible calculation steps that can be reported in a traceable way.

---

## 3. Required additions for peak finding

### 3.1 Add a two-stream spectrum pipeline

FluxForge should process each spectrum in two streams:

- **working spectrum**: optional filtered/smoothed copy for candidate-peak detection only;
- **analysis spectrum**: raw spectrum used for centroid refinement, ROI definition, net-area estimation, uncertainty, and activity.

### Why

For HPGe spectra, smoothing/top-hat/derivative transforms are useful for **detection**, but the final integrated area should be computed from the **raw counts**, not the filtered spectrum.

### Required implementation

Add a pipeline stage with the following rules:

- peak search may use:
  - light smoothing,
  - derivative filtering,
  - top-hat transforms,
  - wavelet-like seed detection;
- final count extraction must use:
  - the raw channel counts,
  - raw per-channel uncertainties,
  - raw live-time / real-time metadata.

### GitHub issues to create

- Add `SpectrumView.raw_counts` and `SpectrumView.search_counts`
- Add configuration switch `peak_search.filter_mode`
- Disallow area integration on filtered arrays in production analysis mode

---

### 3.2 Add a calibrated resolution model as a required dependency

Peak finding and ROI sizing should depend on a fitted detector resolution model:

\[
\mathrm{FWHM}(E)
\]

Recommended support:

- polynomial in \(\sqrt{E}\),
- or another monotone positive model already common in HPGe workflows.

### Required implementation

FluxForge shall store, per detector:

- energy calibration parameters,
- FWHM calibration parameters,
- date/version of calibration,
- energy range of validity,
- QC status.

### Why

A fixed ROI width in channels is not acceptable for a standards-oriented HPGe workflow. The peak width must change with energy.

### GitHub issues to create

- Add detector `energy_calibration`
- Add detector `resolution_calibration`
- Add `fwhm_at_energy(E)` utility
- Add calibration validity checks versus requested energy range

---

### 3.3 Add a tiered peak-finding strategy

FluxForge should use three peak-analysis modes.

#### Mode A — isolated singlet

Use when:

- one local maximum is present,
- no obvious overlap exists,
- local continuum is approximately linear.

#### Mode B — difficult singlet / structured continuum

Use when:

- the continuum bends across the peak,
- Compton structure is significant,
- weak lines sit on a sloped or curved background.

#### Mode C — multiplet / overlap

Use when:

- neighboring centroids are within roughly 1–1.5 FWHM,
- tailing is obvious,
- there is visual or library evidence of a doublet/multiplet.

### Required implementation

Peak classification should be automatic, with manual override.

### GitHub issues to create

- Add `PeakCandidate.classification`
- Add automatic overlap classifier based on centroid spacing and valley depth
- Add manual override in GUI for `singlet`, `difficult_singlet`, `multiplet`

---

## 4. Required additions for ROI selection and peak counting

### 4.1 Add explicit ROI proposal logic based on FWHM(E)

For each seed peak at energy \(E_p\), propose an initial ROI using the detector resolution model.

Recommended defaults:

- **ordinary singlet initial width**: about \(\pm 1.25\,\mathrm{FWHM}(E_p)\)
- **high-background singlet initial width**: about \(\pm 0.6\,\mathrm{FWHM}(E_p)\)
- **fit window for multiplets**: about \(\pm 2\) to \(\pm 3\,\mathrm{FWHM}(E_p)\)

These are implementation defaults, not hard physical constants.

### Required implementation

The ROI engine shall:

- convert energy-domain ROI rules to channel indices using calibration;
- enforce detector bounds;
- expand or shrink windows when adjacent peaks are found;
- store the **reason** the ROI was chosen.

### GitHub issues to create

- Add `roi_strategy = fwhm_scaled`
- Add `ROIProposal` object with `left`, `right`, `method`, `parameters`
- Add provenance record for every final ROI

---

### 4.2 Add a Covell-style net-area method for isolated singlets

For isolated peaks, FluxForge should implement a local continuum subtraction method equivalent to the classical endpoint/trapezoid/Covell approach.

A basic form is:

\[
N_{\text{net}} = \sum_{i=L}^{R} G_i - \sum_{i=L}^{R} \hat{B}_i
\]

where:

- \(G_i\) is the gross count in channel \(i\),
- \(\hat{B}_i\) is the estimated local continuum under the peak,
- \([L,R]\) is the final ROI.

For a simple linear-background model, \(\hat{B}_i\) can be interpolated from left and right background side windows.

### Required implementation

Add a `roi_integrator.covell_linear` method that:

- chooses side windows outside the peak body,
- estimates the local continuum,
- subtracts continuum channel-by-channel,
- propagates counting uncertainty.

### Why

This is the simplest robust net-area method for singlets and maps naturally to thin-wire dosimetry use.

### GitHub issues to create

- Implement `integrate_peak_covell()`
- Add uncertainty propagation for background-subtracted area
- Add diagnostic plot showing gross, background, and net area

---

### 4.3 Add a structured-background fallback for difficult singlets

For peaks on curved continua, fixed endpoint subtraction can fail. FluxForge should add a fallback that re-evaluates local background anchors using a robust search for local minima or low-gradient side regions.

### Required implementation

Add a `difficult_singlet` mode that:

- starts from the FWHM-based ROI,
- searches outward for stable background anchor regions,
- fits a low-order local background model,
- recomputes net area,
- records edge sensitivity.

### Why

This is needed for weak activation lines in structured Compton regions.

### GitHub issues to create

- Implement `find_background_anchors()`
- Add local linear/quadratic background options
- Add edge-sensitivity QC metric

---

### 4.4 Add constrained local peak fitting for multiplets

For overlapping peaks, FluxForge should not rely on simple ROI summation. It should fit a local model such as:

\[
S(E) = \sum_{k=1}^{n} P_k(E; A_k, \mu_k, \sigma_k, \tau_k) + C(E)
\]

where:

- \(P_k\) is a Gaussian or Gaussian-plus-tail component,
- \(A_k\) is the amplitude/area parameter,
- \(\mu_k\) is centroid,
- \(\sigma_k\) or FWHM is constrained by calibration,
- \(\tau_k\) optionally represents tailing,
- \(C(E)\) is a local continuum model.

### Required implementation

Multiplet fitting should support:

- shared width constraints from the resolution model,
- optional centroid priors from a nuclide line library,
- optional fixed centroid spacing for known doublets,
- covariance output.

### Why

ASTM E3376 is intended for measurements where overlapping peaks and peak-to-continuum issues are not important, which means FluxForge must explicitly recognize when spectra leave that regime and then switch methods rather than silently integrating bad ROIs. [R5]

### GitHub issues to create

- Add `fit_local_multiplet()`
- Add Gaussian and Gaussian-plus-tail peak shapes
- Add covariance-aware area uncertainty extraction
- Add fit quality metrics: reduced chi-square, residual pattern, parameter bounds hits

---

### 4.5 Add explicit peak acceptance / rejection logic

Each candidate peak should end with one of the following states:

- `accepted_for_activity`
- `accepted_for_identification_only`
- `rejected_low_significance`
- `rejected_overlap_unresolved`
- `rejected_bad_fit`
- `manual_review_required`

### Required implementation

Acceptance logic should consider:

- decision threshold / detection significance,
- ROI stability,
- fit residuals,
- centroid agreement with library,
- consistency between multiple lines from the same nuclide.

### GitHub issues to create

- Add final peak decision state machine
- Add QC report per accepted/rejected line

---

## 5. Required additions for data sources used in activity calculation

This is the most important structural addition after ROI logic.

### 5.1 Add a formal data-source hierarchy for activity analysis

FluxForge should not pull half-lives and gamma intensities from arbitrary mixed sources. It should implement a ranked hierarchy.

### Recommended hierarchy

#### Tier 1 — evaluated decay data for half-life and photon emission probability

Use, in this order:

1. **DDEP / LNHB recommended decay data** when the radionuclide is available there. The LNHB DDEP tables are continuously updated, and the latest public table index now includes entries published through **Metrologia 63 (2026)**. [R8-R10]
2. **ENSDF-based sources** when DDEP is unavailable or when the nuclide/level of interest is not covered by DDEP. Official access routes include **NNDC NuDat 3** and the **IAEA LiveChart of Nuclides**, both of which draw from evaluated decay datasets. [R11-R13]

#### Tier 2 — detector response data

Use laboratory-controlled, detector-specific data for:

- energy calibration,
- FWHM calibration,
- full-energy peak efficiency calibration,
- coincidence-summing correction factors if used,
- geometry-transfer factors if used.

This information must come from **FluxForge calibration records**, not public nuclear-data libraries.

#### Tier 3 — measurement metadata

Use experimental records for:

- sample ID,
- sample mass,
- wire type / alloy type,
- count start time,
- live time,
- real time,
- dead time,
- detector ID,
- irradiation end or reactor shutdown time,
- cooling interval.

#### Tier 4 — optional correction data

Use these only when needed:

- attenuation coefficients, e.g. NIST XCOM, for self-attenuation corrections; [R14]
- dosimetry reaction data, e.g. IRDFF-II, for later conversion from activity to fluence under ASTM E261/E264/E481; [R15-R16]
- sample-geometry correction factors from Monte Carlo or validated semi-analytic transfer calculations.

---

### 5.2 Add a strict activity-data provenance model

For every reported activity, FluxForge should record:

- radionuclide,
- gamma energy used,
- emission probability source and version,
- half-life source and version,
- detector efficiency source and calibration version,
- whether coincidence-summing, self-attenuation, dead-time, and geometry corrections were applied,
- reference time to which the activity was decay-corrected.

### GitHub issues to create

- Add `DecayDataRecord`
- Add `EfficiencyCalibrationRecord`
- Add `ActivityResult.provenance`
- Add versioned nuclear-data source registry

---

## 6. Required activity model

### 6.1 Add a standard line-by-line activity equation

For each accepted full-energy line, FluxForge should compute the activity at a chosen reference time using a model of the form:

\[
A_{\mathrm{ref}} =
\frac{N_{\mathrm{net}}}
{\varepsilon(E)\, I_\gamma\, t_{\mathrm{live}}\, C_{\mathrm{count}}}
\times
C_{\mathrm{dead}}
\times
C_{\mathrm{geom}}
\times
C_{\mathrm{self}}
\times
C_{\mathrm{sum}}
\times
C_{\mathrm{decay\ to\ ref}}
\]

where:

- \(N_{\mathrm{net}}\) = net full-energy peak area,
- \(\varepsilon(E)\) = full-energy peak efficiency,
- \(I_\gamma\) = photon emission probability for that line,
- \(t_{\mathrm{live}}\) = live time,
- \(C_{\mathrm{count}}\) = correction for decay during counting,
- \(C_{\mathrm{dead}}\) = dead-time / rate-loss correction if needed,
- \(C_{\mathrm{geom}}\) = geometry-transfer correction,
- \(C_{\mathrm{self}}\) = self-attenuation correction,
- \(C_{\mathrm{sum}}\) = coincidence summing correction,
- \(C_{\mathrm{decay\ to\ ref}}\) = correction from count time to the requested reference time.

### Required implementation

At minimum, the first release of `astm_inl_dosimetry` should support:

- \(N_{\mathrm{net}}\),
- \(\varepsilon(E)\),
- \(I_\gamma\),
- \(t_{\mathrm{live}}\),
- decay during count,
- decay to reference time,
- dead-time metadata handling.

Geometry, self-attenuation, and coincidence summing may be optional in v1, but the API should already allow them.

### GitHub issues to create

- Implement `compute_activity_from_line()`
- Add decay-during-count correction utility
- Add decay-to-reference-time utility
- Add optional correction hooks for geometry, self-attenuation, coincidence summing

---

### 6.2 Add a multi-line fusion rule for final activity

A nuclide activity should not always be taken from a single line if multiple good lines exist.

### Required implementation

For all lines accepted for a given radionuclide:

- compute per-line activity and uncertainty;
- reject inconsistent outliers under configurable QC rules;
- combine the remaining lines using an uncertainty-weighted mean or a more conservative robust estimator;
- report both per-line and fused activity values.

### Why

This is one of the strongest automated QC mechanisms for gamma spectrometry.

### GitHub issues to create

- Add `fuse_line_activities()`
- Add nuclide-level line-consistency QC
- Add plot/table showing per-line activity agreement

---

### 6.3 Add multi-detector aggregation

Because recent INL dosimetry reports describe averaging activities across multiple HPGe detectors, FluxForge should support repeated counts of the same sample on different detectors. [R4]

### Required implementation

Add a `SampleMeasurementGroup` that can aggregate:

- multiple detector results,
- repeated counts on one detector,
- detector-by-detector activity means and standard deviations,
- final aggregate value at a common reference time.

### GitHub issues to create

- Add `MeasurementGroup`
- Add common-reference-time normalization before aggregation
- Add detector-to-detector spread report

---

## 7. Required pseudocode

## 7.1 Peak finding and classification

```text
function analyze_spectrum(spectrum, detector_cal, line_library, config):
    raw = spectrum.raw_counts
    search = build_search_copy(raw, config.search_filter)

    energy_cal = detector_cal.energy_calibration
    fwhm_model = detector_cal.resolution_model

    seed_peaks = detect_candidate_peaks(search, energy_cal, fwhm_model, config)

    accepted_candidates = []
    for seed in seed_peaks:
        E0 = channel_to_energy(seed.channel, energy_cal)
        roi0 = propose_roi_from_fwhm(E0, fwhm_model, config)
        neighbors = inspect_local_neighbors(search, roi0)

        if looks_like_multiplet(neighbors, roi0, fwhm_model):
            cls = "multiplet"
        elif looks_like_structured_background(raw, roi0):
            cls = "difficult_singlet"
        else:
            cls = "singlet"

        accepted_candidates.append(
            PeakCandidate(seed=seed, roi0=roi0, classification=cls)
        )

    return accepted_candidates
```

---

## 7.2 ROI and net-area extraction

```text
function extract_net_area(raw_spectrum, candidate, detector_cal, config):
    if candidate.classification == "singlet":
        roi = finalize_roi_singlet(candidate.roi0, detector_cal, config)
        bg = fit_local_linear_background(raw_spectrum, roi, config)
        net, unc = integrate_covell(raw_spectrum, roi, bg)
        return PeakAreaResult(method="covell_linear", roi=roi, net=net, unc=unc)

    if candidate.classification == "difficult_singlet":
        roi = refine_roi_with_background_anchors(raw_spectrum, candidate.roi0, config)
        bg = fit_local_background(raw_spectrum, roi, model=config.bg_model)
        net, unc, sensitivity = integrate_with_edge_sensitivity(raw_spectrum, roi, bg)
        return PeakAreaResult(method="adaptive_local_bg", roi=roi, net=net, unc=unc,
                              edge_sensitivity=sensitivity)

    if candidate.classification == "multiplet":
        fit_window = expand_fit_window(candidate.roi0, detector_cal, config)
        fit = fit_local_multiplet_model(raw_spectrum, fit_window, detector_cal, config)
        line = select_target_component(fit, candidate)
        return PeakAreaResult(method="multiplet_fit", roi=fit_window,
                              net=line.area, unc=line.area_unc,
                              covariance=fit.covariance)
```

---

## 7.3 Line-by-line activity calculation

```text
function compute_line_activity(peak_area, line_meta, detector_cal, measurement_meta, corrections):
    # Required evaluated decay data
    I_gamma = line_meta.emission_probability
    half_life = line_meta.half_life
    energy = line_meta.gamma_energy

    # Detector-specific calibration
    eff = evaluate_efficiency(detector_cal.efficiency_curve, energy)

    # Basic timing values
    t_live = measurement_meta.live_time
    t_real = measurement_meta.real_time
    t_count = measurement_meta.count_duration
    dt_ref = measurement_meta.time_from_count_to_reference

    C_count = decay_during_count_correction(half_life, t_count)
    C_decay_ref = decay_to_reference_correction(half_life, dt_ref)
    C_dead = compute_dead_time_correction(measurement_meta, corrections)
    C_geom = corrections.geometry_factor_or_1
    C_self = corrections.self_attenuation_factor_or_1
    C_sum = corrections.coincidence_summing_factor_or_1

    A_ref = (peak_area.net / (eff * I_gamma * t_live * C_count)) \
            * C_dead * C_geom * C_self * C_sum * C_decay_ref

    u_A = propagate_uncertainty(
        inputs=[peak_area, eff, I_gamma, half_life, t_live,
                C_count, C_dead, C_geom, C_self, C_sum, C_decay_ref]
    )

    return ActivityResult(activity=A_ref, uncertainty=u_A, reference_time=measurement_meta.reference_time)
```

---

## 7.4 Nuclide-level and detector-level fusion

```text
function combine_activities(activity_results, config):
    grouped_by_nuclide = group_by_nuclide(activity_results)
    fused = []

    for nuclide, line_results in grouped_by_nuclide.items():
        good_lines = reject_inconsistent_lines(line_results, config.line_qc)
        fused_line_activity = weighted_or_robust_mean(good_lines)
        fused.append((nuclide, fused_line_activity))

    return fused

function combine_measurement_group(sample_measurements, config):
    # Normalize all activities to common reference time first
    normalized = [normalize_to_common_reference_time(x) for x in sample_measurements]
    return aggregate_across_detectors(normalized, config.detector_qc)
```

---

## 8. Data structures that must be added

### 8.1 `DecayDataRecord`

Required fields:

- `nuclide`
- `state` (ground / metastable)
- `gamma_energy_keV`
- `energy_unc_keV`
- `emission_probability`
- `emission_probability_unc`
- `half_life`
- `half_life_unc`
- `source_name`
- `source_version`
- `source_url`
- `evaluation_date`
- `priority_rank`

### 8.2 `DetectorCalibrationRecord`

Required fields:

- `detector_id`
- `energy_calibration`
- `resolution_calibration`
- `efficiency_calibration`
- `efficiency_fit_domain`
- `calibration_source_measurements`
- `calibration_date`
- `valid_until`
- `point_source_like_geometry_only` (boolean)

### 8.3 `PeakAreaResult`

Required fields:

- `peak_id`
- `method`
- `classification`
- `roi_left_channel`
- `roi_right_channel`
- `centroid_keV`
- `net_area`
- `net_area_unc`
- `background_model`
- `fit_qc`
- `edge_sensitivity`
- `manual_override`

### 8.4 `ActivityResult`

Required fields:

- `sample_id`
- `detector_id`
- `nuclide`
- `line_energy_keV`
- `activity`
- `uncertainty`
- `reference_time`
- `decay_data_record_id`
- `efficiency_record_id`
- `corrections_applied`
- `acceptance_state`

---

## 9. Practical defaults for v1

These defaults are recommended for the first INL-aligned implementation.

### Peak search defaults

- use a lightly filtered search copy;
- minimum prominence based on local noise estimate;
- ignore energies outside calibrated domain;
- ignore lines below user-configured low-energy cutoff.

### Net-area defaults

- singlet: Covell/local-linear subtraction;
- difficult singlet: adaptive background-anchor mode;
- multiplet: constrained local fit.

### Activity defaults

- first-choice decay data: DDEP/LNHB if present;
- second-choice decay data: ENSDF via NuDat/IAEA LiveChart;
- detector response: local laboratory calibration records only;
- output reference time: reactor shutdown or user-selected reference time.

### Dosimetry-oriented defaults

- sample classes: `co_al_wire`, `ni_wire`, `ag_wire`, `small_activated_sample`;
- report per-line activity, fused nuclide activity, and detector-aggregated activity;
- preserve all per-detector intermediate values.

---

## 10. What is out of scope for this issue set

These items are important, but they should be separate issue groups:

- full ASTM E261 spectral unfolding;
- final conversion from activity to fluence/reaction rate under every ASTM branch;
- full k0-NAA implementation;
- automated coincidence-summing Monte Carlo generation;
- full self-attenuation transport modeling for arbitrary bulk geometries.

The first goal is simpler: **get the peak finding, ROI counting, and activity engine correct and traceable for ASTM/INL-style dosimetry spectra.**

---

## 11. Priority issue order

### Priority 1 — must-have

1. Add `astm_inl_dosimetry` profile
2. Add detector energy / FWHM / efficiency calibration records
3. Add raw-vs-search spectrum split
4. Add FWHM-based ROI proposal
5. Add Covell/local-linear singlet area extraction
6. Add constrained multiplet fitting
7. Add strict decay-data hierarchy
8. Add line-by-line activity calculation
9. Add provenance for every activity result

### Priority 2 — strongly recommended

10. Add difficult-singlet adaptive background mode
11. Add nuclide-level multi-line fusion
12. Add multi-detector aggregation
13. Add detector-to-detector spread QC
14. Add ASTM-style result report template

### Priority 3 — next expansion

15. Add ASTM E262 / E264 / E481 reaction-rate calculators
16. Add IRDFF-II-backed dosimetry conversion layer
17. Add self-attenuation and geometry-transfer plugins

---

## 12. References

**[R1]** ASTM International, **ASTM E261-16(2021)**, *Standard Practice for Determining Neutron Fluence, Fluence Rate, and Spectra by Radioactivation Techniques*. ASTM store summary page. https://store.astm.org/e0261-16r21.html

**[R2]** ASTM International, **ASTM E262-17(2024)e1**, *Standard Test Method for Determining Thermal Neutron Reaction Rates and Thermal Neutron Fluence Rates by Radioactivation Techniques*. ASTM store summary page. https://store.astm.org/e0262-17r24e01.html

**[R3]** ASTM International, **ASTM E481-23**, *Standard Practice for Measuring Neutron Fluence Rates by Radioactivation of Cobalt and Silver*. ASTM store summary page. https://store.astm.org/e0481-23.html

**[R4]** ASTM International, **ASTM E264-25**, *Standard Test Method for Measuring Fast-Neutron Reaction Rates by Radioactivation of Nickel*. ASTM store summary page. https://store.astm.org/e0264-25.html

**[R5]** ASTM International, **ASTM E3376-23**, *Standard Practice for Calibration and Usage of Germanium Detectors in Radiation Metrology for Reactor Dosimetry*. ASTM store summary page. https://store.astm.org/e3376-23.html

**[R6]** ASTM International, **ASTM E181-23**, *Standard Guide for Detector Calibration and Analysis of Radionuclides in Radiation Metrology for Reactor Dosimetry*. ASTM store summary page. https://store.astm.org/standards/e181

**[R7]** ASTM International, **ASTM E844-25**, *Standard Guide for Sensor Set Design and Irradiation for Reactor Surveillance*. ASTM store summary page. https://store.astm.org/e0844-25.html

**[R8]** McCary, K.M., Walker, B.J., Reichenberger, M.A., **Results of Neutron Dosimetry Measurements for the Advanced Test Reactor Cycle 173-C** (INL/RPT--25-84769-Rev000, 2025). OSTI summary page. https://www.osti.gov/biblio/2566754

**[R9]** Advanced Test Reactor Neutron Dosimetry Report (INL/RPT--24-80650-Rev.001, 2024). OSTI summary page. https://www.osti.gov/biblio/2477044

**[R10]** Laboratoire National Henri Becquerel (LNHB), *Nuclear data table / recommended decay data*; public index showing current recommended-data table entries through **Metrologia 63 (2026)**. https://www.lnhb.fr/home/nuclear-data/nuclear-data-table/

**[R11]** IAEA, *LiveChart of Nuclides – Advanced version*. https://www.iaea.org/resources/databases/livechart-of-nuclides-advanced-version

**[R12]** IAEA Nuclear Data Services, *LiveChart of Nuclides*. https://www-nds.iaea.org/livechart/

**[R13]** NNDC, Brookhaven National Laboratory, *NuDat 3 User Guide*. https://www.nndc.bnl.gov/nudat3/guide/

**[R14]** NIST, *XCOM: Photon Cross Sections Database*. https://www.nist.gov/pml/xcom-photon-cross-sections-database

**[R15]** ASTM Work Item **WK91070**, revision of ASTM E261 to update nuclear data to **IRDFF-II** recommended data. https://www.astm.org/membership-participation/technical-committees/workitems/workitem-wk91070

**[R16]** IAEA / IRDFF resources referenced by ASTM reactor-dosimetry standards and work items; see ASTM E264-25 summary note and ASTM WK91070. Primary public context pages include ASTM E264-25 and the ASTM work item above.

