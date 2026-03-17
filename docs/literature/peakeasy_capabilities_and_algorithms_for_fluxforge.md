# PeakEasy (LANL) capabilities and algorithm write-up for FluxForge

## Purpose

This note summarizes what PeakEasy **publicly documents** about its capabilities and what can be **safely inferred** about its algorithms from the LANL public site, release notes, and public literature. The goal is to help implement the **same class of functionality** in FluxForge without pretending we know internals that PeakEasy has not publicly disclosed.

A key constraint is important up front:

- PeakEasy is a **LANL-distributed executable** with government-use licensing restrictions.
- The goal for FluxForge should be to **reproduce useful behavior and workflow patterns**, not to copy proprietary code or claim algorithmic equivalence where the public documentation does not prove it.

## Confidence labels used in this note

- **Documented**: explicitly stated in public LANL pages or public literature.
- **Strong inference**: not stated directly, but strongly implied by the documented behavior.
- **Unknown / not public**: internal details that the public material did not reveal.

---

## 1. What PeakEasy clearly is

### 1.1 Core role

**Documented**

PeakEasy is a **gamma-ray spectroscopy analysis and radionuclide reference tool** intended to assist in rapid radionuclide identification and analysis of gamma-ray spectra. LANL states that it can **read and display spectra from over 200 different file formats**, convert spectra into several popular formats, and search a large nuclide library based on the **ENDF/B-VIII.0 decay library**. LANL also states that PeakEasy **does not perform automated identification**. That last point is very important for FluxForge design: PeakEasy is not presented as a fully automatic nuclide-ID engine. It is a **calibration, fitting, searching, and analyst-assistance tool**. 

### 1.2 Scale of the reference library

**Documented**

PeakEasy’s public site states that the gamma-ray library includes **over 3800 nuclides** from **ENDF/B-VIII.0**, and the broader public-facing feature summary says **over 4000 nuclides and gamma-ray sources** are available in the PeakEasy libraries. The same LANL page states that it also includes gamma rays from **over 200 (n,α) and (γ,α) reactions**. Release notes add that in version 5.x the **X-ray library** was significantly expanded, and a **new neutron capture and scattering library** was added.

### 1.3 Platform and workflow style

**Documented**

PeakEasy is a **Windows-only** application. Public descriptions and later LANL reports frame it as a tool for **interactive analysis** rather than large-scale automated comparison. A 2023 LANL technical report on the GammaSpec package describes PeakEasy as having a **robust and tested feature set for calibrating, identifying, and fitting gamma-ray spectra**, but says its intended use was **not batch processing and automated comparison of large numbers of spectra**.

**Implication for FluxForge**

FluxForge should not treat “PeakEasy-like” as meaning “fully automated radionuclide analysis.” A better design target is:

1. **excellent interactive fitting and reference lookup**, and
2. **optional automation around those tools**, not hidden black-box automation.

---

## 2. PeakEasy capabilities that should be treated as real feature targets

### 2.1 Spectrum ingestion and conversion

**Documented**

PeakEasy can:

- read and display spectra from **200+ file formats**,
- convert spectra to several popular external formats,
- extract **derived/processed spectra** from certain files,
- extract **embedded image data** from certain file formats,
- extract **GPS locations** from files,
- display count-rate information and count-rate chart data.

**FluxForge recommendation**

FluxForge should implement a **reader plugin architecture** rather than a monolithic parser. PeakEasy’s file-format breadth is a workflow advantage as much as an algorithmic one.

### 2.2 Library-assisted reference work

**Documented**

PeakEasy provides:

- searchable nuclide libraries,
- multiple search options,
- user-defined nuclide lists,
- user-editable or user-created mixtures of nuclides,
- daughter/parent relationships in the library tab,
- nuclide age adjustment for daughter-yield calculations.

**FluxForge recommendation**

For NAA and activation work, this is highly relevant. FluxForge should have:

- a decoupled **reference library layer**,
- nuclide mixtures,
- parent-daughter handling,
- age-dependent line-yield evaluation,
- reaction-product libraries for activation and dosimetry workflows.

### 2.3 Calibration support

**Documented**

PeakEasy advertises **easy and rapid energy calibration adjustments using slider bars**. A later LANL report also describes PeakEasy as having a robust feature set for **calibrating** spectra.

**Strong inference**

If calibration can be adjusted live through slider bars and peak fits update in real time, PeakEasy is almost certainly organized around an **interactive calibration state** that drives the displayed line positions and fitting/ROI interpretation.

**FluxForge recommendation**

Implement:

- an interactive energy calibration editor,
- resolution-calibration storage,
- real-time redraw of expected line markers and fit windows after calibration adjustment.

### 2.4 Peak analysis and fitting

**Documented**

The most important public statement is that PeakEasy includes a **multiple gaussian peak analysis engine with real-time fit results**. Release notes also show that:

- highlighted peak information includes **Net Area CPS**,
- PeakEasy has a **Search/Find All Peaks** command,
- ROI fits and **calculated continuums** are displayed,
- after subtracting background, **channel counts may go negative**,
- there is a **Sum/Escape Peak Tool**.

**Strong inference**

These public statements imply that PeakEasy has at least the following internal objects:

- a **peak candidate / highlight layer**,
- an **ROI or local fit window**,
- a **multi-Gaussian fit model**,
- an explicit **continuum/background model** within the fit region,
- net-area computation from fit or ROI outputs,
- some handling of **escape/sum-peak interpretation**.

### 2.5 ROI analysis and batch tools

**Documented**

PeakEasy’s public feature list includes **batch mode processing** for:

- summing,
- file converting,
- appending,
- **region-of-interest analysis**.

Release notes also show:

- records/spectra can be summed in the count-rate chart,
- multiple detectors with different calibrations may be summed,
- there was explicit work to improve handling of differing calibrations in summations.

**FluxForge recommendation**

Separate the workflow into:

- single-spectrum interactive analysis, and
- batch transformations that reuse the same core measurement primitives.

### 2.6 Interactive use rather than hidden automation

**Documented**

PeakEasy explicitly says it **does not perform automated identification**. A 2024 journal paper used PeakEasy as one of several **peak integration** tools alongside Genie and Hypermet PC, reinforcing that it is a credible analysis/fitting tool even when not used as a fully automatic nuclide-identification engine.

**FluxForge recommendation**

Do not force FluxForge into a single “auto ID” identity. Preserve an explicit **analyst-driven mode** where:

- peaks are found/highlighted,
- fitting is interactive,
- nuclide matching is library-assisted,
- final radionuclide conclusions remain inspectable and overridable.

---

## 3. What PeakEasy publicly reveals about its algorithms

## 3.1 Peak fitting model

### What is explicit

**Documented**

PeakEasy uses a **multiple Gaussian peak analysis engine** with **real-time fit results**.

### What this almost certainly means

**Strong inference**

At minimum, the local fit model is something like:

\[
S(E) = B(E) + \sum_{k=1}^{N} A_k \exp\left[-\frac{(E-\mu_k)^2}{2\sigma_k^2}\right]
\]

where:

- \(B(E)\) is a local continuum/background term,
- each Gaussian has amplitude \(A_k\), centroid \(\mu_k\), and width \(\sigma_k\),
- the fit is updated interactively as the user changes the region or calibration.

### What is *not* public

**Unknown / not public**

The public sources do **not** reveal:

- whether widths are independent or constrained/shared,
- whether asymmetric tails are supported,
- whether the continuum is linear, polynomial, step-like, or mixed,
- the optimizer used,
- the weighting model used in the objective function.

### FluxForge recommendation

To reproduce PeakEasy’s publicly visible behavior, implement the following without claiming exact equivalence:

1. **Local multi-Gaussian ROI fitter**.
2. Optional **shared-width or resolution-constrained width** mode.
3. Continuum choices:
   - constant,
   - linear,
   - optional step term,
   - optional low-energy tail term later.
4. Real-time display of:
   - individual component peaks,
   - total fit,
   - continuum,
   - net area,
   - net area CPS,
   - residuals.

---

## 3.2 Peak finding / highlighting

### What is explicit

**Documented**

PeakEasy provides a **Search/Find All Peaks** action and can highlight all peaks in the spectrum.

### What is likely true

**Strong inference**

Because PeakEasy is not advertised as a fully automatic identification engine, the “find all peaks” function is best understood as a **candidate-peak search/highlighting tool**, not as a final radionuclide-decision engine. The public material supports a model where:

1. the software finds candidate local maxima / fit-worthy structures,
2. the user inspects or selects peaks,
3. PeakEasy fits one or more Gaussian components in a selected region,
4. the user compares the fit against library lines.

### What is not public

**Unknown / not public**

The public sources do not disclose:

- whether the search uses derivatives,
- whether it uses matched filters,
- whether it uses second-derivative or top-hat logic,
- whether smoothing is used internally for candidate detection,
- how thresholds are defined.

### FluxForge recommendation

Implement this as a modular search stage, not a hard-coded PeakEasy clone. A strong design would be:

- **candidate search on a working copy** of the spectrum,
- highlight all candidate peaks,
- hand off selected windows to the multi-Gaussian fitter,
- keep raw-spectrum counts for final integration/statistics.

A practical default for FluxForge would still be the tiered approach already described in your existing notes:

- FWHM-based ROI proposal,
- raw-spectrum integration for simple singlets,
- fit escalation for overlaps or structured backgrounds.

This aligns well with your current design direction and gives a PeakEasy-like interactive layer on top of it.

---

## 3.3 Background / continuum handling

### What is explicit

**Documented**

The release notes state that PeakEasy used to display ROI fits that went negative as a zero line, and this was corrected so **fits and calculated continuums** display correctly. Release notes also say that after subtracting background, channel counts are allowed to become **negative**.

### What this implies

**Strong inference**

PeakEasy almost certainly has:

- a **local continuum model** used inside ROI fits,
- a **background-subtracted display mode** or background-subtracted working spectrum,
- numerical handling that preserves signed residual structure rather than clipping at zero.

That is a very good design choice for FluxForge as well.

### FluxForge recommendation

Implement background/continuum logic with three separate concepts:

1. **Measured background spectrum subtraction**,
2. **local ROI continuum model**, and
3. **display residual / signed-channel support**.

Do not collapse all three into one “background” toggle.

---

## 3.4 ROI-level outputs

### What is explicit

**Documented**

PeakEasy explicitly supports **region-of-interest analysis** in batch mode, and the highlighted peak information now includes **Net Area CPS**.

### What this means for FluxForge

FluxForge should treat the following as first-class ROI outputs:

- gross counts,
- net counts,
- net count rate / CPS,
- centroid,
- FWHM or fitted width,
- continuum estimate,
- fit status / fit quality,
- uncertainty.

These outputs should be available both:

- interactively for a selected region, and
- programmatically in batch processing.

---

## 3.5 Nuclide matching and identification

### What is explicit

**Documented**

PeakEasy states that the library is searchable and that it **does not perform automated identification**.

### Best interpretation

**Strong inference**

PeakEasy likely supports **manual or analyst-assisted identification** by overlaying or comparing expected gamma lines from selected nuclides against the calibrated spectrum and fitted peaks.

### FluxForge recommendation

This is a major architectural lesson:

- keep **peak finding/fitting** separate from **nuclide decision logic**,
- keep **nuclide matching** separate from **activity calculation**,
- preserve a mode where the user explicitly chooses candidate nuclides and inspects consistency.

This is especially valuable for activation and NAA work, where a line may be real but **not uniquely diagnostic**.

---

## 3.6 Activity / yield support

### What is explicit

**Documented**

PeakEasy’s public material emphasizes radionuclide libraries, daughter yields, nuclide age, mixtures, and peak net area/CPS, but it does **not** publicly document a full standards-style activity equation or uncertainty model.

### What is likely

**Strong inference**

Because it is a gamma spectroscopy analysis tool with library-assisted nuclide information and fitted peak areas, PeakEasy almost certainly supports at least some form of:

- count-rate comparison to reference lines,
- yield-aware interpretation,
- relative analysis of candidate nuclides,
- possibly activity-style outputs in some workflows.

However, the public documents do **not** reveal the exact correction chain.

### FluxForge recommendation

Do not infer more than is documented. For FluxForge, activity should continue to be implemented using your standards-aligned model from your existing notes:

\[
A = \frac{N_{\text{net}}}{t_{\text{live}}\,\varepsilon(E)\,I_\gamma\,C}
\]

with correction hooks for:

- decay,
- self-attenuation,
- geometry transfer,
- summing,
- rate-related losses,
- background treatment,
- uncertainty propagation.

That is stronger and more transparent than anything publicly documented for PeakEasy.

---

## 4. PeakEasy features that are especially worth copying into FluxForge

## 4.1 Keep analyst control central

PeakEasy’s public positioning strongly suggests a philosophy of:

- help the analyst find peaks quickly,
- fit peaks well,
- expose strong reference data,
- avoid pretending the software alone has solved radionuclide identification.

This is an excellent design pattern for FluxForge.

## 4.2 Implement an interactive local-fit inspector

This is probably the single highest-value PeakEasy-like feature to add if it is not already present. The feature should allow the user to:

- select or auto-propose an ROI,
- add/remove peak components,
- fit multiple Gaussian peaks in the window,
- view net area and CPS in real time,
- see continuum and residuals,
- lock centroids or widths when needed,
- export fit results.

## 4.3 Add a full reference workspace, not just a line table

PeakEasy’s public library functionality is richer than a static line list. FluxForge should have:

- nuclide search by name / Z / A / energy,
- daughter-parent links,
- age-adjusted yields,
- user-defined nuclide sets,
- editable mixtures,
- activation reaction products and not just decay gamma lines.

## 4.4 Add “find all peaks” as a visualization tool

Do not force “find all peaks” to mean “accept all peaks.”

PeakEasy’s public stance supports a design where “find all peaks” means:

- search candidates,
- highlight candidates,
- let the user promote selected candidates into fitted or reported peaks.

That is a safer design for HPGe activation work than auto-accepting every local maximum.

## 4.5 Add robust batch utilities around the same core objects

PeakEasy’s batch functions are practical and worth copying:

- sum spectra,
- append records,
- convert file types,
- run ROI analysis over many spectra.

FluxForge should ensure these are thin wrappers around the same calibrated spectrum and ROI objects used in interactive mode.

## 4.6 Preserve negative values after subtraction

PeakEasy’s release notes show this explicitly. FluxForge should not clamp background-subtracted channel values or residuals at zero. Negative structure is informative for:

- diagnosing over-subtraction,
- checking fit bias,
- showing that a continuum model is wrong,
- validating summed or background-subtracted spectra.

---

## 5. What not to over-claim about PeakEasy

The public sources do **not** justify claiming that PeakEasy publicly documents:

- the exact peak-search filter,
- the exact nonlinear optimizer,
- the exact ROI-boundary rule,
- tail or asymmetry model details,
- the exact uncertainty propagation scheme,
- ISO 11929-style detection-limit implementation,
- automatic radionuclide identification logic.

For FluxForge, these should therefore be treated as **design choices**, not “PeakEasy algorithms.”

---

## 6. Recommended FluxForge implementation plan based on PeakEasy

## 6.1 Layer 1 — reference and ingestion

Implement:

- reader registry for many detector/file formats,
- extraction of metadata such as GPS, detector info, embedded images,
- export/conversion tools,
- summed-spectrum and appended-record workflows.

## 6.2 Layer 2 — interactive analysis surface

Implement:

- interactive energy calibration adjustments,
- peak highlighting / find-all-peaks,
- selectable ROI windows,
- real-time multi-Gaussian fitting,
- continuum display,
- residual display,
- net area and net area CPS display.

## 6.3 Layer 3 — reference-driven interpretation

Implement:

- nuclide library explorer,
- activation-reaction library explorer,
- daughter-parent links,
- age-adjusted gamma-yield calculations,
- user-defined nuclide sets,
- nuclide-mixture builder.

## 6.4 Layer 4 — standards-grade quantification

Keep FluxForge’s own stronger standards path for:

- FWHM-based ROI proposal,
- raw-spectrum net area for singlets,
- multiplet deconvolution when required,
- activity calculation,
- comparator NAA,
- dosimetry outputs,
- uncertainty propagation,
- QC flags.

In other words:

- use PeakEasy as inspiration for the **interactive analyst workflow**,
- use your current standards work as the basis for the **final quantitative engine**.

This hybrid is better than trying to turn FluxForge into a literal imitation.

---

## 7. Minimal pseudocode for a PeakEasy-like interactive layer in FluxForge

```python
class SpectrumSession:
    def __init__(self, spectrum, metadata, energy_cal, resolution_cal, library):
        self.spectrum = spectrum
        self.metadata = metadata
        self.energy_cal = energy_cal
        self.resolution_cal = resolution_cal
        self.library = library
        self.highlighted_peaks = []
        self.selected_roi = None
        self.fit_result = None

    def find_all_peaks(self, search_config):
        working = self.spectrum.copy()
        # Candidate search only; do not treat as final ID.
        self.highlighted_peaks = candidate_peak_search(working, search_config)
        return self.highlighted_peaks

    def adjust_energy_calibration(self, new_cal):
        self.energy_cal = new_cal
        refresh_display_lines(self.library, self.energy_cal)
        if self.selected_roi is not None:
            self.refit_selected_roi()

    def select_roi(self, e_min, e_max):
        self.selected_roi = extract_roi(self.spectrum, self.energy_cal, e_min, e_max)
        return self.selected_roi

    def fit_selected_roi(self, peak_seeds, model_config):
        # PeakEasy-like behavior target: multi-Gaussian + explicit continuum,
        # real-time result updates.
        self.fit_result = fit_multi_gaussian_roi(
            roi=self.selected_roi,
            peak_seeds=peak_seeds,
            model_config=model_config,
            resolution_cal=self.resolution_cal,
        )
        return self.fit_result

    def peak_summary(self):
        if self.fit_result is None:
            return None
        return {
            "centroids_keV": self.fit_result.centroids_keV,
            "fwhm_keV": self.fit_result.fwhm_keV,
            "net_area": self.fit_result.net_area,
            "net_area_cps": self.fit_result.net_area / self.metadata.live_time_s,
            "continuum": self.fit_result.continuum_params,
            "chi2_red": self.fit_result.chi2_red,
        }

    def library_match(self, tolerance_keV):
        if self.fit_result is None:
            return []
        return match_fit_peaks_to_library(
            centroids_keV=self.fit_result.centroids_keV,
            library=self.library,
            tolerance_keV=tolerance_keV,
        )
```

### Underlying fit target

```python
def fit_multi_gaussian_roi(roi, peak_seeds, model_config, resolution_cal):
    # Minimum public-behavior target based on PeakEasy docs:
    # - multiple Gaussian peaks
    # - explicit continuum
    # - real-time fit updates
    #
    # A robust FluxForge implementation can go beyond PeakEasy public docs by
    # supporting shared-width constraints, step continua, and tail terms.

    params0 = initialize_from_seeds(roi, peak_seeds, resolution_cal, model_config)

    def model(ch, params):
        y = continuum_model(ch, params.continuum)
        for pk in params.peaks:
            y += gaussian(ch, pk.area, pk.center, pk.sigma)
        if model_config.use_step:
            y += step_background(ch, params.step)
        if model_config.use_tail:
            y += low_energy_tail(ch, params.tail)
        return y

    result = weighted_least_squares(
        x=roi.channels,
        y=roi.counts,
        model=model,
        params0=params0,
        weights=poisson_like_weights(roi.counts),
        allow_negative_residuals=True,
    )
    return result
```

---

## 8. Final practical takeaways

1. PeakEasy is best understood as an **interactive gamma-spectrum analysis workstation**, not a black-box automated ID package.
2. The most important explicitly documented algorithmic feature is the **multiple Gaussian peak analysis engine with real-time fit results**.
3. Public materials support implementing **peak highlighting, ROI analysis, continuum-aware local fitting, library-assisted matching, and batch utilities**.
4. Public materials do **not** support claiming knowledge of PeakEasy’s exact internal search filters, optimizer, or uncertainty engine.
5. For FluxForge, the best path is to combine:
   - PeakEasy-like **interactive fitting and library workflow**, with
   - your existing **standards-oriented quantification pipeline**.

---

## 9. References

### Public PeakEasy / LANL sources

1. PeakEasy About page, Los Alamos National Laboratory.
2. PeakEasy homepage, Los Alamos National Laboratory.
3. PeakEasy 5.21 release notes (LA-UR-24-27616), Los Alamos National Laboratory.
4. Government Use Notice and Acknowledgment for PeakEasy 5.x, Los Alamos National Laboratory.

### Public literature referencing PeakEasy

5. McGlinchey, D., *The GammaSpec Package for Automated Gamma-ray Analysis (V.1.0)*, LANL / OSTI, 2023.
6. *Determination of fluorine in reference materials by chopped-beam cold neutron prompt gamma-ray activation analysis (CB-CNPGAA)*, Journal of Radioanalytical and Nuclear Chemistry, 2024.

### Alignment with current FluxForge design notes

7. `peak_finding.md`
8. `peak_finding_additions_rafm_naa.md`
