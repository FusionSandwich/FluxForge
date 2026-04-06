# FluxForge NAA / Activation / InterSpec-STAYSL Additions

**Document type:** additive delta only  
**Purpose:** this file adds missing requirements to the existing FluxForge roadmap documents. It does **not** replace or rewrite the current plan files.  
**Intended merge targets:**
- `FluxForge_Final_Additions.md`
- `FluxForge_Additions_v3_Final.md`
- the existing Phase 3B / offline spectroscopy parity tracker

---

## 1. Scope of this additions file

This document exists to add the capabilities that were still missing or not stated explicitly enough after reviewing:

1. the existing FluxForge roadmap and execution-status documents,
2. the official InterSpec public releases page,
3. the official InterSpec tutorials page,
4. the official InterSpec short-usage-videos page, and
5. the STAYSL PNNL public project page and user guide.

This file is therefore a **delta** document. It should be merged into the current roadmap as new workstreams / epics / issues. It should **not** replace the current roadmap or restart numbering for already-completed work.

---

## 2. Governing rule for this delta

### Rule D1 — Keep the current roadmap intact

Anything already present in the existing plan or already implemented in the repository should remain where it is. This file only adds the missing experimental-analysis, NAA, InterSpec-parity, and STAYSL-grade inverse-analysis requirements.

### Rule D2 — FluxForge owns the full experimental side

FluxForge must own the full **experimental analysis** side of the benchmark workflow, including all intermediate data products needed between measured HPGe spectra and the final experimental observables used by the separate model-comparison tool.

### Rule D3 — The separate comparison tool owns the model side

The separate comparison tool should continue to own the modeled / top-half workflow:
- transport-derived flux spectra,
- reaction-rate predictions,
- activation/inventory predictions,
- modeled photon sources,
- modeled HPGe spectra,
- experiment-vs-code comparison logic.

FluxForge must export results in a machine-readable, provenance-rich way that makes this handoff straightforward.

---

## 3. What this file adds at a high level

This delta adds five major workstreams:

1. **Full HPGe activation and NAA quantification inside FluxForge**
2. **Time-aware activity / mass / decay / daughter analysis**
3. **Irradiation planning, count planning, and masking-isotope analysis**
4. **Remaining InterSpec capability parity from official releases/tutorials/videos**
5. **Explicit STAYSL-grade experimental inverse-analysis requirements**

---

## 4. New workstream A — Full activation and NAA analysis inside FluxForge

FluxForge should not stop at peak fitting or activity extraction. For experimental work, it must be able to carry the user through a full activation / NAA workflow inside the same environment.

### A1. Experimental quantity ladder

FluxForge must support the full chain below as first-class internal objects:

`Spectrum -> peak areas -> net count rates -> line activities at count time -> nuclide activity at count time -> activity at arbitrary time -> reaction observable / production estimate -> isotope or element mass estimate -> concentration / comparator result / k0 result`

### A2. NAA quantification modes

FluxForge must support all of the following as explicit, selectable workflows:

1. **Direct activity workflow**
   - determine nuclide activity at count start, count midpoint, count end, EOI, or any arbitrary timestamp;
   - determine line-specific and nuclide-combined activities;
   - propagate counting, efficiency, branching-ratio, decay, and timing uncertainties.

2. **Comparator NAA workflow**
   - unknown vs standard;
   - same irradiation / different irradiation handling;
   - decay correction between irradiation, cooldown, and count times;
   - mass and concentration recovery relative to the comparator.

3. **Relative comparator workflow**
   - comparison against previously characterized in-house references;
   - support for archived standards and calibration history.

4. **k0-NAA workflow**
   - detector/facility characterization inputs;
   - monitor / comparator handling;
   - thermal/epithermal treatment;
   - governed report outputs suitable for audited workflows.

5. **Activation-only workflow**
   - when the user wants isotope production, activity, cooldown behavior, or daughter growth without completing a full elemental NAA reduction.

### A3. Mass and concentration estimation

FluxForge must be able to estimate, for a selected nuclide or analyte:
- mass of the product nuclide at count time,
- mass of the product nuclide at EOI,
- mass of the original target isotope inferred from activation relations,
- elemental concentration in the sample,
- concentration with dilution / sample-mass normalization,
- minimum detectable mass / concentration.

This must support both single-line and multi-line solutions and must explain which assumptions were used.

### A4. Inventory-style experimental object

FluxForge must introduce an internal **experimental inventory object** that stores:
- identified nuclides,
- activities,
- uncertainties,
- timestamps,
- daughter links,
- inferred masses / concentrations,
- sample metadata,
- irradiation metadata,
- geometry / efficiency assumptions,
- selected workflow method.

This object is the main handoff object to the separate model-comparison tool.

---

## 5. New workstream B — Time-aware activity, decay, daughter, and back/forward projection

### B1. Determine activity at EOI and arbitrary times

FluxForge must provide a dedicated **time-state solver** that can:
- solve activity at count start,
- solve activity at count midpoint,
- solve activity at count end,
- back-extrapolate to end of irradiation (EOI),
- project activity forward to any future time,
- account for decay during counting,
- account for parent-daughter ingrowth when relevant,
- support piecewise irradiation / cooldown / count histories.

### B2. Future activity prediction

For any isotope or grouped inventory, FluxForge must predict:
- future activity vs time,
- future count rate vs time,
- best count window for a chosen isotope,
- when a target line becomes cleaner due to interfering isotope decay,
- when activity drops below a detection threshold.

### B3. Back-decay / forward-decay calculator

A dedicated calculator must allow the analyst to provide:
- known activity at time `t1`,
- desired evaluation time `t2`,
- optional parent-daughter chain assumptions,
- optional irradiation history,

and then compute the activity or inventory state at `t2`.

### B4. Daughter-product explorer

FluxForge must have a daughter/ancestor exploration workspace that shows:
- likely parents of an observed nuclide,
- likely daughters of a selected nuclide,
- expected line emergence/disappearance over time,
- decay chain branch fractions,
- which daughter lines are likely to become visible later,
- which apparent peaks may come from ingrowth rather than direct production.

### B5. Time-history visualization

For each selected isotope or inventory, FluxForge must provide plots of:
- activity vs time,
- line count rate vs time,
- saturation fraction vs irradiation time,
- daughter ingrowth vs time,
- time of maximum signal-to-interference ratio,
- time of minimum detection limit.

---

## 6. New workstream C — Irradiation, cooldown, and count optimization

### C1. Irradiation-time optimization

FluxForge must provide an optimization tool that, for a selected isotope / reaction / analyte, can estimate a recommended irradiation time based on:
- half-life,
- expected flux or reaction observable,
- planned cooldown,
- planned count duration,
- detector efficiency,
- sample mass,
- target precision or target MDA,
- expected masking isotopes.

Outputs must include:
- recommended irradiation time,
- recommended cooldown time,
- recommended count time,
- expected activity and count rate at the selected count time,
- saturation fraction,
- diminishing-returns warning,
- dead-time warning.

### C2. Count-time optimization

Given a target isotope and current estimate of activity, FluxForge must recommend the count time needed to reach:
- target relative uncertainty,
- target detection significance,
- target MDA,
- target signal-to-background ratio.

### C3. Multi-objective planner

FluxForge should allow optimization for:
- one isotope,
- a prioritized isotope list,
- one isotope while suppressing interference from another,
- one isotope under a maximum allowable dead-time or dose-rate limit,
- the best schedule for a sample set with limited detector time.

### C4. Irradiation planner report

The planner must emit a report showing:
- assumed nuclear data,
- assumed efficiencies,
- timing assumptions,
- optimization objective,
- recommended irradiation/cool/count schedule,
- sensitivity to uncertainty in flux, efficiency, and background.

---

## 7. New workstream D — Masking isotope and interference analysis

### D1. Masking-isotope engine

FluxForge must have a tool that answers:
- what isotopes are most likely masking my selected isotope?
- which specific gamma lines are creating the masking?
- is the interference from direct line overlap, Compton continuum, escape peaks, annihilation peaks, sum peaks, x-rays, or background features?

### D2. Ranked interference report

For a selected isotope and line, FluxForge must produce a ranked list of likely interferers with:
- interfering nuclide,
- interfering line energy,
- expected overlap severity,
- whether the interferer is direct or indirect,
- expected decay-away time,
- recommended alternate lines,
- recommended better count time if waiting helps.

### D3. Time-dependent de-masking

The masking engine must be time-aware. It must estimate whether an interfering isotope will decay away faster or slower than the target isotope and recommend the optimal count window for line separation.

### D4. Alternative-line recommendation

When a preferred line is masked, FluxForge must suggest:
- alternate lines of the same nuclide,
- alternate NAA route for the analyte when applicable,
- whether deconvolution is likely reliable,
- whether the sample likely needs re-counting under different geometry / time / shielding conditions.

### D5. Interference provenance

All reports must distinguish between:
- proven overlap from fitted peaks,
- likely overlap from library proximity,
- Compton/background suspicion,
- user-forced interpretation.

---

## 8. New workstream E — Remaining InterSpec parity additions from official review

This section focuses only on capabilities that should be explicitly added to the FluxForge plan after reviewing the official InterSpec public releases/tutorials/videos pages. It is not a promise to clone every UI detail or every historical bugfix. The goal is to cover the **user-facing capability families**.

### E1. Relative-efficiency / isotopics workflows

FluxForge must explicitly plan for two distinct relative-efficiency style workflows:

1. **Isotopics by peaks**
   - relative-activity / enrichment inference from fitted peaks,
   - no absolute detector response required,
   - suitable for uranium/plutonium enrichment and general relative activity problems.

2. **Isotopics by nuclides**
   - constrained multi-peak / multi-nuclide fit,
   - shared FWHM / branching-ratio / line-relationship enforcement,
   - explicit support for uranium and plutonium enrichment and isotopics,
   - generalizable to other nuclide families.

### E2. Detection-limit workspaces

FluxForge must explicitly include two detection-limit tools:

1. **Simple ROI detection-limit tool**
   - quick limit for one ROI / one line.

2. **Advanced detection-limit tool**
   - multiple ROIs,
   - activity limit or detection-distance limit,
   - optional shielding,
   - deconvolution-aware peak/data-shape treatment.

### E3. Detector response function lifecycle

The plan must explicitly include:
- guided detector-response-function creation from known standards,
- response-function fit refinement,
- response-function storage and reuse,
- import/export of detector response functions,
- automatic association of preferred response function to detector model or serial number,
- quick-view detector-response summary cards.

### E4. File query and archive workbench

The plan must explicitly include a dedicated file-query tool that can:
- recursively search spectrum directories,
- filter by detector model, filename, RIID result, CPS, live/real time, dates, GPS, and metadata,
- cache results for fast repeated search,
- bulk-open or bulk-stage files for comparison,
- support archive/reanalysis of older campaigns.

### E5. Session resume / auto-store

FluxForge must support:
- automatic storage of analyst work state,
- reopening a spectrum and resuming prior analysis,
- versioned saved work states,
- explicit session provenance inside `.ffs` and exported files.

### E6. Batch analysis from exemplar workflow

FluxForge must support batch processing in which the analyst:
- fully configures one exemplar spectrum,
- saves the analysis state,
- applies the same peak/activity/shielding/ID settings across many similar spectra,
- reviews per-spectrum deviations rather than starting from scratch each time.

### E7. Reference photopeak and manual-first workbench

The plan must explicitly keep InterSpec's strong manual-analysis philosophy:
- rich reference photopeak overlaying,
- fast manual assignment,
- manual correction always available after automation,
- right-click peak editing,
- explicit peak-editor workspace,
- direct line-to-peak and peak-to-line navigation.

### E8. Dose and attenuation calculators

Add explicit tools for:
- dose-rate calculations,
- source-to-dose / dose-to-distance estimation,
- gamma/x-ray attenuation cross-section lookup / calculator,
- quick shielding what-if calculations for experimental interpretation.

### E9. Map and geospatial review

The current GPS/map direction should be extended to include:
- review of mapped measurements,
- selecting visible points and summing them into foreground/background/secondary spectra,
- route / campaign review for field spectroscopy datasets.

### E10. Math / command terminal

FluxForge should include a scriptable analysis terminal inside the GUI for quick calculations and scripted actions on the active spectrum, rather than forcing every advanced user action into the full external CLI.

### E11. Spectrum export and portable encapsulation

The plan should explicitly include:
- flexible export of spectrum plus analysis state,
- export subsets / roles / selected spectra,
- export to portable machine-readable formats,
- optional QR/URI style compact sharing for small spectra or lightweight review states.

### E12. Example-problem parity for capability validation

The plan must include regression/demo problems analogous to the official InterSpec tutorial examples, including:
- nuclide ID and quantification,
- energy calibration,
- detector-response creation,
- buried source / shielding depth style problems,
- uranium enrichment and mass,
- trace contamination / trace sources,
- relative efficiency analysis,
- batch analysis.

---

## 9. New workstream F — Explicit STAYSL-grade experimental inverse-analysis requirements

FluxForge does **not** need to become a clone of the STAYSL PNNL toolchain, but the plan must explicitly cover the experimental-analysis capabilities that users expect when they say they want STAYSL-class functionality on the experimental side.

### F1. Corrected activation-rate engine

FluxForge must explicitly compute and store corrected activation rates using corrections for the relevant experimental phenomena, including as applicable:
- decay during irradiation,
- decay during cooldown,
- decay during counting,
- neutron self-shielding / cover / sample corrections,
- burn-up corrections where required,
- flux-history / non-constant irradiation corrections,
- geometry- or monitor-specific correction factors.

### F2. SigPhi-style experimental workspace

FluxForge should include a workspace analogous in function to the role of a SigPhi calculator:
- convert measured activities into corrected reaction observables,
- show every correction separately,
- expose uncertainty contributions by source,
- support repeated monitors / replicate samples,
- flag physically inconsistent monitors.

### F3. Generalized least-squares spectral adjustment

The plan must explicitly include a STAYSL-grade generalized least-squares adjustment mode that uses:
- measured reaction observables,
- prior flux spectrum,
- activation cross sections,
- covariance information,
- activity covariance,
- flux covariance,
- cross-section covariance,
- chi-squared minimization with transparent diagnostics.

This requirement is stronger than generic unfolding. It must be treated as a governed inverse-analysis workflow.

### F4. Covariance-aware outputs

FluxForge must explicitly plan to output:
- adjusted flux spectrum,
- adjusted flux covariance matrix,
- broad-group summaries,
- reaction rates,
- spectral-averaged activation cross sections,
- residuals and chi-squared diagnostics,
- monitor leverage / influence diagnostics.

### F5. STAYSL-style run families

The plan should explicitly include the equivalent of distinct run families for:
- normal spectral-adjustment runs,
- integral-flux / threshold-oriented runs,
- activity-driven convenience runs,
- monitor-consistency / sensitivity runs.

### F6. Library / reaction infrastructure

FluxForge must plan for:
- reaction library management,
- covariance library management,
- group-structure management,
- support for modern IRDFF-derived or equivalent data sources,
- explicit provenance of the nuclear-data set used.

### F7. SHIELD / self-shielding equivalent requirements

FluxForge must plan for a self-shielding correction capability for relevant monitors and sample geometries, including:
- wire / foil dimensions,
- isotropic vs beam-like assumptions,
- energy-dependent correction factors,
- stored correction-factor provenance.

### F8. BCF / flux-history equivalent requirements

FluxForge must plan for explicit flux-history correction handling:
- constant irradiation,
- piecewise constant irradiation,
- reactor power history import,
- effective full-power style corrections,
- correction audit trail.

### F9. Experimental consistency and monitor diagnostics

The STAYSL-grade plan must include monitor diagnostics such as:
- outlier monitor detection,
- inconsistent reaction set detection,
- leave-one-out adjustment tests,
- monitor contribution / leverage ranking,
- correlation visualizations.

---

## 10. New workstream G — Experimental handoff contract to the separate comparison tool

FluxForge must emit a stable machine-readable handoff bundle for the separate model-comparison tool.

### G1. Required export object

For each analyzed sample / campaign, FluxForge must export:
- sample metadata,
- detector metadata,
- count geometry,
- energy / efficiency calibration version,
- fitted peaks,
- line activities,
- nuclide activities,
- EOI activities,
- corrected activation rates,
- inferred inventory / masses / concentrations,
- uncertainty decomposition,
- daughter-chain assumptions,
- selected NAA workflow mode,
- selected inverse-analysis mode,
- nuclear-data provenance,
- all timestamps,
- masking/interference report,
- planned-vs-actual irradiation/cool/count schedule,
- optional adjusted flux spectrum and covariance.

### G2. Export formats

At minimum support:
- JSON bundle for direct tool ingestion,
- CSV tables for human inspection,
- N42-compatible exports where appropriate,
- report PDF/HTML for audit trail.

### G3. Comparison-readiness flag

Each export bundle should carry machine-readable readiness flags such as:
- `peak_fit_complete`,
- `activity_complete`,
- `eoi_backcalc_complete`,
- `naa_quant_complete`,
- `inverse_adjustment_complete`,
- `comparison_ready`.

---

## 11. What to add to the GitHub tracker

This should be opened as one new epic plus child issues rather than a new full replacement roadmap.

## Epic E12 — NAA / Activation Experimental Analysis and InterSpec-STAYSL Delta

### Child issue group A — NAA core
1. Add experimental inventory object and arbitrary-time activity solver
2. Add EOI back-extrapolation and future-activity projection workspace
3. Add comparator NAA workflow
4. Add k0-NAA governed workflow
5. Add mass / concentration inference engine
6. Add daughter / ancestor explorer and time-evolution plots

### Child issue group B — planning and masking
7. Add irradiation/cool/count optimization tool
8. Add masking-isotope ranking engine
9. Add alternative-line recommendation and de-masking planner
10. Add count-time / MDA optimizer

### Child issue group C — InterSpec parity additions
11. Add explicit relative-efficiency / isotopics-by-peaks workspace
12. Add constrained isotopics-by-nuclides workspace
13. Add advanced detection-limit workspace
14. Add detector-response lifecycle manager
15. Add file-query / archive / campaign search tool
16. Add session auto-store / resume / exemplar batch workflow
17. Add dose / attenuation / quick shielding calculators
18. Add in-GUI math/command terminal
19. Add tutorial/example parity dataset pack and regression workflows

### Child issue group D — STAYSL-grade inverse analysis
20. Add corrected activation-rate engine with explicit correction breakdown
21. Add SigPhi-style workspace for corrected reaction observables
22. Add generalized least-squares spectral-adjustment mode
23. Add covariance-aware output bundle and diagnostics
24. Add self-shielding / flux-history correction modules
25. Add monitor influence / leave-one-out / consistency diagnostics

### Child issue group E — handoff and reporting
26. Add comparison-tool handoff JSON schema
27. Add comparison-readiness validation checks
28. Add NAA / inverse-analysis report templates
29. Add batch/campaign aggregate export for large irradiation studies

---

## 12. Merge guidance against the current roadmap

### Merge rule M1
Do **not** replace the current Phase 3B plan. Instead:
- attach this delta to Phase 3B,
- use it to expand `3.17+` child issues,
- add any overflow work as a dedicated follow-on milestone after current offline-parity closure if needed.

### Merge rule M2
Features already clearly present in the current plan / repo should not be restated as fresh roadmap promises. This file is for the missing requirements only.

### Merge rule M3
When a capability exists in InterSpec but is not central to HPGe activation / NAA, it should still be captured if it materially improves analyst throughput, experimental provenance, or reusable batch analysis.

### Merge rule M4
When a capability exists in STAYSL as part of a multi-tool suite, FluxForge should implement the **experimental-analysis function**, not necessarily the exact historical program boundaries.

---

## 13. Definition of done for this additions file

This delta is considered properly merged only when the main roadmap explicitly includes:

- EOI and arbitrary-time activity solving,
- future activity and daughter forecasting,
- irradiation/cool/count optimization,
- masking-isotope ranking,
- full comparator / k0 / activation analysis workflows,
- relative-efficiency / isotopics-by-peaks and isotopics-by-nuclides workflows,
- advanced detection-limit workspace,
- detector-response lifecycle management,
- file-query/archive/search workflow,
- session auto-store / resume / exemplar batch analysis,
- dose / attenuation calculators,
- STAYSL-grade corrected activation-rate workflow,
- generalized least-squares covariance-aware spectral adjustment,
- self-shielding and flux-history correction handling,
- machine-readable experimental handoff bundles for the comparison tool.

