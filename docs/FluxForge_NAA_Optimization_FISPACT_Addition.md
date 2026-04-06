# FluxForge - NAA Activation, FISPACT-Style Analytics, and Irradiation Optimization Addition

## Purpose

This document is an **additive roadmap supplement**. It does **not** replace:

- `FluxForge_Final_Additions.md`
- `FluxForge_Additions_v3_Final.md`
- `ROADMAP_EXECUTION_STATUS.md`
- the prior additive NAA / InterSpec / STAYSL supplement

It extends the experimental-analysis scope of FluxForge so the program can complete the **bottom half** of the experiment-code benchmark workflow for HPGe activation and NAA work, while still exporting machine-readable outputs to the separate model-comparison tool.

This supplement focuses on four missing capability families:

1. **Inventory-at-irradiation and arbitrary-time activity analytics**
2. **FISPACT-style post-irradiation observables, plots, and uncertainty-aware trend analysis**
3. **Line masking / interference intelligence and second-irradiation optimization**
4. **Dose-oriented shutdown / long-term decay studies for microreactor and fusion workflows**

---

## Governing Rule for This Addition

FluxForge remains the **experimental and inventory-interrogation environment**. It may perform:

- spectrum reduction
- calibration
- peak fitting
- nuclide identification
- activity extraction
- decay correction
- activation / NAA inference
- inventory evolution and observables analysis
- cooldown / count planning
- second-irradiation experimental optimization
- uncertainty propagation
- standards-aware reporting

The separate companion tool remains the **model-side environment** for:

- transport-side source terms
- neutron/photon source prediction from simulation
- code-to-code and code-to-experiment comparison
- full modeled top-half workflow orchestration

FluxForge must therefore export every experimental product in a form that the comparison tool can consume without re-parsing GUI state.

---

## 1. New Top-Level Capability Goal

Add a new additive goal to the offline-parity / NAA roadmap:

### Phase 3N - Activation, NAA, and Irradiation Optimization

This phase begins **after** the current formal next step (`3.17`) and runs as a structured extension of the existing Phase 3B parity work.

It has eight submodules:

- **3N.1** Inventory-at-irradiation reconstruction
- **3N.2** Time-evolution analytics and Bateman plotting
- **3N.3** FISPACT-style observables and plot suite
- **3N.4** Masking / line-interference analysis
- **3N.5** NAA mass / concentration inference
- **3N.6** Irradiation / cooldown / count optimization
- **3N.7** Second-irradiation scenario planner
- **3N.8** Machine-readable export contract for downstream comparison tools

---

## 2. Data-Model Additions Required

The current activity and decay workflow is not enough by itself. Add the following analysis objects.

### 2.1 `IrradiationSchedule`

Represents one or more irradiation and cooldown segments.

```python
@dataclass
class IrradiationSegment:
    duration_s: float
    flux_value: float | None
    flux_spectrum_id: str | None
    power_fraction: float | None
    label: str

@dataclass
class CooldownSegment:
    duration_s: float
    label: str

@dataclass
class CountSegment:
    live_time_s: float
    real_time_s: float
    detector_id: str | None
    geometry_id: str | None
    label: str

@dataclass
class IrradiationSchedule:
    irradiation_segments: list[IrradiationSegment]
    cooldown_segments: list[CooldownSegment]
    count_segments: list[CountSegment]
    reference_time_mode: str  # "EOI" | "count_start" | "count_end"
```

### 2.2 `InventoryState`

```python
@dataclass
class InventoryState:
    nuclide: str
    atoms: float
    atoms_sigma: float | None
    activity_bq: float
    activity_sigma_bq: float | None
    mass_g: float | None
    mass_sigma_g: float | None
    time_s: float
    source: str  # measured | inferred | propagated
```

### 2.3 `ObservableTimeSeries`

```python
@dataclass
class ObservablePoint:
    time_s: float
    value: float
    sigma: float | None

@dataclass
class ObservableTimeSeries:
    name: str
    units: str
    points: list[ObservablePoint]
    contributors: list[str] | None
```

### 2.4 `LineMaskingResult`

```python
@dataclass
class MaskingCandidate:
    interfering_nuclide: str
    interfering_line_keV: float
    target_line_keV: float
    delta_keV: float
    overlap_score: float
    expected_counts: float
    expected_counts_sigma: float | None
    continuum_score: float
    pileup_score: float
    rank_score: float

@dataclass
class LineMaskingResult:
    target_nuclide: str
    target_line_keV: float
    candidates: list[MaskingCandidate]
```

### 2.5 `OptimizationScenario`

```python
@dataclass
class OptimizationScenario:
    isotope_of_interest: str
    candidate_lines_keV: list[float]
    irradiation_time_grid_s: list[float]
    cooldown_time_grid_s: list[float]
    count_time_grid_s: list[float]
    optional_second_irradiation_grid_s: list[float]
    objective: str
    dose_constraints: dict[str, float]
    deadtime_limit: float | None
    min_detectability_margin: float | None
```

---

## 3. Inventory-at-Irradiation and Arbitrary-Time Activity

### 3N.1 Scope

FluxForge must be able to determine or infer:

- activity at **count start**
- activity at **count end**
- activity at **end of irradiation (EOI)**
- activity at any arbitrary future or past time point
- atoms and mass of the isotope at those same time points
- likely daughter inventory at those same time points

### 3N.2 Required capabilities

#### A. EOI back-calculation engine
For each fitted line or multi-line nuclide result, FluxForge shall back-propagate activity from count time to EOI using the known schedule and Bateman treatment.

Outputs:

- `activities_at_irradiation.csv`
- `activities_at_count_start.csv`
- `activities_at_count_end.csv`
- `inventory_timeseries.csv`

Minimum columns for `activities_at_irradiation.csv`:

```text
sample_id, nuclide, line_keV, activity_eoi_bq, sigma_bq, relative_sigma,
count_start_time_s, count_live_time_s, inferred_from_lines,
branching_ratio_used, efficiency_used, self_attenuation_factor,
coincidence_correction_factor, decay_correction_factor, provenance_tag
```

#### B. Arbitrary-time calculator
Add a dedicated panel:

**Tab: `Inventory / Time Evolution`**

User controls:
- reference nuclide
- time origin selector (`EOI`, `count start`, `count end`, custom)
- target time list or range
- show atoms / activity / mass / concentration / dose / heat

This panel must calculate forward and backward in time for a chosen inventory state.

#### C. Daughter / ancestor explorer
For any selected nuclide, FluxForge must display:

- immediate parents
- immediate daughters
- dominant pathway from parent to target
- dominant downstream daughters after chosen cooldown
- the fractions of total target activity arising from each parent branch when relevant

---

## 4. Bateman Plots, Half-Life Decay Plots, and Time-Series Visuals

### 4.1 Mandatory plots

Add a plot library under a new package:

```text
src/fluxforge/plots/
    inventory_plots.py
    dose_plots.py
    optimization_plots.py
    uncertainty_plots.py
```

The following plot types are mandatory.

#### A. Bateman chain plots
Purpose: show parent-daughter-granddaughter evolution in time.

Required modes:
- linear y-axis
- log y-axis
- absolute activity
- fractional contribution to total chain activity

#### B. Half-life decay plots
Purpose: show the decay of one selected nuclide or a chosen set of nuclides.

Required overlays:
- measured point(s)
- fitted decay curve
- uncertainty band
- half-life annotation
- marker at EOI / count start / count end / selected future time

#### C. Inventory evolution plots
For one sample, plot any of:
- atoms
- activity
- mass
- concentration
versus time.

#### D. Stacked dominant-contributor plots
For a chosen observable, show total plus top N contributors over cooling time.

Observables must include:
- total activity
- decay heat
- gamma dose rate
- beta heat
- gamma heat
- ingestion hazard index
- inhalation hazard index
- clearance index

#### E. Scenario-comparison plots
Overlay multiple irradiation / cooldown / count scenarios on one canvas.

Examples:
- 30 min vs 2 h irradiation
- 2 h vs 24 h cooldown
- one irradiation vs two-pulse irradiation
- Co-60 target line vs alternative Eu-152 line

### 4.2 Plot export requirements

Every plot must export to:
- PNG
- SVG
- PDF
- CSV of plotted data

All plot exports must include provenance metadata in a sibling JSON file.

---

## 5. Uncertainty Requirements

### 5.1 Uncertainty must never be optional in the underlying engine

The GUI may hide uncertainty bands in Simple mode, but the engine must always carry uncertainty fields wherever available.

### 5.2 Minimum propagated uncertainty sources

For activation and NAA work, propagate at least:

- counting statistics
- background subtraction uncertainty
- calibration uncertainty (energy / FWHM where relevant)
- efficiency-fit uncertainty
- branching-ratio uncertainty when available
- half-life uncertainty when available
- coincidence / correction-factor uncertainty when provided
- sample mass uncertainty
- irradiation-time uncertainty
- cooldown-time uncertainty
- flux or comparator uncertainty for NAA inference
- covariance-aware pathway or inventory uncertainty where library support exists

### 5.3 Required displays

Every activity and inventory result table needs columns for:

- central value
- absolute sigma
- relative sigma
- 95% expanded interval (optional but supported)
- dominant uncertainty source list

### 5.4 Mandatory uncertainty plots

Add:
- shaded confidence bands on decay / dose / heat curves
- tornado chart of uncertainty-source importance for one selected result
- time-dependent relative uncertainty plot

---

## 6. FISPACT-Style Activity Test, Plot Suite, and Parity Features

FluxForge does not need to become FISPACT. But it **does** need to support the analysis and visualization patterns that experimental users expect when reviewing activation and decay behavior.

### 6.1 Add a new validation epic

### Epic: `E12 - Activation and Inventory Analytics Parity`

This epic covers:
- inventory time evolution
- FISPACT-style observables
- dominant-contributor plots
- uncertainty-aware trend plots
- parity fixtures for activation test cases

### 6.2 Minimum FISPACT-style observable set

Implement time-dependent calculation and plotting for:

- total activity
- decay heat
- gamma dose rate
- beta heat
- gamma heat
- ingestion dose metric
- inhalation dose metric
- clearance index
- dominant-nuclide lists for each quantity

### 6.3 Mandatory FISPACT-style plot families

#### Family A - Time decay curves
Generate curves of each observable versus cooling time.

#### Family B - Dominant-nuclide contribution curves
For a selected observable, plot the top contributors versus cooling time.

#### Family C - Importance-diagram analogues
Add a FluxForge experimental-side version of an importance diagram.

For measured/inferred inventories, this means:
- x-axis: cooling time
- y-axis: selected observable or line-of-interest ranking space
- color / area: dominant radionuclide or contributor fraction

Where spectrum families or scenario grids are available, also allow:
- x-axis: irradiation time
- y-axis: cooling time
- color: best line / dominant masker / dominant dose contributor

#### Family D - Nuclide-map style displays
Create a nuclide chart overlay view for inferred inventories:
- nuclides present by Z/N
- color by activity or contribution
- time slider to animate evolution

### 6.4 Activity test harness

Add a dedicated `tests/activation_inventory/` suite.

Required tests:

1. **Single nuclide pure decay test**
   - verifies analytical exponential decay

2. **Two-member Bateman chain test**
   - verifies parent-daughter growth and decay

3. **Multi-member chain test**
   - verifies numerical solver and uncertainty handling

4. **EOI reconstruction test**
   - synthetic count-time activity back to known EOI

5. **Observable regression test**
   - activity / heat / gamma dose rate trend regression

6. **CSV export test**
   - verifies `activities_at_irradiation.csv` and time-series outputs

7. **Plot smoke tests**
   - ensures each plot family renders from fixture data

### 6.5 Benchmark-parity notebook set

Create `examples/activation_inventory/` notebooks demonstrating:
- simple activation and decay timeline
- EOI reconstruction
- dominant-nuclide evolution
- shutdown to 100-year dose trend
- two-irradiation optimization sweep

---

## 7. Masking-Isotope and Line-Interference Analysis

### 7.1 Goal

Given a target isotope and candidate line, FluxForge must determine which nuclides are most likely to mask that line under the inferred inventory and counting geometry.

### 7.2 Ranking engine

Add a `MaskingAnalyzer` that ranks masking candidates by:

- photopeak overlap in energy space
- expected interfering counts in ROI
- Compton continuum burden near the target line
- escape / sum / pileup plausibility
- relative detector resolution at that energy
- decay timing consistency with the chosen cooldown
- expected abundance / activity of the interferer

### 7.3 Required outputs

#### Panel: `Line Interference / Masking`
For a chosen target isotope + line, show:

- likely maskers ranked table
- overlap score
- alternative cleaner lines for the same nuclide
- recommended cooldown changes that reduce masking
- recommended second irradiation schedule if one is allowed

#### CSV export
`masking_candidates.csv`

Columns:
```text
sample_id,target_nuclide,target_line_keV,masking_rank,
interfering_nuclide,interfering_line_keV,delta_keV,
overlap_score,expected_interfering_counts,continuum_score,
pileup_score,recommended_action
```

### 7.4 Required plots

- ROI overlay with all candidate masking lines
- masking score bar chart
- cooldown sensitivity plot: masking score vs cooldown time
- line-choice plot: signal-to-mask ratio for all candidate lines of the target nuclide

---

## 8. Full Activation and NAA Inference Inside FluxForge

### 8.1 Scope

FluxForge must support inference of:
- initial isotope mass
- initial element mass
- concentration in sample
- future activity
- future detectable line strength
- daughter products likely to dominate later times

### 8.2 Two NAA workflow families

Support both:

#### A. Comparator / relative NAA workflow
Use standards or co-irradiated comparators.

#### B. Library / absolute activation workflow
Use cross sections, decay constants, irradiation schedule, and measured activities.

### 8.3 Required calculations

For each target isotope / element:
- measured activity at count time
- inferred activity at EOI
- saturation-corrected production estimate
- inferred atoms at EOI and at irradiation start when applicable
- inferred mass and concentration
- future activity at selected times
- expected line counts at selected future count times

### 8.4 Result tables

Add a new `NAA Results` workspace table with:

```text
element,isotope,line_keV,activity_count_bq,activity_eoi_bq,
atoms_eoi,mass_eoi_g,concentration_ppm,concentration_sigma,
future_activity_bq,target_time_s,dominant_daughters,notes
```

### 8.5 Tutorials to add

- comparator NAA walkthrough
- absolute activation walkthrough
- isotope mass from measured activity walkthrough
- daughter-growth interpretation walkthrough
- selecting the best line for a target isotope walkthrough

---

## 9. Irradiation / Cooldown / Count Optimization

### 9.1 Purpose

FluxForge should not only analyze what happened. It must also answer:

- how long should I irradiate?
- how long should I cool?
- how long should I count?
- which line should I use?
- would a second irradiation help?
- which schedule minimizes dose while preserving detectability?

### 9.2 Optimization objectives

Support at least these objective modes:

1. maximize target counts
2. maximize signal-to-background
3. maximize signal-to-mask ratio
4. minimize relative uncertainty on target activity
5. minimize required count time for fixed detection margin
6. minimize shutdown dose rate while keeping target detectable
7. maximize separation between target and masking isotopes
8. optimize for a chosen future endpoint (`shutdown`, `1 d`, `1 wk`, `1 y`, `100 y`, custom)

### 9.3 Parameter sweep engine

Add a grid / sweep engine that can vary:

- irradiation time
- cooldown time
- count time
- target isotope
- target line
- second irradiation duration
- delay between irradiations
- assumed flux or power state

### 9.4 Required visual outputs

#### A. Heatmaps
- target detectability margin vs irradiation time and cooldown time
- target uncertainty vs cooldown time and count time
- shutdown dose vs irradiation time and cooldown time
- 100-year dose vs irradiation time and material choice / impurity case

#### B. Pareto plots
For multi-objective planning:
- target detectability vs shutdown dose
- target uncertainty vs total campaign time
- signal-to-mask ratio vs total dose burden

#### C. Recommended-schedule summary card
For one selected isotope, display:
- best schedule under the chosen objective
- best alternative schedules
- why they rank well
- what isotope dominates the dose penalty
- what isotope dominates the masking penalty

### 9.5 CSV outputs

- `optimization_grid.csv`
- `recommended_schedules.csv`
- `dose_endpoints.csv`

---

## 10. Second-Irradiation Planner

### 10.1 Why this matters

Many activation / NAA workflows benefit from a second irradiation or staged irradiation strategy to improve detectability, separate half-lives, or reduce masking.

### 10.2 Required capabilities

FluxForge must compare:
- single irradiation
- two-pulse irradiation
- same total irradiation time split across pulses
- first irradiation optimized for one isotope, second for another

### 10.3 Required analysis outputs

For each candidate two-pulse plan, calculate:
- target activity at first count
- target activity at second count
- masking score at both counts
- total shutdown dose after each pulse
- best count windows for short-, medium-, and long-lived products

### 10.4 Required visualizations

- pulse timeline view
- target-vs-mask evolution after pulse 1 and pulse 2
- count-window recommendation chart
- schedule comparison table sorted by objective score

---

## 11. Shutdown, 100-Year, and Long-Term Dose Studies

### 11.1 Required dose endpoints

FluxForge must support dose / hazard review at:
- shutdown
- 1 hour
- 1 day
- 1 week
- 1 month
- 1 year
- 10 years
- 100 years
- custom endpoints

### 11.2 Required observables at those endpoints

- gamma dose rate
- total activity
- decay heat
- dominant radionuclides
- clearance index
- ingestion metric
- inhalation metric

### 11.3 Domain presets

Add presets for:
- `Microreactor quick-look`
- `Fusion materials quick-look`
- `Activation experiment quick-look`

Each preset seeds sensible endpoint lists and report sections.

### 11.4 Long-term plot suite

Required plots:
- log-time dose-rate curve from shutdown to 100 years
- stacked dominant-dose-contributor plot over long cooling time
- activity / dose comparison across materials or impurity assumptions
- endpoint table with traffic-light severity tags

---

## 12. Export Contract for the Companion Tool

### 12.1 Rule

No GUI-only result is acceptable. Every final result used for model comparison must export to a stable machine-readable form.

### 12.2 Required export bundle

Add a new export mode:

### `Export -> Benchmark Experimental Bundle (.ffexp)`

Bundle contents:
- `metadata.json`
- `activities_at_irradiation.csv`
- `activities_at_count.csv`
- `inventory_timeseries.csv`
- `dominant_contributors.csv`
- `dose_endpoints.csv`
- `masking_candidates.csv`
- `optimization_grid.csv` (if optimization run)
- `recommended_schedules.csv` (if optimization run)
- `plots_manifest.json`
- exported plot CSVs / images

### 12.3 Metadata requirements

`metadata.json` must include:
- sample identifiers
- detector / geometry identifiers
- calibration provenance
- efficiency provenance
- correction factors
- schedule definition
- uncertainty model flags
- method selections used in each step
- standards mode status
- timestamp and FluxForge version

---

## 13. Concrete Issue Seed Set

Open the following additive issues after `3.17` is started.

| ID | Title | Area | Priority |
|---|---|---|---|
| 3N.1 | Implement `IrradiationSchedule`, `InventoryState`, and time-evolution dataclasses | area/core | p0 |
| 3N.2 | Implement EOI back-calculation engine and `activities_at_irradiation.csv` export | area/core | p0 |
| 3N.3 | Add Inventory / Time Evolution workspace with arbitrary-time solver | area/gui | p0 |
| 3N.4 | Add Bateman chain plots and half-life decay plots with uncertainty bands | area/gui | p0 |
| 3N.5 | Implement dominant-contributor and FISPACT-style observable curves | area/core | p0 |
| 3N.6 | Add nuclide-map and importance-diagram style visualizations | area/gui | p1 |
| 3N.7 | Implement masking / line-interference ranking engine | area/core | p0 |
| 3N.8 | Add masking GUI panel and CSV export | area/gui | p0 |
| 3N.9 | Implement comparator / relative NAA mass and concentration inference | area/core | p0 |
| 3N.10 | Implement absolute activation inference workflow | area/core | p1 |
| 3N.11 | Add optimization sweep engine for irradiation / cooldown / count time | area/core | p0 |
| 3N.12 | Add optimization heatmaps, Pareto views, and recommended-schedule cards | area/gui | p0 |
| 3N.13 | Implement second-irradiation planner and pulse timeline views | area/core | p1 |
| 3N.14 | Add shutdown-to-100-year dose endpoint suite and reports | area/core | p0 |
| 3N.15 | Add `.ffexp` benchmark experimental export bundle | area/io | p0 |
| 3N.16 | Add activation-inventory regression fixtures and plot smoke tests | area/testing | p0 |

---

## 14. Tutorial / Documentation Additions

Add the following tutorials to the docs set:

1. **Back-calculate activity to EOI**
2. **Create and export activities-at-irradiation CSV**
3. **Interpret Bateman chain plots**
4. **Use half-life plots to validate identification**
5. **Find which isotope masks my target isotope**
6. **Estimate isotope mass from measured activity**
7. **Predict daughter products after cooldown**
8. **Optimize irradiation / cooldown / count for one isotope**
9. **Compare single and second irradiation strategies**
10. **Evaluate shutdown and 100-year dose endpoints**
11. **Export a benchmark experimental bundle for the companion code-comparison tool**

---

## 15. Minimal Acceptance Criteria

This addition is complete only when FluxForge can do all of the following in one workflow:

1. ingest a measured HPGe spectrum and identify / fit the target nuclide
2. compute the target activity at count time and at EOI
3. export `activities_at_irradiation.csv`
4. plot the target's Bateman / decay evolution with uncertainty bands
5. rank likely masking isotopes for a selected target line
6. estimate future activity and dominant daughter products
7. infer target isotope mass or concentration through an NAA workflow
8. compare multiple irradiation / cooldown / count schedules
9. compare one-pulse and two-pulse irradiation strategies
10. generate shutdown, 1-year, and 100-year dose / activity plots
11. export all results in a machine-readable experimental bundle for the separate model-comparison tool

---

## 16. Explicit Non-Goals

This addition does **not** require FluxForge to:

- replace transport calculations
- become a full neutron/photon transport solver
- replace the separate top-half model-comparison environment
- silently invent nuclear data where none are available

It **does** require FluxForge to become a rigorous, uncertainty-aware, experimentally grounded activation and NAA analysis environment with planning and optimization support.
