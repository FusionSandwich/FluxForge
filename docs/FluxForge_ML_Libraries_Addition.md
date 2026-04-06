# FluxForge — ML, Advanced Optimization, and Nuclear Library Stack Addition

## Status and scope

This file is **additive only**. It does **not** replace:
- `FluxForge_Final_Additions.md`
- `FluxForge_Additions_v3_Final.md`
- `ROADMAP_EXECUTION_STATUS.md`
- previous NAA / InterSpec / STAYSL / FISPACT additions

It is intended to be merged into the existing roadmap as an additional requirements document for the currently pending offline-parity and experimental-analysis phases, especially the richer-library, activity/reference, dose, shielding, k0, and optimization work that is already scheduled after Phase 3.16.

---

## 1. New governing decision

FluxForge should stop treating “the library” as a single file and instead adopt a **stacked nuclear-data architecture**.

The immediate issue to solve is:

> current libraries do not expose complete daughter-chain navigation, full half-life uncertainty visibility, recommended metrology overrides, or the specialized coincidence/reaction-gamma data needed for advanced experimental gamma analysis.

The solution is **not** to switch from one incomplete library to another incomplete library. The solution is to make FluxForge use a layered data model with explicit provenance.

### 1.1 Required library layers

FluxForge should use the following layers simultaneously:

1. **Base decay/radiation library**  
   ENDF/B-VIII.0 decay sublibrary for broad production coverage and PeakEasy-style parity.

2. **Evaluated structure/decay backbone**  
   ENSDF for complete evaluated decay structure, levels, gammas, decay modes, and level/half-life metadata.

3. **API and incremental refresh layer**  
   IAEA LiveChart API for direct machine-readable CSV updates and selective sync.

4. **Half-life / isomer / decay-mode overlay**  
   NUBASE2020 for recommended half-lives, uncertainties, isomer states, spins/parities, and decay-mode intensities.

5. **Recommended metrology subset**  
   DDEP / LNHB and IAEA X- and gamma-ray standards for the calibration and key-reference radionuclides where the strongest recommended values matter more than breadth.

6. **Decay engine interoperability layer**  
   Optional SandiaDecay-compatible import/export for InterSpec-style decay-chain calculations and XML-backed interoperability.

7. **Coincidence / cascade extension**  
   paceENSDF-style / ENSDF-derived JSON and coincidence gamma-gamma / gamma-X data for coincidence-aware masking and cascade-sum reasoning.

8. **Reaction-gamma extension**  
   Prompt and reaction-gamma libraries such as CapGam / EGAF / pyEGAF and fast-neutron reaction-gamma libraries (e.g. Berkeley Atlas / Baghdad Atlas class sources) for advanced activation interpretation beyond ordinary decay lines.

### 1.2 Explicit policy for conflicts

Where two sources disagree, FluxForge must preserve both the **displayed recommended value** and the **source provenance**:

- `display_value`
- `display_uncertainty`
- `source_name`
- `source_version`
- `source_priority`
- `alternate_values[]`
- `bibliography[]`

The GUI should always show whether a quantity came from:
- ENDF/B-VIII.0
- ENSDF
- NUBASE2020
- DDEP/LNHB
- IAEA standards
- user override / analyst override

---

## 2. Libraries to add to FluxForge

These are the specific libraries / sources to add to the FluxForge data stack.

### 2.1 ENDF/B-VIII.0 decay sublibrary

**Role:** broad base library and PeakEasy parity target.

**Reason:** PeakEasy explicitly uses the ENDF/B-VIII.0 decay library, exposes >3500 nuclides, supports nuclide age adjustment for daughter yields, lists both parents and daughters, and supports mixtures and user-defined lists. FluxForge should match that baseline.

**Target use in FluxForge:**
- default shipped offline base decay library
- broad isotope coverage for ordinary spectrum work
- baseline decay-radiation and parent/daughter navigation
- compatibility target for PeakEasy-style workflows

### 2.2 ENSDF archive

**Role:** authoritative evaluated structure/decay backbone.

**Reason:** ENSDF contains evaluated nuclear level properties, half-lives, decay modes, gamma energies, intensities, multipolarities, conversion coefficients, and other radiation data in an evaluated form.

**Target use in FluxForge:**
- complete daughter-chain reconstruction
- level-aware gamma and decay-scheme browsing
- half-life uncertainty extraction
- advanced chain and coincidence preprocessing

### 2.3 IAEA LiveChart API

**Role:** machine-readable sync layer.

**Reason:** LiveChart provides direct CSV download via API and makes selective data refresh practical without shipping a full custom parser-only workflow.

**Target use in FluxForge:**
- selective nuclide refresh
- on-demand chain fetch
- direct CSV ingestion into builder scripts
- cache updates for newer evaluations

### 2.4 NUBASE2020

**Role:** half-life / isomer / decay-mode overlay.

**Reason:** NUBASE2020 is evaluated, includes recommended values and uncertainties, covers ground and isomeric states, and is especially strong for state identity, half-life, decay modes, and associated uncertainties.

**Target use in FluxForge:**
- half-life uncertainty visibility
- isomer-aware state identification
- clean canonical nuclide-state IDs
- chain stabilization when ENSDF level detail is more complex than needed for quick analysis

### 2.5 DDEP / LNHB / NUCLÉIDE-LARA

**Role:** recommended-metrology subset and high-confidence override layer.

**Reason:** DDEP/LNHB provide recommended decay data, including half-life, decay mode, branching, and emission energies/intensities with uncertainties, but for a smaller curated subset rather than all nuclides.

**Target use in FluxForge:**
- calibration nuclides
- key assay nuclides
- reference-quality displayed values
- conflict-resolution override when metrology-grade recommendation exists

### 2.6 IAEA X- and gamma-ray standards

**Role:** standards-backed calibration subset.

**Reason:** recommended half-lives and emission probabilities exist for selected radionuclides appropriate for detector efficiency calibration and related use cases.

**Target use in FluxForge:**
- efficiency calibration workflows
- standards-labeled reference lines
- “recommended calibration line” badges in GUI

### 2.7 SandiaDecay-compatible layer

**Role:** decay calculation and interoperability engine.

**Reason:** InterSpec relies on a SandiaDecay-backed ecosystem; FluxForge should be able to interoperate with that style of data and optionally import/export SandiaDecay-compatible XML so analysts can compare or migrate reference workflows.

**Target use in FluxForge:**
- optional alternate decay engine
- verification against InterSpec-style chain calculations
- import/export of editable user decay data

### 2.8 paceENSDF / coincidence JSON layer

**Role:** advanced coincidence and cascade-aware extension.

**Reason:** ENSDF alone does not expose coincidence gamma-gamma and gamma-X intensities in an analysis-ready way. Recent work translating ENSDF to JSON and deriving coincidence data gives exactly the sort of portable queryable data FluxForge needs for masking analysis and cascade reasoning.

**Target use in FluxForge:**
- masking-aware isotope ranking
- coincidence-assisted line selection
- true-coincidence / cascade-sum guidance
- advanced “why this line is bad” diagnostics

### 2.9 CapGam / EGAF / pyEGAF

**Role:** prompt and thermal (n,gamma) library.

**Reason:** experimental activation workflows often need prompt/capture-gamma knowledge that ordinary decay libraries do not provide.

**Target use in FluxForge:**
- capture-gamma reference overlays
- prompt-gamma and reaction-gamma interpretation
- advanced activation / PGAA adjacency features

### 2.10 Fast-neutron reaction-gamma sources

**Role:** n,n'gamma and related reaction-gamma extension.

**Reason:** InterSpec has expanded fast-neutron reaction-gamma content; FluxForge should have an explicit extension path for reaction-induced gamma signatures beyond decay radiation.

**Target use in FluxForge:**
- advanced irradiation diagnostics
- inelastic-scattering and fast-neutron gamma interpretation
- shielding / reaction context for unusual lines

---

## 3. Required data model changes

### 3.1 New canonical schema

Add a normalized internal schema for every nuclide state:

- `nuclide_id`
- `state_id`
- `symbol`
- `Z`
- `A`
- `isomer_label`
- `half_life_value`
- `half_life_unit`
- `half_life_uncertainty_value`
- `half_life_uncertainty_type`
- `decay_modes[]`
- `decay_mode_branching[]`
- `daughter_state_ids[]`
- `parent_state_ids[]`
- `gamma_lines[]`
- `xray_lines[]`
- `coincidence_pairs[]`
- `radiation_source_type` (`decay`, `capture_gamma`, `reaction_gamma`, `coincidence_derived`)
- `provenance[]`
- `bibliography[]`

### 3.2 New chain products

FluxForge should explicitly materialize:

- full chain to stability
- immediate daughters only
- possible parents feeding selected daughter
- secular/transient equilibrium summary
- half-life uncertainty propagation to chain activities
- alternate-library disagreement summary

### 3.3 New exported machine-readable products

Add:

- `nuclide_library_provenance.json`
- `decay_chain_graph.json`
- `half_life_uncertainty_table.csv`
- `chain_activity_timeseries.csv`
- `masking_rankings.csv`
- `dominant_nuclides_by_metric.csv`
- `optimizer_pareto_front.csv`
- `optimizer_objective_grid.parquet`

---

## 4. GUI additions required by the new library stack

### 4.1 Library Source Manager

New workspace / dialog:

- active base library selector
- overlay toggles for ENSDF / NUBASE / DDEP / standards / coincidence / reaction gamma
- per-library version display
- sync / refresh / rebuild buttons
- conflict preview table
- “why is this value shown?” explanation pane

### 4.2 Nuclide Decay Info 2.0

Extend the current workbench to show:

- full daughter chain graph
- reverse-parent graph
- half-life uncertainty and source
- chain completeness indicator
- recommended-value badge
- alternate-evaluation dropdown
- level-by-level decay scheme summary
- specific-activity and dose-context summaries

### 4.3 Library conflict inspector

For a selected nuclide/line:

- ENDF value
- ENSDF value
- NUBASE value
- DDEP / LNHB value if present
- chosen displayed value
- selection rule explanation

### 4.4 Coincidence / masking explorer

Add a tab that shows:

- nearby interfering lines
n- coincidence relationships
- cascade sums
- expected overlap severity
- suggested cleaner alternate lines
- cooling-time effect on each interfering contributor

### 4.5 Calibration standards browser

Add a standards-backed line browser showing:

- recommended calibration nuclides
- recommended half-lives and emission probabilities
- line quality score
- coincidence warning
- detector-efficiency relevance notes

### 4.6 “Interesting isotopes” explorer

New GUI that ranks nuclides by selected metric:

- activity
- gamma dose rate
- decay heat
- inhalation / ingestion metrics when available
- long-term persistence
- shutdown / 1 day / 1 week / 1 year / 100 year significance
- chain dominance / pathway dominance

---

## 5. ML and advanced optimization additions

## 5.1 Do not use ML as a black-box replacement

FluxForge should **not** use ML to directly emit “optimal irradiation time” without a physical model. Instead ML should be used as a **surrogate, ranking, or acquisition layer** on top of a physics-based forward model.

### 5.1.1 Physics-first requirement

The optimizer must remain anchored in:
- activation/production equations
- Bateman-chain evolution
- detector efficiency
- attenuation / self-shielding corrections
- line overlap / masking
- count-rate and dead-time constraints
- dose endpoint calculations
- uncertainty propagation

ML may accelerate or prioritize this search, but must not replace the physical model.

## 5.2 Multiple-peak / multiple-gamma joint optimization

FluxForge should move beyond single-line optimization and optimize **joint objective sets** built from multiple peaks and multiple isotopes at once.

### 5.2.1 Multi-line target objective

For a selected isotope or element, define a joint objective over a set of candidate lines:

- total expected information on inferred activity/mass
- line purity
- line covariance / shared systematic uncertainty
- coincidence or summing risk
- energy-dependent efficiency uncertainty
- masking sensitivity

The optimizer should automatically choose whether the best experiment uses:
- one very clean line
- multiple moderate lines jointly
- a full-spectrum template approach

### 5.2.2 Full-spectrum Bayesian mode

Add an advanced option inspired by recent Bayesian full-spectrum HPGe analysis:

- infer isotope activities jointly from the whole spectrum or broad ROIs
- include energy-scale mismatch correction
- propagate full posterior uncertainty via MCMC or variational approximation
- allow multiple template spectra per isotope where detector non-linearity matters

This mode should be used when single-peak metrics are unstable or strongly coupled.

## 5.3 Multi-objective Bayesian optimization

For expensive design spaces (irradiation time, cooldown, count time, detector position, line subset, second-irradiation schedule), add a multi-objective Bayesian optimization layer above the deterministic sweep.

### 5.3.1 Objectives to optimize jointly

Examples:
- maximize target identifiability
- minimize relative uncertainty in inferred mass
- maximize line purity
- minimize dead-time risk
- minimize total campaign time
- maximize shutdown-dose relevance
- maximize long-term radiological significance contrast
- maximize expected information gain of a second irradiation

### 5.3.2 Why BO is appropriate

The nuclear forward model can become expensive when:
- many isotopes are tracked
- full uncertainty propagation is included
- multiple detector geometries are compared
- second-irradiation branches are evaluated
- shutdown and long-term dose endpoints are included simultaneously

In these cases a Gaussian-process or other surrogate-assisted multi-objective optimizer is appropriate.

### 5.3.3 Required outputs

- Pareto front
- knee-point recommendations
- scalarization presets
- objective importance sliders
- uncertainty-aware ranking
- explanation of why one schedule dominates another

## 5.4 Sequential / second-irradiation design by information gain

For follow-up experiments, FluxForge should compute **expected information gain** rather than simply maximizing raw activity.

Examples:
- pick the second irradiation that best separates two candidate isotope explanations
- select a cooldown window that kills the dominant masker while preserving the target
- prioritize irradiation schedules that increase confidence in long-lived shutdown-relevant products

### 5.4.1 Required algorithmic options

- expected reduction in posterior mass/activity uncertainty
- expected Bayes-factor separation between hypotheses
- expected reduction in masking ambiguity
- expected increase in pathway confidence for shutdown-dominant nuclides

## 5.5 Isotope-interest ranking for shutdown and other activation properties

FluxForge should implement an “interesting isotope” ranking engine using FISPACT-inspired dominant-nuclide and pathway analysis ideas.

### 5.5.1 Rankable response metrics

At user-selected times (`EOI`, `count start`, `shutdown`, `1 day`, `1 week`, `1 year`, `10 year`, `100 year`, custom):

- activity
- gamma dose rate
- total dose proxy
- decay heat
- long-lived waste relevance
- clearance index or waste proxy if available
- measurement feasibility
- uncertainty contribution

### 5.5.2 Ranking modes

- top contributors by metric
- top contributors to uncertainty in metric
- top maskers for isotope of interest
- top pathway ancestors feeding a dominant isotope
- top isotopes whose discrimination most improves the decision objective

### 5.5.3 Advanced methods to include

Not just ML:

1. **Graph-theoretic pathway analysis** to identify dominant source-to-product chains.
2. **Sensitivity analysis** to identify which reactions and decay constants drive the target metric.
3. **Reduced-order models** that keep only important nuclides/pathways for fast optimization.
4. **Expected information gain / Bayesian experimental design** for second-irradiation choice.
5. **Multi-objective BO** for expensive trade-off exploration.
6. **Full-spectrum Bayesian inversion** when multiple peaks jointly inform isotope activity.
7. **Physics-informed neural surrogates** only after the above baselines exist.

## 5.6 What ML can realistically help with

### Good ML uses
- surrogate prediction of objective surfaces
- line-masking severity prediction from spectral neighborhoods
- fast proposal of promising irradiation/cooldown/count regions
- clustering of isotopes by time-dependent dominance profile
- recommending which isotopes are likely to matter for shutdown / 100-year dose endpoints
- amortized approximation to expensive posterior calculations

### Bad ML uses
- opaque direct recommendation without provenance
- replacing Bateman physics
- replacing library-backed activity inference
- ignoring uncertainty or source provenance

---

## 6. Research-inspired implementation path

### 6.1 Near-term (must build first)

1. deterministic multi-line forward predictor
2. chain-aware masking engine
3. uncertainty propagation for line and mass inference
4. interesting-isotope ranking by metric and time
5. full daughter/parent explorer with uncertainty display

### 6.2 Mid-term (high value)

6. Bayesian full-spectrum activity inference
7. coincidence/cascade-aware masking model
8. multi-objective Pareto optimizer
9. second-irradiation information-gain planner
10. dominant-pathway explorer for shutdown/dose metrics

### 6.3 Later research track

11. GP/BO surrogate optimizer
12. physics-informed neural surrogate for fast schedule ranking
13. adaptive experiment design loop
14. combined experimental + modeled objective fusion with the separate top-box tool

---

## 7. Roadmap additions by existing step family

## 7.1 Additions under current Step 3.18 (richer libraries / source-age overlays)

Add these sub-items:

- **3.18A** — ship ENDF/B-VIII.0 decay base library and import pipeline
- **3.18B** — add ENSDF parser / archive ingestion path
- **3.18C** — add LiveChart CSV sync and cache builder
- **3.18D** — add NUBASE2020 overlay for half-life uncertainty and isomers
- **3.18E** — add DDEP/LNHB and IAEA standards override layer
- **3.18F** — add SandiaDecay-compatible XML import/export bridge
- **3.18G** — add library conflict-resolution and provenance schema
- **3.18H** — add full daughter-chain and reverse-parent explorer GUI

## 7.2 Additions under current Step 3.19 (dose / shielding / attenuation)

- **3.19A** — add “Interesting Isotopes” ranking workspace
- **3.19B** — add dominant-nuclide ranking by shutdown/1y/100y/custom time
- **3.19C** — add pathway and sensitivity visualizations for radiological metrics
- **3.19D** — add measurement-feasibility overlays to dose-significance views

## 7.3 Additions under current Step 3.20 (batch compare / k0 / offline workspaces)

- **3.20A** — add irradiation/cooldown/count optimization workspace
- **3.20B** — add second-irradiation planner
- **3.20C** — add multi-line / multi-isotope objective builder
- **3.20D** — add activities/masses/dose endpoint batch sweeps and Pareto export

## 7.4 Additions under current Step 3.25 (new workspaces)

Create dedicated workspaces for:

- Library Source Manager
- Nuclide Decay Info 2.0
- Coincidence / Masking Explorer
- Interesting Isotopes Explorer
- Irradiation Schedule Optimizer
- Second Irradiation Planner
- Calibration Standards Browser

---

## 8. Acceptance criteria

The additions in this file are only complete when FluxForge can do all of the following:

1. Display a selected nuclide with its full daughter chain to stability.
2. Display reverse-parent candidates feeding a selected daughter.
3. Show half-life value, uncertainty, and source provenance in the GUI.
4. Show alternate values from multiple libraries where they disagree.
5. Use an ENDF/B-VIII.0-backed shipped base library, achieving PeakEasy-style parity.
6. Use recommended overrides for calibration / metrology nuclides from DDEP / IAEA standards.
7. Export provenance for every displayed nuclear-data value.
8. Rank masking isotopes for a selected line.
9. Optimize schedules using multiple lines jointly, not just one peak.
10. Rank isotopes by shutdown / 100-year / custom radiological importance.
11. Provide a pathway-based explanation of why an isotope matters.
12. Export machine-readable optimizer and library metadata for the separate top-box comparison tool.

---

## 9. Important implementation note

Because the FluxForge repository is not mounted in this workspace, the items above are provided as a roadmap addition and drop-in resource manifests rather than as committed repository changes. The intended merge path is:

- copy the manifest files into `src/fluxforge/resources/`
- add builder/import code under `src/fluxforge/data/` and `scripts/`
- wire the new GUI workspaces into the existing Qt shell under the pending 3.18–3.25 slice

---

## 10. InterSpec compatibility checkpoint (2026-04-06)

This section records the compatibility decision after source-level review of the
InterSpec data stack and current FluxForge implementation status.

### 10.1 What InterSpec actually uses

InterSpec does not rely on a single flat line list. It uses layered files and
loaders:

1. `sandia.decay.xml` for decay-chain and radiation backbone.
2. `sandia.reactiongamma.xml` for capture and reaction-gamma signatures.
3. `PhotoPeak.lis` for fast line lookup and isotope ID paths.
4. `add_ref_line.xml` and `dynamic_ref_lines.xml` for reference-line overlays.
5. `more_nuclide_info.xml` for curated nuclide annotations and analyst context.

### 10.2 FluxForge implementation decision

FluxForge should not replace its current stacked architecture with direct
InterSpec file dependence. FluxForge should:

1. Continue using authoritative third-party nuclear data sources as the primary
   value authority.
2. Add optional InterSpec-compatible import adapters for Sandia-style XML and
   reference overlays.
3. Keep FluxForge as the normalization and provenance authority so all displayed
   values remain explainable and reproducible.

In short: adopt third-party data where possible, but keep a FluxForge-governed
schema and conflict policy instead of a one-off opaque custom library.

### 10.3 New 3.18 sub-steps for InterSpec parity

Add these sub-items under Step `3.18`:

- **3.18I** - SandiaDecay adapter: import `sandia.decay.xml` into normalized
  decay tables with parent/daughter links and uncertainty fields.
- **3.18J** - Reaction-gamma adapter: import
  `sandia.reactiongamma.xml` into a dedicated reaction-gamma table with
  explicit reaction type tags.
- **3.18K** - Reference overlay adapter: ingest `PhotoPeak.lis`,
  `add_ref_line.xml`, and `dynamic_ref_lines.xml` as non-authoritative overlay
  layers.
- **3.18L** - Value-governance rule engine: enforce priority and provenance
  display (`display_value`, `display_uncertainty`, `source_name`,
  `source_version`, `alternate_values`).
- **3.18M** - Cross-library validation harness: compare selected nuclides and
  lines across FluxForge base sources, Sandia imports, and metrology overrides.

### 10.4 Required quality gates

Before enabling any InterSpec-compatible source by default, all of the following
must pass:

1. Schema validation for XML and tabular overlays, with explicit parse errors.
2. Golden-nuclide comparison tests for half-life, gamma energy, and intensity.
3. Reaction-gamma sanity checks (units, yield normalization, reaction labels).
4. Provenance-completeness checks: every surfaced value must carry source and
   version metadata.
5. Regression tests proving no degradation of existing peak-identification and
   activity workflows when the adapters are enabled.

