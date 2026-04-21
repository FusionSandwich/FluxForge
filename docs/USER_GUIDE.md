# FluxForge User Guide

## 1. What FluxForge Is For

FluxForge is a reproducible analysis environment for:

- gamma-spectrum ingest, review, peak fitting, and ROI analysis
- activity review, reaction-rate estimation, and inventory propagation
- reactor dosimetry and neutron-spectrum unfolding
- k0-NAA characterization and reporting
- schedule optimization and second-irradiation planning
- parity, validation, and release-gate governance workflows

FluxForge supports both command-line and desktop-GUI use:

- use the CLI when you want reproducible, scriptable workflows and explicit artifacts
- use the GUI when you want interactive spectrum review, drag-driven ROI editing, and linked analysis panels

## 2. Installation and Command Discovery

Use [docs/INSTALLATION.md](/groupspace/cnerg/users/smandych/projects/ALARA/FluxForge/docs/INSTALLATION.md:1)
for the full step-by-step setup guide.

The short version is:

```bash
pip install -e .
```

or, for the GUI and reporting extras:

```bash
pip install -e '.[native-gui,reporting]'
```

After installation, the most important discovery commands are:

```bash
fluxforge --help
fluxforge commands
fluxforge commands --family spectrum
fluxforge gui --help
```

Use [docs/CLI_REFERENCE.md](/groupspace/cnerg/users/smandych/projects/ALARA/FluxForge/docs/CLI_REFERENCE.md:1)
for the full grouped command list.

## 3. Capability Families and When to Use Them

### Spectrum Analysis

Use these commands when your starting point is a measured spectrum file and you
need an artifact, plot, peak list, or ROI result.

| Family | Use it when | Main commands | Typical inputs |
|---|---|---|---|
| Spectrum ingest | You need to normalize a raw spectrum into a FluxForge artifact | `ingest`, `ingest-batch` | `.ASC`, `.Spe`, `.spe`, `.txt`, artifact JSON |
| Spectrum plotting | You want a calibrated spectrum plot, manual ROI overlays, or feature guides | `spectrum-plot` | measured spectrum plus optional manual ROI file |
| Peak and ROI review | You want automatic peaks, one ROI result, or repeated-ROI statistics | `peaks`, `roi-analyze`, `roi-statistics` | ingested spectrum artifacts or measured spectrum plus manual ROI file |

Starter data:

- `examples/RAFM_irradiation/raw_gamma_spec/`
- `examples/manual_peak_inspection/`

### Activity, Inventory, and Libraries

Use these commands after you have a peak report and want isotope-level or
time-dependent activation results.

| Workflow | Main commands | Typical outputs |
|---|---|---|
| Library inspection and registration | `library-list`, `library-register`, `library-remove` | governed source listings and registration changes |
| Line activity and isotope review | `activity`, `activity-review` | activity JSON, line tables, isotope tables |
| Reaction-rate conversion | `rates` | reaction-rate artifact |
| Inventory propagation | `inventory-review` | time-series JSON, CSV exports, optional plots |

### Planning and Optimization

Use these commands when you want to decide what to count, when to count it, or
whether a follow-on irradiation would improve detectability.

| Workflow | Main commands | Typical outputs |
|---|---|---|
| Isotope triage | `isotope-priority` | ranked isotope JSON and optional CSV |
| Masking review | `masking-review` | alternate-line guidance and masking tables |
| Schedule scoring | `optimization-sweep` | ranked candidates and objective diagnostics |
| Follow-on irradiation planning | `second-irradiation-plan` | recommended schedules and propagated inventory |
| Bundle export | `ffexp-export` | portable `.ffexp` workflow bundle |

### Dosimetry and Unfolding

Use these commands when you are building a dosimetry workflow or adjusting a
neutron spectrum from measured reaction-rate data.

| Workflow | Main commands | Typical outputs |
|---|---|---|
| ASTM workflows | `astm-e2005`, `astm-e261`, `astm-e262`, `astm-e3376` | standards-oriented JSON outputs |
| Reaction browser | `reactions` | table or JSON reaction catalog |
| Response and unfolding | `response`, `unfold`, `compare` | response matrix, unfolded spectrum, comparison metrics |
| Reporting and plots | `report`, `plots` | report bundles and plot directories |

Starter data:

- `examples/astm_e261_plan.json`
- `examples/unfolding_benchmark/`

### k0-NAA

Use these commands for normalization, characterization, analysis, aggregation,
and reporting in k0-style workflows.

| Workflow | Main commands |
|---|---|
| Normalize peak observations | `k0-normalize` |
| Build detector characterization | `k0-detector` |
| Build facility characterization | `k0-facility` |
| Run first-pass analysis | `k0-analyze` |
| Aggregate and QA/QC | `k0-aggregate`, `k0-qaqc` |
| Build final report or governed import | `k0-report`, `k0-import-kayzero` |

### Validation and Reference Workflows

Use these commands when you want bundled replay workflows, parity checks, or
release-readiness evidence.

| Workflow | Main commands | Typical inputs |
|---|---|---|
| Parity and crosswalk governance | `parity-check`, `phase5-crosswalk-report`, `phase5-release-gate` | committed test and QA assets |
| RAFM validation and benchmarking | `rafm-validate`, `rafm-qg-benchmark`, `rafm-compare-branches` | `examples/RAFM_irradiation/` |
| Planning replay workflows | `phase6-ldrd-worked-example`, `phase6-ldrd-second-irradiation-repo` | committed RAFM results and planning assets |

## 4. Common User Paths

### Path A: First CLI Analysis

Use this path when you want to learn the core artifact chain:

```bash
fluxforge ingest \
  --input examples/RAFM_irradiation/raw_gamma_spec/RAFM4/RAFM4-B_15dEOI.ASC \
  --profile rafm_25cm \
  --output /tmp/rafm4_b_ingest.json

fluxforge peaks \
  --spectrum-file /tmp/rafm4_b_ingest.json \
  --output /tmp/rafm4_b_peaks.json
```

At that point you have an ingested spectrum artifact and a peak-report
artifact. Move on to `activity-review` after you have a line-assigned peak
report, or use one of the maintained replay workflows when you want a bundled
end-to-end activity and planning chain.

### Path B: First GUI Session

Use this path when you want interactive review:

```bash
fluxforge gui --project-dir .
```

Recommended first files:

- foreground: `examples/RAFM_irradiation/raw_gamma_spec/RAFM4/RAFM4-B_15dEOI.ASC`
- background: `examples/RAFM_irradiation/background.ASC`

Recommended first actions:

1. Load the foreground and background spectra.
2. Zoom into a peak-rich region.
3. Drag the ROI directly on the main canvas.
4. Drag the background sidebands if you want sideband-style ROI analysis.
5. Drag a selected peak centroid to refine the fit anchor.
6. Use the right-click menu for manual peak insertion, deletion, line-ID clearing, and source-role assignment.

### Path C: Maintained Replay Workflow

Use this path when you want a fully bundled, reviewable example:

```bash
fluxforge rafm-validate \
  --example-root examples/RAFM_irradiation \
  --results-root /tmp/rafm_validation \
  --no-fail
```

## 5. Data and Example Assets

The most important bundled user-facing data locations are:

| Asset family | Path | Used by |
|---|---|---|
| RAFM raw spectra | `examples/RAFM_irradiation/raw_gamma_spec/` | validation, GUI onboarding, manual peak review |
| RAFM background | `examples/RAFM_irradiation/background.ASC` | GUI and plotting workflows |
| Manual ROI CSV overlays | `examples/manual_peak_inspection/` | SSH-safe and manual review workflows |
| Spectroscopy validation assets | `examples/spectroscopy_data/` | small parser, calibration, and plotting demos |
| Unfolding benchmark assets | `examples/unfolding_benchmark/` | benchmark workflow |
| Example inventory | `examples/example_inventory.json` | authoritative example catalog |

## 6. Where to Find More Detail

- Setup and environment troubleshooting:
  [docs/INSTALLATION.md](/groupspace/cnerg/users/smandych/projects/ALARA/FluxForge/docs/INSTALLATION.md:1)
- Full CLI catalog:
  [docs/CLI_REFERENCE.md](/groupspace/cnerg/users/smandych/projects/ALARA/FluxForge/docs/CLI_REFERENCE.md:1)
- Maintained cookbook workflows:
  [docs/EXAMPLE_WORKFLOWS.md](/groupspace/cnerg/users/smandych/projects/ALARA/FluxForge/docs/EXAMPLE_WORKFLOWS.md:1)
- Full example inventory:
  [examples/README.md](/groupspace/cnerg/users/smandych/projects/ALARA/FluxForge/examples/README.md:1)
- Quick onboarding:
  [docs/tutorials/0_quick_start.md](/groupspace/cnerg/users/smandych/projects/ALARA/FluxForge/docs/tutorials/0_quick_start.md:1)
- Guided first session:
  [docs/tutorials/1_getting_started.md](/groupspace/cnerg/users/smandych/projects/ALARA/FluxForge/docs/tutorials/1_getting_started.md:1)
