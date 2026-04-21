# FluxForge

FluxForge is an open-source toolkit for reproducible gamma spectroscopy,
activation and inventory review, reactor dosimetry, neutron-spectrum
unfolding, k0-NAA workflows, and irradiation-planning studies.

FluxForge has two primary user entrypoints:

- `fluxforge` for the command-line interface
- `fluxforge gui` or `fluxforge-gui` for the desktop GUI

If you are new to the project, start with the installation steps below, then
run `fluxforge commands` to see the grouped CLI surface.

## What FluxForge Covers

| Capability family | Typical use | Main entry points |
|---|---|---|
| Spectrum analysis | Ingest measured spectra, plot calibrated views, detect peaks, and analyze ROIs | `ingest`, `ingest-batch`, `spectrum-plot`, `peaks`, `roi-analyze`, `roi-statistics` |
| Activity and inventory review | Convert peak outputs into line activity, isotope review, reaction rates, and time-propagated inventory views | `activity`, `activity-review`, `inventory-review`, `rates` |
| Planning and optimization | Rank isotopes, review masking, optimize schedules, plan follow-on irradiation, and export bundles | `isotope-priority`, `masking-review`, `optimization-sweep`, `second-irradiation-plan`, `ffexp-export` |
| Reactor dosimetry and unfolding | Build response matrices, run ASTM-style workflows, unfold spectra, compare results, and report | `astm-e2005`, `astm-e261`, `astm-e262`, `astm-e3376`, `response`, `unfold`, `compare`, `report`, `reactions` |
| k0-NAA | Normalize peak observations, characterize detector and facility state, analyze, aggregate, QA/QC, and report | `k0-normalize`, `k0-detector`, `k0-facility`, `k0-analyze`, `k0-aggregate`, `k0-qaqc`, `k0-report`, `k0-import-kayzero` |
| Validation and replay | Run parity, crosswalk, release-gate, and bundled RAFM replay workflows | `parity-check`, `phase5-crosswalk-report`, `phase5-release-gate`, `rafm-validate`, `phase6-ldrd-worked-example` |
| Desktop GUI | Review spectra interactively, drag ROIs and peak centroids, and inspect linked analysis panels | `fluxforge gui`, `fluxforge-gui` |

## Step-by-Step Setup

Detailed setup instructions live in [docs/INSTALLATION.md](/groupspace/cnerg/users/smandych/projects/ALARA/FluxForge/docs/INSTALLATION.md:1). The shortest supported path is:

### 1. Create and activate an environment

Using `venv`:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Using Conda:

```bash
conda env create -f environment.yml
conda activate fluxforge
python -m pip install --upgrade pip
```

### 2. Choose an install profile

| Profile | Command | Use when |
|---|---|---|
| CLI-only | `pip install -e .` | You want the command-line workflows and bundled examples |
| Full user install | `pip install -e '.[native-gui,reporting]'` | You want the CLI plus the Qt GUI and reporting extras |
| Developer/test extras | `pip install -e '.[dev,gui-test]'` | You are contributing, running QA probes, or extending the test surface |

### 3. Verify the install

Run these from the repository root after installation:

```bash
fluxforge --help
fluxforge commands
fluxforge gui --help
```

If you installed the GUI extras, also verify:

```bash
fluxforge-gui --help
```

## The First 5 Commands to Run

These five commands give most users the fastest path to the full surface:

```bash
fluxforge --help
fluxforge commands
fluxforge commands --family spectrum
fluxforge gui --help
fluxforge phase6-ldrd-worked-example --help
```

Use `fluxforge <command> --help` for flags. Use
[docs/CLI_REFERENCE.md](/groupspace/cnerg/users/smandych/projects/ALARA/FluxForge/docs/CLI_REFERENCE.md:1)
for the full grouped reference.

## First CLI Workflow

This is the shortest verified analysis chain using committed RAFM data:

```bash
fluxforge ingest \
  --input examples/RAFM_irradiation/raw_gamma_spec/RAFM4/RAFM4-B_15dEOI.ASC \
  --profile rafm_25cm \
  --output /tmp/rafm4_b_ingest.json

fluxforge peaks \
  --spectrum-file /tmp/rafm4_b_ingest.json \
  --output /tmp/rafm4_b_peaks.json
```

What you get:

- an ingested spectrum artifact
- a peak-report artifact

Use a maintained replay workflow such as `fluxforge rafm-validate ...` or
`fluxforge phase6-ldrd-worked-example ...` when you want a fully populated
activity/inventory/planning output chain backed by committed reference assets.

## First GUI Workflow

Install the full user profile first:

```bash
pip install -e '.[native-gui,reporting]'
```

Then launch the GUI:

```bash
fluxforge gui --project-dir .
```

Recommended first files:

- foreground spectrum: `examples/RAFM_irradiation/raw_gamma_spec/RAFM4/RAFM4-B_15dEOI.ASC`
- background spectrum: `examples/RAFM_irradiation/background.ASC`

Recommended first interactions:

1. Load the foreground and background spectra.
2. Zoom into a photopeak-rich region.
3. Drag the ROI directly on the main canvas.
4. Drag the sideband handles if you want ROI-sideband background estimation.
5. Select a peak and drag its centroid to refine it.
6. Right-click near a peak to use context actions such as select peak, use peak ROI, clear identification, delete peak, add manual peak, and assign foreground/background/overlay roles.

If you prefer the direct GUI entrypoint, `fluxforge-gui --project-dir .` launches the same modern Qt shell when the `native-gui` extra is installed.

Important:

- use `fluxforge gui` or `fluxforge-gui` after installation
- do not rely on `python -m fluxforge.gui.app` as the primary user path

If you see `ModuleNotFoundError: No module named 'fluxforge.gui'`, the usual causes are:

- FluxForge was not installed into the active environment
- the active Python is older than the required Python 3.11+
- the `native-gui` extra was not installed for a GUI workflow

Fix it with:

```bash
python -m pip install --upgrade pip
pip install -e '.[native-gui,reporting]'
fluxforge gui --help
```

## Maintained Starter Examples

These are the best first examples because they use committed inputs and have
clear output targets:

| Workflow | Command | Inputs | Typical outputs |
|---|---|---|---|
| RAFM validation | `fluxforge rafm-validate --example-root examples/RAFM_irradiation --results-root /tmp/rafm_validation --no-fail` | `examples/RAFM_irradiation/raw_gamma_spec/` and committed comparison assets | analysis JSON, comparison tables, validation summaries |
| Manual peak inspection | `fluxforge spectrum-plot ... --manual-peaks-file examples/manual_peak_inspection/manual_flux_wire_ti_rafm_1a.csv` | RAFM flux-wire spectrum plus manual ROI CSV | plot PNG plus manual peak JSON |
| Phase 6 worked example | `fluxforge phase6-ldrd-worked-example --sample-id RAFM4-C_15dEOI --output-root /tmp/phase6_ldrd_worked_example` | committed RAFM assets | activity/inventory review, masking review, optimization outputs, planning bundle |
| Unfolding benchmark | `python examples/unfolding_benchmark/run_benchmark.py` | bundled benchmark response matrix, measurements, and truth spectrum | benchmark plots and JSON metrics |

For the full example inventory, use
[examples/README.md](/groupspace/cnerg/users/smandych/projects/ALARA/FluxForge/examples/README.md:1)
and [examples/example_inventory.json](/groupspace/cnerg/users/smandych/projects/ALARA/FluxForge/examples/example_inventory.json:1).

## Documentation Map

Start here:

- [docs/INSTALLATION.md](/groupspace/cnerg/users/smandych/projects/ALARA/FluxForge/docs/INSTALLATION.md:1)
- [docs/CLI_REFERENCE.md](/groupspace/cnerg/users/smandych/projects/ALARA/FluxForge/docs/CLI_REFERENCE.md:1)
- [docs/USER_GUIDE.md](/groupspace/cnerg/users/smandych/projects/ALARA/FluxForge/docs/USER_GUIDE.md:1)
- [docs/EXAMPLE_WORKFLOWS.md](/groupspace/cnerg/users/smandych/projects/ALARA/FluxForge/docs/EXAMPLE_WORKFLOWS.md:1)
- [examples/README.md](/groupspace/cnerg/users/smandych/projects/ALARA/FluxForge/examples/README.md:1)

Workflow-specific references:

- `docs/workflows/`
- `docs/optimization_of_irradiation/`
- `docs/ASTM_standards/`

## Testing and QA

Run the standard test suite:

```bash
pip install -e '.[dev,gui-test]'
pytest -q
```

Run the heavier Phase 5 gate scripts when needed:

```bash
tools/qa/run_phase5_full_suite.sh
tools/qa/run_phase5_release_gate.sh
```

## Contributing

See `CONTRIBUTING.md` for development and verification expectations.
