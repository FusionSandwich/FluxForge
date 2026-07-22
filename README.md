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
| Desktop GUI | Review spectra interactively, zoom/pan, set ROI boundaries, and inspect linked analysis panels | `fluxforge gui`, `fluxforge-gui` |

## Quick GUI Setup

Use 64-bit Python **3.11 or 3.12**. Python 3.13 is not currently supported
because FluxForge uses NumPy 1.26. The complete prerequisite and
troubleshooting guide is [docs/INSTALLATION.md](docs/INSTALLATION.md).

### Windows PowerShell

```powershell
git clone https://github.com/FusionSandwich/FluxForge.git
Set-Location FluxForge
git switch optimization-workflows
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[native-gui,reporting]"
.\.venv\Scripts\fluxforge.exe gui --project-dir .
```

### Linux

Install the Qt runtime prerequisites first; the complete package list is in
the [installation guide](docs/INSTALLATION.md#linux-copy-and-paste-setup).

```bash
git clone https://github.com/FusionSandwich/FluxForge.git
cd FluxForge
git switch optimization-workflows
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[native-gui,reporting]'
fluxforge gui --project-dir .
```

Verify either installation with:

```bash
fluxforge --help
fluxforge-gui --help
python -c "from fluxforge.gui.qt_compat import QT_AVAILABLE, QT_IMPORT_ERROR; assert QT_AVAILABLE, QT_IMPORT_ERROR; print('Qt GUI ready')"
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
[docs/CLI_REFERENCE.md](docs/CLI_REFERENCE.md)
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

1. Use **File > Open Spectrum** to load the foreground `.ASC` file.
2. Use the mouse wheel to zoom and left-drag to pan the calibrated energy axis.
3. Click **Select ROI**, then drag either boundary on the main canvas.
4. Open **ROI Tools** to choose the background model and run the bounded analysis.
5. Click **Reset View** to return to the full spectrum or **Clear ROI** to remove the boundaries.
6. Use the spectrum tabs and background controls to compare foreground, background, and overlay spectra.

If you prefer the direct GUI entrypoint, `fluxforge-gui --project-dir .` launches the same modern Qt shell when the `native-gui` extra is installed.

Important:

- use `fluxforge gui` or `fluxforge-gui` after installation
- do not rely on `python -m fluxforge.gui.app` as the primary user path

If you see `ModuleNotFoundError: No module named 'fluxforge.gui'`, the usual causes are:

- FluxForge was not installed into the active environment
- the active Python is not a supported 64-bit Python 3.11 or 3.12
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
[examples/README.md](examples/README.md)
and [examples/example_inventory.json](examples/example_inventory.json).

## Documentation Map

Start here:

- [docs/INSTALLATION.md](docs/INSTALLATION.md)
- [docs/CLI_REFERENCE.md](docs/CLI_REFERENCE.md)
- [docs/USER_GUIDE.md](docs/USER_GUIDE.md)
- [docs/EXAMPLE_WORKFLOWS.md](docs/EXAMPLE_WORKFLOWS.md)
- [examples/README.md](examples/README.md)

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
