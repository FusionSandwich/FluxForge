# FluxForge

**HPGe-driven flux-wire / foil activation analysis • Neutron spectrum unfolding • Model validation • CLI & GUI Tools**

FluxForge is a pure-Python package, dual CLI, and desktop GUI that converts HPGe-derived spectrum counts into activities, reaction rates, and infers neutron flux spectra with generalized least squares and Monte Carlo uncertainty propagation. It acts as an open-source, reproducible replacement for standard tools like STAYSL, PeakEasy, and QuantumGold, conforming directly to ASTM E3376, ASTM E261, and INL standard metrology practices.

FluxForge is intentionally self-contained. It must install, package, and run without any dependency on sibling repositories or anything under `../testing`; that tree is for inspiration, audits, and optional developer-side comparisons only.

FluxForge's primary redesign path is now a native `PySide6 + PyQtGraph` desktop shell under `src/fluxforge/gui/`. It does **not** require a browser, an embedded web runtime, or any internet connection for supported offline workflows.

The Stage 0 roadmap governance layer now lives in the repository under
`.github/project-management/`, `docs/adr/`, and `CONTRIBUTING.md`. The
next-generation `PySide6 + PyQtGraph` shell is now the primary GUI target. That
modern path now carries the completed Phase 1 through Phase 3 GUI roadmap plus
the user-directed predictive dashboard slice, including calibration, analysis,
unfolding, standards/QA, reporting, batch workflows, and direct sidebar ASTM
review actions, without pulling archived Tk widgets back into the redesign.

## Key Capabilities
- **Full Workflow Parity**: Implements raw ASCII/IEC spectral processing, deterministic Peak Identification, Activity/Reaction rate generation matching Quantum Gold and PeakEasy.
- **Standards Compliant**: Direct integration with ASTM E3376 two-stream analysis, FWHM-scaled Covell continuum subtraction, and ASTM E261 reactor dosimetry schemas.
- **Spectrum Unfolding**: Multi-algorithm backend featuring Iterative GRAVEL, MLEM, and GLS with optional non-negativity enforcement and robust Monte Carlo uncertainty propagation.
- **Nuclear Data Integrations**: Bundled access to ENDF/B-VIII.0, IRDFF-II test schemas, and custom user dosimetry libraries.
- **Modern calibration workspace**: The Phase 2.1 Qt shell now includes a unified energy + FWHM calibration dialog with embedded spectrum review, residual-first plots, and ASTM E181 order locking.
- **Interactive plot review**: Spectrum inspection defaults to log counts with isotope-colored peak markers, and Activity/Rates now include live zoomable plot panels alongside the existing unfold diagnostics.
- **Rigorous Test Suite**: Backed by 1131 passing unit and integration tests in this workspace, spanning MCNP workflows, ASTM paths, unfolding parity, GUI logic, transport/IO integrations, and predictive Qt workflows.

## Getting started
The project maintains low external dependency overhead to ensure seamless offline, air-gapped lab execution. Install in editable mode and run the CLI or GUI:

```bash
# Optional: reproducible dev environment
conda env create -f environment.yml
conda activate fluxforge

pip install -e .
# Optional: developer lint/test extras
pip install -e '.[dev]'

# Optional: install the modern native GUI stack
pip install -e '.[native-gui]'

# Optional: enable Jinja2 + WeasyPrint report export support
pip install -e '.[reporting]'

# Optional: legacy Tk desktop automation coverage
pip install -e '.[gui-test]'

# Optional: enforce offline-only execution
export FLUXFORGE_OFFLINE=1

# Launch the modern interactive GUI
python -m fluxforge.gui.app
# or: fluxforge-gui

# Launch the archived Tk GUI fallback
python -m fluxforge_gui.app
# or: fluxforge-gui-legacy

# Or use the CLI
python -m fluxforge.cli.app --help
# or: fluxforge --help
```

For offline delivery and native packaging helpers, use the repo-local tooling:

```bash
# Build a wheelhouse for air-gapped installs
python tools/build_offline_wheelhouse.py

# Build native CLI/GUI bundles with PyInstaller
python tools/build_native_bundle.py --target both
```

The legacy desktop regression path still includes a real interactive Tk run that opens the archived GUI, loads a spectrum, edits ROI/calibration controls, exports artifacts, runs the report plot suite, captures screenshots, and verifies copied CLI commands:

```bash
python -m pytest -q tests/test_gui_desktop_native.py
```

For repeatable legacy GUI evidence capture, run the desktop driver directly:

```bash
PYTHONPATH=src FLUXFORGE_OFFLINE=1 xvfb-run -a \
  python tests/gui_desktop_driver.py artifacts/gui_review/current_linux
```

The latest local native-review run writes screenshots such as `01-launch.png`, `03-roi-calibration.png`, and `06-report-plots.png` under `artifacts/gui_review/current_linux/` for manual inspection. The committed Linux screenshot baselines live under `tests/data/gui_review_baselines/linux/`.

For the redesigned Qt calibration workspace, generate the native Phase 2.1 review gallery with:

```bash
QT_QPA_PLATFORM=offscreen PYTHONPATH=src \
  python tests/gui_calibration_workspace_probe.py \
  artifacts/gui_review/phase2_calibration_workspace
```

To apply the roadmap tracker on GitHub after pushing planning changes, use the
`Sync Project Planning` workflow. The repository stores the milestone, label, board,
epic, and seed-issue definitions as code.

Synthetic validation and inference routines expect JSON inputs; see `src/fluxforge/cli/app.py` for expected schemas. For dedicated ASTM workflows, explore `examples/astm_e261_plan.json` or run the testing parity scripts under `examples/RAFM_irradiation/`.
