# FluxForge

**HPGe-driven flux-wire / foil activation analysis • Neutron spectrum unfolding • Model validation • CLI & GUI Tools**

FluxForge is a pure-Python package, dual CLI, and desktop GUI that converts HPGe-derived spectrum counts into activities, reaction rates, and infers neutron flux spectra with generalized least squares and Monte Carlo uncertainty propagation. It acts as an open-source, reproducible replacement for standard tools like STAYSL, PeakEasy, and QuantumGold, conforming directly to ASTM E3376, ASTM E261, and INL standard metrology practices.

## Key Capabilities
- **Full Workflow Parity**: Implements raw ASCII/IEC spectral processing, deterministic Peak Identification, Activity/Reaction rate generation matching Quantum Gold and PeakEasy.
- **Standards Compliant**: Direct integration with ASTM E3376 two-stream analysis, FWHM-scaled Covell continuum subtraction, and ASTM E261 reactor dosimetry schemas.
- **Spectrum Unfolding**: Multi-algorithm backend featuring Iterative GRAVEL, MLEM, and GLS with optional non-negativity enforcement and robust Monte Carlo uncertainty propagation.
- **Nuclear Data Integrations**: Bundled access to ENDF/B-VIII.0, IRDFF-II test schemas, and custom user dosimetry libraries.
- **GUI and CLI parity**: Fully featured UI using Tkinter+Matplotlib available everywhere, mapping directly onto highly scriptable CLI functions.
- **Rigorous Test Suite**: Backed by 960+ unit and integration tests spanning MCNP workflows, ASTM paths, GUI logic, and transport/IO integrations.

## Getting started
The project maintains low external dependency overhead to ensure seamless offline, air-gapped lab execution. Install in editable mode and run the CLI or GUI:

```bash
# Optional: reproducible dev environment
conda env create -f environment.yml
conda activate fluxforge

pip install -e .

# Launch the interactive GUI
python -m fluxforge_gui.app
# or: fluxforge-gui

# Or use the CLI
python -m fluxforge.cli.app --help
# or: fluxforge --help
```

Synthetic validation and inference routines expect JSON inputs; see `src/fluxforge/cli/app.py` for expected schemas. For dedicated ASTM workflows, explore `examples/astm_e261_plan.json` or run the testing parity scripts under `examples/RAFM_irradiation/`.
