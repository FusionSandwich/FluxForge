# Testing Repo Capability Matrix (radioactivedecay, py-findpeaks, PyGammaSpec, curie)

Scope
- Audit targets: `testing/radioactivedecay`, `testing/py-findpeaks`, `testing/PyGammaSpec`, `testing/curie`
- FluxForge column reflects current repo state.
- Legend: YES = present, PARTIAL = partial/limited, NO = missing.

## radioactivedecay
Core capabilities
- Inventory class with decay over time, branching, metastable states.
- Unit conversions for activity, mass, moles, and atom counts.
- Cumulative decays and decay chain querying (progeny, branching fractions, decay modes).
- Nuclide data lookup (half-life, decay modes, atomic mass).
- High-precision mode using SymPy (InventoryHP).
- Plotting of decay curves and decay chain diagrams.

Supported I/O
- CSV/TSV inventory reader: `radioactivedecay/fileio.py` (`read_csv`).
- Decay dataset packaged under `radioactivedecay/icrp107_ame2020_nubase2020/`.

Example workflows + datasets
- README and tests: inventory decay, cumulative decays, nuclide queries, plots.
- Example usage in README (Mo-99 decay over 20 h, cumulative decays).

Outputs
- Dicts of activities/masses/moles/atoms; cumulative decays.
- Matplotlib plots (decay curves, decay chain diagrams).

## py-findpeaks
Core capabilities
- Survey of peak detection algorithms with code samples.
- Algorithms include SciPy `find_peaks`, `find_peaks_cwt`, `argrelextrema`, PeakUtils, and several single-file detectors.

Supported I/O
- None (array-based; in-memory vectors).

Example workflows + datasets
- `testing/py-findpeaks/tests/` scripts use `vector.py` and `lows_and_highs.py` arrays.
- README examples show parameter variations (prominence, distance, height).

Outputs
- Peak index arrays printed to stdout.
- Reference plots under `testing/py-findpeaks/images/`.

## PyGammaSpec
Core capabilities
- `GammaSpectrum` class for simple spectrum operations.
- PRA histogram import and counts-per-second normalization.
- Spectrum arithmetic (add/subtract) and moving-average smoothing.
- Polynomial energy calibration with save/load.
- Peak search (SciPy `find_peaks`) and Gaussian + polynomial baseline fitting.
- Gamma and x-ray line lookup; decay product line prediction (via radioactivedecay).
- Spectrum plotting with background subtraction and peak/line annotations.

Supported I/O
- PRA histogram ASCII files (channel, counts).
- Calibration file format with points + coefficients (`docs/utils/calibration.txt`).
- Gamma/x-ray line data CSVs in `src/pygammaspec/data/`.

Example workflows + datasets
- Docs under `docs/Guide/` using:
  - `docs/utils/background.txt`
  - `docs/utils/weak_radium.txt`
  - `docs/utils/calibration.txt`

Outputs
- Peak tables (dicts), fit arrays, and plots.
- Optional image export via plotting functions.

## curie
Core capabilities
- HPGe spectroscopy: peak finding/fitting, SNIP background, and peak summaries.
- Energy, efficiency, and resolution calibration; auto-calibration helpers.
- Attenuation and geometry corrections.
- DecayChain solver with production schedules; fit initial activities or production rates to measured counts.
- Isotope data lookup (half-life, gamma lines, dose rate, decay products).
- Reaction/cross-section library access (ENDF, TENDL, IRDFF) and plotting.
- Element/compound properties: mass coefficients, attenuation, stopping power, range.
- Stacked-target energy loss modeling.

Supported I/O
- Spectrum input: `.Spe`, `.Chn`, `.CNF`, `.IEC`.
- Export: CSV/JSON/DB plus spectrum re-save.
- Stack/compound definitions from CSV/JSON/DB.

Example workflows + datasets
- `examples/spectroscopy_examples.py` with `examples/eu_calib_7cm.Spe`.
- `examples/isotope_decay_examples.py` for decay chains and isotope lookups.
- `examples/reaction_examples.py` for cross-section plots.
- `examples/stack_examples.py` with `examples/test_stack.csv` and `examples/example_compounds.json`.

Outputs
- Peak fit tables, calibration files, plots, cross-section curves.

---

## Capability Matrix (Feature x Repo x FluxForge)

| Feature | radioactivedecay | py-findpeaks | PyGammaSpec | curie | FluxForge |
| --- | --- | --- | --- | --- | --- |
| Inventory with unit conversions (activity, mass, moles, atoms) | YES | NO | NO | NO | YES |
| Decay chain solver with branching/metastable | YES | NO | PARTIAL | YES | PARTIAL |
| Time-varying production schedule + fit to counts | NO | NO | NO | YES | PARTIAL |
| High-precision decay solver (SymPy) | YES | NO | NO | NO | NO |
| Nuclide data lookup (half-life, decay modes, atomic mass) | YES | NO | PARTIAL | YES | PARTIAL |
| Gamma + x-ray line database with energy search | NO | NO | YES | YES | YES |
| HPGe spectrum I/O (SPE/CHN/CNF/IEC) | NO | NO | NO | YES | YES |
| PRA histogram I/O | NO | NO | YES | NO | YES |
| Spectrum arithmetic + smoothing | NO | NO | YES | PARTIAL | YES |
| Peak finding algorithms (SciPy and variants) | NO | YES | YES | YES | YES |
| Peak fitting (Gaussian/Hypermet + baseline) | NO | NO | YES | YES | YES |
| Energy calibration (manual polynomial + auto) | NO | NO | YES | YES | YES |
| Efficiency + resolution calibration | NO | NO | NO | YES | PARTIAL |
| Spectrum plotting with annotations/overlays | PARTIAL | PARTIAL | YES | YES | PARTIAL |
| Export outputs (CSV/JSON/DB, calibration files) | PARTIAL | NO | PARTIAL | YES | PARTIAL |
| Cross-section library access (ENDF/TENDL/IRDFF) | NO | NO | NO | YES | PARTIAL |
| Stopping power / attenuation / range | NO | NO | NO | YES | YES |
| Stacked target energy loss modeling | NO | NO | NO | YES | YES |
