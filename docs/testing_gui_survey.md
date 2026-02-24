# GUI Survey of `testing/` Codebases

This lists GUI capabilities (if any), how they work, and what they use for each
repo under `testing/`.

## actigamma
- GUI: none.
- How it works: library-only; examples generate plots programmatically.
- Tech: Python; optional matplotlib for plotting in examples.

## becquerel
- GUI: none.
- How it works: library-only; plotting via user code.
- Tech: Python; matplotlib for plots.

## curie
- GUI: none.
- How it works: library-only; plotting helper functions can show matplotlib
  figures.
- Tech: Python; matplotlib.

## gamma_spec_analysis
- GUI: none.
- How it works: functions + notebooks; plotting helpers for spectra/peaks.
- Tech: Python; matplotlib (notebook usage).

## gmapy
- GUI: none.
- How it works: library + Jupyter notebook examples.
- Tech: Python; Jupyter for interactive usage.

## hdtv
- GUI: yes (interactive spectrum/matrix display).
- How it works: command-line shell with keybindings drives GUI windows for
  spectrum/matrix views, peak search, and fitting; supports batch commands.
- Features: loads compressed/uncompressed and ROOT spectra plus 2D matrices,
  calibration tools, peak find/fit, keyboard-driven viewport controls.
- Tech: Python + C++; PyROOT/ROOT for GUI/plotting; prompt_toolkit for CLI.

## irrad_spectroscopy
- GUI: none.
- How it works: library + Jupyter notebook walkthroughs; plots produced by code.
- Tech: Python; matplotlib; Jupyter.

## NAA-ANN-1
- GUI: none.
- How it works: Fortran data augmentation + Jupyter notebook ANN workflow.
- Tech: Fortran; Python/Jupyter.

## Neutron-Spectrometry
- GUI: no dedicated GUI; CLI tools produce plots.
- How it works: C++ command-line apps; plotting done via ROOT outputs.
- Tech: C++; ROOT for plotting/visuals.

## Neutron-Unfolding
- GUI: none.
- How it works: Python scripts run algorithms and plot results.
- Tech: Python; matplotlib.

## npat
- GUI: none.
- How it works: library-only; plotting via API methods.
- Tech: Python; matplotlib.

## peakingduck
- GUI: none.
- How it works: peak-finding library; plotting adapter for visualizations.
- Tech: C++ (pybind) + Python; matplotlib wrapper.

## pyunfold
- GUI: none.
- How it works: library-only; user code handles plots.
- Tech: Python.

## SpecKit
- GUI: yes (desktop GUI).
- How it works: Tkinter app with tabbed workflow; file pickers drive CSV I/O;
  real-time plots embedded in the UI.
- Features: data prep, inversion controls, uncertainty/error plots, spectrum
  comparison, and export of figures/CSVs.
- Tech: Python; tkinter/ttk; matplotlib (TkAgg backend).

## PyGammaSpec
- GUI: none.
- How it works: library-only; plotting utilities for spectra in code or docs.
- Tech: Python; matplotlib.

## Gamma-MCA
- GUI: yes (web/PWA).
- How it works: progressive web app with Plotly-based spectrum viewer; file
  import/export and optional live plotting via WebSerial/WebUSB.
- Features: multi-format import (CSV/TKA/XML/JSON), JSON/XML export with sample
  info + calibration + multiple spectra, polynomial calibration, auto peak
  detection, isotope list overlay, serial console, light/dark themes, offline
  install.
- Tech: TypeScript/JS; Plotly.js; Bootstrap; runs in modern browsers
  (Chromium required for WebSerial/WebUSB).

## py-findpeaks
- GUI: none.
- How it works: collection of peak-finding algorithm examples and comparisons.
- Tech: Python; SciPy/Numpy (varies by algorithm).
