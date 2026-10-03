# Issue 21: complete Conda test environment

Expanded `environment.yml` to include the dev, native GUI, desktop-test, ML,
bundle and reporting extras, PyUnfold, lxml, Cairo and Pango. NumPy remains
1.26.x and TensorFlow remains 2.15.1. Native Qt and PyQtGraph are supplied by
conda-forge rather than mixing a pip Qt binary with the Conda native runtime.

Validation uses an isolated Windows Conda-compatible environment created with
Micromamba 2.9.0 and Python 3.11.16. The final import smoke check passes all
19 required modules, runs a TensorFlow CPU matrix multiplication and creates a
WeasyPrint PDF. `python -m pip check` reports no broken requirements.

The targeted ML/environment tests completed: **22 passed, 2 skipped**. Both
skips are Linux-style pip CUDA-library discovery; TensorFlow-dependent tests
executed and did not skip for a missing TensorFlow installation.

Installation exposed two reproducible Windows problems: excessive package
cache path length and a pip Qt DLL mismatch in the Conda process. A shorter
Micromamba root and the conda-forge Qt provider resolved them. The latter is
encoded in the manifest; no system runtime, shell profile or other agent's
environment was changed.

The full suite will run against a frozen checkout using the existing isolated
acceptance runner. Full-suite results remain pending; this issue is not yet
declared complete. See `docs/TEST_ENVIRONMENT.md` for commands and the smoke
check's nonzero failure behavior.
