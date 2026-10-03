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

Full collection at frozen commit `36ed262` contained **2,081 cases**. The first
run was interrupted; its recorded cases were retained and only the unreported
cases were resumed. Combined coverage is complete: **2,052 passed, 26 skipped,
3 failed, 0 missing**. The result is not a passing full-suite receipt.

Two failures shared a parity field-name bug: the manifest's declared
`energy_keV_abs` tolerance was ignored for `energies_keV[]` and `first_peak_keV`.
The comparator now honors that declaration, retains explicit-field precedence,
and rejects changes beyond the declared tolerance. Follow-up parity/irradiation
and production-catalog checks pass (21 tests) on the branch incorporating the
other agent's published activation fixes.

The third failure was a legacy desktop coordinate-click wait for a calibration
button. A separate run using the driver's existing Windows CI widget-event mode
passes. The original real-coordinate timeout is retained; this follow-up does
not claim a passing physical mouse-input replay.

Skips cover external transport/unfolding reference files, Linux-only CUDA library
discovery, and a POSIX directory-fsync scenario. They do not skip TensorFlow or Qt due
to a missing installation. The complete environment is supplied and usable;
single-revision full-suite acceptance and scientific source qualification remain
separate open work. See `docs/TEST_ENVIRONMENT.md` for commands and the smoke
check's nonzero failure behavior.
