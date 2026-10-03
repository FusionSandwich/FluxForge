# Complete test environment

Issue [#21](https://github.com/FusionSandwich/FluxForge/issues/21) uses
`environment.yml` as the shared local test environment. Create it from the repo
root so that the editable install binds to this checkout:

```console
conda env create --file environment.yml
conda activate fluxforge
python tools/check_test_environment.py
python -m pytest -q tests/test_naa_ann.py tests/test_tensorflow_env.py
python -m pytest -q --junitxml=artifacts/full-suite.xml
```

The manifest uses conda-forge, Python 3.11, NumPy 1.26.x and TensorFlow 2.15.1.
The editable install includes the dev, native GUI, desktop test, ML, reporting
and packaging extras. PyUnfold is also installed for its optional reference
tests. Cairo and Pango provide the native libraries used by PDF reporting.
PySide6 and PyQtGraph come from conda-forge so their native dependencies share
the environment's ABI. `matplotlib-base` avoids a redundant GUI backend.

For a headless test run, set `MPLBACKEND=Agg` and `QT_QPA_PLATFORM=offscreen` in
the test process. TensorFlow CPU tests need no CUDA installation. On a busy
machine, `TF_NUM_INTRAOP_THREADS=1`, `TF_NUM_INTEROP_THREADS=1`, and
`OMP_NUM_THREADS=1` keep the ML checks bounded.

Micromamba can create and run the same manifest without changing the shell's
activation configuration. See its [official installation guide](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html).
Use a short root-prefix path on Windows to keep extracted package paths within
the platform path limit:

```console
micromamba create --root-prefix C:\mm --prefix C:\ffenv --file environment.yml --yes
micromamba run --root-prefix C:\mm --prefix C:\ffenv python tools/check_test_environment.py
```

The smoke check verifies actual imports, exact TensorFlow/NumPy compatibility,
CPU execution and PDF generation. It exits nonzero for missing packages,
native-library failures or version drift. A passing smoke check does not replace
the targeted or full suite. Tests requiring CUDA, a desktop display, external
transport fixtures or a licensed application can still skip for those stated
reasons; missing TensorFlow is not an acceptable skip in this environment.
