# [Environment] Provide one full Conda environment for FluxForge tests

## Goal
Make it straightforward to install a single working environment that runs the complete FluxForge test suite, including optional-library paths.

## Scope
- Expand `environment.yml` with the packages needed for the full suite.
- Pin the tested TensorFlow and NumPy combination used by the passing environment.
- Include editable install of the repo.
- Validate the environment with import smoke tests and pytest runs.

## Acceptance Criteria
- `environment.yml` installs TensorFlow, `h5py`, and `pyyaml`.
- The targeted pytest command and full test suite run in that environment.
- TensorFlow-dependent tests no longer skip because TensorFlow is missing.
