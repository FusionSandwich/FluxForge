# [Tests] Add RAFM background integration and regression coverage

## Goal
Verify RAFM shared-background behavior and prevent regression.

## Scope
- Add tests for scale modes and exact uncertainty formulas.
- Add warning-path tests for missing background and clipping.
- Add RAFM integration tests using `examples/RAFM_irradiation/background.ASC`.

## Acceptance Criteria
- New tests pass in CI/local environment.
- Regression subset passes:
  - `tests/test_spectrum_math.py`
  - `tests/test_flux_wire_parity.py`
  - `tests/test_cli_app.py`
  - new background/RAFM tests.
