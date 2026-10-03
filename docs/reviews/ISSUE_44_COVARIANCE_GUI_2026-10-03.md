# Issue 44: covariance and correlation heatmaps

Added a modern GUI Tools action and reusable dialog for declared activity,
nuclear-data and other analytical covariance matrices. Users load an artifact,
switch between covariance/correlation, inspect labeled numeric cells and export
a PNG. Null covariance remains unavailable; zero-variance correlations remain
undefined. Invalid covariance cannot be displayed as a valid uncertainty model.

Validation in the existing Windows Python 3.12 / PySide6 environment:

- Covariance mathematical and Qt behavior tests: **26 passed**.
- Final Qt dialog and production GUI catalog tests: **11 passed**.
- Counterexamples include indefinite tiny-unit matrices, negative variance,
  invalid zero-variance cross terms, ambiguous JSON, unit changes spanning
  1e-300 to 1e300, mixed units, and singular cancellation.
- An offscreen Windows screenshot and PNG export were visually inspected.
  The offscreen screenshot explicitly registered Matplotlib's bundled DejaVu
  Sans font because the Windows offscreen plugin did not discover its default
  font. Product font settings are unchanged.

See `docs/COVARIANCE_INSPECTION.md` for the artifact contract. The illustrative
screenshot is synthetic; no scientific calibration or measurement is inferred.
Native desktop accessibility and other operating systems remain unverified.
Issue #44 remains open pending integration.
