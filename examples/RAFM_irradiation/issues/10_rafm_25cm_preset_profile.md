# [RAFM] Add explicit `rafm_25cm` detector profile

## Goal
Provide one explicit, checked-in profile for RAFM 25 cm counting geometry so users can analyze the bundled irradiation data without manually re-entering shared detector metadata.

## Scope
- Store the profile as repo metadata, not path-based hidden logic.
- Include shared efficiency coefficients, detector geometry parameters, resolution coefficients, and the shared background file reference.
- Expose the profile through CLI and programmatic analysis entrypoints.
- Preserve raw `.ASC` file header energy calibration by default.

## Acceptance Criteria
- `--profile rafm_25cm` works on `ingest` and `ingest-batch`.
- Flux-wire and RAFM analysis functions accept `profile_name="rafm_25cm"`.
- Explicit user overrides still take precedence over the profile.
- Raw `.ASC` spectra keep their own `A,B,C` calibration unless explicitly overridden.
