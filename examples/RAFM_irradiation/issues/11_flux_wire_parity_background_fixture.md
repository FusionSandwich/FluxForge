# [Tests] Make flux-wire parity fixtures self-contained with shared background

## Goal
Ensure copied RAFM flux-wire parity tests do not depend on example-path lookups or external repositories for the shared measured background.

## Scope
- Copy the shared `background.ASC` into the flux-wire raw fixture tree.
- Update parity tests to pass the bundled background into raw-spectrum analysis.
- Validate the profile-driven background path in the RAFM integration test.

## Acceptance Criteria
- `test_flux_wire_parity.py` no longer triggers the "no background spectrum was provided" warning.
- Parity fixtures remain self-contained under `tests/data/`.
- RAFM preset integration tests confirm the shared background is resolved from committed repo data.
