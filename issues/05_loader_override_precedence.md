# [IO] Unify calibration/efficiency override precedence in loaders

## Goal
Ensure `.ASC/.TXT/.txt` loading supports header values plus user overrides consistently.

## Scope
- Extend Genie and flux-wire readers with optional override inputs.
- Parse header efficiency coefficients when present.
- Apply user overrides last.

## Acceptance Criteria
- Header-only, user-only, and mixed override tests pass.
- Override precedence is deterministic and documented.
