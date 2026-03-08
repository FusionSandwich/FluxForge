# [Core] Add measured background subtraction API

## Goal
Implement reusable measured-background subtraction in FluxForge core.

## Scope
- Add `subtract_measured_background(sample, background, mode, manual_scale, negative_policy, warn_missing)`.
- Support scale modes: `live`, `real`, `manual`.
- Record subtraction metadata (`scale_factor`, `mode`, `negative_bins`).

## Acceptance Criteria
- Correct subtraction math and variance propagation per channel.
- Supports missing-background warning path.
- Unit tests cover all scale modes and manual-mode validation.
