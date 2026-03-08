# Flux-Wire Raw Count Parity Split

## Problem
QG `GROSS/NET` peak-count columns do not map cleanly onto FluxForge's background-adjusted activity-count path. Using the same count field for both QG parity and activity conversion mixed two different accounting conventions and hid where the disagreement actually came from.

## Current change
- FluxForge now keeps flux-wire QG count parity on a raw-spectrum local-ROI basis:
  - shared `rafm_25cm` energy calibration
  - ROI width `4.0 * FWHM`
  - background width `1 channel`
  - background gap `0 * FWHM`
- Background-adjusted net counts remain in the activity path.

## Why this matters
- QG gross/net parity can improve without corrupting the background-adjusted activity calculation.
- Remaining differences can now be labeled honestly as count-path problems versus efficiency/activity problems.

## Remaining work
- Decide whether the raw-count comparison fields should be surfaced in more user-facing exports.
- Add dedicated regression tests for the raw-count comparison fields on `Co-Cd`, `Ti-RAFM-1a`, and `Sc`.
