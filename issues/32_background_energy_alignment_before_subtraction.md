# Background Energy Alignment Before Subtraction

## Problem
The RAFM raw spectra were being analyzed with the shared `@25 cm` sample calibration while `background.ASC` still carried its own raw-file calibration. Subtracting those spectra channel-by-channel mixes different energy grids.

## Change
- FluxForge now resamples the measured background onto the sample energy grid before subtraction when the energy calibrations differ.
- The RAFM workflow also loads `background.ASC` with the shared `rafm_25cm` calibration override.

## Outcome
- The background-adjusted activity path is now energy-consistent.
- This does **not** remove the remaining QG gross/net count mismatches, which confirms those are now peak/ROI protocol issues rather than a background-calibration mismatch.
