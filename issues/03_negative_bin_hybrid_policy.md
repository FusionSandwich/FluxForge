# [Core] Implement hybrid negative-bin policy with warnings

## Goal
Keep signed subtraction output for accounting while clipping only for algorithms that require non-negative counts.

## Scope
- Add helper for algorithm-specific non-negative clipping.
- Emit warnings when clipping is applied.
- Document interpretation of negative bins.

## Acceptance Criteria
- Hybrid mode keeps negative bins in stored spectrum.
- Algorithms that require non-negative inputs receive clipped arrays.
- Warning tests pass for clipping paths.
