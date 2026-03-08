# [Analysis] Integrate measured background into HPGe processor

## Goal
Apply measured background subtraction before HPGe peak/activity analysis.

## Scope
- Add background parameters to HPGe analysis entrypoints.
- Clip negative bins only for algorithms that require non-negative counts.
- Include subtraction-informed uncertainty in activity uncertainty floor.

## Acceptance Criteria
- HPGe analysis runs with and without supplied background.
- Missing background emits warning and continues.
- Activity uncertainty includes subtraction-informed contribution.
