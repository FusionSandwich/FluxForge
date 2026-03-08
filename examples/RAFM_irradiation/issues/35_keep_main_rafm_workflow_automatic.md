# Keep Main RAFM Workflow Automatic

## Requirement
The committed RAFM and flux-wire validation workflow must remain fully automatic.

## Why
- The RAFM example is the primary regression and validation case for FluxForge.
- It should exercise the automatic peak identification, counting, isotope assignment, and activity pipeline.
- Manual ROIs are useful for inspection and debugging, but they should not become part of the main RAFM parity workflow.

## Implemented in this pass
- The RAFM runbook now states explicitly that the main `examples/RAFM_irradiation/` workflow is automatic.
- Manual ROI inspection was moved into a separate example under `examples/manual_peak_inspection/`.
- The quick start now points to that separate manual example instead of mixing it into the main RAFM instructions.

## Follow-up
- Keep count-parity and activity-parity fixes inside the automatic analysis path.
- Use the manual example only for debugging, user inspection, and future GUI parity.
