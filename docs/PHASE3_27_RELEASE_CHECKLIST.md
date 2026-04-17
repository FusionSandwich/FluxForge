# Phase 3.27 GUI Release-Blocking Checklist

Status: active checklist
Last Updated: 2026-04-17

Use this checklist before marking Step 3.27 complete.

## Test and Verification

- [ ] Targeted Qt regression tests for new parity surfaces passed.
- [ ] Full FluxForge test suite passed (`pytest -q -rs`).
- [ ] `gui-acceptance-check` CLI report indicates `ready=true` for release payload.

## Native Probe Evidence

- [ ] `artifacts/gui_review/phase325_probe/index.html` exists and is current.
- [ ] `artifacts/gui_review/phase326_probe/index.html` exists and is current.
- [ ] `artifacts/gui_review/phase327_probe/index.html` exists and is current.

## Artifact Review and Manual Validation

- [ ] Artifact galleries reviewed for interaction correctness.
- [ ] Modern Qt GUI launched natively with `/usr/bin/python`.
- [ ] Laptop and desktop sizing checks completed with no clipped controls.

## Documentation and Traceability

- [ ] `docs/ROADMAP_EXECUTION_STATUS.md` updated with 3.26 and 3.27 evidence.
- [ ] `docs/FLUXFORGE_CONSOLIDATED_MASTER.md` status rows updated when sequence gates close.
- [ ] `docs/GUI_PLAN.md` updated if acceptance expectations changed.
- [ ] `docs/FluxForge_Testing_Master.md` updated if test contract changed.
- [ ] `docs/PHASE3_EXECUTION_HANDOFF.md` refreshed with final branch state and follow-up notes.
