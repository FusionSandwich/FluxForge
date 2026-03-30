# ADR-005: Three-Mode GUI and Standards Locking

**Status:** Accepted

## Context

The earlier planning baseline distinguished only between Simple and Expert visual
complexity. The merged roadmap requires a third mode that enforces standards-defined
workflows without removing free-form analysis.

## Decision

FluxForge adopts three GUI modes:

- `Simple`
- `Expert`
- `Standards`

Standards mode applies explicit workflow locking for the active standard while preserving
the current analysis state and the full Expert mode for non-compliance work.

## Rationale

Standards compliance is a workflow constraint, not just a visual simplification. A
separate mode cleanly expresses that distinction and avoids overloading the meaning of
Simple mode.

## Alternatives Rejected

- Keeping only Simple and Expert — rejected because it cannot express standards locking.
- Hiding standards rules inside Expert mode — rejected because it would make compliance
  ambiguous and error-prone.

## Consequences

- Mode state must include an active standards context when Standards mode is selected.
- Standards-controlled inputs must be visibly locked rather than silently overridden.
- Session and report provenance must record the active mode and standard.

## Compliance with Project Rules

- Issue-First Execution: standards-mode work is tracked through ADR and feature issues.
- Additive Capability Policy: Standards mode does not replace Expert mode.
- User-Choice Policy: Expert mode still exposes alternative valid methods.
- Standards-Locked Workflows: this ADR defines the operating model.
