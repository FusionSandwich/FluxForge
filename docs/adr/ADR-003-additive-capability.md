# ADR-003: Additive Capability Policy

**Status:** Accepted

## Context

FluxForge is expanding rapidly across GUI, spectroscopy, standards, reporting, and
unfolding. Without a written policy, newer implementations could displace still-valid
capabilities simply because they are newer.

## Decision

FluxForge adopts an additive capability policy: new valid analytical capabilities must be
added without deleting existing valid capabilities unless the existing capability is
demonstrably broken, unsafe, or no longer scientifically defensible.

## Rationale

Scientific software depends on reproducibility, operator trust, and historical continuity.
Removing valid methods creates unnecessary churn and breaks workflows that still need to be
supported.

## Alternatives Rejected

- Replacing older methods whenever a preferred method appears — rejected because it erodes
  trust and breaks reproducibility.
- Treating additive preservation as a case-by-case convention — rejected because the
  roadmap needs an explicit review rule, not an informal norm.

## Consequences

- Existing GUI and analytical surfaces remain until parity exists.
- Registry, mode, and reporting designs must preserve method provenance.
- PR review must check whether new work removes an existing valid path.

## Compliance with Project Rules

- Issue-First Execution: policy impact is reviewable through tracked issues.
- Additive Capability Policy: this ADR formalizes the rule itself.
- User-Choice Policy: preserved methods remain selectable.
- Standards-Locked Workflows: standards locking constrains standards mode only.
