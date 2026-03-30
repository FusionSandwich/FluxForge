# ADR-007: Offline-First Delivery

**Status:** Accepted

## Context

FluxForge targets air-gapped and offline laboratory environments. The roadmap adds richer
GUI, reporting, standards, and data features, but those additions cannot depend on live
internet access.

## Decision

FluxForge remains offline-first. Core analysis, reporting, standards validation, and the
planned GUI migration must function without a browser runtime or network dependency for
supported workflows. Bundled data and explicit provenance replace hidden runtime fetches.

## Rationale

Air-gapped deployment is a hard operating constraint for part of the user base. Offline
execution is also a reproducibility and packaging advantage.

## Alternatives Rejected

- Online-first enrichment features as a baseline assumption — rejected because they break
  target deployment environments.
- Embedded browser runtimes as a strategic dependency — rejected because they increase
  packaging complexity and undermine the native desktop direction.

## Consequences

- New roadmap features must define offline data and asset strategies.
- Remote lookups remain optional enrichments, not required workflow steps.
- Packaging and provenance work must account for bundled libraries and templates.

## Compliance with Project Rules

- Issue-First Execution: offline constraints are reviewable through tracked issues.
- Additive Capability Policy: offline capability is preserved as features expand.
- User-Choice Policy: optional online enrichments cannot displace offline paths.
- Standards-Locked Workflows: compliance workflows must remain runnable offline.
