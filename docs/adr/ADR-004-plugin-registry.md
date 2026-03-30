# ADR-004: Plugin and Registry Architecture

**Status:** Accepted

## Context

The roadmap adds multiple interchangeable method families: render backends, peak fitters,
unfolders, calibration models, nuclide identification engines, and standards modules.
Without a shared registration pattern, those capabilities would fragment across ad hoc
imports and conditionals.

## Decision

FluxForge will use a generic plugin registry layer with named registries for each
analytical dimension. Built-in capabilities register through that layer, and future
extensions are added without changing every consumer.

## Rationale

A registry layer makes the additive policy practical. It creates a single place to record
available methods, defaults, metadata, and future capability flags.

## Alternatives Rejected

- Hard-coded method selection in each workflow — rejected because it scales poorly.
- Dynamic discovery without a typed registry core — rejected because the roadmap first
  needs explicit, testable built-in registration.

## Consequences

- `src/fluxforge/plugins/registry.py` becomes a core extension surface.
- UI method selectors can be populated from registry metadata.
- Standards mode can lock or restrict methods without deleting them.

## Compliance with Project Rules

- Issue-First Execution: registry work is tracked as Epic E10 and seed issues.
- Additive Capability Policy: registries preserve parallel valid methods.
- User-Choice Policy: registry metadata supports recommended defaults and alternatives.
- Standards-Locked Workflows: standards registries define lockable workflow surfaces.
