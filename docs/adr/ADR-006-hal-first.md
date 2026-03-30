# ADR-006: HAL-First Architecture

**Status:** Accepted

## Context

Live acquisition is a later phase, but the roadmap requires the GUI, session model, and
status surfaces to be ready for hardware state from the beginning. Delaying hardware
abstractions until device support arrives would force later architectural rewrites.

## Decision

FluxForge defines the hardware abstraction layer in Phase 1, including base device
interfaces, status models, registries, and mock devices. Concrete acquisition drivers may
arrive later, but the architecture must reserve their integration points now.

## Rationale

HAL-first scaffolding lets the GUI, session model, and reporting surfaces evolve with
stable device contracts. Mock devices provide testable placeholders without requiring
hardware-specific dependencies.

## Alternatives Rejected

- Postponing all HAL work until live acquisition starts — rejected because it would force
  downstream refactors in the GUI and data model.
- Binding device handling directly into the GUI — rejected because device logic needs a
  reusable, testable layer.

## Consequences

- A base HAL package exists before production drivers.
- Session models can carry hardware provenance fields now.
- Status LEDs and dashboard placeholders can be wired without real devices.

## Compliance with Project Rules

- Issue-First Execution: HAL scaffolding is represented in the Stage 0 seed set.
- Additive Capability Policy: mock devices augment offline workflows rather than replacing
  them.
- User-Choice Policy: hardware-backed and file-backed workflows can coexist.
- Standards-Locked Workflows: standards workflows may use the same device metadata model.
