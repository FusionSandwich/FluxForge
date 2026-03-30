# ADR-001: GUI Stack and Migration Path

**Status:** Accepted

## Context

FluxForge currently ships a native Tkinter + ttk + Matplotlib desktop GUI. The roadmap
requires a dockable, multi-panel, higher-performance interface with stronger layout
persistence, richer interaction models, and long-term support for standards-locked
workflows.

## Decision

FluxForge adopts PySide6 as the strategic GUI framework for the next-generation desktop
application. The existing Tk GUI remains supported until the PySide6 shell reaches
functional parity for the workflows it replaces.

## Rationale

PySide6 provides the widget system, docking model, theming hooks, persistence surfaces,
and ecosystem compatibility needed for the roadmap. Keeping the current Tk GUI alive
during migration preserves existing operator workflows and respects the additive
capability policy.

## Alternatives Rejected

- Dear PyGui as the primary shell — faster to prototype, but weaker for the complex,
  dock-heavy desktop workflow the roadmap specifies.
- Continuing exclusively with Tkinter — adequate for the existing prototype, but not the
  strongest fit for the planned docking and standards-locking architecture.
- Browser or Electron shell — rejected due to heavier runtime, packaging overhead, and
  weaker offline-first alignment.

## Consequences

- New strategic GUI modules live under `src/fluxforge/gui/`.
- PySide6 remains additive until parity exists; the Tk GUI is not removed as part of the
  migration scaffold.
- Packaging will need native GUI extras for the PySide6 stack.

## Compliance with Project Rules

- Issue-First Execution: tracked through Stage 0 ADR and seed issue scaffolding.
- Additive Capability Policy: preserves the Tk GUI during migration.
- User-Choice Policy: leaves room for multiple front ends during the transition.
- Standards-Locked Workflows: supports the richer state and widget model needed later.
