# ADR-002: Renderer Abstraction and Backend Policy

**Status:** Accepted

## Context

The roadmap requires a high-interaction spectrum canvas with multiple visual layers,
coordinated selections, and future backend flexibility. The current GUI uses Matplotlib,
but the planned shell calls for a rendering abstraction.

## Decision

FluxForge will isolate spectrum rendering behind a `SpectrumCanvas` abstraction. PyQtGraph
is the first production backend. Vispy remains an optional backend stub, and future
lower-level rendering work is documented as a later path rather than a current dependency.

## Rationale

PyQtGraph is the fastest practical route to the planned interaction model in Python while
preserving contributor accessibility. An abstraction layer avoids coupling the roadmap to
a single renderer and keeps future backend experiments additive.

## Alternatives Rejected

- Binding the GUI directly to one rendering library — rejected because it would make
  backend changes unnecessarily invasive.
- Making Vispy the default immediately — rejected because the roadmap explicitly softens
  that swap and keeps PyQtGraph first.
- Staying on Matplotlib as the strategic path — rejected for the next-generation canvas
  because the interaction and redraw model is weaker for the target UX.

## Consequences

- Render backends must register through the plugin registry layer.
- PyQtGraph is the recommended default backend.
- Vispy support is additive and may remain disabled until enough capability exists.

## Compliance with Project Rules

- Issue-First Execution: renderer work is split into explicit epic and feature issues.
- Additive Capability Policy: new backends do not delete existing ones.
- User-Choice Policy: backend selection can remain user-visible where appropriate.
- Standards-Locked Workflows: backend choice must not alter standards computations.
