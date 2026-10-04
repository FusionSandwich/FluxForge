# ADR-008: Single supported Qt GUI and Tk source archive

**Status:** Accepted (2026-10-03)

## Context

FluxForge retained both a Tkinter prototype and the newer PySide6/Qt interface.
The owner explicitly requested archiving the older interface and focusing
improvements on the newest interface, across spectrum review, navigation,
and sample setup through unfolding.

## Decision

Qt under `src/fluxforge/gui/` is the sole supported and shipped GUI. Preserve
the Tk package and its tests/probes under `archive/legacy_gui/`, remove its
installed launcher and active CI jobs, and route native bundles through Qt.
Supersede ADR-001's transitional support policy. Preserve scientific APIs and
toolkit-independent standards preview helpers in the active package.

## Consequences

- Both public GUI launch paths and native bundles use Qt.
- Historical Tk source can be inspected or run explicitly from the archive.
- New GUI work and release acceptance target Qt on Windows and Linux.
- Shell retirement does not remove scientific methods, standards, or CLI commands.
- The workflow toolbar shares existing actions and their state/standards locks.
- Measured unfolding opens without an example; no synthetic rates populate that
  workspace. RAFM CSV retains its explicitly labelled simplified response.

This is an owner-authorized shell retirement under ADR-003, not a claim that
all Tk interaction behavior has reached Qt parity.
