# Current integration and Qt GUI review — 2026-10-04

The published science integration at `701288a30f418ed7dc331e6deda0044eb220e72e`
had lost the Qt consolidation and report work. Five original consolidation
checks failed independently against that exact revision. This branch merges the
published GUI/report head `a85f3bf716155e569a5cadef249e5387ed84d2a3` (PRs #218
and #233) and preserves the newer science behavior, including unavailable
reaction rows and exact parity-field tolerance precedence. The science branch
was fetched again before publication; there were no additional commits to merge.

## Repairs

- The installed `fluxforge-gui` and frozen-app launcher use `fluxforge.gui.app`.
  Legacy Tk source/tests remain archived outside the distributable package.
- Restored workflow shortcuts, peak-review focusing, scrollable analysis forms,
  measured rate/physical response imports, report snapshots and instrument provenance.
- A calibration dialog now retains its original document, spectrum identity and
  acquisition counts. Switching selection calibrates the original acquisition;
  replacing/reloading the acquisition rejects the old fit and closes the stale
  dialog. Undo/redo preserves the active selection. Applying an inactive fit no
  longer changes the active spectrum label or selection broadcast.
- Calibration plots scroll at a readable minimum height. Apply and Close remain
  outside the scrolling forms; the dialog fits a 1280 × 800 logical-pixel window.
- Mode/theme controls occupy separate rows, and canvas metadata and buttons use
  separate rows. The native main window can now shrink to 1280 pixels wide.

The acquisition defects were reproduced as three failing behavioral tests before
repair. Separate layout tests reproduced the 960-pixel minimum calibration height
and oversized main window before repair. Five new regression checks are registered
in the CI calibration step.

## Validation

Evidence is under `artifacts/validation/qt_gui_review_20261004`. Native Windows
Qt was used, with offline scientific data and isolated in-memory settings for
the layout captures. The final native captures show real spectra, readable plots,
report preview and unfolding controls. A generated report bundle has matching
manifest hashes, typed tables, and a captured spectrum image. Its presentation
snapshot does not establish scientific admission.

The final `verification.json` lists 197 distinct passing cases and the wheel checksum.
Passing groups cover science/parity/reporting, shell/consolidation, report bundles
and provenance, rate/history/queue workspaces, production controls, calibration,
session persistence, and the new acquisition/layout regressions. These are
targeted receipts across integration and repair revisions, not a new full-suite
claim. Two combined native runs timed out after 360 seconds; their logs and
timeout receipts are retained. All constituent groups passed in smaller runs
with numerical-library thread counts bounded to one. The wheel is inspected for
Qt source and entry points and the absence of legacy GUI modules. The Windows
installer/frozen executable was not rebuilt.

## Additional findings

1. **Unsaved-session protection (P1).** `FluxForgeMainWindow.closeEvent` checks
   conversion workers and saves layout but does not consult `_document_dirty`.
   Opening another `.ffs` session also replaces the document without a dirty
   check. A user can lose unsaved calibration/ROI edits. Save/Discard/Cancel
   behavior should cover both paths and cancellation of Save As.
2. **Blocking exports (P2).** Report HTML/PDF/bundle exports run synchronously
   from button handlers. Large snapshots or PDF generation can freeze the GUI.
   Capture immutable GUI state on the main thread, then render/write in a worker
   with progress, failure recovery, and safe shutdown.
3. **Smaller screens (P2).** The native main-window capture at 1280 pixels wide
   retains an approximately 833-pixel minimum height. Further compact layout work
   is needed for 720-pixel logical-height desktops at high display scaling.

These follow-ups are review findings, not repaired behavior in this branch.
The original frozen three-failure evidence remains unchanged in PR #219.
