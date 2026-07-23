# FluxForge Session and Workspace Format

FluxForge `.ffs` files are self-contained analysis sessions. New files use the
version 2 envelope and embed a validated `WorkspaceDocument`:

```json
{
  "format": "fluxforge_session",
  "format_version": 2,
  "document": {
    "schema": "fluxforge.workspace_document.v2",
    "schema_version": 2
  },
  "recent_files": [],
  "device_snapshot": [],
  "metadata": {},
  "created_at": "2026-07-22T12:00:00+00:00"
}
```

The document, rather than the GUI widgets, owns spectra, spectrum roles, ROIs,
peak models and assignments, detector profiles and calibrations, fit
diagnostics, viewports, pinned nuclides, tags, plot settings, workflow state,
and provenance. `SelectionBus` remains transient UI state and is never written
as the authoritative ROI model.

## Compatibility

- Version 1 sessions are migrated in memory to version 2 when opened.
- Missing version fields are interpreted as version 1 only when the payload has
  the legacy session shape.
- Unsupported future versions, malformed references, non-finite values,
  reversed ranges, and invalid covariance matrices fail with a field-specific
  error instead of being clipped or silently discarded.
- New saves always write version 2. FluxForge does not write a legacy version 1
  file.

Legacy fields that cannot be interpreted without guessing are retained in the
document extensions area. Existing spectrum arrays, calibration dictionaries,
GPS/device metadata, recent files, and acquisition snapshots are preserved.

## Safe saves

FluxForge serializes and validates the complete payload before touching the
destination. It then writes and synchronizes a temporary file in the same
directory and performs one atomic replacement. If writing or replacement
fails, the previous session remains byte-identical. On Windows, close any
application holding the file and allow OneDrive or other synchronization tools
to release the lock before retrying.

## Workflow presets are not sessions

Named workflow presets store reusable interface configuration such as mode,
library selection, plot preferences, and panel settings. They do not store or
reopen spectrum paths, spectrum roles, peaks, pinned nuclides, detector state,
or analysis results. Use **File > Save Session** when scientific state must be
reproducible.
