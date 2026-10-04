# Qt run reports

Issue #49 adds **Generate Report** to the supported Qt **Export Report** dialog.
Choose a bundled template and an export file path. Generate Report captures the
current workspace and writes a ZIP beside that path (`run.html` becomes
`run.zip`). Existing ZIPs are preserved; choose a new filename for another run.
Export HTML and Export PDF remain available as individual files.

Each ZIP contains:

- `report.html`, with embedded plot images for standalone viewing;
- `snapshot.json`, with typed analysis parameters, current input values, table
  headers/rows, selected and resolved libraries, irradiation segments, viewport
  settings, and the canonical workspace document;
- `views/*.png`, captured from the actual visible Qt plots at their current
  zoom, scale and overlay settings;
- `manifest.json`, with byte sizes and SHA-256 hashes for the report, snapshot
  and images;
- `report.pdf` when **Include PDF in run bundle** is selected and a working
  WeasyPrint installation is available.

Hidden production tabs' instantiated tables and inputs are included. Plots are
captured only when visible in the workspace or an open child dialog; the export
does not open tabs or regenerate analysis. Developer panels and the report
dialog's own controls are excluded. Empty tables remain explicit. GUI capture
runs on the Qt thread, and preview and export use the same detached capture.
Text from inputs and displayed tables is escaped in the report.

These are presentation snapshots. They do not establish scientific admission,
ASTM compliance, uncertainty qualification, or complete acquisition provenance.
The canonical workspace preserves the recorded source identities and analysis
state. Use session export to resume work; this ZIP is not a session loader.
HTML requires the reporting extra's Jinja2 dependency. PDF additionally requires
WeasyPrint and its native libraries. A missing PDF backend or failed rendering
does not create a ZIP. Large tables/plots and PDF rendering can take time; export
currently runs synchronously.
