# Qt run reports and instrument provenance

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
ASTM compliance or uncertainty qualification.
The canonical workspace preserves the recorded source identities and analysis
state. Use session export to resume work; this ZIP is not a session loader.
HTML requires the reporting extra's Jinja2 dependency. PDF additionally requires
WeasyPrint and its native libraries. A missing PDF backend or failed rendering
does not create a ZIP. Large tables/plots and PDF rendering can take time; export
currently runs synchronously.

## Instructional acquisition records (#68)

Every capture includes an `instrument_provenance` record for each loaded
acquisition, bound to its workspace spectrum ID, source path/hash and detector.
Live/real time and MCA energy calibration come from that spectrum, retaining
unknown timing as unavailable and inconsistent timing as incomplete.
Amplifier gain (dimensionless), shaping time (microseconds), high voltage
(volts) and count geometry can come from these explicit imported metadata keys:

```json
{"instrument_settings": {
  "amplifier_gain": 12.5,
  "shaping_time_us": 6.0,
  "high_voltage_v": 2500.0,
  "count_geometry": "25 cm from detector endcap, sample centered on axis"
}}
```

Place this mapping in the spectrum's `metadata`. Numeric units are fixed by the
key names; ambiguous keys or unit conversions are not guessed. Missing fields
stay null with an explicit missing-settings list. Invalid/nonfinite values fail
capture. The Qt export dialog accepts explicit user entries for the active
spectrum and labels their source `user_entered`; a blank entry uses imported
metadata. Switch spectra in the workspace to record each acquisition's settings.
Entries are kept separately by spectrum ID and restored when switching back.
They remain in the dialog and exported artifact, and do not edit raw files or
persist into the saved session.

Select **Require complete instrument settings (instructional report)** to block
HTML, PDF and ZIP export until every loaded acquisition has all required fields,
positive live/real times in the correct order, and recorded MCA energy
coefficients. The engine independently checks the instructional receipt against
the captured workspace and explicit user entries before rendering. A complete
settings record establishes documented provenance, not measurement validity or
scientific admission. Ordinary review reports can export incomplete provenance
with its missing fields visible.
