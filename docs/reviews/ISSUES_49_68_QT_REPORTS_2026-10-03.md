# Qt report bundles and acquisition provenance

Draft review: https://github.com/FusionSandwich/FluxForge/pull/233.
Addresses open issues #49 and #68; both remain pending integration.

Application source tested: `3163f1ccf8cb01727632d3ae013e7f6314508473`.
The branch incorporates the published Qt-only migration from PR #218 at
`b618f5e` and acceptance fixes through PR #216's `a4d1b24`. Tk sources and tests
remain under `archive/legacy_gui`; the console GUI launcher and native bundle
entrypoint use `fluxforge.gui.app`. The existing parity field mapping retains
explicit-field precedence while integrating the acceptance branch's other fixes.

## Behavior

Generate Report captures visible Qt plots without changing their viewport,
production tables, current inputs and canonical workspace state into HTML,
JSON and PNG assets, with a SHA-256 manifest and optional PDF. Preview and export
share a detached capture. Existing ZIPs are preserved. The preview has a white
page background so dark Qt themes do not obscure report text.

Acquisition records retain spectrum/source/detector identity, measured live/real
time and MCA calibration. Instrument gain, shaping time, voltage and geometry
come only from documented metadata or explicit spectrum-bound user entries.
Ordinary reports expose missing settings; instructional reports require every
loaded acquisition's settings. Rendering independently checks the instructional
receipt against the captured workspace, preventing a stale or forged receipt
from passing the gate. Manual entries stay in the dialog and artifact rather
than editing raw data or session persistence.

## Validation

- Frozen targeted run: **129 passed**, zero failures/errors/skips, three
  deselected optional-backend cases. Those three cases passed separately:
  **132 targeted tests passed** in total. Coverage includes report capture,
  provenance, Qt consolidation/control catalog, reaction-rate inputs, queue
  dismissal, parity precedence, joint covariance and RAFM independence/workflows.
- Runtime for pytest: Python 3.12.14, NumPy 1.26.4, SciPy 1.17.1.
- Scoped Black, Flake8 and Git whitespace checks passed.
- The Conda reporting environment produced a real **61,993-byte PDF**, six
  captured tables and one visible Qt spectrum plot. Visual inspection verified
  the current Qt report dialog, plot and readable preview. The probe explicitly
  registered Matplotlib's bundled DejaVu font in Qt for rendering; product font
  settings were unchanged.
- A clean source-snapshot wheel contains the new report modules and template,
  excludes `fluxforge_gui`, exposes the Qt GUI launcher, and renders the template
  from an isolated wheel import.

The frozen run emitted the existing ill-conditioned RMLE matrix warning in the
committed diagnostic example. This report does not qualify that example's
numerical result or claim a complete final full-suite pass.

Clean wheel SHA-256:
`10f761da5e27c3f8ecf25c82d7c32bd3116105aba40e9fde542bf3645c4a9a12`.
PDF SHA-256:
`939c54772677448ed105ad6f9cac9253b0a0c7f5eaef52589de3d7cffe02b8ea`.

Local logs/JUnit, source archive, wheel, isolated wheel import, ZIP, PDF and
screenshots are preserved under
`artifacts/validation/issue_work_20261003/`. They are not added as generated
repository artifacts. The first incremental wheel failed validation because
`build/lib` retained old Tk modules from a pre-archive build. That failed artifact
is preserved separately and excluded from acceptance. The local stale build
cache was moved out of the build path. Build release wheels from a clean checkout
or committed source snapshot, and inspect their contents after GUI migration.

Exports remain synchronous; large captures/PDFs can take time. Validation was
Windows/offscreen, with a real native PDF backend. Linux acceptance, executable
packaging, full-suite acceptance and scientific qualification are not claimed.
Failure follow-ups are owned by PR #219; their original frozen evidence remains
preserved and this feature work does not duplicate them.
