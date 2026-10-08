# Archived Tkinter GUI

Archived on 2026-10-03 at the owner's request. The PySide6/Qt GUI in
`src/fluxforge/gui/` is now the sole supported desktop application.

This archive preserves the former `src/fluxforge_gui/` package, its six
test/probe files, and a snapshot of the previous CI workflow. The source package
is excluded from wheel/package discovery and native bundles. Its console script
has been removed. Normal test discovery and CI cover the Qt GUI.

All new GUI work belongs in `src/fluxforge/gui/`. The shared scientific and CLI
implementations remain active. Toolkit-independent ASTM E261/E262 previews now
live in `src/fluxforge/reporting/standards_preview.py`, so standards tests no
longer import Tkinter.

For historical investigation only, run from the repository root with an
environment containing the usual FluxForge dependencies and Tkinter:

```powershell
$env:PYTHONPATH = "$(Get-Location)\archive\legacy_gui\src;$(Get-Location)\src"
python -m fluxforge_gui.app --project-dir .
python -m pytest archive/legacy_gui/tests/test_gui_app.py
```

On Linux use `PYTHONPATH=archive/legacy_gui/src:src`. Desktop probes additionally
need an interactive desktop or Xvfb and the `gui-test` extra. The archived test
setup and probe paths have been adjusted for this location; historical parity
records retain their original references. The CI snapshot is reference material
and is not an active workflow.

Use `fluxforge gui` or `fluxforge-gui` for current work.
