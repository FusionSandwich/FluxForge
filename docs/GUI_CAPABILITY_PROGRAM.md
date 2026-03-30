# FluxForge GUI Capability Program (Archived Redirect)

This file is no longer a controlling GUI capability roadmap document.

Use these documents for the active GUI direction:

- `docs/FluxForge_Final_Additions.md`
- `docs/FluxForge_Additions_v3_Final.md`
- `docs/FluxForge_Improvement_Guide.docx`

Use this file for current execution state:

- `docs/ROADMAP_EXECUTION_STATUS.md`

Archived legacy reference material remains available here:

- `docs/GUI_PLAN_old.md`
- `docs/GUI_CAPABILITY_PROGRAM_old.md`

Those archived documents are still useful for identifying existing Tk-era
features that need to be carried into the redesign, but they should not be used
to decide the current GUI architecture. The controlling architecture is the Qt
shell under `src/fluxforge/gui/`, with `src/fluxforge_gui/` retained only as a
legacy/archive fallback during migration.
