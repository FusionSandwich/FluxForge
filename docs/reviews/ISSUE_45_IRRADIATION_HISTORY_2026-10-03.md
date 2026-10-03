# Issue 45: irradiation history input form

Added Tools → Irradiation History with explicit Start/Stop/Relative power rows.
Rows map to the existing `IrradiationSegment` model. Leading and internal gaps
become zero-power segments; a supplied shutdown is never removed. Chronology,
overlap, finite numbers, positive durations and missing power are checked before
application or export. No nominal power, duration or measured specimen value is
inserted into an incomplete row.

The dialog emits validated segments to its current main window, retains edits
when reopened, and loads/exports a versioned JSON timeline with the existing
`irradiation.segments` backend shape. Imported duration rows must agree with the
original timeline. Invalid imports preserve rows; invalid exports preserve files.
These user-entered histories do not certify operating logs or supply uncertainty.
Session persistence is documented as a limitation; JSON provides explicit saving.

Validation: 14 model checks and 2 Qt behavior checks passed in the complete
Windows Conda environment. They include independent decay arithmetic across
shutdowns, invalid chronology/power inputs, button-driven application/removal,
import/export and main-window retention. The production action catalog covers
all six controls and the menu action; the production catalog suite passed.

No changes were made to the activation engine being corrected by the other
local agent. The frontend uses its existing segment model.
