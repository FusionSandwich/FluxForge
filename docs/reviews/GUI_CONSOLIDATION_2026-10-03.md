# GUI consolidation — 2026-10-03

The owner requested archiving the older GUI and improving the newer one across
spectrum review, navigation, and sample setup through unfolding.

## Result

- Qt in `src/fluxforge/gui/` is the sole shipped desktop interface. Both normal
  launch paths and the PyInstaller GUI launcher target Qt.
- All 11 original Tk source modules are preserved unchanged under
  `archive/legacy_gui/src/fluxforge_gui/`. Six historical tests/probes and the
  former CI workflow are preserved beside them. The archive is excluded from
  wheels, native bundles, and normal test discovery/CI.
- ASTM E261/E262 preview helpers remain active in the toolkit-independent
  reporting package, and standards tests no longer import Tkinter.
- The analysis workflow toolbar reuses menu actions for Open Spectrum, Find
  Peaks, Review Peaks, Calibrate, Irradiation, Unfold, and Export. Peak search
  reveals the review panel; Review Peaks restores a hidden analysis dock.
- Dense analysis forms scroll within their existing pages rather than forcing
  an oversized window. The installed wheel was visually checked at 1560×980.
- Unfolding opens from a clean workspace with no synthetic measurements. It
  accepts measured RAFM rate CSV, or reaction-rates JSON followed by a matching
  response bundle JSON with physical energy boundaries. Response/rate reaction
  order, finite values, and energy boundaries are checked. A later reordered
  rate import cannot silently reuse an incompatible response.
- The latest reaction-rate and spectrum folder-queue workspaces from
  `codex/remaining-issues-20261003` at `3ed85ee` were integrated before final QA.
- ADR-008 records the owner's GUI retirement decision and supersedes ADR-001's
  transitional Tk support policy. The active GUI and installation plans now
  direct all improvements to Qt.

## Validation

Local Windows validation used Python 3.12.14 and PySide6 6.11.2. Qt checks used
the offscreen backend; each group ran in its own pytest process.

| Group | Result |
|---|---|
| CLI, ASTM E261/E262, tracker assets, implementation steps | 116 passed |
| Modern Qt shell and tracker assets | 29 passed |
| Production controls, copy, and action catalog | 9 passed |
| Consolidation, reaction-rate, folder-queue, irradiation-history workflows | 10 passed |
| Unfolding workspace | 11 passed |
| Peak/ROI/activity interactions, calibration, session persistence | 28 passed |

Tracker tests overlap the first two groups; these counts should not be summed
as distinct cases. Black checks, compilation, and `git diff --check` also passed.

The final wheel built successfully. Archive/source preservation, package
contents, and entrypoints were verified. The wheel was extracted into an
isolated import directory and initialized the GUI with themes and standards
previews, without importing the archived package.

The committed RAFM-rate replay retains the existing explicitly labelled
simplified response and ill-conditioned-matrix warning. This work does not
qualify those measurements or replace scientific acceptance. Native executable
bundles were routed to Qt but were not built locally; Linux and remote CI results
are not claimed by this local report.

The initial automation attempt waited for the peak-review confirmation dialog;
the test now explicitly accepts it. An experimental test-only Qt cleanup fixture
caused a Windows access violation when disposing earlier dialogs; it was removed.
The final groups above passed without that fixture.

## Review and adoption

Review branch: `codex/archive-legacy-gui`, based on
`codex/remaining-issues-20261003`. This is a stacked change for review; the main
checkout and installed user environment were left available to the ongoing
FluxForge analysis chats. After merging, reinstall FluxForge from the updated
checkout with `.[native-gui,reporting]` to use the consolidated release paths.
