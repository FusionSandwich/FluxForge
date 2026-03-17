# PR Draft: Remove GUI Matplotlib Warnings and Add Native Desktop Acceptance

## Suggested Title

Remove off-screen Matplotlib warnings and add native desktop GUI acceptance

## Summary

- attach an Agg canvas before off-screen GUI/report figure layout and export so Matplotlib deprecation warnings no longer appear in regression runs,
- add a real desktop-driven GUI acceptance workflow with screenshots, artifact checks, ROI/calibration interaction, and copied CLI verification,
- make the Spectrum workspace controls scrollable so ROI, calibration, efficiency, and fit tools remain reachable in the native desktop UI,
- add packaging extras and CI jobs for Linux and Windows native desktop GUI validation.

## Root Cause

Off-screen report and GUI helper figures were created from bare `Figure(...)` instances. On older Matplotlib versions, calling `tight_layout()` on those figures fell back to the deprecated renderer-acquisition path, which emitted `MatplotlibDeprecationWarning` messages during tests and exports.

## What Changed

- added `src/fluxforge_gui/mpl_helpers.py` to guarantee an Agg-backed canvas for off-screen figures before layout/export,
- routed GUI/report helper figures through that helper in `src/fluxforge_gui/reporting.py` and `src/fluxforge_gui/spectrum_ops.py`,
- added a regression test that treats `MatplotlibDeprecationWarning` as an error for the affected render/export helpers,
- added `tests/gui_desktop_driver.py` and `tests/test_gui_desktop_native.py` for real desktop acceptance evidence,
- updated the Spectrum tab layout with a scrollable controls rail,
- added `gui-test` extras and Linux/Windows native desktop jobs in `.github/workflows/quality-checks.yml`.

## User-Facing Impact

- no intended change to the visible analysis outputs beyond cleaner regression behavior,
- Spectrum-tab controls are easier to reach on typical desktop layouts,
- GUI acceptance now verifies actual screenshots, saved artifacts, and copied CLI commands instead of relying on startup-only coverage.

## Validation

- `python -m pytest -q tests/test_gui_app.py tests/test_gui_desktop_native.py -W error::matplotlib.MatplotlibDeprecationWarning`
- full targeted regression suite rerun after the warning fix and native desktop changes

## Notes

- Linux native desktop acceptance currently runs under Xvfb with a real Tk window, screenshots, and event-driven interaction.
- Windows native desktop acceptance is wired into CI with the same test entry point and a Windows-specific input backend.
