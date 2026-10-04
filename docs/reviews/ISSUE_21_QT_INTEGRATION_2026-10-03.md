# Follow-up adoption of the current Qt GUI

The owner's follow-up requested checking published FluxForge updates and using
the new GUI. This worktree now includes PR #218's published
`codex/archive-legacy-gui` at **`b618f5e3bf41323d613b8193b43a8ec26f94ec90`**.
Integration revision **`bade0779be0169f1c69267db380cf67b68d7a5f3`** merges that
branch into the isolated diagnostic branch. Normal console/CLI/native-bundle
launchers target **`fluxforge.gui` (PySide6/Qt)**. Tk is archived and excluded
from active test/package discovery. The diagnostic driver and its two geometry
tests moved beside the archived driver, with source-hash paths adjusted.

No other chat's checkout or shared installed environment was edited. The draft
diagnostic PR is now stacked on #218; the consolidation still awaits review and
merge into a release. This report describes this worktree, not a claim that
GitHub `main` or the user's existing installation has been upgraded.

## Current-source checks

All groups below use **the same committed integration revision `bade0779`** and
the complete Conda environment, Python 3.11 and PySide6 6.11.2. `PYTHONPATH`
points to this worktree's `src`; offline mode is enabled. Each group runs in its
own bounded pytest subprocess. The final passing case sets do not overlap.

| Group | Profile | Result |
| --- | --- | --- |
| GUI consolidation, toolbar, scrolling, measured-data unfolding | Qt offscreen | 5 passed |
| Modern shell, sample loading, actions and renderer interactions | Qt offscreen | 23 passed |
| Irradiation history, reaction rate and spectrum folder queue | Qt offscreen | 5 passed |
| Energy/FWHM/efficiency calibration workspace | Qt offscreen | 20 passed |
| Workspace sessions | Qt offscreen | 5 passed |
| Reference parity runner, energy tolerances and manifests | Same process environment, no desktop input | 15 passed |
| Production controls, action catalog and visible copy | Native Windows Qt | 9 passed |

Total: **82 selected cases passed, 0 skipped**. This is targeted coverage, not a
full-suite result. The first combined run was stopped after failing the source
directory assertion and progressing into production tests. A separate first
consolidation run passed four workflow cases but failed the same assertion:
the old source directory contained only ignored `__pycache__/*.pyc` from the
earlier Tk reproductions. The cache was preserved under the local evidence
directory, removing the stale source-directory appearance. No original source
or failure evidence was deleted. All five consolidation cases then passed.

The first production group in the offscreen profile exceeded its 90-second
subprocess limit. Its partial output and timeout record are retained and are
not counted as a completed pass. The native Windows Qt replay, with a
180-second bound, completed **9 passed in 115.06 seconds**. A separate isolated
production-copy diagnostic also passed. This evidence does not establish the
cause of the slower batch or a timing fix; native production acceptance is
separate from the incomplete offscreen attempt.

The numerical warning on the committed RAFM simplified response and calibration
conditioning warnings remain visible in the logs. These UI regressions do not
qualify scientific accuracy or external source data.

## Native Qt rendering

An isolated-settings snapshot script constructed the current Qt main window,
showed its workflow toolbar, loaded the explicitly requested bundled demo
spectrum, rendered screenshots, and closed it. Its receipt verifies that the
GUI module came from this worktree's `src`, normal entrypoints target Qt, no
`fluxforge_gui` module was imported, and archived Tk sources do not exist under
the active package directory. It records source hashes, prefix, Qt version,
font families, native `windows` platform and actual geometry.

The native window used Segoe UI and readable labels; the logical size was
1539 by 844 (the screenshot includes the desktop scaling factor). The example
screenshot is demonstration data with zero automatically detected peaks;
peak-detection behavior is exercised by the workflow tests, not claimed from
that screenshot alone. User settings were isolated in memory.

The first offscreen snapshot showed missing font glyphs and an inflated minimum
width of 2108 pixels. That failed diagnostic image and its log remain evidence.
The native Windows renderer displayed readable text with a minimum-size hint
of 1448 by 805. No Qt product layout was changed to compensate for an offscreen
font environment. Native Windows rendering, offscreen testing and historical
Tk coordinate input are distinct profiles.

## Other published updates and required reconciliation

The published remaining-issues branch remains `3ed85ee`; the new GUI branch
includes it. The latest inspected acceptance branch was
**`c8985a3205c790e9804e082507d716c6e78ae8e1`**, with newer reaction-rate power
reference, South scenario and replay-provenance work. That parallel scientific
branch was inspected, not wholesale
merged into this GUI validation branch.

Its comparator differs from the #217 implementation retained here. An isolated
AST probe executes the two comparator functions directly from each published
source revision, without modifying the current module or rerunning scientific
algorithms under mixed source. Current `bade0779` passes **8/8** scalar/array,
absolute/relative explicit-field precedence checks in either dictionary order.
The acceptance branch passes **4/8**: when generic `energy_keV_*` appears first,
it accepts a 1 keV difference despite the stricter explicit field limit. Preserve
the verified #217 field-specific precedence and its regression tests when
reconciling the acceptance branch. This is an integration requirement, not a
second comparator implementation. Exact inputs, outputs, source hashes and
branch revisions are in `published_parity_precedence.json`.

The two original parity failures remain resolved by that retained fix. The old
Tk coordinate failure remains reproducible in historical evidence; after GUI
consolidation its test is retired from the active Qt suite. This is retirement
of that acceptance surface, not a passing physical Tk replay. Frozen coverage
at `36ed262` is unchanged: 2,052 passed, 26 skipped, 3 failed out of 2,081.

## Reproduction and evidence

From this worktree:

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path
$env:FLUXFORGE_OFFLINE = '1'
$env:MPLBACKEND = 'Agg'
$env:QT_QPA_PLATFORM = 'offscreen'
$mm = 'C:/Users/Josh/.codex/environments/fluxforge-issue21-tools/Library/bin/micromamba.exe'
$prefix = 'C:/Users/Josh/.codex/environments/fluxforge-issue21-complete'
& $mm --no-rc --root-prefix C:/Users/Josh/.mm21 run --prefix $prefix python artifacts/validation/three_failures_20261003/run_qt_checks.py qt_consolidation qt_shell qt_new_workspaces qt_calibration qt_sessions qt_parity
$env:QT_QPA_PLATFORM = 'windows'
$env:QT_CHECK_RUN_LABEL = 'native_'
$env:QT_CHECK_TIMEOUT = '180'
& $mm --no-rc --root-prefix C:/Users/Josh/.mm21 run --prefix $prefix python artifacts/validation/three_failures_20261003/run_qt_checks.py qt_production
& $mm --no-rc --root-prefix C:/Users/Josh/.mm21 run --prefix $prefix python artifacts/validation/three_failures_20261003/qt_snapshot.py
```

Current Qt logs, JUnit results, bounded-run summaries, snapshots and the
published comparator probe are in `artifacts/validation/three_failures_20261003`.
The original failure investigation and immutable historical receipts remain
beside them. The active GUI work here uses Qt; the old Tk artifacts are retained
for review of the original failure record.
