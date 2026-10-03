# Recorded validation failures: independent follow-up

Two original parity failures are resolved by the existing PR #217 fix. The
legacy Tk desktop coordinate workflow remains reproducible. Its calibration
timeout is a clipped-control input failure, not evidence of a failed calibration
algorithm or missing runtime dependency.

This work started from published `codex/remaining-issues-20261003` at
`3ed85ee0ca9df39f0dbfe50499c485bca45d9d55`, in its own managed worktree.
Inspection of **Check FluxForge features and samples** showed that chat was
also fixing the parity comparator. **Archive old GUI and improve new one**
owns moving Tk source/tests/launchers into a source archive and improving Qt.
No legacy product source or next-batch issue implementation was changed here.

| Original failure | Reproduction and follow-up | Disposition |
| --- | --- | --- |
| `test_reference_parity_suite_runs_all_fixture_families` | Reproduces at exact frozen `36ed262`; passes at published `3ed85ee` and diagnostic commit `51654ce` | Resolved by existing `a88a2c4` in PR #217 |
| `test_reference_parity_suite_supports_scope_and_fixture_filters` | Same frozen failure and passing published/final follow-ups | Resolved by existing `a88a2c4` in PR #217 |
| `test_native_desktop_gui_workflow_generates_evidence` | Unmodified Windows coordinate replay at `3ed85ee`: calibration timeout, 39.46 s. Diagnostic coordinate replay: clipped target rejection, 26.16 s. Windows CI event mode at `51654ce`: passes, 39.04 s | Coordinate workflow unresolved; event-mode pass is separate evidence |

The original frozen coverage remains **2,081 collected; 2,052 passed, 26
skipped, 3 failed, 0 unreported**. The frozen original/resumed artifacts were
read only. `original_failures.json` extracts the three original failure records;
the JSON receipt records hashes of the original case ledgers, environment files,
GUI failure log, and summary. Those files' hashes were rechecked unchanged.

## Parity verification

The manifest declares `energy_keV_abs=8` and `peak_count_abs=2`. Before the
existing fix, the comparator's token matching missed `energies_keV[]` and
`first_peak_keV`; small rounding differences consequently used its `1e-9`
default. No fixture values or limits were changed here.

All 15 runner/tolerance/manifest tests pass at `3ed85ee`. A separate JSON-level
probe performs 25 checks: actual fixture output matches with its declared
limits and fails without them; both energy aliases accept positive/negative
boundary errors and reject errors beyond the boundary; explicit scalar/array
field limits take precedence in either dictionary order for absolute and
relative limits; relative aliases work; channel differences, changed array
lengths, and excessive peak-count errors fail. This validates the published
fix independently without duplicating its implementation. Probe source and
observations are included with the evidence.

## Desktop diagnosis and repair scope

On the 1920 by 1080 desktop, the driver calculated the Fit calibration center
as **(523, 708)**. `root.winfo_containing` identified the spectrum plot canvas
at that coordinate, rather than the calibration button. The Tk controls panel
is fixed to approximately 430 pixels, supports vertical scrolling only, and
clips the calibration button row horizontally. An observed run recorded no
fit callback. Its screenshot shows the clipping. The unchanged test reproduces
the same calibration timeout, so this is not merely an intermittent timing
claim. The legacy layout and the driver's blind coordinate dispatch jointly
cause the failure; the numerical fit remains unimplicated.

Diagnostic commit **`51654ce6814c0547a52705369018cf495501beeb`** changes only the
desktop driver and two real-Tk geometry contract tests. Coordinate dispatch
checks the widget hit at the calculated center before sending a physical click.
A mismatch fails immediately with target, hit widget, coordinates and screen
size. The driver preserves `run.json` on failure, including completed steps,
exception, failure screenshot, Python/prefix, exact source file hashes and
input profile. It retains the existing event-mode behavior. It does not silently
invoke an inaccessible control to turn a coordinate failure into a pass.

The first guarded run launched with the modified driver while HEAD was still
`3ed85ee`; the same code was committed as `51654ce` during the run. The receipt
labels this explicitly as a modified-source run and records its exact source
hashes. The subsequent event-mode and final focused runs used committed
`51654ce` source. Final focused verification is **17 passed, 0 skipped**;
Black checks, scoped Flake8 and `git diff --check` also pass. The earlier geometry
test attempt lacked foreground placement; the final tests explicitly raise
their test windows, as the desktop driver does.

`GITHUB_ACTIONS=false` selects Windows coordinate widget clicks. `true` selects
Tk invoke/generated events for widget clicks and tree selection. Both profiles
retain the pre-existing programmatic notebook-selection and entry-text fallbacks
and synchronous CLI dispatch. The successful event-mode test verifies the full
existing workflow assertions and output artifacts; it is not a passing physical
mouse-input replay or a full product scientific-accuracy qualification.

## Integration and reproduction

This diagnostic change is stacked on PR #217's published branch. Retain its
existing comparator fix; the acceptance branch has its own concurrent parity
work, so reconcile that during integration instead of applying another copy.
Coordinate with **Archive old GUI and improve new one** to relocate this driver
and its diagnostic tests beside the archived Tk tests, or retire them from the
active suite when that archive lands. Keeping legacy coordinate acceptance
active would require an accessible Tk layout and a new passing coordinate run;
that product change belongs to the archival decision. No Tk layout repair is
claimed by this draft.

Use the complete environment and set `PYTHONPATH` to this checkout's `src`:

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path
$env:FLUXFORGE_OFFLINE = '1'
$mm = 'C:/Users/Josh/.codex/environments/fluxforge-issue21-tools/Library/bin/micromamba.exe'
$prefix = 'C:/Users/Josh/.codex/environments/fluxforge-issue21-complete'
& $mm --no-rc --root-prefix C:/Users/Josh/.mm21 run --prefix $prefix python -m pytest -q tests/test_reference_parity_runner.py tests/test_reference_parity_energy_tolerance.py tests/test_parity_fixture_manifests.py tests/test_gui_desktop_driver_diagnostics.py
& $mm --no-rc --root-prefix C:/Users/Josh/.mm21 run --prefix $prefix python artifacts/validation/three_failures_20261003/parity_probe.py
$env:GITHUB_ACTIONS = 'false' # Reproduces the clipped coordinate target
& $mm --no-rc --root-prefix C:/Users/Josh/.mm21 run --prefix $prefix python -m pytest -q tests/test_gui_desktop_native.py
$env:GITHUB_ACTIONS = 'true' # Existing CI widget-event profile
& $mm --no-rc --root-prefix C:/Users/Josh/.mm21 run --prefix $prefix python -m pytest -q tests/test_gui_desktop_native.py
```

Compact logs, JUnit results, probe results, original failure records, click
trace, GUI run receipts and screenshots are in
`artifacts/validation/three_failures_20261003`. Large generated desktop outputs
remain local in the worktree. The portable follow-up receipt is
`ISSUE_21_THREE_FAILURES_FOLLOWUP_2026-10-03.json` beside this report.

**No final single-revision full suite was run or claimed.** These targeted runs
do not revise the frozen suite counts, and do not close source/scientific
qualification or unrelated open issues.
