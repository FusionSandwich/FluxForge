# ChatGPT Pro continuation prompt — FluxForge validation

Copy the text below into ChatGPT Pro with repository access. If the repository
is private, connect it or provide the relevant source/documents. Local drive
paths identify evidence on Josh's computer; a chat without computer access
cannot inspect them or claim to run native tests.

---

Continue FluxForge testing and feature validation from the `recovery-validation`
branch of https://github.com/FusionSandwich/FluxForge. Inspect the current branch
HEAD and changes before acting. The previous starting HEAD was
`78aee3ee9da3d4c44dd77dec05cb674321971c79`; it is a historical baseline, not the
current implementation. Do not revert to it. This is an unfinished scientific
and application-validation effort, not a qualified release.

First read these repository documents in order:

1. `docs/RECOVERY_TODO_20260916.md`
2. `docs/INL_BACKGROUND_IMPLEMENTATION_RECEIPT.md` (current results and blockers)
3. `docs/BACKGROUND_COVARIANCE_CONTRACT.md`
4. `docs/INL_BACKGROUND_FIX_PLAN_20260918.md`
5. `docs/FEATURE_VALIDATION_MATRIX.md`
6. `docs/GUI_TEST_COVERAGE_LEDGER.md`
7. `docs/SCIENTIFIC_VALIDATION_20260916.md`

The older `VALIDATION_RESUME_20260918.md`, background follow-up and old new-chat
prompt describe the starting grid-rejection state. Current code supports
conservative alignment; do not repeat their obsolete diagnosis as current.

## Working preferences and environment

Keep one native FluxForge window open for testing. Josh explicitly asked us to
stop repeatedly opening and closing it. Do not run the broad native pytest
suite or old GUI probe scripts if they cycle windows. Exercise the existing
window and save screenshots/state between checks. Use bounded Luna assistance
for independent review when available; previous Luna sessions reached a usage
limit. Never claim an unavailable tool/model/browser was used.

For a local agent, use `C:/Users/Josh/projects/FluxForge-validation` (junction
to `D:/C_drive_offload_2026-09-17/local_data/projects/FluxForge-validation`).
The old `C:/Users/Josh/projects/FluxForge` directory is empty. Interpreter:
`C:/Users/Josh/projects/FluxForge-recovery-20260916/windows-env/Scripts/python.exe`.
Set `PYTHONPATH` to the validation checkout's `src`; native Qt uses
`QT_QPA_PLATFORM=windows`. Jinja2 3.1.6 and MarkupSafe 3.0.3 were added for
reporting tests. These are source tests, not installed-wheel qualification.

Local evidence is at `D:/FluxForge-background-fix-20260918-v1`, including logs,
JUnit XML, fixture hashes, original dirty-state snapshots, method/covariance
audits and followup-commands.md. The persistent window helper is
`persistent_gui.py`; its bounded action request/receipt directory is
`persistent-gui-v1`. Check its recorded PID against live processes before use;
a ready.json file alone does not prove the window is still open. If the chat
has no execution access, review the code and produce concrete patch/test
instructions for the local agent; label unexecuted checks honestly.

## What was implemented

Optional full sparse CSR count covariance now survives spectrum JSON, CSV and
FFS sessions. Signed counts require explicit uncertainty/covariance. Validation
checks dimensions, finite values, symmetry, PSD and diagonal consistency.
Conservative overlap subtraction uses midpoint energy-bin edges, strict target
coverage and fixed live/manual scaling: `net=s-aWb`, `Cnet=Cs+a²WCbWᵀ`.
Original calibrations/counts stay unchanged; cropped source counts are recorded.
ROI/sideband sums use `wᵀCw`; Gaussian fits use full observation covariance and
amplitude-width covariance for area uncertainty. Unsupported consumers reject
correlations explicitly. Reference-parity loading preserves covariance.

GUI fixes avoid reapplying identical global stylesheets and replace the
canonical document on reset, preventing old background roles from reappearing.
The corrected CSV -> GUI -> FFS -> GUI check covers that reset bug.

## Evidence and limits

Consult the current receipt for the latest persistent-window results. Completed
receipts include 174 passes/1 failure for numerical INL work (all ten real
alignment oracles pass; original raw ten-bin regression still fails), 67 passes
for the latest numerical follow-up, and 105 passes/4 failures/1 skip in the
broad GUI run that completed without crashing. Its report failures were caused
by absent Jinja2; label/forecast tests needed explicit sample/reference setup.
Subsequent focused checks exercised those fixes. The final broad rerun was
interrupted at the user's request, so it is not an aggregate passing receipt.
The original crash cause is not conclusively established. Conditioning warnings
in calibration/RMLE remain meaningful. Mocked PDF-dialog tests do not qualify
native PDF generation; no completed gallery audit or fresh wheel is claimed.

## Next work, in priority order

1. Review the current diff for correctness and regression risks. The persistent
   data/covariance round trip passed, but its screenshot shows a 0–1 keV viewport
   instead of the full spectrum. Investigate autorange/viewport restoration and
   exercise Reset View in the existing window. Do not confuse correct data in
   the plot buffer with correct visual framing. Verify the
   persistent-window reset/corrected-file/session behavior and meaningful error
   recovery without cycling the application. Keep exported/displayed values
   and full covariance consistent. Retain invalid coverage/singular-fit tests.
2. Repair the raw INL counting-to-reaction pathway scientifically. The legacy
   targeted QG path uses historical reference overrides and cannot consume
   correlated net counts. Do not remove its guard to obtain a pass. Implement
   an explicit covariance-aware estimator with fixed ROI/sideband weights or
   supported fits, provenance and no historical count/activity substitution.
   Carry cross-peak and shared-background cross-sample covariance through
   reaction reduction into GLS. Record conditional ROI-selection assumptions.
3. Resolve/document remaining workflow failures: two RAFM profile-selected
   sample axes extend below the background interval; strict coverage rejects
   them. Two activity-parity fixtures omit required net-count uncertainty.
   Do not replace real calibrations, zero-pad uncovered measurements, fabricate
   uncertainty, relax tolerances, skip subtraction or hide failures with xfail.
4. Reconcile every feature-matrix row against actual code/tests. Inventory CLI,
   native GUI, readers, sessions/undo, calibration, peak search/fitting, ROI,
   activity/efficiency, batch/reporting, QA/predictive, nuclear-data/standards,
   planning/optimization, k0/NAA and packaging. Record tested, unsupported,
   unavailable and untested separately, with evidence rather than feature names.
5. Validate each applicable unfolding method on traceable INL inputs or retain
   explicit blockers: discrete assignment, GLS/response-covariance GLS, GRAVEL,
   MLEM/covariance MLEM, gradient descent, regularized gradient/Tikhonov, MAXED,
   RMLE, ML Seed, MCMC and optional PyUnfold. Check response provenance,
   conditioning, refold residuals, independent numerical expectations and full
   uncertainty semantics. Finite outputs or expected array lengths do not prove
   acceptance. Core GLS can accept full covariance; existing reaction handoffs
   and other adapters do not thereby become covariance-qualified.
6. Preserve physical qualification blockers: detector/geometry calibration,
   background suitability, efficiency, timing and nuclear-data provenance.
   Build a fresh package and test outside the checkout before any installed
   qualification claim. Update the receipt/matrix/ledger with exact commands,
   hashes, failures, skips and limitations.

Do not perform reactor/transport simulations. Preserve historical study files
and original local handoffs; never publish the local_handoffs directory. Do not
reset work, delete Git refs or overwrite historical evidence. The September 21
push was explicitly authorized; inspect status/remote before further Git work,
and do not infer ongoing authorization to publish unrelated future changes.
G1/G2/G3/G4 remain incomplete until their complete evidence gates pass. Deliver
concrete improvements and a candid remaining-work list; never turn documented
blockers into a completion claim.
