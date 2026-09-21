# Measured-background implementation receipt

Status: OPEN; source implementation under regression review. No G1/G2/G3/G4
promotion or installed-package qualification is claimed. On September 21 Josh
explicitly authorized committing and pushing the accumulated repository work.

Workspace: `C:/Users/Josh/projects/FluxForge-validation`, branch
`recovery-validation`, starting HEAD `78aee3ee9da3d4c44dd77dec05cb674321971c79`.
Evidence: `D:/FluxForge-background-fix-20260918-v1`. Initial dirty changes,
fixture hashes, diagnostics and unsuccessful runs are retained there.

## Implemented and independently checked

See [the numerical contract](BACKGROUND_COVARIANCE_CONTRACT.md). Spectrum CSR
covariance survives JSON, CSV and FFS round trips, including signed counts.
Conservative overlap subtraction retains original calibrations and counts,
requires full target coverage, and records cropped source counts. Independent
hand calculations and cumulative-integral oracles cover all ten real INL files.
ROI/sideband variance uses full covariance; Gaussian fits use correlated
observation weighting and amplitude-width covariance in area uncertainty.
Unsupported quantitative consumers reject correlations explicitly.

Completed source receipts before the GUI crash investigation:

| Receipt | Result | Meaning |
| --- | --- | --- |
| baseline | 1 failed, 2 passed | Original raw unfolding grid guard and interface rejection baseline |
| storage-v2 | 79 passed | Spectrum, schema, reader and persistence checks |
| subtraction | 44 passed | Conservative subtraction and legacy behavior |
| consumers-v2 | 49 passed | ROI, Gaussian fit and uncertainty oracles |
| interfaces-v3 | 3 passed | Real CLI success/rejection; native GUI selection, subtraction, save/reopen, ROI error/recovery |
| numerical-inl-v2 | 174 passed, 1 failed | Ten real rebin oracles pass; legacy targeted QG reduction explicitly rejects covariance |
| broad-core | 238 passed, 1 failed, 1 skipped | One synthetic CLI mock omitted calibration; corrected fixture awaits follow-up receipt. Windows skip is POSIX atomic-save behavior |
| broad-native-v2 | Abnormal exit; incomplete | Several assertions failed before native process exit; no aggregate pass claim |
| consumer-followup-v2 | 124 passed, 3 failed | CLI synthetic fixture fix passes; remaining failures described below |
| native-isolate-v1 | 25 passed, 1 failed | Stopped cleanly at reference-label assertion; 377.99 seconds |
| native-focused-v1 | 4 passed, 1 failed | Two-window restore and original real background workflows pass; label assertion remains |
| numerical-followup-v3 | 67 passed | Latest covariance/consumer/artifact/GLS checks |
| broad-native-v3 | 105 passed, 4 failed, 1 skipped | Completed without crash in 119.32 seconds; reporting dependency and fixture issues identified |
| native-fixes-v1 | 10 passed, 1 failed | Reporting/predictive fixes pass; label fixture also needs loaded spectrum |
| native-fixes-v2 | 3 passed, 1 failed | Label toggle passes; expanded corrected CSV/session check exposes stale background after reset |

Commands and outputs are retained in the evidence directory. The interpreter
is `C:/Users/Josh/projects/FluxForge-recovery-20260916/windows-env/Scripts/python.exe`,
with source `PYTHONPATH` and `QT_QPA_PLATFORM=windows` for native tests.

## GUI crash investigation

The user observed the GUI crash during the broad native slice. The process
returned `-805306369`; its log contains 90-second stack snapshots in global
Qt stylesheet application while creating windows. This is evidence of slow
stylesheet work, not a demonstrated crash cause. Main-window mode updates now
avoid reapplying an identical application stylesheet. Fresh `broad-native-v3`
completed the full slice without abnormal exit. This does not conclusively
identify the original crash cause.

Installed the declared optional HTML dependency (Jinja2 3.1.6, MarkupSafe 3.0.3)
in the validation environment. Report dialogs had correctly refused to open
without it. PDF dialog tests mock PDF generation. Forecast and label-toggle
fixtures now explicitly load an example; labels select a known gamma emitter.
The expanded corrected CSV -> GUI -> FFS -> GUI test found reset retained old
document-only background roles. Reset now replaces the canonical document;
`broad-native-v4` was interrupted at the user's request to stop opening and
closing windows (progress output reached 63%; no final aggregate receipt).
Further native checks use one persistent FluxForge window driven by the bounded
QA actions in `D:/FluxForge-background-fix-20260918-v1/persistent_gui.py`.

September 21 persistent-window checks passed (`load-background.json` and
`corrected-roundtrip.json` in `persistent-gui-v1`): real subtraction agrees with
the canvas count buffer; corrected CSV and FFS reopening preserve exact CSR
covariance; reset clears spectra and roles. The same window remains open.
Visual inspection of `corrected-roundtrip.png` nevertheless shows a 0–1 keV
viewport rather than the full spectrum after reopening. The data-buffer checks
do not prove correct initial framing. Investigate viewport/autorange restoration
and exercise Reset View in the persistent window before claiming visual QA.

## Scientific blockers and unsupported boundaries

The unchanged original raw ten-bin regression still fails. The blocker has
moved from grid alignment to the targeted QG reducer: that path uses legacy
estimators and historical processed-count/activity overrides. It cannot be
accepted by silently discarding covariance or substituting reference values.
Shared background covariance across samples and peak/reaction covariance are
not supplied end to end to the unfolding handoff.

All INL unfolding acceptance remains blocked, including legacy discrete/GLS,
GRAVEL, MLEM, covariance MLEM, gradient descent, regularized gradient/Tikhonov,
MAXED, RMLE, ML Seed, MCMC, and response-covariance GLS. Core GLS supports full
measurement covariance, but the current raw reaction reduction does not provide
it. Other adapters commonly consume marginal errors only. Synthetic solver
tests establish software behavior, not measured INL uncertainty qualification.
The optional PyUnfold route is unavailable in the current environment.

The consumer follow-up exposes two RAFM example failures: their profile-selected
sample axis starts below the background interval, so strict coverage rejects
them. No calibration replacement or zero-padding is applied. The reference
parity aggregate also fails because two activity fixtures omit net-count
uncertainty; the required-uncertainty guard already exists at the starting HEAD.
The detailed `parity-diagnostic-v1.json` retains both errors. These are open
workflow/fixture blockers, not accepted runs.

Physical background suitability, detector/geometry calibration, efficiency,
timing and nuclear-data provenance remain unqualified. The Luna method audit
in the evidence directory includes a background-disabled diagnostic; it is
excluded from acceptance evidence and does not replace the failing regression.

Correlated inputs are explicitly unsupported in legacy QG/HPGe, quantitative
SNIP/minima continua, non-Gaussian/Bayesian fits, legacy overlay parity, SPE and
RAFM corrected-array/CSV exports. Reference-parity input loading now preserves
the covariance instead of silently rebuilding a diagonal spectrum.

## Remaining verification

Resolve the viewport/framing issue and finish persistent-window feature checks.
The documentation's historical Playwright gallery
audit script was not found in this checkout; no gallery audit is claimed.
