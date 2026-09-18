# Windows validation receipt — 2026-09-16

## Verified local installation

This receipt covers the recovered local checkout at
`C:\Users\Josh\projects\FluxForge-validation`, based on commit
`c297542da781291c3d1461af181b79349bda3087` plus the uncommitted recovery
changes present on 2026-09-16. A generic upstream clone does not contain those
changes. Use the verified local installation below when reproducing this result.

- Python: CPython 3.12.10, 64-bit Windows
- Environment: `C:\Users\Josh\projects\FluxForge-recovery-20260916\windows-env`
- Python executable: `C:\Users\Josh\projects\FluxForge-recovery-20260916\windows-env\Scripts\python.exe`
- CLI launcher: `C:\Users\Josh\projects\FluxForge-recovery-20260916\windows-env\Scripts\fluxforge.exe`
- GUI launcher: `C:\Users\Josh\projects\FluxForge-recovery-20260916\windows-env\Scripts\fluxforge-gui.exe`
- Installed package version: `fluxforge 0.1.0`
- Final wheel: `D:\FluxForge-validation-temp-20260916\wheelhouse-final-attenuation\fluxforge-0.1.0-py3-none-any.whl`
- Final wheel SHA-256: `e0e502b7e2f416bd914a8cf9f4fea45916d934ae0cafc81be6c687c0390610e9`

The final wheel contains 270 files. It contains no `fluxforge_gui` package,
does not install `fluxforge-gui-legacy`, and exposes only these console scripts:

- `fluxforge = fluxforge.cli.app:main`
- `fluxforge-gui = fluxforge.gui.app:main`

The installed `fluxforge` and `fluxforge.gui` modules resolve under the fresh
environment's `Lib\site-packages`; the source checkout is absent from
`sys.path`. `pip check`, `fluxforge --help`, and `fluxforge-gui --help` all
returned exit status 0.

## Native Windows result

An automated native `qwindows` startup created and displayed the production
main window, reported the title `FluxForge — HPGe Analysis`, opened with zero
spectra, processed Windows events, and closed cleanly. PySide6 6.11.2 was used.
This is an automated startup check and is not a manual usability review.

The numerical and GUI wheel used for the native workflow tests had SHA-256
`9d4d0b0ccded499083835e7afe4220cb61c96be6fd137a3f2f135447e7c8ff7c`.
The source then received one command-catalog text correction. A content-level
comparison between that wheel and the final wheel found exactly two changed
entries after the command-catalog correction: `fluxforge/cli/command_catalog.py`
and the generated wheel `RECORD`. A later dimensional correction changed only
`fluxforge/io/flux_wire.py` and `RECORD` relative to wheel
`991c5f860221863fc2688d351ff8994f4237c3894d8783382eea545a53425b56`.
There were no added or removed wheel entries in either comparison. The final
installed wheel completed the affected backend slice with 24 passes and the two
packaging failures described below; its affected Qt, package identity, and
dependency checks passed. Unaffected native results remain applicable because
their installed modules were byte-identical in the wheel comparison.

## Efficiency attenuation correction

The detector model now evaluates the reported equation as

`ε = A exp[-(μ_Al T1 + μ_Ge DL)/cos(θ)] [1 - exp(-μ_Ge DI/cos(θ))] p(ln E)`.

The XCOM tables provide mass attenuation `μ/ρ` in cm²/g. The implementation now
uses `μ = (μ/ρ)ρ` in cm⁻¹ before multiplying by the aluminum-window,
germanium-dead-layer, and active-detector lengths in cm. The resulting
Beer-Lambert exponents are dimensionless, consistent with the
[NIST attenuation relation](https://physics.nist.gov/PhysRefData/XrayMassCoef/chap2.html).

No calibration parameters were retuned. The bundled `rafm_25cm` parameter set
uses a 6.45 cm active depth and now evaluates to approximately 0.001066 and
0.001025 at the Co-60 1173.228 and 1332.492 keV lines. Under a candidate percent
interpretation, the separate local export with a 1.39 cm depth corresponds to
fractions 0.000388913 and 0.000354430; the report/QG values imply 0.000391376 and
0.000360421. Those parameter conventions disagree, so the bundled coefficients
remain unqualified under the dimensionally corrected model until their binding
is verified or the curve is refit. The narrow export comparison is a consistency
lead, not independent curve validation. No accepted activity result was
generated from the corrected profile.

## Test receipts

All installed-only pytest runs used `--noconftest --import-mode=importlib` from
outside the checkout, so `tests/conftest.py` could not inject `src`.

| Coverage | Result | Log |
| --- | ---: | --- |
| Production contracts, ROI core, workspace document/controller/undo, and peak-report scientific contract | 81 passed, 9 skipped in 13.57 s | `windows/final-installed-core-peak-report.log` |
| Workspace JSON schema after installing `jsonschema` as a test-only dependency | 41 passed in 6.52 s; closes the 9 skips above | `windows/final-installed-workspace-schema.log` |
| Native session save/open, replacement, envelope preservation, calibration undo, and peak-area uncertainty reopen | 5 passed in 90.63 s | `windows/final-installed-workspace-session-qt.log` |
| Production startup, real ASC load, ASTM enablement, path non-persistence, action coverage, and production copy | 9 passed in 199.45 s | `windows/final-installed-production-gui.log` |
| Inventory, masking, and optimization exports; activity review export; mouse-driven ROI analysis/statistics | 5 passed in 103.53 s | `windows/final-installed-export-roi-qt.log` |
| Final command catalog and generated CLI reference | 4 passed in 6.79 s | `windows/final-installed-command-catalog.log` |
| Source dimensional, calibration, loader, and profile-energy checks | 16 passed in 9.00 s | `windows/source-efficiency-linear-attenuation.log` |
| Source raw/processed flux-wire parity and RAFM background integration | 10 passed in 45.40 s | `windows/source-flux-wire-parity-linear-attenuation.log` |
| Final installed affected backend slice | 24 passed, 2 failed in 30.15 s | `windows/final-installed-linear-attenuation-backend.log` |
| Final installed native efficiency-calibration dialog | 1 passed in 29.34 s | `windows/final-installed-linear-attenuation-qt.log` |

The pre-attenuation, non-overlapping composite receipt was 113 passed and 0
skipped. Its unaffected results remain applicable by wheel-member comparison;
the latest affected slice is reported separately above and includes the two
packaging failures described below. The export slice
emitted one known deprecation warning for `datetime.utcnow()` in
`fluxforge.gui.panels.phase6`; it did not affect output or exit status.

The two attenuation follow-up installed-backend failures were the parameterizations of
`test_rafm_profile_supplies_background_and_efficiency_defaults`. The installed
profile resolves its background path to
`windows-env\Lib\examples\RAFM_irradiation\background.ASC`, which is not in the
wheel. Both cases pass from the source checkout. This was a separate packaging
blocker; it is resolved by the 17 September packaged-resource correction below.

The activity/session regression specifically verifies that fitted
`net_counts_uncertainty` survives GUI copies, workspace serialization, save,
and reopen. Legacy workspace peaks without that value deserialize as `None` and
the activity panel shows the explicit refit-required message rather than
manufacturing an uncertainty. Activity review also receives the spectrum's real
count time.

## Exceptions and pending manual checks

- No full Windows installer or PyInstaller executable was built.
- Native startup and Qt interaction were automated. Manual confirmation is
  still pending for the Windows file dialogs, keyboard shortcuts, drag-created
  ROI behavior, save/reopen through visible dialogs, export destination dialogs,
  high-DPI scaling, multi-monitor placement, and an extended interactive
  session.
- The first diagnostic log, `windows/installed-baseline.log`, includes an
  `ImportError` for a nonexistent diagnostic-only `QT_API` name. The corrected
  identity checks are recorded in `windows/final-installed-identity.json` and
  `windows/final-text-installed-identity.json`, both with exit status 0. The
  latest attenuation wheel identity is recorded in
  `windows/final-attenuation-installed-identity.json`, also with exit status 0.
- One initial quiet combined test invocation was interrupted after remaining
  CPU-active without producing a result. Its files were rerun in bounded slices
  with visible progress and completed successfully as listed above.
- `jsonschema 4.26.0` and its dependencies were installed after the FluxForge
  wheel as test-only packages. They are not FluxForge runtime requirements and
  did not change the wheel.
- The original installed RAFM background-path defect is resolved in the 17 September
  wheel. Profile-driven subtraction with incompatible energy grids remains blocked
  by the explicit validation guard described below.

## Artifacts

Windows logs and receipts are under
`C:\Users\Josh\projects\FluxForge-recovery-20260916\windows`. Build staging and
pytest temporary data were placed under
`D:\FluxForge-validation-temp-20260916` to avoid further pressure on the system
drive. The original source checkout at `C:\Users\Josh\projects\FluxForge` was
not used or modified by this validation.


## Integrated feature follow-up — 17 September 2026

The installed follow-up wheel is `D:\FluxForge-feature-validation-20260916\integration-20260917\wheelhouse-final\fluxforge-0.1.0-py3-none-any.whl`, SHA-256 `81c41a1d107d8dde8bf56d7c63713f82750d8434644d51e9767bc4ab4832faf2`. Its 271 members exclude the historical package and retain the two maintained entry points. Every packaged source/data file matches the frozen build snapshot; the original source checkout still matches all 1,591 preserved hashes.

The three RAFM profiles now resolve the packaged `fluxforge/data/rafm_background.ASC`, with the original asset hash verified. This resolves the missing-file defect. It does not establish that the background shares a measurement's energy calibration or is physically suitable.

The first integrated candidate (`00720a4f...`) passed 99 installed core/reader checks and one separate HTML reporting check. The reporting check uses Jinja2 3.1.6 and MarkupSafe 3.0.3 from the isolated D: dependency directory; they are not bundled into the wheel. The final refresh changes only `io/n42.py`, `gui/panels/modern_shell_center.py` and `RECORD`: N42 rejects unsupported counts before writing; the desktop visibly retains raw foreground when background subtraction is invalid and recovers on valid selection. Affected final checks are recorded below; unchanged modules retain the preceding scoped evidence.

The historical RAFM/flux-wire slice has two passes and eight failures, all at the explicit incompatible-energy-grid guard. These are remaining workflow blockers, not missing package assets. No calibration is overwritten and no unsupported interpolation is restored to pass them. The older passing background/parity receipts are superseded for these workflows.

N42 output remains a local schema subset. Full metadata/official-schema interchange, CHN timing/date/trailer conformance, general covariance-preserving background rebinning, PDF rendering, and the earlier manual Windows checks remain unqualified.

Build logs, installed import identities, exact wheel-member comparison, tests and exit codes are in `D:\FluxForge-feature-validation-20260916\integration-20260917\installed-validation`; `wheel-identity-final.json` is in its parent. The initial build attempt used a runtime-only environment lacking setuptools; the preserved successful build used the existing build environment without adding runtime dependencies.


Final affected results:

| Slice | Outcome |
|---|---|
| Final installed N42 reader/export | 18 passed, no failures or skips |
| Candidate core/readers carried by unchanged member hashes | 99 passed; report and native nodes separately selected |
| HTML report with isolated optional dependency | 1 passed |
| Historical mismatched-grid workflows | 2 passed, 8 failures at the intentional grid guard |
| Refreshed native desktop | Blocked before first result; no passing claim |

The combined four-node native run stalled at startup. An isolated test's 45-second traceback stopped while constructing `QApplication`, and a minimal PySide6 `qwindows` probe also stalled in `QApplication([])` beyond 30 seconds. This reproduces before application initialization; the diagnostic evidence does not identify the underlying Windows/Qt cause. Only task-owned test processes were interrupted, and none remain. Source Qt regressions pass under the offscreen platform, including invalid-background display and subsequent recovery, but they do not replace native qualification. Retry the native slice in a fresh interactive Windows session. Final receipt: `installed-validation/FINAL_VALIDATION_RECEIPT.json` under the follow-up directory above.

## Integrated follow-up verification - 18 September 2026

The validation checkout now resolves through the authoritative D: offload copy. Its Git worktree pointer was repaired without changing the branch or commit. Background normalization and explicit energy-axis fixes are integrated; finite signed spectra with supplied uncertainties now persist through FFS sessions. Conservative histogram rebinning with full sparse covariance is implemented and tested as a standalone primitive. It is not yet wired into the existing background workflows.

Wheel SHA-256: `fc179bd1b3b6fcfec0ce33bee9ea464545786b84725e8d8e4ef94c3278e1c023` (272 members). Installed in the separate recovery environment. Tests ran outside the source checkout and verified imports from site-packages.

- Installed numerical/reader/session/schema checks: 191 passed, one POSIX-only skip.
- Installed native Windows session/recovery checks: 11 passed; optional HTML test skipped in that environment.
- The same HTML check with the isolated optional dependency: one passed.
- Native source workflow run: 45 passed, one optional HTML skip. These overlap the installed checks and are not additional distinct coverage.
- Original-source preservation: all 1591 archived-manifest files match the authoritative offload copy.

Receipts: `D:\FluxForge-integrated-followup-20260917\installed-core.txt`, `installed-native.txt`, `installed-html.txt`, `wheel-identity.json`, and `original-preservation.json`.

Native Qt initialization and the selected session workflows are no longer blocked in this run. Full DPI, multi-monitor and endurance qualification remains open. Eight historical mismatched-grid workflows remain blocked by the existing guard until covariance is propagated through spectrum/session storage, ROI estimators and peak fitting. Physical calibration/time qualification and external dosimetry parity remain separate open gates.
