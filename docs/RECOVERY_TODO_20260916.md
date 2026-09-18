# Recovery execution checklist — 16 September 2026

Current scope: scientific correctness, all available INL reductions, native Windows Qt workflows, and recoverable legacy isolation. Checked items represent completed work; an open item is not an acceptance claim.

The separate feature-validation task completed its initial pass: 107 passing checks, 34 failed expectations grouped into 17 findings, and one platform skip. Its live checklist and detailed receipts are under `D:\FluxForge-feature-validation-20260916`. Follow-up fixes and regression checks are now in progress; these counts describe the initial pass, not current acceptance.

Follow-up work was separated by files: reader/calibration/CLI fixes were prepared in the independent task; background, ROI uncertainty and shared-width fitting were corrected together; library recovery, forecasts and HTML escaping were completed in another bounded pass. Profile packaging and SQLite proposals have passed 12 integrated source checks. Follow-up numerical and recovery fixes are integrated; the bounded reader proposal is now integrated and its 62 focused checks pass in the combined source tree. Physical experimental qualification remains open.

## Preservation and source selection

- [x] Preserve the original checkout, dirty files, refs, reflog and pre-fetch history; verify the bundle and 1,591-file source archive.
- [x] Compare Windows and Debian copies and preserve unpublished study changes.
- [x] Select the integrated source in `C:\Users\Josh\projects\FluxForge-validation` on `recovery-validation` and the study in `C:\Users\Josh\projects\rafm-analysis` on `measurement-validation`.
- [x] Keep original measurements and historical outputs untouched; use separate, versioned output directories.
- [x] Verify all 1,591 original source/archive-manifest files remain byte-identical after the work.
- [x] Retain the final uncommitted source patches and changed-file archives in `C:\Users\Josh\projects\FluxForge-recovery-20260916\final-local-changes`, with a verified manifest.

## C — Scientific correctness

- [x] Correct GLS covariance validation, pseudoinverse, exact-constraint checks, observation-space diagnostics and response-uncertainty propagation.
- [x] Verify the posterior against independent equations: 70 inverse-contract checks pass (10 unavailable external-reference skips), plus 100 covariance/solver regression checks.
- [x] Correct elapsed/live-time activity integration and require actual net-area uncertainty; remove the implicit Poisson uncertainty assigned to an activity in Bq.
- [x] Carry peak uncertainty through the desktop workspace, session persistence and activity review; missing values require refitting.
- [x] Finish default CLI peak-area, covariance and timing integration checks: 107 CLI/artifact/contract tests pass. Activity requires net-area uncertainty; reaction-rate conversion requires an explicit EOI reference.
- [x] Correct and independently test detector-model attenuation units and refresh the installed wheel. Mass coefficients now include density before multiplication by geometric thickness. Source checks pass; installed numerical checks pass, with two separate background-asset failures tracked under Windows.
- [ ] Qualify physical calibration, timing, nuclear data and shared covariance for experimental acceptance.
- [ ] Establish external dosimetry parity. Generic software tests do not establish STAYSL or PeakEasy parity.

## D — Existing INL/UWNR data

- [x] Recover the 32-count study register and identify 29 report/spectrum clock conflicts.
- [x] Implement a separate raw-spectrum reduction with hash-based aliases, signed ROI areas, counting uncertainty, per-peak failure reporting and overwrite protection.
- [x] Parse 47 distinct accessible spectrum byte identities from 129 paths, including 26 audited count identities; all parsed. Verify 527 source files unchanged.
- [x] Preserve a first provisional ROI pass with 564 strongest-candidate rows. This pass omits 2,204 detected candidates and uses fixed-width sidebands without nonlinear fits; it is not complete peak analysis.
- [x] Remove the candidate cap in v2: all 2,768 detected candidates reduced, with zero parse/area failures. Keep fixed-width sideband and unresolved-overlap limitations explicit; this is not fitted or accepted activity analysis.
- [x] Compare raw/QG area candidates: 221 unique and 36 ambiguous associations under the stated exploratory matching rule; no parity acceptance claimed.
- [x] Inspect newly supplied `raw_gamma_spec_cambio_converted (1).zip`: all 107 file members match the previously discovered corpus. All 29 CHN and 29 SPE files decode to count arrays equal to their existing ASC counterparts; 29 SPC members remain undecoded. It adds no missing A-series spectra.
- [x] Inspect newly supplied `qg_activation_by_isotope (1).csv` and `writeup (1) (1).pdf`: 93 historical derived activity rows and a five-page methodology; neither supplies a measured absolute-efficiency calibration.
- [x] Finish QuantumGold and Lab 6 evidence review. Recover a 1,960-row historical `eff.csv` table; percent scaling agrees with two report-derived Co-60 points within 0.63% and 1.66%. Units, conflicting detector dimensions, negative low-energy values and suspicious attenuation columns still prevent calibration acceptance. Lab 6 supplies energy calibration only.
- [ ] Resolve detector/date/geometry calibration bindings and irradiation/count timing conflicts before accepting activity, EOI, reaction-rate or calculation/experiment results.
- [x] Update study coverage, result contract and Paper 3 readiness notes. Reconcile all 93 historical activity CSV rows against the 88-row audit without merging or altering either source. T4/T10 are ready as audits; T1/T2 partial; T3/T5–T9 blocked.

## W — Native Windows Qt

- [x] Create a separate non-editable install and verify imports from its installed package outside the source checkout.
- [x] Launch and close a visible native Windows Qt window.
- [x] Exercise installed core checks (72 pass, 9 optional-schema skips), session checks (5 pass) and production-shell checks (9 pass).
- [x] Fix activity export's missing uncertainty wiring and verify source session/activity checks, including a legacy-session refit message.
- [x] Rebuild the clean wheel and verify installed-only identity, native launch, core/peak contracts (81 passed), session persistence (5 passed), production workflows (9 passed), and ROI/exports (5 passed).
- [x] Add the optional schema test dependency and pass all 41 workspace/schema checks; the earlier nine optional-schema skips are now covered. Verify the final text-only example update changes no numerical or GUI code.
- [x] Record the final wheel identity and 113 distinct installed checks (all pass) in [the Windows receipt](WINDOWS_VALIDATION_20260916.md). Manual usability, DPI/multiple-monitor/long-session checks and an installer remain unqualified.
- [x] Verify the installed-profile correction. All three profiles resolve the packaged background, its original hash matches, and the installed profile tests pass. The earlier 113-pass receipt is superseded for the newly exposed incompatible-grid workflows.

## Feature regression follow-up

- [x] Preserve the independent feature report and executable reproductions on D:.
- [x] Integrate the packaged RAFM background and Windows SQLite proposals. The packaged background matches the original SHA-256 `505565653785c1e704f175e32e09ae1d69352fd5c891ff413f5acda29f633374`; physical suitability remains unqualified.
- [x] Verify the integrated profile/SQLite source checks: 12 passed, with original background hash checked. Receipt: `D:\FluxForge-feature-validation-20260916\logs\integrated-profile-sqlite.log`.
- [x] Verify profile/SQLite corrections in the rebuilt installed package; the core/reader slice has 99 passing checks.
- [x] Preserve N42/SPE calibration and counts; reject malformed SPE/CHN inputs and incompatible summation; correct calibration inversion and CLI dispatch. The isolated proposal passes 195 focused/compatibility checks, and all 62 focused cases pass after integration. N42 also rejects unsupported processed-count exports before touching the destination (18 installed N42 checks pass). Full N42 schema/export conformance and CHN timing/date/trailer conventions remain separate limitations.
- [x] Preserve signed GUI background results, reject unsupported grid transforms, propagate ROI estimator variance and correct shared-width constant-background fitting: 30 supported-path checks pass.
- [x] Implement and verify strict background count-time normalization and explicit energy-axis preservation in the isolated D: follow-up source: 112 focused checks pass. See [follow-up receipt](BACKGROUND_FOLLOWUP_20260917.md).
- [x] Integrate the background follow-up patch, repair the relocated worktree pointer and refresh the separate recovery installation after disk space was restored.
- [x] Preserve signed fractional spectra and explicit uncertainty in FFS sessions; align Python and JSON Schema validation.
- [x] Implement and independently test conservative histogram rebinning with sparse covariance (14 checks).
- [ ] Carry rebinned covariance through spectrum/session storage, ROI estimators and peak fitting before enabling mismatched-grid background subtraction.
- [ ] Restore valid processing for incompatible RAFM background grids. Eight historical workflow checks now fail explicitly at the grid guard; calibration values are not overwritten to make them pass.
- [x] Recover duplicate-timestamp forecasts and unavailable/invalid custom libraries; escape report metadata. Six focused checks and 30 existing regressions pass; recovery reasons are visible in the library summary.
- [x] Build the combined 271-member wheel after integration; Final SHA-256 `81c41a1d107d8dde8bf56d7c63713f82750d8434644d51e9767bc4ab4832faf2`. Source remained unchanged during build and all packaged source files matched the staging snapshot.
- [x] Record affected installed outcomes: 99 core/reader passes, 18 final N42 passes, one optional-dependency HTML pass, and eight incompatible-grid workflow failures. The native refresh is explicitly blocked at `QApplication` startup, including a minimal independent probe.
- [x] Retain a new hash-verified patch/archive snapshot in `D:\FluxForge-recovery-20260916\final-local-changes-v2-20260917`; retain the earlier checkpoint unchanged.

- [x] Retry native Windows startup and bounded source workflows after offload: minimal initialization passes; 45 source checks pass. Refreshed installed wheel passes 191 core/reader/session/schema checks, 11 native session/recovery checks and one isolated HTML check. Only the POSIX-only check remains inapplicable.
- [x] Verify all 1591 original archived-manifest source files at the authoritative offload location; no content changes.
- [ ] Complete broader interactive desktop qualification, including DPI, multiple monitors and long sessions. The native automated pass does not close these gates.

## A — Archive and consolidation

- [x] Preserve all 16 selected legacy source/package files in a hash-verified archive and restore them to a separate directory.
- [x] Instantiate and close the restored historical interface; document that this does not prove complete manual workflow parity.
- [x] Exclude the historical package and launcher from normal installation; select modern Qt in the frozen launcher.
- [x] Document the retained legacy feature map, including the unported constrained multiplet fitter.
- [x] Verify the final installed wheel excludes the historical implementation and exposes `fluxforge-gui` as the maintained GUI command.

## Current limitations

- Some university OneDrive calibration leads are offline placeholders; the cloud provider is not running.
- The C: drive previously filled during validation; subsequent offload restored free space. The selected checkout resolves through a junction to the authoritative D: copy. The two newly generated main preservation archives were moved to `D:\FluxForge-recovery-20260916\archives` with matching hashes, recovering about 200 MB; originals remain in place. New disposable test/build files use a dedicated D: directory.
- Source changes are local and uncommitted. Nothing has been pushed, merged, published or represented as accepted experimental results.

Evidence: [recovery state](RELEASE_RECOVERY_STATE.md), [scientific validation](SCIENTIFIC_VALIDATION_20260916.md), [Windows validation](WINDOWS_VALIDATION_20260916.md), [legacy restoration](LEGACY_GUI_RECOVERY.md), and `C:\Users\Josh\projects\FluxForge-recovery-20260916`.
