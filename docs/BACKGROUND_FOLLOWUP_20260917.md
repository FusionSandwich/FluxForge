# Background subtraction follow-up - 17 September 2026

Integrated into the selected validation checkout after the drive offload. Its stale Git worktree pointer was repaired to the verified D: repository metadata; no branch or commit was changed. The recovery installation has been refreshed with the integrated wheel; new receipts are under `D:\FluxForge-integrated-followup-20260917`.

Automatic live/real-time scaling now requires a finite, positive selected count time for both spectra. Unknown, zero, negative or non-finite times raise an actionable error instead of silently selecting unity or producing invalid counts. Explicit manual scaling remains available without count-time metadata. Subtraction retains an explicitly supplied energy axis as an independent copy.

The new regression cases reproduced 21 failures before the change, with two valid-path checks already passing. After the fix, 112 focused checks pass, covering normalization, signed counts, ROI uncertainty, peak backgrounds and reader/export regressions. Receipts are `before-tests.txt` and `after-tests.txt` in the follow-up directory. A separately installed candidate wheel on D: also passes 45 focused checks, with imports verified outside the source tree and packaged source bytes checked. Wheel SHA-256: `8ddbad73aaa12e393a8d92624280b751629bc8ac0e9c679909b64618bedc6692`. The existing C: installation is unchanged. These checks do not establish physical experimental acceptance.

## Mismatched RAFM grids

The shipped background calibration is [-1.502, 0.4991, 2.239e-7]; one representative sample family uses [0.541, 0.498, 2.605e-7]. Both have 8192 channels, but their energy difference changes from -2.043 to +4.512 keV across the array. The bin edges are not nested; over 8180 background bins overlap multiple sample bins. Neither channelwise subtraction nor a single integer shift is valid from these headers.

A valid general solution needs an overlap matrix W, rebinned background Wb and net covariance C = diag(sample variance) + scale^2 W diag(background variance) W^T. The off-diagonal covariance must survive spectrum/session serialization and enter ROI linear estimators and peak fitting. Per-channel uncertainty alone cannot represent it. The existing grid guard remains in force; eight historical workflows remain blocked.

Required follow-up: a sparse covariance representation, conservative bin-overlap mapping with an explicit coverage policy, ROI variance w^T C w, covariance-aware fit weighting and spectrum/session roundtrip tests. This broader work is not claimed complete here.

## Further corrections and validation

Signed fractional counts now survive workspace/session save and reopen with their explicit uncertainty and energy metadata. Persisted signed counts without an uncertainty array are rejected before automatic Poisson defaults can be applied. The document schema enforces the same requirement. Workspace, migration and atomic-save checks with schema validation enabled: 69 passed, one POSIX-only skip.

The conservative histogram rebinning primitive now returns sparse overlap weights, full propagated covariance, coverage fractions and discarded counts. Fourteen independent tests cover hand-calculated shared-bin covariance, singular and invalid matrices, overflow, cropping and an 8192-bin shifted grid. This primitive is not yet connected to background subtraction: spectrum/session covariance storage and every affected ROI and fit consumer must be integrated first. The existing mismatch guard remains intact.

A bounded native Windows source run completed with 45 passes and one optional HTML-dependency skip. It covers session replacement/reopen, calibration undo, recoverable GUI errors and a real bundled spectrum. A minimal native Qt initialization also passed. An earlier uninstrumented batch was interrupted without results; slow application-wide styling was observed in the successful instrumented run. These automated checks do not establish DPI, multi-monitor or long-session usability.

Integrated wheel SHA-256: `fc179bd1b3b6fcfec0ce33bee9ea464545786b84725e8d8e4ef94c3278e1c023`; 272 members. Package source bytes match the staging snapshot, the source was unchanged during build, and the historical interface is excluded.

Final installed verification: 191 core/reader/session/schema passes, 11 native session/recovery passes, and one HTML pass with its optional dependency. One POSIX-only check remains inapplicable on Windows. Receipt directory: `D:\FluxForge-integrated-followup-20260917`.
