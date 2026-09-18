# Scientific validation — 16 September 2026

This is a scoped software and numerical review. It does not establish accepted INL activities, a qualified adjusted neutron spectrum, or standards compliance. Study measurements require the physical and reference-data bindings recorded in the study register.

## Result chain and acceptance boundaries

| Stage | Method / units | Required evidence and present boundary |
|---|---|---|
| Spectrum | Recorded channel counts and live/real seconds | Preserve original bytes and headers; conflicting report/ASC clocks remain unresolved. |
| Energy calibration | Channel-to-energy mapping, keV | Bind coefficients, calibration date, residuals and detector identity. Header calibration alone does not certify its validity. |
| Background / ROI | Signed foreground minus scaled background, counts | Propagate foreground and background variance; retain negative estimates and source live-time ratios. |
| Peak fit | Peak integral and fit covariance, counts² | Area uncertainty must follow the fitted/background model; square root of net area is not a general substitute. Overlap covariance remains relevant. |
| Gamma identification | Evaluated transition energies and emission probabilities | Pin actual source/version, line identity, units and uncertainty; a nearby candidate energy alone is insufficient. |
| Activity | Count integral divided by efficiency, emission probability and exposure integral, Bq | Separate live-time normalization from real elapsed decay. A constant live fraction is an explicit approximation, not a general correction for time-varying losses. |
| EOI / inventory | Exponential decay or coupled decay equations; Bq / atoms | Bind actual count and irradiation times/timezones, half-lives, previous inventory and parent feeding. Filename cooldowns are insufficient. |
| Reaction observable | Activity divided by irradiation buildup and target convention | Bind each physical monitor's own history, target atoms, channel, cross-section and correction definitions. Activity uncertainty cannot be inferred as Poisson counts. |
| Covariance | Counting terms plus shared calibration, emission and timing terms | Shared nuisance terms do not become independent when combining lines or repeated spectra. Missing covariance remains a limitation. |
| Model comparison | Same physical observable and reference interval | No accepted calculation/experiment ratios until matching measurement/model provenance exists. |
| Spectrum adjustment | Linear Gaussian prior plus response/observations/covariances | Generic algorithm checks do not certify activation dosimetry. Nuclear-data covariance, response units, energy groups and monitor identifiability must be qualified separately. |

## Corrected inverse-analysis behavior

The GLS checks now compare its posterior with an independently solved quadratic objective in precision space, including correlated inputs and a non-square response. They also test singular but consistent measurements, contradictory exact constraints, signed background-subtracted observations, monitor leverage and covariance validity.

The Monte Carlo response-uncertainty path previously returned only variation among conditional mean spectra. It now includes the mean conditional posterior covariance. The zero-response-uncertainty limit therefore retains the prior/measurement uncertainty. A deterministic two-response test checks both terms independently.

Covariance inputs must be symmetric and positive semidefinite. A spectral pseudoinverse replaces inversion through normal equations. The reported condition number includes correlations. The innovation chi-square uses the rank of its covariance as its degrees of freedom; it is explicitly a prior-predictive statistic. Influence is the diagonal of the measurement-space response-times-gain matrix, one value per monitor.

Unsupported covariance policies now raise an explicit error. Linearized augmentation requires supplied element variances; the separate Monte Carlo interface retains its documented element-standard-deviation input. Both remain approximations without cross-element response correlations.

Existing nonnegativity clipping remains a compatibility behavior. Whenever clipping occurs, diagnostics explicitly reject the claim of a valid constrained posterior: the returned covariance is the unconstrained linear Gaussian covariance. Such a result cannot be accepted as a constrained dosimetry posterior. Monte Carlo diagnostics identify when nominal-response diagnostics accompany an averaged spectrum.

## Independent references and limitations

The efficiency evidence review identified an additional dimensional error in the detector model: centimeter layer thicknesses were multiplied by mass attenuation coefficients in cm²/g. The implementation now obtains linear coefficients in cm⁻¹, including material density. [NIST's attenuation definition](https://physics.nist.gov/PhysRefData/XrayMassCoef/chap2.html) uses mass thickness with mass attenuation, or equivalently geometric thickness with linear attenuation. An independent density/path-length equation check and focused calibration checks pass. This establishes the dimensional correction, not the physical calibration parameters or an exported efficiency table.

No empirical coefficients or dimensions were retuned to reproduce QuantumGold. With the existing 6.45 cm profile thickness, corrected model values at the two Co-60 energies are approximately 0.001066 and 0.001025. The report-derived values are approximately 0.0003914 and 0.0003604; these are not interchangeable calibration results. The recovered table names a different 1.39 thickness parameter and has a plausible percent-unit interpretation, but its provenance, units and validity remain unresolved. Existing profile coefficients require qualification under the corrected model before absolute activity acceptance.

The default segmented CLI peak path now fits raw counts in channel space. Its integrated area no longer changes with energy-calibration gain, and its area uncertainty includes amplitude-width covariance. Peak reports carry elapsed counting time through serialization. The simple `activity` command now uses the same finite-count decay integral as activity review, requires a supplied area uncertainty and explicit efficiency/emission/half-life values, and labels the result as count-start or EOI activity. An explicit `--cooling-time-s` establishes the EOI interval; `rates` rejects inputs without an EOI reference. Scalar calibration inputs in this command are treated as fixed, and the output states this uncertainty limitation.

Activity review retains a shared efficiency term when combining lines and preserves signed estimates. This is a stated correlation assumption; it does not recover a measured energy-dependent calibration covariance. Low-level legacy helpers and provisional study single-peak outputs can still use diagonal parameter uncertainty; they are not qualified as covariance-complete measurements. Overlap/background model inadequacy, nuclear-data covariance and physical metadata remain separate acceptance gates.

The [PNNL STAYSL guide](https://www.pnnl.gov/main/publications/external/technical_reports/PNNL-22253.pdf), section 8.5, describes adjustment using activity, prior-spectrum and cross-section covariance. It supports the distinction between dosimetry adjustment and generic unfolding; no STAYSL parity is claimed here.

[NIST's combined decay/dead-time study](https://www.nist.gov/publications/corrections-combined-effects-decay-and-dead-time-live-timed-counting-short-lived) explains why separately applied corrections can fail for short-lived sources with varying losses. The present constant-live-fraction numerical oracle is a controlled approximation and cannot establish an instrument-specific correction in that regime. [NIST TN 2073](https://nvlpubs.nist.gov/nistpubs/TechnicalNotes/NIST.TN.2073.pdf), appendix B, supplies additional gamma-spectrometry correction context.

## Receipts

- `C:\Users\Josh\projects\FluxForge-recovery-20260916\science\inverse-contract-tests.log`: 70 passed, 10 skipped. External implementation fixtures are absent; skips do not establish comparison parity.
- The broader inverse run was stopped while the large reference baseline was still running. It is not a passing receipt. The 10-bin regression was also changed to write into a temporary test directory rather than overwrite historical artifacts.
- `science/covariance-regression.log`: 100 passed.
- `science/activity_correctness_junit.xml`: 24 targeted activity checks passed.
- `science/peak-activity-integration-final.log`: 14 focused peak/timing/activity checks passed.
- `science/cli-artifacts-verified.log`: 107 CLI, artifact and scientific-contract checks passed, with 17 existing timestamp-deprecation warnings. An earlier run failed because three mocks lacked the new uncertainty fields; the mocks now supply explicit synthetic uncertainties. An initial temporary-directory setup failure is retained separately.
- `windows/source-efficiency-linear-attenuation.log`: 16 focused detector/efficiency checks passed. `windows/source-flux-wire-parity-linear-attenuation.log`: 10 existing workflow regressions passed; their historical parity names do not certify the recovered physical calibration.
- Native interface receipts are maintained separately; counts from overlapping test slices must not be summed as distinct tests.
- PeakEasy: `deferred_external_validation`. No fresh execution or parity claim. Historical comparisons require identical source observations and conventions.


## Feature follow-up — 17 September 2026

The GUI background path now uses the shared signed subtraction and uncertainty calculation. Unequal energy grids are rejected: point interpolation of histogram counts was not count conserving, and the spectrum container cannot retain the induced bin covariance. Manual background scales must be finite and nonnegative. Same-grid independent-count propagation, ROI sideband estimator weights (including shared ROI/sideband bins), and both constant/linear shared-width doublet fits pass 30 focused and existing checks.

This exposes eight historical RAFM/flux-wire workflow failures. The sample and raw background have equal channel indices but energy differences from -4.5115 to +2.043 keV. The actual implicit profile-background loader uses a different override and still differs from this sample by -1.0742 to +2.235 keV. Their calibrations are preserved. These workflows remain blocked until scientifically supported alignment and covariance propagation are supplied; the prior passing interpolation results are superseded for this use.

SNIP and linear-minima continuum uncertainties retain an explicit integrated-count approximation because their fitted-background covariance is unavailable. Shared-width index correction does not establish covariance-complete experimental multiplet areas. Provisional study reductions remain unchanged and do not use the newly rejected external-background interpolation.

Receipts: `D:\FluxForge-feature-validation-20260916\numerical-fixes` and `integration-20260917/implicit-profile-grid-difference.json` under the same validation directory.

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
