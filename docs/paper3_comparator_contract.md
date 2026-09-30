# Paper 3 source-bound unfolding comparison contract

Status: **synthetic method checks only**. No Paper 3 measured spectrum or
independent published-code validation has been produced.

## Candidate implementations

- Primary: FluxForge `unfold_gls_physical`, with explicit physical response,
  prior, covariance, monitor identities, and source hashes.
- Independent activation comparator: the **original**
  [SpecKit repository](https://github.com/lifangchen2021/SpecKit) associated
  with the [SoftwareX 2025 article](https://doi.org/10.1016/j.softx.2025.102456).
  Its documented inversion uses activity coefficients, least squares, and log
  smoothness. At a future qualified run, record the exact upstream commit,
  unmodified source hashes, configuration, and executable environment. Verify
  its actual accepted units and objective from that pinned source before
  adapting the frozen Paper 3 inputs. No original SpecKit source was acquired
  or executed for this task.
- The local `Downloads/speckit_cleanroom_reimpl.zip`, SHA256
  `9BEFCC2696AD00B0D74D3FE198276F10767607F88668DEFE8C5CADF3DE7781E7`,
  says in its README that it is a *clean-room, core-only reimplementation*.
  It is not the upstream SpecKit implementation and does not count as
  independent published-code validation. `examples/speckit_benchmark` holds
  data assets and has no runner.
- FluxForge GRAVEL and MLEM are useful internal method checks. SciPy NNLS is
  an independent numerical baseline on a small nonnegative synthetic system.
  None substitutes for the original SpecKit comparison.
- PyUnfold models a cause/effect count distribution with a conditional
  probability response, as [its derivation](https://jrbourbeau.github.io/pyunfold/mathematics.html)
  states. Direct activation rate and cross-section inputs are dimensionally
  incompatible. The FluxForge wrapper now rejects that direct route and
  returns no invented full covariance.

## Frozen input bundle

The paper team must qualify and freeze one bundle before any experimental
comparison: exact energy edges; group-integral flux definition and units;
physical row IDs including sample, cover, reaction, and product; activities
and irradiation/decay chronology; rate conversion; sample-specific response
with cross-section, mass/abundance, Cd/body and location effects; response
uncertainty; full observation covariance; prior and prior covariance; and
holdout IDs selected before viewing validation rates. Record source-file and
numerical-array SHA256 hashes plus code commits. Preserve all original rows,
including exclusions and reasons.

Each implementation must see the **same mathematical forward problem**. If
SpecKit uses activity observations, transform both measured rates and every
response row through the same frozen sample-specific activity operator, and
transform covariance with its Jacobian. Check that the transformed forward
fold reproduces the frozen activities before inversion. Do not compare output
vectors until group-integral versus differential units, normalization, and
energy ordering match. If a method cannot accept correlated covariance or
response uncertainty, report that limitation. A source-qualified real-data
diagonal-only run of the original SpecKit may be shown as a separate
**sensitivity diagnostic** after the physical input gates clear. Quantify the
omitted-correlation effect by comparing FluxForge fits with the full and
diagonalized covariance on the same frozen inputs, and identify any response
uncertainty omitted by SpecKit. Do not describe the diagonal-only run as an
equivalent-likelihood fit or independent validation of the full-covariance
primary inference. A synthetic diagonal case remains useful for exact
cross-code acceptance before the real-data sensitivity run.

## Acceptance evidence

1. On a source-qualified synthetic known-truth case, report the exact `R`,
   `y`, prior, energy edges, covariance, code revisions, and numerical hashes.
   Compare FluxForge GLS, GRAVEL, MLEM, and independent SciPy NNLS using the
   same group-integral operator. Record each forward fold, residual, rank,
   prior sensitivity, and failure mode. The local focused test performs this
   small comparison; it is a solver check, not experimental evidence.
2. For the original SpecKit run, retain raw outputs and the adapter receipt.
   Recompute all predictions independently as `R @ flux` on the frozen grid.
   Report regularization settings and objective terms separately from
   postfit residuals. Evaluate rank-supported combinations and prior
   sensitivity; do not infer all 20 groups from fewer independent response
   directions.
3. Keep fit and sealed holdout rows separate. Show holdout predictions and
   residuals using the same frozen model without tuning on their activities.
   A fit statistic alone is not validation, especially with regularization or
   correlated errors.

The original Paper 3 activity, response, covariance, and chronology gates
remain open. No experimental rerun is authorized by this document.

## Local synthetic comparison receipt

`tests/test_physical_gls.py::test_known_truth_cross_method_forward_fold_on_same_operator`
uses asymmetric energy edges `[1, 3, 20]` eV and a full-rank, three-row,
two-group operator in `cm2`: `1e-24 * [[2, 1], [0.1, 1], [0.5, 2]]`.
The group-integral truth is `1e12 * [3, 4] n/cm2/s`; the deliberately wrong
prior is `1e12 * [1, 8]`. Forward folding produces
`1e-12 * [10, 4.3, 9.5] reactions/target_atom/s`. The observation covariance
is diagonal with entries `1e-26 (reactions/target_atom/s)^2`. All methods
received those same numerical `R`, `y`, and initial prior; SciPy NNLS uses
unit-preserving rescaling for numerical stability. This is an exact synthetic
truth with no response uncertainty or activity conversion.

| Method | Flux (`1e12 n/cm2/s`) | Maximum relative forward-fold error | Note |
| --- | --- | ---: | --- |
| FluxForge physical GLS | `[2.999847, 4.000146]` | `3.03e-5` | Explicit broad prior covariance |
| FluxForge GRAVEL | `[2.993676, 4.004113]` | `8.54e-4` | 36 iterations; convergence stopping applies |
| FluxForge MLEM | `[3.000000, 4.000000]` | `1.54e-10` | 101 iterations; chi-square early stop disabled |
| SciPy NNLS | `[3.000000, 4.000000]` | `1.88e-16` | Independent Lawson–Hanson numerical baseline |

The exact numerical-response hash from the GLS receipt is
`8ba3fe0b593cfe7a0a1530797e21fd1e6cfead9799d90b2ccfaaab3361c5bd29`;
the prior hash is
`771bd160cdd4af3c1344787d8d6a4a6520577a15fc1b8f2e843ceddb2902fe46`.
The test asserts forward folds and known truth to 0.5%. A rank-deficient
counterexample in the same test file checks prior dependence and nonzero
uncertainty in an unsupported group. The comparison is not an original
SpecKit run and gives no Paper 3 experimental conclusion.
