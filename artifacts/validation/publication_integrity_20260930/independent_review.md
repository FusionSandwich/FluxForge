# Independent continuation review — 2026-09-30

Reviewer: GPT-6.1 Sol; root retains final software/publication acceptance. Scientific admission remains false.

Initial frozen implementation: `ec62698eb56cdbe70df07b293091e6fe78f8cbf2`, parent integrated `f0797d4`. Worktree: `C:\Users\joshu\fluxforge-worktrees\unfolding-publication-integrity`. Reviewed changed cache, MC, source/component budget, covariance/holdout, abundance and physical-GLS receipt logic, regressions, continuation validator and decisive receipts. Followup `8fe3d8adefa97da75051d97bf29f75e59d997b55` inspected independently. No acquisition, remote work, external publication or source edits by reviewer.

## Findings and resolution

1. [P2, resolved at 8fe3d8] Valid singular shared-error aggregation rejected by cancellation roundoff. Identical operators, rates `[1,1]`, errors `[.14,.23]`, `C=outer(errors,errors)` gave BLUE weights `[2.55555556,-1.55555556]`. Direct `T C T^T` was `-1.7732728865540703e-17`, and covariance validation at this residual's own scale incorrectly rejected valid input. Also reproduced errors `[1,3]`. Followup uses the Gram matrix of the transformed covariance factor, preserving nonnegative zero-variance limits. Two new exact regressions independently passed.

2. [P2, resolved at 8fe3d8] Fully shared/zero covariance could erase deterministic replicate inconsistency. Same operators, rates `[1,2]`, `C=.01*ones((2,2))` originally returned mean `1.5`, variance `.01`, although the difference has zero variance under the supplied common-operator model. Followup explicitly checks covariance-nullspace residuals with scale-aware tolerances. Six new shared/zero covariance regressions at rate scales `1e-30`, `1`, `1e30` independently passed.

3. [P2, pending] RAFM source declaration `null` silently becomes string `None`. `kwargs = dict(source=str(spec.get('source', '')),...)` in `build_flux_wire_reactions` coerces a missing JSON source into nonempty text, so complete source coverage is falsely declared. Reproduced with config `rate_uncertainty_components={name:{'relative':.01,'source':None} for name in REQUIRED_COMPONENTS}`, otherwise the existing itemized RAFM regression's metadata/timing/activity fixture. All declared component sources were `'None'`, budget.complete was True, and `rate_covariance([budget],require_complete=True)` succeeded. Preserve absence or reject nonstring source values and add a null-source regression. Actual source-joined budgets remain incomplete and are unaffected.

## Decisive passing evidence

Independently ran initial `tests/test_publication_integrity.py`: 35 passed in 5.33s. Environment: existing `D:\FluxForgeQA\envs\fluxforge-py312\Scripts\python.exe`; checkout src in PYTHONPATH, offline, bytecode/cache disabled, Agg backend. Independently ran the eight new singular regressions after inspecting 8fe3d8 (2 passed in 1.56s, 6 passed in 1.54s). Root logs inspected: earlier 303 affected tests passed; adjacent 45 passed, 10 skipped.

Final ec62698 actual receipts were independently read and hashed:
- `artifacts/validation/publication_integrity_20260930/operator/receipt.json`: `097901096691d4f5b02ad9cb69fd9639d381eff9a03dea60d79d03a772e31dfb`.
- `artifacts/validation/publication_integrity_20260930/continuation.json`: `a72ea08be1a4430d01467ad826199138743f7f614db6ccef5e926334c5c77807`.

Both record exact commit ec62698. All source/input manifests match existing file bytes at inspection; changed source-file hashes match reviewed code. Against frozen PR210, energy boundaries, 13 nominal and uncertainty rows, diagnostic prior and forward predictions are exactly equal. Validator checks bounded workflow equivalence at `rtol=1e-12` and binds the earlier operator receipt. Density changes match warm/fresh rebuilds; rate-only changes retain cache; missing covered response rejects. All 13 actual rate budgets correctly list six missing measured components and fail strict qualification. Actual eight-draw MC has eight finite draws, zero converged draws and an unconverged main fit: default qualification unavailable, explicit capped qualification diagnostic. Physical-GLS receipt and overall receipt retain scientific_admission=False.

Independent additional holdout probes at covariance scales `1e-35`, `1`, `1e35` match direct correlated Gaussian conditioning. Noiseless incompatible held-out residual rejects. Explicit Ni-58 abundance `1.0` and `.999999` are retained, missing abundance resolves to `.6808`. Tests cover changed current source/spec/grid and public result mutation, all-draw qualification without survivor selection, signed shared component covariance, coverage overlap, and the total-error physical-GLS statistic definition.

## Scope limits and current decision

The rank-seven operator over 20 groups, input/response/covariance mismatch, SpecKit original settings/quasi-single benchmark, metrology/calibration/history/geometry and comparator/source qualification remain unresolved source-audit matters. Synthetic prior and response-error terms exercise software interfaces only. No measured uncertainty, spectrum, confidence interval or publication validation is admitted.

Acceptance remains pending finding 3 and its focused independent recheck. Earlier actual receipts bind ec62698; root will preserve final replay/test evidence for the repaired revision. The reviewer has not rerun a broad campaign.
## Final resolution and bounded software acceptance

Final reviewed source commit: `098936f0901bf50e377655fed4f3c76dbb7b7d70`. This conclusion supersedes the pending decision above.

Finding 3 is resolved by `454187df54e60d7582ddfc15a077600f4c33268f`: RAFM stops string coercion; component boundary normalizes absent None to blank, rejects nonstring source values, and validates correlation/coverage/name strings. Null and blank sources correctly keep source coverage incomplete; strict qualification rejects them. Independently inspected this narrow patch and its six null/blank/nonstring regressions.

Root additionally found and repaired finite Monte Carlo draws whose sample covariance overflows. At `098936f`, both converged and capped estimators preserve all draw statuses but report the ensemble unavailable, usable count zero, NaN flux uncertainty, no covariance, and explicit ensemble_error when the derived covariance/std is nonfinite. Independently inspected the guard and both regression cases. This does not invent missing uncertainty or discard failed draws.

Independently executed the final focused selection covering source, covariance overflow and both singular fixes: `21 passed, 30 deselected in 2.87s` at exact final HEAD. Together with initial 35-case regression execution, direct multiscale holdout and explicit abundance probes, and independently verified hash-bound actual ec62698 receipts, the bounded software contract is supported. These test invocations overlap and are not presented as a count of unique tests.

No actionable finding remains within this review scope. Software acceptance at exact final source HEAD is supported. Root's final broad test and actual replay evidence for 098936f was still being generated at this review's close; root should preserve/check it before final publication. This reviewer does not claim those pending receipts have passed. Earlier actual receipts remain applicable to nominal software compatibility; repaired failure branches are covered by the focused counterexamples and followup regressions. Scientific admission remains false and all source-audit limitations above remain in force.
Final reviewed source/test SHA-256:

```json
{
  "src/fluxforge/workflows/spectrum_unfolding.py": "d2a887bf26b0c09736739ce41f323973f2130f1c841f0203c7efddeec7ebca44",
  "src/fluxforge/uncertainty/covariance.py": "f71cbb6658ef64854cb5460db54fafee711dce7fb3e3e5270c03955d13b175c6",
  "src/fluxforge/uncertainty/reaction_rate_budget.py": "c4771c6f48dfae088cbf462ea28906ed4ba3cb163ecc3aeed083e46c26469f4c",
  "src/fluxforge/analysis/holdout_validation.py": "a5be0838b05ee4519f72eb7e15947380451acc144f2ee0319bc7e83d8a90441a",
  "src/fluxforge/analysis/physical_gls.py": "59053177666ddf0dc35f9a0a7dcbcd9db60e1eaba699503c3e0fc6925533f591",
  "src/fluxforge/examples/rafm_workflow.py": "10b21224776d593167caed090b1fa6aedab7dafdfba5e372862b87cc8c0d566a",
  "tests/test_publication_integrity.py": "0f651a5aed2b4dc38f990e543f7dfea183c58c5f79b9eff01ae4e29f9dbfc6d0",
  "tools/validate_publication_integrity_rafm.py": "fb259dc221fec5efd449a914f79807adbe55b28327324974b9da5d2797a7a5bc"
}
```
