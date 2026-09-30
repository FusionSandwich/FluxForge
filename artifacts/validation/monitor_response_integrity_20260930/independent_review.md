# Independent bounded adversarial review — 2026-09-30

Reviewer: GPT-6.1 Sol. Root retains final acceptance and scientific admission.

Reviewed implementation commit `2756b00f655bc90001f8e27220268a83da972f07`, parent `9d04eff6ebc867227ae439ecdcc37ddb40732b7f`, checkout `C:\Users\joshu\fluxforge-worktrees\response-identity-validation`. Reviewed changed physical response/aggregation logic, new integrity tests, actual-data validator, original #208/#209 issue specifications and decisive before_v2/after receipts. No source mutation, acquisition, remote work, GitHub action or delegation.

## Actionable finding

[P2] Remove the uncertainty floor from the dimensionless scaled-weight denominator.

`src/fluxforge/workflows/spectrum_unfolding.py`, aggregation calculation `sum(weights * measured_rates[indices]) / max(weight_sum, floor)`.

The new weights are `(min(sigmas) / sigmas)**2`, so at least one weight is exactly one and the weight sum is at least one. `floor` remains a caller-supplied uncertainty floor, now unrelated to the dimensionless weight normalization. A positive finite floor larger than the replicate count passes validation but silently reduces the weighted mean. Reproduced on the exact reviewed implementation:

```python
u = SpectrumUnfolder.__new__(SpectrumUnfolder)
out = u._aggregate_duplicate_reaction_rows(
    np.ones((2, 2)), ['r', 'r'],
    np.array([2., 4.]), np.array([1., 2.]), floor=100.,
)
# Both sigmas floor to 100, both scaled weights are 1.
# Expected mean: 3. Actual mean: 0.06.
# Returned uncertainty: 70.71067811865474 (correct).
```

Recommendation: divide by `weight_sum` directly and add a bounded regression with `floor` above the replicate count. Default `floor=1e-30` and the actual-data receipts are unaffected. This is a narrow helper parameter failure, not evidence of changed nominal physical operators or an incorrect actual-data result.

## Passing decisive evidence

Independent execution: 87 focused tests passed in 11.86s (`tests/test_monitor_response_integrity.py`, `tests/test_monitor_response.py`), existing Python `D:\FluxForgeQA\envs\fluxforge-py312\Scripts\python.exe`, offline, bytecode/cache disabled, Agg backend. Root's final suite receipt reports 216 passed.

Reviewed input identity uses round-trip float precision and full SHA-256: all CoverLayer fields and body geometry, dimension, number density, uncertainty, energy/value tables and source are present. Default and explicit cover density/mass converge to the same key. Body tables are copied into immutable tuples. Aggregation separately checks exact reaction, nominal operator, total response uncertainty and supplied uncertainty-component model; one-ULP differences remain separate in the focused tests. Membership lists preserve original row identity and ordering. Constructors and physical build/workflow/kernel boundaries reject nonfinite/negative physical inputs; legitimate zero kernel and total-table limits pass.

Actual-data receipts: `before_v2/receipt.json`, `after/receipt.json`, matching output SHA manifests verified independently. After receipt SHA-256: `411b29b0ce47221c34b36ed5ed620ee5490a102777748fc735aa27e5f28fdf2d`. Shared validation-script SHA-256: `1e68c1a1b8879ef31b04966b7a4789b4ea823767174aa69bdb28c0f617d43ed4`. All 32 after-manifest input hashes match current bytes, including joined raw ASC/QG/saved-analysis sources and local evaluated archives. Checkout metadata JSON hashes match the baseline; mass-review text matches after universal newline normalization despite LF/CRLF byte differences.

All 13 source/rate records, physical nominal and uncertainty rows, diagnostic prior and forward predictions are exactly equal before/after. Unaggregated GRAVEL/MLEM responses, rates, flux and predictions are exactly equal. Both revisions form seven valid aggregate groups: four Co/Sc bare/Cd observations and three Ti reaction groups with three members each. Scaled aggregation introduces only rounding differences: max relative aggregate-rate delta `2.393110074207898e-16`, aggregate flux delta GRAVEL `5.028124901900647e-15`, MLEM `1.4505987097196185e-15`, aggregate prediction delta GRAVEL `3.5960422338543227e-16`, MLEM `4.581412478166071e-16`.

Four cover variants (density, atomic mass, one-ULP thickness, thickness uncertainty) each retain two rows in the repair. The uncertainty-only variant has equal nominal rows but unequal uncertainty rows. All 12 deliberately mutated invalid cover copies reject at the workflow boundary. Actual Cd absorption archive has 180,848 coordinates including 357 equal adjacent printed coordinates; replay accepts it while maintaining strict caller body grids.

Reviewed source SHA-256:
- monitor_response.py: `33d3c41e1b56a648ba75f6bcc0493912ed67953b2b58876b2fff43f6530a6748`
- spectrum_unfolding.py: `a0eeb9f02a69258f5c4411cfb694b8641428c40a96115e846ae218c207491fec`

## Conclusion

The original #208/#209 counterexamples are addressed within the bounded reviewed path and actual-data compatibility is supported. One actionable denominator finding remains; recommend the small fix and focused recheck before software acceptance. This review does not admit a measured spectrum, confidence interval, scientific validation, full covariance, as-built geometry or comparator agreement.
## Resolution and final bounded acceptance

The finding above is resolved by follow-up implementation commit `f378b218a119d46609b9f7d79dbf380b1584cda8`. Independently inspected the one-line calculation repair and new `test_large_uncertainty_floor_does_not_scale_weighted_mean`. The exact adversarial reproduction now returns mean `3.0` and uncertainty `70.71067811865474`. The focused new regression passed (`1 passed, 75 deselected in 1.76s`), offline using the same existing environment. Current workflow SHA-256: `46b5a74b91d5f3a3035bf754682d2f4725e457c90413d3c7b603fdc65c24cf56`; monitor_response.py and validator remain at the hashes above.

This resolution supersedes the earlier open-finding conclusion. No actionable finding remains within the bounded review scope. Software acceptance of the #208/#209 repair is supported for this exact implementation plus its inspected follow-up. Root should retain the final 217-case-suite and after_v2 replay receipts, which were still running when this narrow independent check completed; this reviewer does not claim those pending outputs have passed. The prior actual-data receipts remain relevant to unchanged nominal/default behavior: the denominator repair changes only a nondefault helper floor branch and is separately covered by the reproduced counterexample. Scientific admission and final publication acceptance remain with root; all recorded scientific limitations continue to apply.