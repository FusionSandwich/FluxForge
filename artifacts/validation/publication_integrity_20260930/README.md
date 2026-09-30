# Final software evidence

Final reviewed implementation: `098936f0901bf50e377655fed4f3c76dbb7b7d70`.
The following publication commit adds receipts only. PR205, PR207 and PR210
remain unchanged and are ancestors of this feature.

Use `continuation_final.json` and `operator_final/receipt.json` for the final
source revision. `continuation.json` and `operator/receipt.json` retain the
earlier ec62698 replay for provenance. The operator tool is the unchanged PR210
compatibility harness, so its legacy text describing PR205 as separate refers
to that harness's scope; this integrated feature also exercises physical GLS in
the continuation receipt.

Final continuation SHA256:
`aa603cd6ceaa7b6a7f410395a79d9a57e73344c08e1c22b36a09fbda1f840e77`.
Final operator SHA256:
`d764b1f131088b976db747ee18e74fd6d9621b1645e554fcc83bf8f6c50365d0`.

The replay verifies original source hashes, 13 observation identities/rates,
exact nominal and uncertainty operator rows, exact fixed-prior forward folds,
and valid GRAVEL/MLEM workflows against frozen PR210 at rtol 1e-12, atol zero.
It confirms 13 observations retain seven compatible aggregation groups, four
changed cover copies remain separate and 12 invalid physical copies reject.
The changed-density warm cache matches a fresh response; rate-only changes
reuse it; a missing covered response specification rejects even with a cache.

Actual one-iteration MC records every requested draw. Default converged mode
is unavailable; explicit capped mode is diagnostic. All 13 actual rate budgets
still lack six measured components and reject strict qualification. Source
bytes are unchanged. Full physical GLS uses an explicitly synthetic prior and
response-error covariance only to verify the interface/statistic definition.

`independent_review.md` records bounded GPT-6.1 Sol acceptance, including the
singular covariance, deterministic contrast, absent source and overflow fixes.
`verification.json` and `final_tests.txt` record final local verification.

Scientific admission is false. Rank 7 over 20 groups and the unresolved
common-operator input/response/covariance discrepancy remain visible. No rates
or model floors were tuned for agreement. The measured source/calibration,
history, cover/body geometry and covariance gates and unresolved original
SpecKit qualification remain owned by the publication audit chat.
