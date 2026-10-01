# QG source QC replay — 2026-09-30

Source implementation reviewed at b718b780f266fb36786899a5ffb8b5548a5dfef3, stacked on PR212 da0f0680a4d652c3428a9f592057f82940a1f95d. This evidence commit does not change source or tests.

`receipt.json` binds four original Downloads reports, the pre-change `baseline.json`, five source-owner audit files, all 32 prior publication inputs and six implementation/reference files. All original imported values, legacy comparison fields and existing line-summary inconsistency rows remain baseline-equal. Three Ti reports each have two Sc48 yield-convention source flags; the Cu64 low-percent control has none.

This is a source-only replay, with `raw_parity_evaluated=False`. It is not a raw-spectrum refit, inversion or measured uncertainty qualification. Raw RAD INT values, both percent/fraction hypotheses, ROI/summary unit distinctions and exact source lines are preserved. There is no activity, summary or rate correction. Original GammaLib/settings, calibration/history and full uncertainty remain unqualified; scientific admission is false.

See the source-bound independent review and test logs in this directory. Output hashes in `artifact_sha256.json` are computed after copying all evidence and exclude that manifest itself. Local original-input paths are provenance references and are not portable download links.