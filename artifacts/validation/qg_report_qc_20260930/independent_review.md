# Independent QG report-QC review — 2026-09-30

Reviewer: GPT-6.1 Sol. Initial frozen implementation `22912d767ba14853bf5d7cf461860346b330abb3`, parent PR212 `da0f0680a4d652c3428a9f592057f82940a1f95d`, worktree `C:\Users\joshu\fluxforge-worktrees\qg-report-yield-qc`. Reviewed six-file diff, review document, original-report validator and decisive evidence. No source edits, acquisitions, remote execution, source correction or publication by reviewer. Root retains final acceptance.

## Finding

[P2, pending] Missing/unknown ROI activity units still produce a fabricated activity comparison score when a raw match exists. Importer correctly records unknown unit and `reference_line_activity_bq=None`. `build_line_diagnostic_records` then coerces this missing reference into local zero and returns `line_activity_en_score = raw_activity/raw_uncertainty`. This contradicts unqualified reference units and produces a large numerical comparison in report/CSV.

Reproduced at exact frozen implementation: parse the existing Cu control fixture, construct a raw match with equal original counts and original line activity, raw uncertainty 1 Bq, and change only the in-memory ROI `activity_unit` to `widgets`. Returned fields:

```text
reference_line_activity_bq: None
relative_line_activity_error: None
line_activity_en_score: 3172380.0
diagnostic_bucket: matched
source_qc_bucket: line_activity_unit_unqualified
```

Recommendation: gate activity comparison scores on a qualified reference activity and add a matched missing/unknown-unit regression. Preserve the separate source-QC and count-parity buckets. Original actual-report replay has known ROI units and is unaffected. This is a reporting-boundary finding, not a solver or measured-source defect.

## Decisive passing evidence

Independent focused execution: `tests/test_qg_report_qc.py`, 18 passed in 4.18s. Existing Python `D:\FluxForgeQA\envs\fluxforge-py312\Scripts\python.exe`, checkout src PYTHONPATH, offline, UTF-8, no bytecode/pytest cache writes, Agg backend. Root reports final affected suite 113 passed; reviewer did not duplicate it.

Actual receipt `D:\FluxForgeQA\receipts\qg_report_qc_20260930\actual_reports\receipt.json` independently hashed to `27a84e60e932b7fec5ad1272a9263217351a20264e360398ea72b83b7ad3cf86`. Baseline independently hashed to `5757f22b5dba3dbc7a30650c9356fa974d87ebf2dd7687022e3eace634c61fef`. All 42 input hashes match current bytes (32 original PR212 inputs plus root evidence, baseline and four original reports). Six source/data/test/script hashes match reviewed files.

The three original Downloads Ti reports each retain two Sc48 yield-convention flags (six total). The Cu64 0.47 control remains consistent with the percent hypothesis and unflagged. All original summary/line values and legacy comparison fields remain baseline-equal under the validator's explicit projections; existing line-summary inconsistency rows remain equal. Reviewed validator verifies source-byte stability before and after computation and preserves these projections.

Independently verified receipt line and summary source-file hashes, exact original decoded source lines and source-line numbers for all four reports. Aggregate CSV source buckets and line references match receipt rows. Every row in this source-only replay records `raw_parity_evaluated=False`; empty raw-match lists do not establish missing actual raw lines. Actual receipt retains scientific_admission=False, report_only=True and rate_or_summary_correction=False.

Tests and source inspection show source flags survive absent raw matches, do not replace raw count/activity parity buckets, do not mutate imported or raw runtime values, and remain coherent in CSV/per-spectrum reports. Summary bucket plumbing uses the same source bucket field; it is reporting-only and does not feed acceptance, activities, rates or numerical solver choices. Percent/fraction hypotheses remain separate; no generic <=1 yield repair or whole-summary/rate correction is introduced. Distant/absent references stay unqualified. Independent synthetic equidistant reference-line probe returns ambiguous_reference_line. Missing/unknown ROI units remain source-unqualified, subject to the activity-score finding above.

Bundled reference provenance is explicitly decay_2012 via actigamma, source JSON SHA `159e6dccd10532323f986dbe45d95b7d804b53c965d254863fbe63cae565c355`. Separately browsed the official NNDC ENSDF Sc48 beta-decay PDF, November 2021 evaluation by Jun Chen, pages 2–3: the two gamma relative intensities are 1000, normalization .100 produces 100 photons per 100 parent decays, approximately one photon per decay. This supports the independent reference statement, not a vendor GammaLib.mdb processing mechanism. Official reference: https://www.nndc.bnl.gov/ensnds/48/Ti/beta_decay.pdf.

## Scope and decision

No measured calibration, original gamma library/settings, irradiation history, source uncertainty, unfolded spectrum or publication admission is qualified. Implied efficiencies are conditional algebra; line-yield hypotheses cannot justify dividing a whole nuclide summary or rate by 100. Existing PR212 source/covariance/publication gates remain in force. Acceptance is pending the one reporting-boundary repair and focused recheck.
## Resolution and final bounded software acceptance

Final independently reviewed source commit: `b718b780f266fb36786899a5ffb8b5548a5dfef3`. The earlier pending finding is resolved. Qualified-reference gating now leaves activity delta, combined uncertainty and activity En score unavailable when the ROI unit is missing/unknown. Separate source-QC/count-parity behavior and raw values are preserved. Independently inspected the narrow patch and ran both matched missing/unknown-unit regressions.

The owner also repaired a provenance edge case identified in their independent check: bundled reference values are read from the same file bytes whose SHA is recorded, rather than potentially mutated cached decay-library entries. Independently inspected that patch and ran its regression. All three new cases passed (`3 passed, 18 deselected in 4.16s`) under the same offline environment. Root's complete repaired QG module run was inspected: 21 passed in 68.23s. These test invocations overlap and are not a unique-test total.

Final reviewed SHA-256:
- qg_report_qc.py: `7ed90522de4ef57f36fe5ae899816a5d04d495b270c7cbbf17c4f12451be4122`
- rafm_workflow.py: `f5478777e0f5d6fa2c0a9c1ea259d642ebb0e0286af0c660dd7a3fbc39f18c52`
- test_qg_report_qc.py: `80b444f39d8fc0dbaa7f5e712e76820bd43ed4e1a2f3a8cc650fbe2bd52e8651`

No actionable finding remains in the bounded software/reporting scope. Acceptance at this exact final source commit is supported. The original-report receipt above binds initial 22912d7; it was independently verified and remains decisive for actual known-unit source compatibility. The two followup failure-path repairs are independently checked; root should preserve/check a final replay receipt bound to b718b78 before publishing. This reviewer has not claimed such a pending replay completed. Scientific admission remains false and all original-library/calibration/history/publication limitations remain in force.
## Final actual replay gate — independently verified

The final replay is now independently verified at exact source commit `b718b780f266fb36786899a5ffb8b5548a5dfef3`. Receipt `D:\FluxForgeQA\receipts\qg_report_qc_20260930\actual_reports_final\receipt.json` SHA-256 is `b390c21ea8ab6114e227eb1a9fbf2cab7bd2b1919a79d8f3194409f5458753c2`. All six source/data/test/script hashes match frozen worktree bytes; all 42 input hashes match current bytes and are exactly unchanged from the initial replay. Old baseline SHA remains `5757f22b5dba3dbc7a30650c9356fa974d87ebf2dd7687022e3eace634c61fef`.

Final report content is exactly equal to the already inspected initial replay, including original source paths/hashes, line/summary provenance, baseline consistency rows and unchanged-value/legacy-comparison assertions. Independently checked direct baseline projections of summary isotope/activity/uncertainty/unit, line count fields, raw RAD INT and legacy branching fractions, and line/header activities. All four reports pass. Independently reverified exact decoded source lines and aggregate CSV bucket/line-reference coherence.

The final replay retains two convention flags per original Ti report (six total), none for Cu64, and every source-only row has raw_parity_evaluated=False. Report-only status, scientific_admission=False and rate_or_summary_correction=False remain explicit. This supersedes the earlier statement that final replay was pending. No broad tests or source changes were performed for this final gate. Final root suite remains a separate root-owned check.

Bounded software and actual-report acceptance is supported at this exact reviewed source commit and replay hash. No actionable finding remains; no scientific/publication admission is made.