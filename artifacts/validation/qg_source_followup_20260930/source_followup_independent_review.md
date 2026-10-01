# Independent source-followup review — 2026-09-30

Reviewer: GPT-6.1 Sol. Exact reviewed local commit `6fbd878c6007f59b80d2ec578571f54f1868b562`, worktree `C:\Users\joshu\fluxforge-worktrees\qg-report-yield-qc`, published base `0b3a02ee3829e3d196e17ecc93ec4d32ac31a4fe`.

Decision: no actionable gap and no additional runtime defect demonstrated by this new source evidence. Report-only software gates remain preserved. The appendix and new public artifacts are privacy-safe within the inspected diff. Scientific admission remains false; root retains final acceptance/publication.

Independently verified:
- Production, existing tests, tools and example configuration have no diff from published base. New arithmetic JSON binds unchanged runtime/test source hashes; its inspected_commit is the generating base revision, while this review binds the additive followup commit.
- Original three Ti report bytes match their receipt hashes. Independently recomputed N/u(N)-weighted means using math.fsum, without calling the supplied weighted helper. All-four/three-stronger results are Ti-1: 0.41016720724378275 / 0.42412383001462656; Ti-1a: 0.40789678333994495 / 0.44166820618787567; Ti-1b: 0.2342877053219885 / 0.24563860560344766 uCi. These agree with the public arithmetic receipt to floating precision.
- Independently computed start/average factors q/(-expm1(-q)), q=ln(2)*RT/T_half, with the printed half-life 157320 s. Factors are 1.0320851574099623, 1.0320852923864068 and 1.4288928812039008, matching the receipt. Uniform live fraction and unverified vendor count-decay processing remain explicit conditions.
- On every actual Ti report, missing qg_report_activity_includes_count_decay raises; True gives zero count duration; False gives measured real duration. Existing four-test receipt is 4 passed in 14.89s and its source hash is current. Reviewed tests cover real-versus-live timing, no double correction and processed activity/rate declaration behavior. No broad tests were rerun.
- Browsed the public [Quantum 4.04.00 manual](https://ludlums.com/images/product_manuals/QTMmanual.pdf): PDF page 46 specifies intensity per 100 decays; pages 118–119 describe percent efficiency and activity/uncertainty weighting; page 126 provides an activity-reference-date field. Applicability to the deployed version and active library remains unknown.
- Browsed [IAEA historical tabulation](https://nds.iaea.org/sgnucdat/safeg2008.pdf), Table D-2, PDF page 116: the table uses per-100-decay emission probabilities, including near-unit Sc48 lines. This is historical corroboration, not independent detector calibration or current covariance.
- Inspected all newly published appendix/artifact content and searched it for EML identifiers, email/chat URLs and private provenance fields. No private correspondence hash or chat link is present. The private source-owner evidence was not copied or published by reviewer.

No vendor subset/inclusion rule, summary uncertainty, active calibration/history, selective factor-100 mechanism, measured spectrum or scientific validation is inferred. Different matching subsets remain hypotheses; the common timing factor cannot alone explain selective line discrepancies. No source summary/rate correction or configuration change is supported by this evidence.

Reviewed artifact SHA-256:
- Appendix document: `638f81bbfcdd8ef3bace253c71ebe448150b97f98c1d35383d9417be89dac982`
- source_followup_arithmetic.json: `28ace9ed1d788a751f7ade333087a5c2dfdc4fd8f71e4641192bf9c07bb80f46`
- source_followup_arithmetic.py: `14c005174e8dba9c845d0084c01192f7cf6a03bc4b456d96fd46e652c6ffc395`
- source_followup_count_decay_tests.txt: `3a5a53fe80daecb918c7be6af95227b76f9c8ea75e5e7518dece3a13a5c5bc08`