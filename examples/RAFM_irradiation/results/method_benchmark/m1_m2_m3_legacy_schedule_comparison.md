# M1 vs M2 vs M3 vs Legacy Schedule Comparison

Comparison scope: RAFM analysis artifacts in `results/analysis_json/*.json`.

- Candidate schedules compared: 29
- Legacy objective source: `/groupspace/cnerg/users/smandych/projects/ALARA/rafm_irradiation_ldrd_copy/scripts/schedule_optimizer.py` (z_score aggregation)
- Spearman rank correlation (M1 DI-FOM vs Legacy): 0.9419
- Spearman rank correlation (M2 FIM-D vs Legacy): 0.5562
- Spearman rank correlation (M3 MWDCS vs Legacy): 0.9335
- Top-5 overlap (M1 vs Legacy): 5
- Top-5 overlap (M2 vs Legacy): 2
- Top-5 overlap (M3 vs Legacy): 5

## Top Legacy schedules
- RAFM1_Long_72h_EOI
- RAFM1_Long_144h_EOI
- RAFM4-A_15dEOI
- RAFM4-B_15dEOI
- RAFM4-C_15dEOI

## Top DI-FOM schedules
- RAFM1_Long_72h_EOI
- RAFM1_Long_144h_EOI
- RAFM4-A_15dEOI
- RAFM4-B_15dEOI
- RAFM4-C_15dEOI

## Top FIM-D schedules
- RAFM3-B_2hrEOI
- RAFM3-C_300sEOI
- RAFM1
- RAFM1_Long_144h_EOI
- RAFM1_Long_72h_EOI

## Top MWDCS schedules
- RAFM1_Long_72h_EOI
- RAFM1_Long_144h_EOI
- RAFM4-A_15dEOI
- RAFM4-B_15dEOI
- RAFM4-C_15dEOI

CSV output: `/groupspace/cnerg/users/smandych/projects/ALARA/FluxForge/examples/RAFM_irradiation/results/method_benchmark/m1_m2_m3_legacy_schedule_comparison.csv`