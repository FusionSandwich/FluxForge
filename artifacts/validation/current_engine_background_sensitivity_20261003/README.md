# RAFM measured-background sensitivity on the current validation engine

This is a **conditional analysis receipt**, separate from the historical portable-engine numbers in this PR. It exercises the current `4615e61` validation lineage at the clean figure-lifecycle commit `a7bcc680d1f5e06b1d9dae405241fc380087ca2b`. The figure change is not numerical. All three runs used the same 69 staged input files (length-prefixed path-and-content tree SHA-256 `45803d05cfe2c49aa541c9bfc2079d441feb9c7978e81a6a68a732fedc0fdfe1`), `iec_tiered` flux-wire and generic counting, the current engine's integrated-bin-overlap rebinning and covariance, and the existing legacy 25 cm efficiency profile. QG report activities were not substituted for physical estimates. The source-bound runners and scenario receipts are beside this file. The local scripts contain the execution host's absolute paths; adapt those paths to replay them.

## Scope and main result

The workflow analyzed 30 original ASC spectra and paired 29 with QG reports. One ASC spectrum lacks a QG report; two QG reports lack an ASC raw counterpart. The native-only Cu-Cd and near-contact Fe-Cd cases therefore remain outside this 30-ASC campaign (issue #231). The matched activity statistic uses the same **81 finite, matched isotope rows** in all scenarios and compares measurement-time activity to the printed QG report.

| Background treatment | Median absolute relative activity difference from QG | Samples failing validation | Co-Cd Co-60 activity |
| --- | ---: | ---: | ---: |
| Historical North measured `background.ASC` | 41.71% | 26 | 156.23 Bq |
| Native South measured background substituted for the workflow's single background input | 38.45% | 26 | 134.33 Bq |
| Synthetic zero-valued ambient input, a protocol control | 38.28% | 25 | 246.19 Bq |
| QG report, Co-Cd only | — | — | 234.95 Bq |

South improves 41 of the 81 activity errors relative to North and worsens 40. The source schedule identifies Cu-RAFM-1_25cm as a flux wire, although the workflow CSV labels it `RAFM1` (#234); the comparison groups it with the monitors. For the resulting 17 flux-wire isotope rows, the median absolute QG difference rises from 33.50% with North to 42.83% with South; the zero-ambient control gives 23.73%. The South substitution therefore does **not** establish a universal background to subtract. For Co-Cd, South moves activity further from QG than North. The synthetic zero control is closer for Co-Cd but is not a measured-background recommendation or proof of QG's final report setting.
Co-Cd is the one sample that passes the campaign threshold in the zero control but fails with either measured background; 25 other samples still fail.

The Co-Cd North/South physical peak net counts are 6989.5/6974.5 and 5370.09/6472.79, respectively, at the two Co-60 lines. The separate `comparison_net_counts` fields are 10522/11020 and 10522/10839. Those comparison fields must not be substituted into the physical activity formula or used to infer that physical peak counts agree with QG. At the same current-engine assumptions, the Co-Cd reaction-rate estimate moves from `1.40078e-13` to `1.20437e-13` per target atom per second (South/North 0.860); the zero control gives `2.20739e-13`. All are conditional on the efficiency, timing/history, and background choice.

## Background identity and QG comparison

North `background.ASC` has SHA-256 `505565653785c1e704f175e32e09ae1d69352fd5c891ff413f5acda29f633374`; native South ANS has SHA-256 `96f2e47eb2edc68db227157aa08c601be6cd0ec4e46f1abfa114d46e2d509344`. South is an 8192-channel, 14,400 s live-time acquisition on 2025-10-03. Its acquisition follows at least some RAFM sample measurements, and QG reports do not name a measured background file. Applicability to every sample is unresolved (#229).

The [Quantum 4.04 manual](https://ludlums.com/images/product_manuals/QTMmanual.pdf), Appendix C, defines `AnalysisCtrl` bit A as **no ambient correction**. The pinned [saved-header diagnostic in PR #221](https://github.com/FusionSandwich/FluxForge/blob/46096eb/artifacts/validation/quantumgold_documentation_20261003/FINDINGS.txt) finds value `1` at the empirically verified offset in all 32 original ANS files (diagnostic SHA-256 `c63f9f6e095da6dcc0a86d4e4d8a36e150a45299122ac2304258d82a29942c4d`). This supports ambient-off in the saved spectra only. The installed 2025 QG version and final report analysis state are unverified; hence the zero-ambient run is a counterfactual protocol control. The manual's printed offsets differ from the empirically validated study-specific layout.

## Qualification

No scenario passes the campaign's validation threshold. All three have 26 QG internal-consistency flags. North/South/zero have 225/159/224 FluxForge line-consistency flags, but the present count diagnostics mix comparison and physical count bases (see #26 and #32), so their bucket totals are not physical-count parity evidence. The GLS, GRAVEL, and MLEM outputs are `diagnostic_only=true`, `converged=false` with no admitted physical inversion; no neutron-flux spectrum is qualified from these scenarios. Native-only spectra, low-energy response, and efficiency/source applicability remain blockers (#231, #229, #232). The executing local NumPy was 2.5.1, outside the project's declared `<2.0` range; this is a recorded limitation, not a dependency change.

`north_south_comparison.json` contains all paired North/South activity rows, physical/comparison Co-Cd count bases, and status changes. `three_scenario_comparison.json` contains all 81-row comparisons plus the 21 reaction-rate output rows and group summaries. Four reaction-rate rows whose source CSV says end-of-irradiation activity is unavailable are recorded as null with their exclusion reasons, not as the CSV's numeric zero placeholders (#235); the Ni-57 row remains provisional/excluded from unfolding. The receipts preserve scenario-specific source hashes, engine/input identities and run parameters. The adjacent Python runners and comparators show the exact host-bound execution and assertions.
