# Follow-up: South detector background and QG parity

These are source-bound diagnostics on current validation engine `a7bcc680` and the same staged 69-file input tree as the complete campaign runs. `background_forensics.py` checks source identities, broad energy-band rates, and fixed-window Co-Cd controls. `south_scale_probe.py` records a one-spectrum synthetic amplitude stress test. Neither selects a physical scale.

## Source match

- All **32 original ANS** files and all **31 processed QG reports** identify detector `South`. The supplemental native background also identifies `South`. The bundled ASC background calls itself `North 4hr background terminal`; its title does not establish that it represents the South detector.
- Both background live times are 14,400 s. North records **12.54 counts/s**, South **37.74 counts/s**, and the Co-Cd raw sample **41.56 counts/s**. South/North rates span 2.38–3.47 across six broad energy bands. Co-Cd raw/South rates span 1.06–1.14. South is the source-matched measured candidate; North is a sensitivity control, regardless of QG closeness.
- The 30 ASC measurements are dated **2025-07-31 to 2025-08-28**. South background was acquired **2025-10-03** and North **2026-03-02**. South is closer in time and detector identity, but its temporal stability over the gap is unverified. An earlier South background or detector log would be valuable.
- The workflow applies the `rafm_25cm` energy calibration `[-1.694, 0.4996, 6.71e-08]` to ASC samples and North ASC. Native South has a close but distinct polynomial and is rebinned by integrated energy-bin overlap with covariance.

## Different mechanisms

**Co-Cd count extraction.** For Co-60 at 1173/1332 keV, QG prints **11,202/11,439** net counts. FluxForge's zero-ambient IEC control gives **11,244/10,839**. A diagnostic fixed ±5 keV core with two 10–25 keV sidebands gives North net counts about **6858/6016** and South **7477/6996**. The campaign IEC/local-sideband pathway instead gives North **6989/6974** and South **5370/6473**. Thus QG's Co-Cd line counts are much closer to FluxForge's ambient-off control, consistent with the saved ANS flag, while the direction of the measured-background effect depends on estimator/window. QG's final report setting remains unverified. The fixed window is not a qualified peak model. The current physical IEC path uses a fit-dependent ROI and one immediately adjacent channel per side; a joint original-sample/background count model should test its stability. South contains appreciable Co-60-region counts, so some reduction in sample peak area is expected.

This is selective: the stronger bare Co monitor gives **2093.30 Bq** with North and **2093.61 Bq** with South, whereas weak Co-Cd changes from **156.23** to **134.33 Bq**. This points to sample-specific count/background sensitivity at the same nominal profile and yield.

**QG activity conversion.** In RAFM3-A at 300 s EOI, QG reports V-52 activity **95.94 MBq** and 98,446 net line counts. A distinct FluxForge detected-line path gives 100,350 counts and 1.009 MBq; the campaign targeted IEC isotope result is **0.947 MBq**. That IEC activity changes by only about 18 Bq among North, South, and zero-ambient runs. QG prints `RAD INT 1.00` with unspecified unit while the pinned local library uses intensity fraction `1.0`. Normalizing activity by detected-line net counts, QG/FluxForge is **96.9×** for V-52 and **105.2×** for an independent Al-28 line in the same report. Interpreting QG's `1.00` as **1%** and the local `1.0` as **100%** predicts approximately 100×; substituting 1% into FluxForge's conversion with QG's counts gives 98.98 MBq V-52 and 0.236 MBq Al-28 versus QG's 95.94 and 0.248 MBq. This is the leading testable hypothesis, not a confirmed QG library interpretation. The exact QG library, efficiency, and timing settings are unverified. The detected-line count is not the targeted IEC count basis.

The synthetic Co-Cd South-amplitude probe reproduces the complete-run endpoints: factor 0 gives **246.19 Bq** and factor 1 gives **134.33 Bq**. The first-line gross ROI changes between factors 0.25 and 0.5, showing estimator/window sensitivity. Factors are stress parameters, **not** inferred physical corrections. Fitting one to QG would mix QG's possible ambient-off convention with background estimation.

## Working approach

1. Use same-detector South as the physical measured-background candidate across the original South spectra. Keep North solely as a mismatched-source sensitivity check.
2. Keep ambient-off QG reproduction separate from physical South subtraction. The saved ANS flags support ambient-off in saved files, but final report settings remain unknown. Audit count parity before activity conversion.
3. Compare fixed ROI choices and a joint Poisson sample/background model on original Co-Cd counts, with explicit continuum, exposure and covariance checks. Inspect residuals and model dependence before production changes. Issue #239 tracks the joint model.
4. Verify QG's RAD INT unit and underlying isotope library first, then bind detector efficiency and timing per isotope. Do not change local nuclear data to force parity until the QG convention is confirmed.
5. Obtain earlier South background or stability records if available. Otherwise quantify temporal background uncertainty and keep physical activation and flux conclusions conditional.

The complete 30-ASC campaign remains numerically available, but no physical neutron-flux inversion is yet qualified. Issues #229, #231, #232, #234, #235, and #239 track remaining source, integration, and modeling work.
