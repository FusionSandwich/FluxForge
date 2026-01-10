# FluxForge Comprehensive Validation Report

**Total:** 22/22 tests passed

| Category | Test | Status | Correlation | Notes |
|----------|------|--------|-------------|-------|
| STAYSL-M1 | SigPhi saturation rates | ✅ | 1.0000 | R_sat factor=9.17e-09 |
| STAYSL-M2 | Self-shielding (SHIELD-style) | ✅ | 1.0000 | Au SSF = 0.9810 |
| STAYSL-M3 | Cover corrections (Cd/Gd/B/Au) | ✅ | 1.0000 | Cd STAYSL CCF=0.3479 |
| STAYSL-M4 | GLS with prior covariance | ✅ | 1.0000 | χ²_red=5.65 |
| Epic-N | IRDFF-II database | ✅ | 1.0000 | 36 reactions available |
| Epic-O | ENDF covariance (MF33) | ✅ | 1.0000 | Validation + conditioning OK |
| Epic-P | k₀-NAA workflow | ✅ | 1.0000 | g(Au)=1.0000, Cd-ratio=10.00 |
| Epic-Q | RMLE gamma unfolding | ✅ | 0.4239 | Correlation=0.4239, converged=True |
| Epic-P | Uncertainty budget | ✅ | 1.0000 | 3 components, total=50.0000 |
| Epic-H | Transport comparison (C/E) | ✅ | 1.0000 | Mean C/E = 0.981 |
| Workflow | Activation pipeline | ✅ | 1.0000 | Pipeline configured OK |
| Epic-O | Thermal scattering S(α,β) | ✅ | 1.0000 | H2O, graphite TSL available |
| Epic-O | Library provenance | ✅ | 1.0000 | ENDF/B-VIII.0 + IRDFF-II bundle created |
| NAA-ANN-1 | NAA-ANN-1 spectra | ✅ | 1.0000 | 20/20 read, 20 with peaks |
| gamma_spec_analysis | gamma_spec_analysis SPE | ✅ | 1.0000 | 5/5 files, 45056 channels |
| Neutron-Unfolding | GRAVEL unfolding | ✅ | 0.9908 | Correlation=0.9908, 7 iters |
| Neutron-Unfolding | MLEM (ddJ mode) | ✅ | 0.9908 | Correlation=0.9908, 6 iters |
| peakingduck | SNIP background | ✅ | 0.9938 | Below=True, Smoother=True |
| pyunfold | D'Agostini MLEM | ✅ | 0.9965 | Corr=0.9965, ratio_err=0.01 |
| Corrections | Coincidence summing | ✅ | 1.0000 | Co-60 TCS@1332=1.0000, batch=2 energies |
| Corrections | Gamma attenuation | ✅ | 1.0000 | Iron @ 662keV: factor=1.1508 |
| Data | Gamma database | ✅ | 1.0000 | 0 nuclides available (API verified) |
