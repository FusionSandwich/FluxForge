# Source-bound saved background scenario

The original 32 rev4 ANS files store control value 1 at the observed byte 860:
ambient correction disabled and continuum correction enabled. The
[manufacturer manual, Appendix C](https://ludlums.com/images/product_manuals/QTMmanual.pdf)
defines the control bits. Its printed offsets do not describe these study
files: byte 852 reads 0 and byte 1280 reads 8224 for the library-efficiency flag.
The bounded diagnostic requires the observed channel bounds, exact file length,
ROI structure and available report timing/library anchors. It is not a general
ANS reader. All 32 headers passed; 31 have report anchors.

Saved settings can precede the final analysis. Several files have zero saved
ROIs despite report peaks. The saved ambient-off state does not establish final
QuantumGold processing, a background file or installed software version.

The portable monitor comparison now accepts explicit `--background-mode
ambient_off`. It removes separate measured ambient subtraction while retaining
the configured local continuum calculation. The default historical North
background remains unchanged. The receipt records no background hash, no live
scaling factor, sample-only covariance and unresolved QuantumGold background
matching. Efficiencies remain separately selectable; calibration, counting
and systematic uncertainty are still conditional. No reference activity or
count replaces an analysis estimate.

From the repository root:

```powershell
python tools/audit_qg_saved_settings.py --originals examples/RAFM_irradiation/quantumgold_reference/originals --output saved_settings.json
python examples/RAFM_irradiation/run_portable_qg_example.py --raw-sample Co-Cd-RAFM-1 --efficiency-mode legacy_profile --background-mode ambient_off --output co_cd_ambient_off
```

Choose fresh output paths. Keep this scenario separate from the background-
adjusted campaign and from independent physical acceptance. Agreement with a
processed report does not qualify a detector response or an uncertainty budget.
