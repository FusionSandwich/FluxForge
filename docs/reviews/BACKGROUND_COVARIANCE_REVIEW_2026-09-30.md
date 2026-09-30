# Measured-background covariance review

## Change and evidence

This correction addresses the missing counting covariance in #9, #14, #15,
#16 and part of #25. Interpolation retains `W C_background W.T`; subtraction
retains `C_sample + scale^2 W C_background W.T`. ROI sums use the complete
covariance. Local continuum subtraction uses one signed weight vector for ROI
and sidebands, including their cross terms. Sparse covariance survives JSON
round trips, no-background copies, independent-spectrum arithmetic and smoothing.

Exact counterexamples distinguish this implementation from diagonal-only
propagation: a two-channel interpolation has total variance 86 rather than 68;
a shared continuum mode cancels under sideband subtraction, giving variance 30
rather than 97.5. Shape, symmetry, uncertainty and mismatched-count errors fail.
Independent Sol review reproduced repeated interpolation against a dense oracle,
found a CSR copy-alias bug, and verified its correction and regression.

Verification command, using existing local Python and the Agg plotting backend:

```powershell
$env:MPLBACKEND = 'Agg'
$env:PYTHONUTF8 = '1'
python -m pytest tests/test_flux_wire_analysis.py tests/test_flux_wire_parity.py tests/test_spectrum_io_parity.py tests/test_rafm_workflow.py tests/test_no_silent_defaults.py tests/test_irradiation_history.py tests/test_background_covariance.py tests/test_spectrum_math.py tests/test_hpge_background_integration.py tests/test_rafm_background_integration.py -q
```

Result: **108 passed**. The first broader attempt had 102 passes and one local
Tk plotting failure; Agg resolved it without installing or changing dependencies.

## UWNR replay

`tools/validate_rafm_background_covariance.py --output-root <unused-directory>`
replays the committed Co and Co-Cd ASC spectra. A separate oracle interpolates
individual source basis vectors with NumPy, then adds their independent count
variances. It checks both HPGe windows and signed local ROI sums for qg, covell,
gilmore and iec_tiered. The raw Covell workflow runs without a QG report.

| Sample | Live-time scale | Signed negative bins | 1173 keV HPGe window diagonal SD | Covariance/oracle SD | Raw Covell rate per target atom per second |
| --- | ---: | ---: | ---: | ---: | ---: |
| Co-RAFM-1 | 3 | 707 | 218.060 | 226.929 | 1.7767657216980463e-12 |
| Co-Cd-RAFM-1 | 9 | 676 | 278.384 | 336.184 | 2.22270667018277e-13 |

These rates are example outputs, not a complete physical qualification. The
0.46 wt% Co composition remains explicit. Listed masses remain adjusted Co
element masses (4.0661 and 3.6703 mg), giving 4.154979967868026e19 and
3.750528264446526e19 target atoms; there is no second alloy-factor multiplication.

The final local receipt is
`C:\Users\joshu\Documents\UWNR_work\composition_review\covariance_review_2026-09-30_final\covariance_review_receipt.json`.
It records input/source hashes, independent variances, per-line outputs and
actual raw workflow artifacts. Generated spectrum plots were inspected.

## Still open

The count covariance and mass-basis checks pass. HPGe replay uses default unity
efficiency, so its activity outputs are diagnostic. Output inspection also found
an unstable existing Co-Cd 1332 keV fit: it reports about 85,932 peak counts from
a window with only 13,144 signed counts, while fitting a large negative continuum
and a centroid on the window boundary. Its reduced chi-squared is about 346.
This result is not an accurate peak measurement. Investigate its use of Poisson
weights on clipped background-subtracted counts before accepting HPGe results.

The covariance-aware ROI floor is not a GLS fit. Nonlinear SNIP/model uncertainty,
efficiency/emission budgets and covariance between samples sharing a background
remain incomplete. Symmetric indefinite covariance is detected if a requested
weighted sum produces negative variance; construction does not run a global PSD
test. QG reference substitutions remain separate open work in #24/#26.

No issue is closed on the strength of this draft-branch correction alone.
