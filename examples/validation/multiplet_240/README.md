# Overlap qualification tranche for issue #240

This opt-in module compares a constrained doublet with a single Gaussian on
the same original ROI, noise model and continuum. It reuses `peakfit.gaussian`
and the existing erfc step response. It changes no shared workflow, GUI,
activity, efficiency, background or unfolding code. It is a diagnostic layer
for integration under #232/#220, not a claim of end-to-end scientific parity.

Run from the repository root using existing Python:

```powershell
$env:PYTHONPATH='src'
python -m pytest -q tests/test_multiplet_validation.py tests/test_joint_peak_covariance.py tests/test_signed_peak_fit_covariance.py tests/test_targeted_peak_regressions.py
python examples/validation/multiplet_240/run_example.py --output FRESH_RECEIPT.json
```

`final_receipt.json` records exact engine-module/test/example hashes, runtime,
source counts and hash, ROI, model choices, fitted parameters, full covariance,
component covariance, residuals and admission. Output paths must be new. The
test run used installed Python 3.12.10, NumPy 2.5.1, SciPy 1.18.0, pytest 8.4.2;
NumPy is outside the project's declared `<2` range. No environment was changed.
This evidence does not qualify a different dependency environment or the GUI.

## Scientific scope and constraints

- Use unit-spaced channel coordinates. Areas are full Gaussian integrals in
  counts. Sigma lower bounds below one channel are unsupported pending a
  bin-integrated response. The ROI must cover at least three of the broadest
  permitted sigmas plus calibration shift on both outer sides.
- Supply externally sourced common resolution bounds and fixed candidate
  separation; the doublet fits two areas, a shared calibration shift and a
  shared sigma. This is appropriate for a narrow ROI with independently known
  line separation. It does not qualify a blind free-centroid line search.
- The single control may move across the whole candidate interval. An additional
  single control has a fixed upper width of three times the supplied upper
  bound. This is a response-mismatch diagnostic; an adequate broader single
  with comparable BIC makes component admission ambiguous. It does not validate
  the broader width as detector resolution.
- Choose constant or linear nonnegative continuum explicitly. A descending
  erfc step plus linear continuum is allowed only with independently recorded
  response evidence; step area is excluded from Gaussian component areas.
  Existing tail helpers do not yet have qualified joint area/identifiability
  semantics in this layer, so tail requests return `unsupported`.
- Absolute noise covariance propagates to the full fitted covariance. The area
  parameters are Gaussian counts directly. Component covariance is its area
  block; total variance is the sum of every entry in that block, including
  cross terms. Errors are local and conditional on candidate separation,
  response family, ROI and resolution constraints; they omit external
  calibration, yield, efficiency and nuclide-assignment uncertainty.
- The measured `raw_sample` basis requires unscaled nonnegative integer
  observations. Optional covariance must contain at least independent
  observed-count Poisson noise. The default is `sqrt(max(count,1))` and uses a
  high-count WLS approximation. Expected fitted bins below five counts are
  rejected. Explicit `synthetic` data require declared noise and cannot be
  used as a physical-analysis count basis. Signed `background_adjusted` counts
  require declared uncertainty or full covariance and remain distinct from
  raw sample and historical comparison counts.
- The fixed screening policy requires at least eight residual degrees of
  freedom, separation of 0.5 FWHM, absolute area correlation below 0.95,
  covariance correlation condition below 1e8, per-component area SNR of three,
  BIC improvement of ten, and goodness p at least 0.001. These are conservative
  diagnostic choices, not a standards-compliance claim. User-supplied policy
  choices are persisted. Residual vectors, sign runs and lag-one structure are
  exposed. Active fit bounds, invalid covariance, failed controls and failed
  optimizers withhold admission.
- `component_admission` means conditional support for component area estimates
  in this response/count basis. It requires independent support strings for
  both candidate lines. Those strings are caller assertions, not evidence
  verified by this function. Integration must retain the provenance and apply
  existing independent nuclide/yield/efficiency and physical-activity gates.
  Lower residuals alone never establish a nuclide or activity.

## Bounded real-source observation

The fixture is a byte-identical additive copy of historical commit `46096eb`:

`examples/RAFM_irradiation/quantumgold_reference/originals/ASC/RAFM-N-300s.ASC`

SHA-256: `4f2aadb52bcab511b6e8da042ebe689a93f4f6007a481b3f6bcac5c0238188d3`.
No historical core was imported. The original remains unchanged. The
source-bound `FINDINGS.txt` hash and source attribution are in the receipt;
neither the copyrighted manual nor private chat links are included.

One fixed morphology screen of this one spectrum, 100–1800 keV, found seven
significant maxima and no close pair within 0.5–2 profile FWHM. Screening uses
one-channel Gaussian smoothing and prominence of eight times the square root
of the smoothed count. Fits use original unsmoothed counts. This is not a
comprehensive search for shoulders or proof that all 32 study files lack
overlaps. With no observed pair, the bounded real fit is explicitly a
single-peak/unsupported-extra-component control around the strongest maximum,
channels 1682–1712 (about 838.91–853.88 keV), not an observed doublet example.
Its doublet is rejected at a calibration-shift bound with excess residuals;
no components are admitted. The synthetic separated/partial cases qualify,
the unresolved case is non-identifiable, and the single control rejects extras.

The ASC header calibration `[0.541, 0.498, 2.605e-7]` is preserved. The existing
`rafm_25cm` resolution profile is used only for this declared conditional
diagnostic with a fixed ±20% width band; its saved QuantumGold energy
calibration differs. Applicability to this acquisition is unknown. Saved
settings do not establish final report settings or independent calibration.
No per-sample fit was tuned to QuantumGold counts, activities or target values.
No qualified real overlap is claimed from this bounded evidence.

## Integration handoff

The #232/#220 owner can call `qualify_doublet` after obtaining calibrated
candidate separation/resolution and independent line/response evidence. Retain
the entire `to_dict()` receipt and count basis; withhold physical admission on
any non-qualified result or unresolved source applicability. Preserve existing
isolated Co-60 and single-line activity comparison methods. This tranche
modifies only the new module, new tests and this focused example; shared
integration, real calibration qualification and tail qualification remain open.

Independent Sol review found a broad-single false-admission counterexample and
raw-count noise semantics gap. Both now have regression tests and fixes. Review
receipts distinguish the initial findings from the final recheck. Existing joint
fit, signed covariance and targeted single/multiplet regression tests are also
included in the recorded bounded test run.
