# Issue #249 — source-bound South Co-Cd joint-Poisson pilot

This directory records the bounded method pilot requested by issue #249. It is
based on integration head `701288a30f418ed7dc331e6deda0044eb220e72e` and does
not modify the primary checkout, shared workflow drivers, source manifests,
original spectra, or production defaults.

## Source binding

- Co-Cd sample: `examples/RAFM_irradiation/quantumgold_reference/originals/ASC/Co-Cd-RAFM-1.ASC`
  - SHA-256 `7482016df6c9d370b68e5d5646cd64c5fa74def9fe586e58d19d89dab0c87cba`
  - original integer counts; 129600 s live time
- South background: `examples/RAFM_irradiation/quantumgold_reference/supplemental_inputs/South 4hr Background Terminal.ANS`
  - SHA-256 `96f2e47eb2edc68db227157aa08c601be6cd0ec4e46f1abfa114d46e2d509344`
  - 8192 native UInt32 counts at byte 1548; 14400 s live / 14409.5 s real
  - native energy polynomial `[-1.6994324923, 0.4995956719, 9.0240725e-8]`
  - temporal applicability to the August 2025 sample remains **UNRESOLVED**
- Calibration/efficiency shared covariance remains **UNAVAILABLE** and is never
  silently set to zero.

The joint likelihood consumes only original native nonnegative integer sample and
South observations on their own native energy grids. The same fixed sample ROIs
are compared to the existing count-conserving South subtraction plus local
IEC/Covell component, retaining full rebin covariance.

## Main findings

Both Co-60 lines converge numerically under fixed-response South-background
models, but the fits remain physically unqualified because the sample residuals
show strong structure. For the linear sample continuum, the fitted full-response
areas are about 7367 counts at 1173 keV and 6873 counts at 1332 keV. The same
fixed-ROI IEC/Covell controls are about 7407 and 7513 net counts respectively.

Ambient-off increases the fitted areas to roughly 11058 and 11084 counts. That
sensitivity is retained as a control only. It is **not** selected because it
moves toward any QuantumGold-reported quantity; vendor activity targets are not
loaded by the pilot.

The largest absolute sample Poisson deviance residuals are approximately 5.7 at
1173 keV and 7.7 at 1332 keV. A step-continuum alternative reduces deviance but
does not remove the strong-lack-of-fit classification and reaches a nuisance
boundary. Free South normalization remains rank-deficient/unidentifiable. The
bounded evidence therefore does not justify response tuning or a physical
activity qualification.

The integrated current-engine IEC control remains a separate external method
control: Co-Cd Co-60 lines are approximately 116.73 Bq and 147.68 Bq, combining
to 134.325 Bq conditionally. This pilot does not reinterpret that number as a
truth target.

## Qualification boundary

Three states are intentionally separate:

1. **Software execution:** the opt-in pilot and focused regression contract are
   implemented.
2. **Vendor agreement:** not an optimization objective and not an acceptance
   criterion.
3. **Physical qualification:** **NOT QUALIFIED** because model adequacy,
   calibration covariance, and South temporal applicability remain unresolved.

`SCIENTIFIC_RECEIPT.json`, `summary.csv`, and `residuals.svg` preserve the
bounded numerical/model-adequacy evidence. `TEST_RECEIPT.json` is added only
after the supported-runtime CI result is known. Independent Sol review remains a
separate acceptance gate; this chat does not claim independence from itself.
