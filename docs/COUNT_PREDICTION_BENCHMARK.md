# Reaction-family held-out count prediction

This API connects an integrated group-flux spectrum to **absolute accepted
native-channel counts**, freezes posterior count-mean draws and scores a test
vector after freezing. It does not claim that present RAFM measurements are
physically qualified or that FluxForge outperforms nFoils.

## Forward calculation

`physics.count_prediction.activation_count_operator` reuses the canonical
irradiation buildup and real-clock count-decay factor. It includes target atoms,
piecewise relative production, cooling, gamma yield, geometry-specific efficiency,
line-bin probabilities and one live-time acceptance factor. Response coefficients
multiply group-integrated flux. Target atom counts must include the actual
isotopic abundance and specimen mass. No depletion, parent feeding or burnup is
included: use a qualified decay/inventory calculation when those processes matter.
Add isotope operators and ambient/sample-continuum means on their original grids.
Gaussian line profiles are integrated over native edges, retaining out-of-window
tails; more appropriate detector responses can be supplied directly.

`expected_count_draws` accepts fixed or draw-aligned operators and continuum.
Correlated efficiency, nuclear data, irradiation and response draws must be
**joint posterior** draws conditioned on training data. Sampling these nuisances
independently from their prior after fitting is generally wrong. In particular,
the response-marginalized parametric solver does not itself return the conditional
nuclear-response draws needed for a shared-response count prediction. The example
therefore uses fixed synthetic nuclear and detector inputs and explicitly labels
its partial uncertainty. This is not a complete experimental uncertainty budget.

Native integer counts enter the Poisson score without subtraction, smoothing or
fractional rebinning. A constant live fraction is modeled as independent thinning.
Validate that assumption for dead time/pileup; otherwise a different observation
model is required. A valid posterior likelihood cannot repair a wrong peak or
continuum model. Maintain the native background acquisition's own grid/exposure.

## Split, sources and independent evaluation

1. Declare canonical observation, acquisition and reaction-family identities.
   All lines, repeats and coupled bare/Cd or production paths within a family
   stay on one side. The validator checks declared identities, not nuclear
   genealogy or undisclosed data reuse. Hold out enough independent reactions
   while keeping the training model identifiable.
2. Freeze modeling choices, priors, background, calibration, normalization,
   exclusions and adequacy thresholds using training and independent source
   records only. A later count of the same isotope primarily checks decay and
   detector transfer; a new line checks detector response. Neither alone tests
   new spectral response combinations.
3. Supply a standard run-readiness manifest with SHA-256-bound training data,
   held-out data, calibration, irradiation history, responses, analysis config
   and analysis code. Additional source/background records should be listed.
   The manifest checks file identity and byte integrity, not physical admission,
   completeness, certificate applicability or genuine blindness.
4. Call `freeze_count_prediction` without test count values. Archive/commit its
   returned SHA before the evaluator releases values. `score_frozen_prediction`
   requires that expected SHA and refuses a changed packet. Hashing held-out
   source bytes does not itself prove that their values were never inspected.
   Declare synthetic, retrospective or prospective scope accurately.
5. Run common-rate inference comparisons and common-original-count workflow
   comparisons separately. Use matched external information and units, with
   unadjusted transport/prior and covariance-GLS baselines. Retain all failures.
   No normalization, efficiency tuning or response selection on test counts.

The primary score is `log(mean_draw(product_bin(Poisson(count | mean_draw))))`.
Summing marginal bin scores loses shared-nuisance dependence. Predictive variance
includes count noise plus variability in mean counts. Seeded Monte Carlo 50%, 80%
and 95% intervals are predictive intervals, with Monte Carlo error; single-case
containment is not coverage calibration. Zero predictive probability is explicitly
reported. Compare paired joint scores over independent experimental blocks and
require adequate counts/residuals and calibrated coverage before superiority claims.

`analysis.prediction_comparison.compare_calculated_experimental` adds full joint
C/E covariance (including cross-covariance) and residual chi-square. Ratio errors
use a first-order Jacobian; near-zero denominators require a fuller treatment.
Vendor activities are comparison observations, not truth or withheld raw counts.

## Minimum existing measured dataset

One identified exposure with sufficient independent training responses and one
withheld family; original integer spectra with acquisition/live/real times; foil
mass, composition, covers and position; irradiation clocks/power history; applicable
independent efficiency calibration with covariance; justified background records;
versioned response/decay data and uncertainty; canonical split and frozen settings.
Missing uncertainty must remain explicit. Current South RAFM calibration covariance
and historical applicability remain unresolved; this code does not remove those
experimental blockers or admit the existing inadequate Co-Cd fits.

```sh
PYTHONPATH=src python examples/heldout_count_prediction.py NEW_DIRECTORY
PYTHONPATH=src pytest -q tests/test_count_prediction_benchmark.py
```

The example fits two synthetic reaction families, predicts a third through
irradiation/decay/detection, freezes predictions for fitted and unadjusted-prior
baselines, then scores withheld integer counts. It binds all inputs and relevant
source files and writes a portable packet. Its successful execution is software
verification only. A real-data benchmark and an nFoils comparator run remain
necessary before any measured differentiating result can be claimed.
