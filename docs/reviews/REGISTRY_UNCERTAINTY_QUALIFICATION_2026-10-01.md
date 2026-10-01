# Registry uncertainty qualification — October 1, 2026

At frozen PR213 head `f672374`, the registry's generic ridge helper used
`max(trace(R.T @ R), 1)` and lost invariance to measurement row units. For
`R=s*I2`, `sigma_y=s*[.1,.2]`, the fixed unknown sigma should remain `[.1,.2]`;
at `s=1e-12` the old helper reported approximately `[2e-15,4e-15]`.
GRAVEL, MAXED and ML Seed also published this unrelated linear proxy as their
own estimator uncertainty, regardless of prior, nonlinear constraints or convergence.
The earlier workflow Monte Carlo repair did not cover these registry consumers.

The adapters now return `uncertainties=None`, record estimator-specific
unavailability and nonconvergence reasons, and set `supports_uncertainties=False`.
Numerical response rank is assessed only using supplied positive measurement
sigmas and the error-weighted operator; missing weighting leaves rank unavailable.
This conditional numerical assessment does not certify physical identifiability.
Flux estimates, solver objectives, priors and convergence rules are unchanged.

The retained `estimate_unfolding_uncertainties` utility describes only a fixed,
unconstrained, full-column-rank weighted linear least-squares estimator with
independent explicitly supplied measurement standard uncertainties. It whitens
each response row by sigma and uses scaled SVD without an absolute ridge/floor.
Response uncertainty is excluded. Missing/nonpositive sigma and rank-deficient
response are errors, rather than implicit Poisson variance or zero null-space error.
Changing individual row units for both response and sigma preserves the result.
It does not qualify GRAVEL/MAXED/ML Seed uncertainty.

CLI/export paths retain covariance and flux/predicted-rate uncertainty as JSON
null with a required nonempty reason. Artifact schemas and validation preserve
this meaning. CSV report uncertainty cells stay blank and carry the reason;
text reports and the GUI explain unavailability. The GUI omits missing bands
and disables the band toggle when selected methods provide no uncertainty.
RMLE default and fallback percentage uncertainty is also unavailable. The count-domain
identity counterexample `R=I2`, `y=[25,400]`, zero regularization, seeded at `y`,
converges at the same counts while previously publishing `[2.5,40]` instead of the
independent Poisson identity plug-in sigma `[5,20]`. No general square-root-flux
remedy is implied. The registry does not publish backend uncertainty. Direct
Poisson Monte Carlo keeps sampling diagnostics but does not qualify bin uncertainty:
model, replicate convergence, estimator selection and identifiability remain unresolved.
A Gaussian fallback cannot silently republish a different estimator's covariance;
covariance computation failures also return None. Successful direct Gaussian
linear propagation is labelled an unqualified conditional linear proxy; its
positivity and parameter selection are not propagated. No new Poisson fit of
activation rates is used for the validation.

The bounded actual replay uses the frozen 13-row, 20-group input, three
manufactured prior/start choices and the same recorded solver parameters.
All six GRAVEL/MAXED estimates, predictions, residuals and convergence states
must match the historical replay exactly. MAXED's choices change its entropy
prior and initial iterate together, so they are not pure optimizer-start variation.
Consumer exports also exercise these actual arrays with unavailable covariance.
The identity row-unit regression, rank deficiency, nonconvergence, metadata rank,
GUI and JSON/CSV consumer checks supplement that replay.

Measured rates, mass, activities, schedules and source covariance are not corrected.
Coarse Sc/Co group bounds do not establish general pointwise incompatibility;
the separately accepted pointwise refinement remains a source/model investigation.
Calibration, event/history, geometry and full covariance qualification are unresolved.
Scientific admission remains false. Related: issues195 and214; frozen PR213 evidence remains intact.
