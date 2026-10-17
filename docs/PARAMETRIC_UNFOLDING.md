# Parametric unfolding and correlated nuclear responses

`fluxforge.solvers.parametric` supplies a caller-defined spectral model,
explicit bounded parameters and a fixed-kernel, multi-chain Metropolis sampler.
It complements existing groupwise GLS/GRAVEL/MLEM/MCMC. It uses existing NumPy
and SciPy dependencies; nFoils, emcee, SANDY and NJOY are not runtime dependencies.

The spectrum callback returns **group-integrated flux**. Group-averaged cross
sections in cm² multiply flux in particles/cm²/s to give reactions/target atom/s.
`integrate_spectrum` integrates a differential spectrum over explicit edges;
its quadrature order must resolve the caller's narrowest feature. The operator
is particle-agnostic; evaluated data for the actual incident particle must be
supplied. No lithium-source spectral prior is imposed on a reactor field.

For rates y, response R and group flux f(theta), the likelihood is
`Normal(y; R f, V + B(f) C_R B(f)^T)`, with `B = kron(I, f^T)`.
`C_R` is the absolute covariance of `R.flatten(order='C')`, ordered reaction
then energy group. It preserves energy and cross-reaction correlations.
The log determinant of this parameter-dependent covariance is included.
This is an exact marginalization under the declared Gaussian response-error
model, assuming independence from measurement errors and spectrum parameters.
It avoids equally pooling response-conditional posteriors with prior response
weights. It is not a general replacement for non-Gaussian joint nuclear-data
inference; Gaussian responses can include unphysical tails. Measurement–response
cross-covariance and uncertain energy grids require an extended joint model.
Omitting `C_R` explicitly means a fixed response, not zero known nuclear error.
No covariance floor is invented and singular measurement covariance is rejected.

Each Parameter's support is checked before evaluating the model. Custom priors
are densities in the declared parameter coordinates; a log-coordinate model
must include its intended change of measure. Independent overdispersed starts
are supplied explicitly. Results retain chain identity, acceptance, classical
split R-hat, parameter samples and flux samples. R-hat is not rank-normalized;
there is no automatic convergence or physical-qualification flag. Review trace
plots, multiple starts, effective sample size and model adequacy before using
uncertainty scientifically. Finite bounds do not establish identifiability.

Run the self-contained synthetic example:

```sh
PYTHONPATH=src python examples/parametric_unfolding.py
PYTHONPATH=src pytest -q tests/test_parametric_unfolding.py
```

Tests use independent multivariate-normal densities, direct integration over a
latent response, nonuniform-group integration and a known truncated-normal
posterior. Synthetic recovery is software verification, not measured validation.
This implementation follows established Bayesian and Gaussian-marginalization
methods; it does not claim methodological novelty or superiority over nFoils.
