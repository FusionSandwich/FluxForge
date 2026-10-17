# MCMC density and proposal convention

The existing `mcmc_unfold` uniform and smoothness priors define densities
with respect to **linear group flux**. A symmetric random walk in log flux
is log-normal in linear flux, so its Metropolis-Hastings log acceptance
ratio includes `sum(log(proposed) - log(current))`. Omitting this term
instead introduces an inverse-flux factor into the linear-flux target.

The proposal no longer clamps each current state before stepping. Overflow
and underflow proposals are rejected. A caller-local RNG makes seeded calls
reproducible without changing Python's process-wide random state.

`tests/test_mcmc_target_measure.py` checks the proposal ratio against SciPy's
log-normal density and samples an independently known half-normal posterior.
These are numerical target-distribution checks, not experimental qualification.
Acceptance, a finite chain, or thinning does not establish convergence,
effective independence, uncertainty coverage, or a physically identified spectrum.

Supplied errors in this legacy API are diagonal. Omitted errors retain the
historical ten-percent diagnostic assumption, explicitly labeled in diagnostics.
Use independently measured covariance and an explicit nuclear-response model
for a scientific comparison. Historical chains should be regenerated; this
correction changes the sampled distribution.
