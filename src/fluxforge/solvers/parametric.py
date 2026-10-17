"""Parametric unfolding with analytically marginalized Gaussian responses.

The spectrum callback returns group-INTEGRATED flux, not differential ordinates.
The prior is a density in the declared parameter coordinates. Response errors
are Gaussian, independent of measurement errors and of spectrum parameters;
their correlations across reactions and energy groups are retained.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Callable, Sequence

import numpy as np

from fluxforge.core.probability import covariance_matrix, gaussian_log_density


def _vector(values, name, *, nonnegative=False):
    value = np.array(values, dtype=float, copy=True)
    if value.ndim != 1 or value.size == 0 or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a finite nonempty vector")
    if nonnegative and np.any(value < 0):
        raise ValueError(f"{name} must be nonnegative")
    value.setflags(write=False)
    return value


@dataclass(frozen=True)
class GaussianResponseModel:
    """Rate likelihood with full nuclear-response covariance.

    `response[i,g]` multiplies integrated group flux. `response_covariance`
    describes absolute errors in response.flatten(order='C'), i.e. reaction
    major then group. It is NOT an already-folded rate covariance. Missing
    nuclear covariance means conditional-on-fixed-response inference.
    """

    response: np.ndarray
    measurements: np.ndarray
    measurement_covariance: np.ndarray
    response_covariance: np.ndarray | None = None

    def __post_init__(self):
        response = np.array(self.response, dtype=float, copy=True)
        if (
            response.ndim != 2
            or 0 in response.shape
            or not np.isfinite(response).all()
            or np.any(response < 0)
        ):
            raise ValueError("response must be a finite nonnegative matrix")
        rates = _vector(self.measurements, "measurements")
        if rates.size != response.shape[0]:
            raise ValueError("response rows must match measurements")
        observed = covariance_matrix(
            self.measurement_covariance,
            rates.size,
            "measurement_covariance",
            definite=True,
        )
        nuclear = None
        if self.response_covariance is not None:
            nuclear = covariance_matrix(
                self.response_covariance, response.size, "response_covariance"
            )
        response.setflags(write=False)
        object.__setattr__(self, "response", response)
        object.__setattr__(self, "measurements", rates)
        object.__setattr__(self, "measurement_covariance", observed)
        object.__setattr__(self, "response_covariance", nuclear)

    def predict(self, group_flux):
        """Return rate means and covariance conditional on a spectrum."""
        flux = _vector(group_flux, "group_flux", nonnegative=True)
        m, g = self.response.shape
        if flux.size != g:
            raise ValueError("group_flux length must match response columns")
        covariance = self.measurement_covariance.copy()
        if self.response_covariance is not None:
            covariance += np.einsum(
                "g,igjh,h->ij",
                flux,
                self.response_covariance.reshape(m, g, m, g),
                flux,
            )
        if not np.isfinite(covariance).all():
            raise ValueError("folded covariance is not finite")
        mean = self.response @ flux
        if not np.isfinite(mean).all():
            raise ValueError("predicted rates are not finite")
        return mean, covariance

    def log_likelihood(self, group_flux):
        mean, covariance = self.predict(group_flux)
        # The determinant depends on flux when response uncertainty is present.
        return gaussian_log_density(self.measurements - mean, covariance)


@dataclass(frozen=True)
class Parameter:
    """A finite uniform support and random-walk scale in parameter coordinates."""

    name: str
    lower: float
    upper: float
    proposal_scale: float

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("parameter name must be nonempty")
        if not np.isfinite([self.lower, self.upper, self.proposal_scale]).all():
            raise ValueError("parameter bounds and proposal scale must be finite")
        if self.lower >= self.upper or self.proposal_scale <= 0:
            raise ValueError("require lower < upper and positive proposal scale")


def integrate_spectrum(spectrum, energy_edges, theta, *, order=32):
    """Integrate differential spectrum(E, theta) over each energy group.

    Energy edges and differential spectrum must use reciprocal consistent units.
    Uses fixed Gauss-Legendre quadrature; callers must establish quadrature
    accuracy for narrow features. No midpoint or bin-width convention is inferred.
    """
    edges = _vector(energy_edges, "energy_edges")
    if edges.size < 2 or edges[0] < 0 or np.any(np.diff(edges) <= 0):
        raise ValueError("energy_edges must be nonnegative and increasing")
    if isinstance(order, bool) or not isinstance(order, int) or order < 2:
        raise ValueError("quadrature order must be an integer >= 2")
    nodes, weights = np.polynomial.legendre.leggauss(order)
    half_width = np.diff(edges) / 2
    grid = (edges[:-1, None] + edges[1:, None]) / 2 + half_width[:, None] * nodes
    values = np.asarray(spectrum(grid, np.asarray(theta, dtype=float)), dtype=float)
    if values.shape != grid.shape or not np.isfinite(values).all():
        raise ValueError("differential spectrum must return finite grid-shaped values")
    if np.any(values < 0):
        raise ValueError("differential spectrum must be nonnegative")
    return half_width * (values @ weights)


@dataclass(frozen=True)
class ParametricProblem:
    likelihood: GaussianResponseModel
    spectrum: Callable[[np.ndarray], np.ndarray]
    parameters: Sequence[Parameter]
    model_id: str
    log_prior: Callable[[np.ndarray], float] | None = None

    def __post_init__(self):
        parameters = tuple(self.parameters)
        if not parameters or not all(isinstance(p, Parameter) for p in parameters):
            raise ValueError("parameters must contain Parameter objects")
        if len({p.name for p in parameters}) != len(parameters):
            raise ValueError("parameter names must be unique")
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("model_id must identify the caller's spectrum model")
        object.__setattr__(self, "parameters", parameters)

    def log_posterior(self, theta):
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (len(self.parameters),) or not np.isfinite(theta).all():
            return -np.inf
        if any(t < p.lower or t > p.upper for t, p in zip(theta, self.parameters)):
            return -np.inf
        prior = (
            float(self.log_prior(theta))
            if self.log_prior is not None
            else -sum(np.log(p.upper - p.lower) for p in self.parameters)
        )
        if np.isnan(prior) or prior == np.inf:
            raise ValueError("log_prior must be finite or negative infinity")
        if prior == -np.inf:
            return prior
        return prior + self.likelihood.log_likelihood(self.spectrum(theta))


def split_rhat(chains):
    """Classical split R-hat per parameter; not rank-normalized R-hat or ESS."""
    chains = np.asarray(chains, dtype=float)
    if chains.ndim != 3 or chains.shape[0] < 2 or chains.shape[1] < 4:
        raise ValueError("split R-hat requires >=2 chains, >=4 draws per chain")
    n = chains.shape[1] // 2
    split = np.concatenate((chains[:, :n], chains[:, -n:]), axis=0)
    within = np.var(split, axis=1, ddof=1).mean(axis=0)
    between = n * np.var(split.mean(axis=1), axis=0, ddof=1)
    estimate = (n - 1) / n * within + between / n
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(within > 0, np.sqrt(estimate / within), np.inf)


@dataclass
class ParametricSolution:
    parameter_samples: np.ndarray
    flux_samples: np.ndarray
    log_posterior: np.ndarray
    acceptance_rates: np.ndarray
    rhat: np.ndarray
    metadata: dict

    @property
    def flux(self):
        return self.flux_samples.mean(axis=(0, 1))

    def credible_interval(self, probability=0.95):
        if not 0 < probability < 1:
            raise ValueError("probability must lie between zero and one")
        tail = (1 - probability) / 2
        return np.quantile(self.flux_samples, [tail, 1 - tail], axis=(0, 1))


def parametric_unfold(
    problem: ParametricProblem,
    initial_states,
    *,
    draws=2000,
    warmup=1000,
    thin=1,
    seed=None,
):
    """Sample independent chains with a fixed symmetric parameter-space kernel.

    initial_states is (chains, parameters), with >=2 chains supplied by caller.
    Draws is the number retained PER chain; warmup is the number discarded.
    No clipping, adaptation after warmup, or automatic convergence claim occurs.
    """
    for name, value, minimum in (
        ("draws", draws, 4),
        ("warmup", warmup, 0),
        ("thin", thin, 1),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    states = np.array(initial_states, dtype=float, copy=True)
    starts = states.copy()
    ndim = len(problem.parameters)
    if states.ndim != 2 or states.shape[0] < 2 or states.shape[1] != ndim:
        raise ValueError("initial_states must contain >=2 chains of all parameters")
    current = np.array([problem.log_posterior(t) for t in states])
    if not np.isfinite(current).all():
        raise ValueError("all initial states must have finite log posterior")
    rng = np.random.default_rng(seed)
    scales = np.array([p.proposal_scale for p in problem.parameters])
    samples = np.empty((states.shape[0], draws, ndim))
    logs = np.empty(samples.shape[:2])
    accepted = np.zeros(states.shape[0], dtype=int)
    for iteration in range(warmup + draws * thin):
        for c in range(states.shape[0]):
            proposal = states[c] + rng.normal(size=ndim) * scales
            proposed = problem.log_posterior(proposal)
            if np.log(max(rng.random(), np.finfo(float).tiny)) < proposed - current[c]:
                states[c], current[c] = proposal, proposed
                if iteration >= warmup:
                    accepted[c] += 1
        if iteration >= warmup and (iteration - warmup) % thin == 0:
            k = (iteration - warmup) // thin
            samples[:, k], logs[:, k] = states, current
    flux = np.array([[problem.spectrum(t) for t in chain] for chain in samples])
    return ParametricSolution(
        samples,
        flux,
        logs,
        accepted / (draws * thin),
        split_rhat(samples),
        {
            "model_id": problem.model_id,
            "parameter_names": [p.name for p in problem.parameters],
            "posterior_measure": "declared_parameter_coordinates",
            "response_uncertainty": (
                "full_gaussian_marginalization"
                if problem.likelihood.response_covariance is not None
                else "fixed_response"
            ),
            "converged": None,
            "physical_qualified": False,
            "rhat_definition": "classical_split_not_rank_normalized",
            "seed": seed,
            "draws_per_chain": draws,
            "warmup": warmup,
            "thin": thin,
            "initial_states": starts.tolist(),
            "parameter_support": [[p.lower, p.upper] for p in problem.parameters],
            "proposal_scales": scales.tolist(),
            "input_sha256": {
                name: hashlib.sha256(value.astype("<f8").tobytes()).hexdigest()
                for name, value in (
                    ("response", problem.likelihood.response),
                    ("measurements", problem.likelihood.measurements),
                    (
                        "measurement_covariance",
                        problem.likelihood.measurement_covariance,
                    ),
                    ("response_covariance", problem.likelihood.response_covariance),
                )
                if value is not None
            },
        },
    )
