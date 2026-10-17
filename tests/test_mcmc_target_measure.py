"""Independent target-distribution checks for the log-normal proposal."""

import math
import random

import numpy as np
import pytest
from scipy.stats import halfnorm, lognorm

from fluxforge.solvers.mcmc import (
    _log_proposal_ratio,
    _propose_flux,
    mcmc_unfold,
)


def test_hastings_ratio_matches_independent_lognormal_density():
    current, proposed, step = [1.0, 7.0], [2.0, 3.0], 0.4
    expected = sum(
        lognorm.logpdf(c, s=step, scale=p) - lognorm.logpdf(p, s=step, scale=c)
        for c, p in zip(current, proposed)
    )
    assert _log_proposal_ratio(current, proposed) == pytest.approx(expected)
    assert _log_proposal_ratio([1.0], [2.0]) == pytest.approx(math.log(2))


def test_half_normal_posterior_in_linear_flux():
    # y=0, sigma=1, R=1 and flat positive-flux prior => standard half normal.
    # A symmetric-log acceptance rule without Hastings targets a different law.
    result = mcmc_unfold(
        [[1.0]],
        [0.0],
        initial_flux=[0.5],
        measurement_uncertainty=[1.0],
        n_samples=70000,
        burn_in=5000,
        thin=2,
        step_size=1.0,
        prior="uniform",
        adaptive_step=False,
        seed=2801,
    )
    samples = np.asarray(result.samples)[:, 0]
    assert samples.mean() == pytest.approx(halfnorm.mean(), abs=0.025)
    assert np.mean(samples < halfnorm.ppf(0.8)) == pytest.approx(0.8, abs=0.02)
    assert result.diagnostics["posterior_measure"] == "linear_group_flux"
    assert result.diagnostics["physical_qualified"] is False


def test_sampler_does_not_reseed_callers_random_generator():
    state = random.getstate()
    mcmc_unfold(
        [[1]],
        [1],
        measurement_uncertainty=[1],
        n_samples=30,
        burn_in=10,
        thin=1,
        seed=17,
    )
    assert random.getstate() == state


def test_small_states_are_not_clamped_in_proposal():
    class ZeroNoise:
        def gauss(self, mean, sigma):
            return 0.0

    assert _propose_flux([1e-30], floor=1e-20, rng=ZeroNoise()) == pytest.approx(
        [1e-30], rel=1e-12, abs=0
    )


@pytest.mark.parametrize("value", [0.0, float("inf"), float("nan")])
def test_unrepresentable_proposals_rejected(value):
    assert _log_proposal_ratio([1], [value]) == -math.inf


@pytest.mark.parametrize(
    "kwargs",
    [
        {"thin": 0},
        {"n_samples": 0},
        {"burn_in": -1},
        {"step_size": float("nan")},
        {"floor": 0},
        {"smoothness_weight": -1},
        {"target_acceptance": 1},
        {"measurement_uncertainty": [0]},
    ],
)
def test_invalid_controls_rejected(kwargs):
    with pytest.raises(ValueError):
        mcmc_unfold([[1]], [1], **kwargs)
