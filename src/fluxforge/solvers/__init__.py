"""Solver package."""

from fluxforge.solvers.gls import gls_adjust
from fluxforge.solvers.iterative import IterativeSolution, gravel, mlem, gradient_descent
from fluxforge.solvers.mcmc import MCMCSolution, mcmc_unfold, mcmc_convergence_diagnostic
from fluxforge.solvers.regularized import (
    RegularizedSolution,
    regularized_unfold,
    gradient_descent_regularized,
    tikhonov_solve,
    l_curve_corner,
    gcv_select_alpha,
    log_smoothness_penalty,
)
from fluxforge.solvers.test_statistics import (
    TestStatisticType,
    ConvergenceResult,
    ks_test_statistic,
    chi2_test_statistic,
    bayes_factor_test_statistic,
    rmd_test_statistic,
    compute_test_statistic,
    check_convergence,
    uniform_prior,
    jeffreys_prior,
    power_law_prior,
    spline_regularize,
    grouped_spline_regularize,
    SplineRegularizer,
    iterative_bayesian_unfold,
)

__all__ = [
    "gls_adjust",
    "IterativeSolution",
    "gravel",
    "mlem",
    "gradient_descent",
    "MCMCSolution",
    "mcmc_unfold",
    "mcmc_convergence_diagnostic",
    "RegularizedSolution",
    "regularized_unfold",
    "gradient_descent_regularized",
    "tikhonov_solve",
    "l_curve_corner",
    "gcv_select_alpha",
    "log_smoothness_penalty",
    # Test statistics
    "TestStatisticType",
    "ConvergenceResult",
    "ks_test_statistic",
    "chi2_test_statistic",
    "bayes_factor_test_statistic",
    "rmd_test_statistic",
    "compute_test_statistic",
    "check_convergence",
    # Priors
    "uniform_prior",
    "jeffreys_prior",
    "power_law_prior",
    # Regularization
    "spline_regularize",
    "grouped_spline_regularize",
    "SplineRegularizer",
    "iterative_bayesian_unfold",
]
