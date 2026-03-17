"""Tests for optional unfold wrappers and lazy import behavior."""

from __future__ import annotations

import numpy as np
import pytest


def test_unfold_lazy_exports() -> None:
    import fluxforge.unfold as unfold

    ReactionRates = unfold.ReactionRates
    ResponseBundle = unfold.ResponseBundle
    SpectrumFile = unfold.SpectrumFile

    rates = ReactionRates(values=np.array([1.0]), uncertainties=np.array([0.1]))
    response = ResponseBundle(matrix=np.array([[1.0]]), energy_bins=np.array([0.0, 1.0]))
    spec = SpectrumFile(counts=np.array([10.0]), live_time_s=1.0)

    assert rates.values.shape == (1,)
    assert response.matrix.shape == (1, 1)
    assert spec.live_time_s == 1.0

    with pytest.raises(AttributeError):
        getattr(unfold, "does_not_exist")


def test_gamma_rmle_guard_and_solver(monkeypatch: pytest.MonkeyPatch) -> None:
    from fluxforge.unfold import gamma_rmle

    # Guard path
    monkeypatch.setattr(gamma_rmle, "_HAS_PYLOPS", False)
    with pytest.raises(ImportError):
        gamma_rmle._require_pylops()

    # Fake pylops objects for deterministic test
    class FakeOperator:
        def __init__(self, matrix: np.ndarray) -> None:
            self.matrix = np.asarray(matrix, dtype=float)

        def __matmul__(self, vec: np.ndarray) -> np.ndarray:
            return self.matrix @ np.asarray(vec, dtype=float)

    class FakePylops:
        MatrixMult = FakeOperator

    def fake_fista(op, y, niter, eps, tol, show):
        # Return a vector with one negative component to exercise non-neg clamp.
        x = np.linalg.lstsq(op.matrix, y, rcond=None)[0]
        x[0] = -abs(x[0])
        return x, 7, [1.0, 0.5, 0.2]

    monkeypatch.setattr(gamma_rmle, "_HAS_PYLOPS", True)
    monkeypatch.setattr(gamma_rmle, "pylops", FakePylops)
    monkeypatch.setattr(gamma_rmle, "pylops_fista", fake_fista)

    R = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    y = np.array([3.0, 1.0, 4.0])
    solver = gamma_rmle.GammaUnfolderRMLE(R)

    unfolded, residuals = solver.solve_regularized(y, lambda_reg=0.1, n_iter=50)
    assert unfolded.shape == (2,)
    assert residuals.shape == (3,)
    assert np.all(unfolded >= 0.0)

    full = solver.solve_full(y, lambda_reg=0.1, n_iter=50)
    assert full.unfolded_spectrum.shape == (2,)
    assert full.residuals.shape == (3,)
    assert full.n_iterations == 7
    assert len(full.cost_history) == 3
    assert solver.n_channels == 3
    assert solver.n_energy_bins == 2

    with pytest.raises(ValueError):
        solver.solve_regularized(np.array([1.0, 2.0]))  # wrong length


def test_neutron_ibu_guard_and_solver(monkeypatch: pytest.MonkeyPatch) -> None:
    from fluxforge.unfold import neutron_ibu
    from fluxforge.unfold._types import ReactionRates, ResponseBundle

    # Guard path
    monkeypatch.setattr(neutron_ibu, "_HAS_PYUNFOLD", False)
    with pytest.raises(ImportError):
        neutron_ibu._require_pyunfold()

    def fake_pyunfold_unfold(
        *,
        data,
        data_err,
        response,
        response_err,
        efficiencies,
        efficiencies_err,
        prior,
        ts,
        ts_stopping,
        max_iter,
        cov_type,
        return_iterations,
    ):
        # Validate wrapper input mapping is sensible.
        assert len(data) == response.shape[0]
        assert len(efficiencies) == response.shape[1]
        assert ts in {"ks", "chi2", "bf", "rmd"}
        return {
            "unfolded": np.array([2.0, 1.0]),
            "stat_err": np.array([0.2, 0.1]),
            "sys_err": np.array([0.1, 0.2]),
            "num_iterations": 4,
            "ts_iter": 0.03,
            "unfolding_matrix": np.eye(2),
            "custom_diagnostic": 42,
        }

    monkeypatch.setattr(neutron_ibu, "_HAS_PYUNFOLD", True)
    monkeypatch.setattr(neutron_ibu, "_pyunfold_unfold", fake_pyunfold_unfold)

    solver = neutron_ibu.NeutronUnfolderIBU(ts="chi2", ts_stopping=0.05, max_iter=20)
    rates = ReactionRates(values=np.array([10.0, 20.0]), uncertainties=np.array([1.0, 2.0]))
    response = ResponseBundle(
        matrix=np.array([[1.0, 0.2], [0.1, 1.0]]),
        energy_bins=np.array([0.0, 1.0, 2.0]),
    )

    result = solver.solve(rates, response, prior_flux=np.array([3.0, 1.0]))
    np.testing.assert_allclose(result.unfolded_flux, np.array([2.0, 1.0]))
    np.testing.assert_allclose(np.diag(result.flux_covariance), np.array([0.05, 0.05]))
    assert result.n_iterations == 4
    assert result.test_statistic == 0.03
    assert result.diagnostics["custom_diagnostic"] == 42

    comparison = solver.compare_with_gls(np.array([2.1, 1.1]), result, rtol=0.2)
    assert isinstance(comparison["agrees"], bool)
    assert "max_relative_difference" in comparison

    with pytest.raises(ValueError):
        solver.compare_with_gls(np.array([1.0, 2.0, 3.0]), result)

    with pytest.raises(ValueError):
        solver.solve(
            ReactionRates(values=np.array([1.0]), uncertainties=np.array([0.1])),
            response,
        )
    with pytest.raises(ValueError):
        solver.solve(rates, response, prior_flux=np.array([1.0, 2.0, 3.0]))
