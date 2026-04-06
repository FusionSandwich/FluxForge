from __future__ import annotations

from argparse import Namespace
import random
from types import SimpleNamespace

import numpy as np
import pytest

from fluxforge.cli import app
from fluxforge.solvers.advanced_unfolding import CovarianceModel, mlem_with_covariance
from fluxforge.solvers.gls import gls_adjust, gls_adjust_with_response_cov
from fluxforge.solvers.iterative import gradient_descent, gravel, mlem
from fluxforge.solvers.mcmc import mcmc_unfold
from fluxforge.solvers.regularized import regularized_unfold
from fluxforge.solvers.rmle import (
    PoissonPenalty,
    PoissonRMLEConfig,
    RegularizationType,
    ResponseMatrix,
    SpectrumData,
    poisson_rmle_unfolding,
    rmle_unfolding,
)
from fluxforge.unfold._types import ReactionRates, ResponseBundle
from fluxforge.unfolding import GravelUnfolder, MLSeedUnfolder, MaxedUnfolder, RMLEUnfolder
from fluxforge.workflows import spectrum_unfolding as spectrum_workflow
from fluxforge.workflows.spectrum_unfolding import FluxWireMeasurement, SpectrumUnfolder


@pytest.fixture
def neutron_case() -> SimpleNamespace:
    response = np.eye(3, dtype=float)
    true_flux = np.array([120.0, 60.0, 20.0], dtype=float)
    measurements = response @ true_flux
    uncertainties = np.array([6.0, 4.0, 2.0], dtype=float)
    prior = np.array([80.0, 80.0, 80.0], dtype=float)
    measurement_cov = np.diag(uncertainties**2)
    prior_cov = np.diag((0.5 * prior) ** 2)
    response_cov = np.full_like(response, 0.02)
    return SimpleNamespace(
        response=response,
        true_flux=true_flux,
        measurements=measurements,
        uncertainties=uncertainties,
        prior=prior,
        measurement_cov=measurement_cov,
        prior_cov=prior_cov,
        response_cov=response_cov,
    )


@pytest.fixture
def gamma_case() -> SimpleNamespace:
    n_channels = 60
    n_bins = 6
    response = np.zeros((n_channels, n_bins), dtype=float)
    block = n_channels // n_bins
    for idx in range(n_bins):
        lo = idx * block
        hi = (idx + 1) * block
        response[lo:hi, idx] = 1.0
    response = ResponseMatrix(matrix=response).normalize_columns()
    true_source = np.array([0.0, 50.0, 0.0, 80.0, 0.0, 20.0], dtype=float)
    counts = response.matrix @ true_source
    spectrum = SpectrumData(counts=counts)
    return SimpleNamespace(
        response=response,
        true_source=true_source,
        counts=counts,
        spectrum=spectrum,
    )


def _refold_relative_error(
    response_matrix: np.ndarray,
    flux: np.ndarray,
    measurements: np.ndarray,
) -> float:
    predicted = np.asarray(response_matrix, dtype=float) @ np.asarray(flux, dtype=float)
    baseline = max(float(np.linalg.norm(measurements)), 1e-12)
    return float(np.linalg.norm(predicted - measurements) / baseline)


def _fake_pylops_module(monkeypatch: pytest.MonkeyPatch):
    from fluxforge.unfold import gamma_rmle

    class FakeOperator:
        def __init__(self, matrix: np.ndarray) -> None:
            self.matrix = np.asarray(matrix, dtype=float)

        def __matmul__(self, vec: np.ndarray) -> np.ndarray:
            return self.matrix @ np.asarray(vec, dtype=float)

    class FakePylops:
        MatrixMult = FakeOperator

    def fake_fista(op, y, niter, eps, tol, show):
        x = np.linalg.lstsq(op.matrix, y, rcond=None)[0]
        return np.maximum(x, 0.0), 6, [1.0, 0.4, 0.1]

    monkeypatch.setattr(gamma_rmle, "_HAS_PYLOPS", True)
    monkeypatch.setattr(gamma_rmle, "pylops", FakePylops)
    monkeypatch.setattr(gamma_rmle, "pylops_fista", fake_fista)
    return gamma_rmle


@pytest.mark.parametrize(
    ("name", "threshold"),
    [
        ("gls", 1e-2),
        ("gls_mc", 3e-2),
        ("gravel", 1e-3),
        ("mlem", 1e-3),
        ("gradient_descent", 5e-2),
        ("mlem_cov", 1e-3),
        ("regularized_gradient", 0.25),
        ("regularized_tikhonov", 1e-5),
        ("mcmc", 0.2),
        ("registry_gravel", 1e-3),
        ("registry_maxed", 5e-2),
        ("registry_rmle", 1e-2),
        ("registry_ml_seed", 5e-2),
    ],
)
def test_public_neutron_unfolding_paths_cover_identity_case(
    neutron_case: SimpleNamespace,
    name: str,
    threshold: float,
):
    case = neutron_case
    uncertainty = None

    if name == "gls":
        result = gls_adjust(
            case.response,
            case.measurements,
            case.measurement_cov,
            case.prior,
            case.prior_cov,
            enforce_nonnegativity=True,
        )
        flux = np.asarray(result.flux, dtype=float)
        uncertainty = np.asarray(result.flux_uncertainty, dtype=float)
    elif name == "gls_mc":
        random.seed(42)
        result = gls_adjust_with_response_cov(
            case.response,
            case.response_cov,
            case.measurements,
            case.measurement_cov,
            case.prior,
            case.prior_cov,
            n_samples=4,
            enforce_nonnegativity=True,
        )
        flux = np.asarray(result.flux, dtype=float)
        uncertainty = np.asarray(result.flux_uncertainty, dtype=float)
    elif name == "gravel":
        result = gravel(
            case.response.tolist(),
            case.measurements.tolist(),
            initial_flux=case.prior.tolist(),
            measurement_uncertainty=case.uncertainties.tolist(),
            max_iters=200,
            tolerance=1e-8,
            chi2_tolerance=1e-10,
        )
        flux = np.asarray(result.flux, dtype=float)
    elif name == "mlem":
        result = mlem(
            case.response.tolist(),
            case.measurements.tolist(),
            initial_flux=case.prior.tolist(),
            measurement_uncertainty=case.uncertainties.tolist(),
            max_iters=200,
            tolerance=1e-8,
            chi2_tolerance=1e-10,
        )
        flux = np.asarray(result.flux, dtype=float)
    elif name == "gradient_descent":
        result = gradient_descent(
            case.response.tolist(),
            case.measurements.tolist(),
            initial_flux=case.prior.tolist(),
            measurement_uncertainty=case.uncertainties.tolist(),
            max_iters=600,
            tolerance=1e-8,
            chi2_tolerance=1e-8,
            learning_rate=0.5,
            smoothness_weight=0.0,
        )
        flux = np.asarray(result.flux, dtype=float)
    elif name == "mlem_cov":
        result = mlem_with_covariance(
            case.response.tolist(),
            case.measurements.tolist(),
            initial_flux=case.prior.tolist(),
            cov_model=CovarianceModel.POISSON,
            measurement_uncertainty=case.uncertainties.tolist(),
            max_iters=200,
            tolerance=1e-8,
            convergence_mode="relative",
            smoothness_weight=0.0,
        )
        flux = np.asarray(result.flux, dtype=float)
        uncertainty = np.asarray(result.flux_uncertainty, dtype=float)
    elif name == "regularized_gradient":
        result = regularized_unfold(
            case.response,
            case.measurements,
            measurement_errors=case.uncertainties,
            prior_spectrum=None,
            method="gradient",
            reg_type="log_smooth",
            reg_param=1e-6,
            learning_rate=0.5,
            max_epochs=1200,
            min_improvement=1e-8,
            patience=200,
        )
        flux = np.asarray(result.spectrum, dtype=float)
        uncertainty = np.asarray(result.uncertainties, dtype=float)
    elif name == "regularized_tikhonov":
        result = regularized_unfold(
            case.response,
            case.measurements,
            measurement_errors=case.uncertainties,
            prior_spectrum=case.prior,
            method="tikhonov",
            reg_type="second_deriv",
            reg_param=1e-6,
        )
        flux = np.asarray(result.spectrum, dtype=float)
        uncertainty = np.asarray(result.uncertainties, dtype=float)
    elif name == "mcmc":
        result = mcmc_unfold(
            case.response.tolist(),
            case.measurements.tolist(),
            initial_flux=case.prior.tolist(),
            measurement_uncertainty=case.uncertainties.tolist(),
            n_samples=400,
            burn_in=100,
            thin=2,
            seed=42,
            prior="uniform",
            adaptive_step=False,
        )
        flux = np.asarray(result.flux, dtype=float)
        uncertainty = np.asarray(result.credible_upper, dtype=float) - np.asarray(
            result.credible_lower,
            dtype=float,
        )
    elif name == "registry_maxed":
        result = MaxedUnfolder(
            max_iterations=300,
            entropy_weight=0.02,
        ).unfold(
            case.measurements,
            case.response,
            initial_flux=case.prior,
            measurement_uncertainty=case.uncertainties,
        )
        flux = np.asarray(result.flux, dtype=float)
        uncertainty = np.asarray(result.uncertainties, dtype=float)
    elif name == "registry_rmle":
        result = RMLEUnfolder(
            max_iterations=300,
            auto_regularization=True,
        ).unfold(
            case.measurements,
            case.response,
            measurement_uncertainty=case.uncertainties,
        )
        flux = np.asarray(result.flux, dtype=float)
        uncertainty = np.asarray(result.uncertainties, dtype=float)
    elif name == "registry_ml_seed":
        result = MLSeedUnfolder().unfold(
            case.measurements,
            case.response,
            initial_flux=case.prior,
            measurement_uncertainty=case.uncertainties,
            confidence_threshold=0.4,
        )
        flux = np.asarray(result.flux, dtype=float)
        uncertainty = np.asarray(result.uncertainties, dtype=float)
    else:
        result = GravelUnfolder(
            max_iterations=200,
            tolerance=1e-8,
            chi2_tolerance=1e-10,
        ).unfold(
            case.measurements,
            case.response,
            initial_flux=case.prior,
            measurement_uncertainty=case.uncertainties,
        )
        flux = np.asarray(result.flux, dtype=float)

    assert flux.shape == case.true_flux.shape
    assert np.all(flux >= 0.0)
    assert _refold_relative_error(case.response, flux, case.measurements) <= threshold
    if uncertainty is not None:
        assert uncertainty.shape == case.true_flux.shape
        assert np.all(np.isfinite(uncertainty))


@pytest.mark.parametrize(
    "name",
    [
        "gls",
        "gls_mc",
        "gravel",
        "mlem",
        "gradient_descent",
        "mlem_cov",
        "regularized",
        "mcmc",
        "registry_gravel",
        "registry_rmle",
        "registry_ml_seed",
    ],
)
def test_public_neutron_unfolding_paths_reject_negative_measurements(
    neutron_case: SimpleNamespace,
    name: str,
):
    case = neutron_case
    measurements = case.measurements.copy()
    measurements[0] *= -1.0

    with pytest.raises(ValueError, match="must be non-negative"):
        if name == "gls":
            gls_adjust(
                case.response,
                measurements,
                case.measurement_cov,
                case.prior,
                case.prior_cov,
            )
        elif name == "gls_mc":
            gls_adjust_with_response_cov(
                case.response,
                case.response_cov,
                measurements,
                case.measurement_cov,
                case.prior,
                case.prior_cov,
                n_samples=2,
            )
        elif name == "gravel":
            gravel(
                case.response.tolist(),
                measurements.tolist(),
                initial_flux=case.prior.tolist(),
                measurement_uncertainty=case.uncertainties.tolist(),
            )
        elif name == "mlem":
            mlem(
                case.response.tolist(),
                measurements.tolist(),
                initial_flux=case.prior.tolist(),
                measurement_uncertainty=case.uncertainties.tolist(),
            )
        elif name == "gradient_descent":
            gradient_descent(
                case.response.tolist(),
                measurements.tolist(),
                initial_flux=case.prior.tolist(),
                measurement_uncertainty=case.uncertainties.tolist(),
            )
        elif name == "mlem_cov":
            mlem_with_covariance(
                case.response.tolist(),
                measurements.tolist(),
                initial_flux=case.prior.tolist(),
                measurement_uncertainty=case.uncertainties.tolist(),
            )
        elif name == "regularized":
            regularized_unfold(
                case.response,
                measurements,
                measurement_errors=case.uncertainties,
                prior_spectrum=case.prior,
            )
        elif name == "mcmc":
            mcmc_unfold(
                case.response.tolist(),
                measurements.tolist(),
                initial_flux=case.prior.tolist(),
                measurement_uncertainty=case.uncertainties.tolist(),
                n_samples=50,
                burn_in=10,
                thin=1,
            )
        else:
            if name == "registry_rmle":
                RMLEUnfolder().unfold(
                    measurements,
                    case.response,
                    measurement_uncertainty=case.uncertainties,
                )
            elif name == "registry_ml_seed":
                MLSeedUnfolder().unfold(
                    measurements,
                    case.response,
                    initial_flux=case.prior,
                    measurement_uncertainty=case.uncertainties,
                )
            else:
                GravelUnfolder().unfold(
                    measurements,
                    case.response,
                    initial_flux=case.prior,
                    measurement_uncertainty=case.uncertainties,
                )

@pytest.mark.parametrize("method", ["GRAVEL", "MLEM", "MAXED", "RMLE", "ML_SEED"])
def test_spectrum_unfolder_public_workflow_runs_real_solver(
    monkeypatch: pytest.MonkeyPatch,
    method: str,
):
    def fake_build_response_matrix(self):
        return np.eye(3, dtype=float), ["rx1", "rx2", "rx3"], np.zeros(3)

    monkeypatch.setattr(
        spectrum_workflow,
        "IRDFFDatabase",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(SpectrumUnfolder, "_build_response_matrix", fake_build_response_matrix)

    unfolder = SpectrumUnfolder(
        custom_energy_edges=np.array([1.0, 2.0, 4.0, 8.0]),
        verbose=False,
    )
    unfolder.measurements = [
        FluxWireMeasurement("rx1", activity_Bq=120.0, uncertainty_Bq=6.0),
        FluxWireMeasurement("rx2", activity_Bq=60.0, uncertainty_Bq=4.0),
        FluxWireMeasurement("rx3", activity_Bq=20.0, uncertainty_Bq=2.0),
    ]
    unfolder.set_initial_guess(np.array([80.0, 80.0, 80.0]), source="test")

    result = unfolder.unfold(method=method, max_iterations=200, tolerance=1e-8)

    assert result.method == method
    assert result.flux.shape == (3,)
    assert result.flux_uncertainty.shape == (3,)
    assert np.all(result.flux >= 0.0)
    assert result.predicted_rates.shape == (3,)
    threshold = 5e-2 if method in {"MAXED", "ML_SEED"} else 1e-2
    assert _refold_relative_error(np.eye(3), result.flux, np.array([120.0, 60.0, 20.0])) < threshold


def test_quick_unfold_runs_public_workflow_with_small_synthetic_response(
    monkeypatch: pytest.MonkeyPatch,
):
    def fake_build_response_matrix(self):
        return np.eye(3, dtype=float), ["rx1", "rx2", "rx3"], np.zeros(3)

    monkeypatch.setattr(
        spectrum_workflow,
        "IRDFFDatabase",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        spectrum_workflow,
        "get_flux_wire_energy_groups",
        lambda: np.array([1.0, 2.0, 4.0, 8.0]),
    )
    monkeypatch.setattr(SpectrumUnfolder, "_build_response_matrix", fake_build_response_matrix)

    result = spectrum_workflow.quick_unfold(
        {"rx1": 120.0, "rx2": 60.0, "rx3": 20.0},
        uncertainties={"rx1": 6.0, "rx2": 4.0, "rx3": 2.0},
        initial_spectrum=np.array([80.0, 80.0, 80.0]),
        method="GRAVEL",
        verbose=False,
    )

    assert result.flux.shape == (3,)
    assert result.flux_uncertainty.shape == (3,)
    assert result.reactions_used == ["rx1", "rx2", "rx3"]
    assert _refold_relative_error(np.eye(3), result.flux, np.array([120.0, 60.0, 20.0])) < 1e-2


def test_spectrum_unfolder_allows_ml_seed_initialization_for_gravel_and_rmle(
    monkeypatch: pytest.MonkeyPatch,
):
    def fake_build_response_matrix(self):
        return np.eye(3, dtype=float), ["rx1", "rx2", "rx3"], np.zeros(3)

    monkeypatch.setattr(
        spectrum_workflow,
        "IRDFFDatabase",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(SpectrumUnfolder, "_build_response_matrix", fake_build_response_matrix)

    unfolder = SpectrumUnfolder(
        custom_energy_edges=np.array([1.0, 2.0, 4.0, 8.0]),
        verbose=False,
    )
    unfolder.measurements = [
        FluxWireMeasurement("rx1", activity_Bq=120.0, uncertainty_Bq=6.0),
        FluxWireMeasurement("rx2", activity_Bq=60.0, uncertainty_Bq=4.0),
        FluxWireMeasurement("rx3", activity_Bq=20.0, uncertainty_Bq=2.0),
    ]
    unfolder.set_initial_guess(np.array([10.0, 10.0, 10.0]), source="test")

    gravel_result = unfolder.unfold(
        method="GRAVEL",
        max_iterations=200,
        tolerance=1e-8,
        use_ml_seed=True,
        ml_seed_threshold=0.4,
    )
    rmle_result = unfolder.unfold(
        method="RMLE",
        max_iterations=200,
        tolerance=1e-8,
        use_ml_seed=True,
        ml_seed_threshold=0.4,
    )

    assert gravel_result.metadata["seed_with_ml"] is True
    assert gravel_result.metadata["seed_accepted"] is True
    assert gravel_result.metadata["seed_confidence_score"] >= 0.4
    assert rmle_result.metadata["seed_with_ml"] is True
    assert rmle_result.metadata["seed_accepted"] is True
    assert rmle_result.metadata["seed_confidence_score"] >= 0.4


@pytest.mark.parametrize("method", ["gls", "gravel", "mlem", "maxed", "rmle", "ml_seed"])
def test_cmd_unfold_exposes_values_and_uncertainties_for_all_methods(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    method: str,
):
    written: dict[str, object] = {}

    monkeypatch.setattr(
        app,
        "read_response_bundle",
        lambda _: {
            "matrix": np.eye(3, dtype=float).tolist(),
            "boundaries_eV": [1e-5, 1.0, 10.0, 100.0],
            "reactions": ["r1", "r2", "r3"],
        },
    )
    monkeypatch.setattr(
        app,
        "read_reaction_rates",
        lambda _: {
            "rates": [
                {"rate": 120.0, "uncertainty": 6.0},
                {"rate": 60.0, "uncertainty": 4.0},
                {"rate": 20.0, "uncertainty": 2.0},
            ]
        },
    )
    monkeypatch.setattr(
        app,
        "write_unfold_result",
        lambda output, **kwargs: written.update({"output": output, "payload": kwargs}),
    )

    app.cmd_unfold(
        Namespace(
            rates_file=tmp_path / "rates.json",
            response_file=tmp_path / "response.json",
            prior_flux_file=None,
            method=method,
            prior_uncertainty=0.25,
            prior_cov_model="diagonal",
            prior_correlation_length=1.0,
            max_iters=200,
            tolerance=1e-8,
            chi2_tolerance=1e-10,
            relaxation=0.8,
            floor=1e-20,
            convergence_mode="relative",
            use_ml_seed=True,
            ml_seed_threshold=0.4,
            enforce_nonnegativity=True,
            verbose_solver=False,
            output=tmp_path / f"{method}.json",
            validate=False,
        )
    )

    payload = written["payload"]
    diagnostics = payload["diagnostics"]
    assert payload["method"] == method
    assert len(payload["flux"]) == 3
    assert len(diagnostics["measured_rates"]) == 3
    assert len(diagnostics["predicted_rates"]) == 3
    assert len(diagnostics["predicted_rate_uncertainties"]) == 3
    assert len(diagnostics["flux_uncertainty"]) == 3
    assert len(diagnostics["rate_pulls"]) == 3
    if method in {"gravel", "rmle"}:
        assert diagnostics["seed_confidence_score"] >= 0.4
    if method == "ml_seed":
        assert diagnostics["confidence_score"] >= 0.4


def test_cmd_unfold_rejects_negative_measurements(monkeypatch: pytest.MonkeyPatch, tmp_path):
    monkeypatch.setattr(
        app,
        "read_response_bundle",
        lambda _: {
            "matrix": np.eye(2, dtype=float).tolist(),
            "boundaries_eV": [1e-5, 1.0, 10.0],
            "reactions": ["r1", "r2"],
        },
    )
    monkeypatch.setattr(
        app,
        "read_reaction_rates",
        lambda _: {
            "rates": [
                {"rate": -1.0, "uncertainty": 0.1},
                {"rate": 0.5, "uncertainty": 0.1},
            ]
        },
    )

    with pytest.raises(ValueError, match="must be non-negative"):
        app.cmd_unfold(
            Namespace(
                rates_file=tmp_path / "rates.json",
                response_file=tmp_path / "response.json",
                prior_flux_file=None,
                method="gls",
                prior_uncertainty=0.25,
                prior_cov_model="diagonal",
                prior_correlation_length=1.0,
                max_iters=50,
                tolerance=1e-4,
                chi2_tolerance=0.01,
                relaxation=0.8,
                floor=1e-20,
                convergence_mode="relative",
                use_ml_seed=False,
                ml_seed_threshold=0.6,
                enforce_nonnegativity=True,
                verbose_solver=False,
                output=tmp_path / "bad.json",
                validate=False,
            )
        )


def test_gamma_unfolding_paths_cover_public_entrypoints(
    gamma_case: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
):
    ls_result = rmle_unfolding(
        spectrum=gamma_case.spectrum,
        response=gamma_case.response,
        regularization=RegularizationType.TIKHONOV,
        reg_param=0.01,
    )
    poisson_result = poisson_rmle_unfolding(
        spectrum=gamma_case.spectrum,
        response=gamma_case.response,
        config=PoissonRMLEConfig(
            penalty=PoissonPenalty.NONE,
            alpha=0.0,
            background_mode="none",
            max_iterations=200,
            tolerance=1e-9,
            positivity=True,
            guardrail_max_reduced_chi2=1e6,
            mc_samples=0,
        ),
    )

    gamma_rmle = _fake_pylops_module(monkeypatch)
    wrapper = gamma_rmle.GammaUnfolderRMLE(gamma_case.response.matrix)
    wrapped_result = wrapper.solve_full(gamma_case.counts, lambda_reg=0.01, n_iter=20)

    assert ls_result.solution.shape == gamma_case.true_source.shape
    assert poisson_result.solution.shape == gamma_case.true_source.shape
    assert wrapped_result.unfolded_spectrum.shape == gamma_case.true_source.shape
    assert np.all(ls_result.solution >= 0.0)
    assert np.all(poisson_result.solution >= 0.0)
    assert np.all(wrapped_result.unfolded_spectrum >= 0.0)
    assert ls_result.uncertainty.shape == gamma_case.true_source.shape
    assert poisson_result.uncertainty.shape == gamma_case.true_source.shape
    assert _refold_relative_error(
        gamma_case.response.matrix,
        poisson_result.solution,
        gamma_case.counts,
    ) < 0.3


def test_gamma_unfolding_paths_reject_negative_counts(
    gamma_case: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
):
    bad_counts = gamma_case.counts.copy()
    bad_counts[0] = -1.0
    bad_spectrum = SpectrumData(counts=bad_counts, uncertainty=np.ones_like(bad_counts))

    with pytest.raises(ValueError, match="must be non-negative"):
        rmle_unfolding(bad_spectrum, gamma_case.response)

    with pytest.raises(ValueError, match="must be non-negative"):
        poisson_rmle_unfolding(bad_spectrum, gamma_case.response)

    gamma_rmle = _fake_pylops_module(monkeypatch)
    wrapper = gamma_rmle.GammaUnfolderRMLE(gamma_case.response.matrix)
    with pytest.raises(ValueError, match="must be non-negative"):
        wrapper.solve_regularized(bad_counts, lambda_reg=0.1)


def test_neutron_ibu_wrapper_rejects_negative_inputs(monkeypatch: pytest.MonkeyPatch):
    from fluxforge.unfold import neutron_ibu

    monkeypatch.setattr(neutron_ibu, "_HAS_PYUNFOLD", True)
    monkeypatch.setattr(
        neutron_ibu,
        "_pyunfold_unfold",
        lambda **kwargs: {
            "unfolded": np.array([1.0, 1.0]),
            "stat_err": np.array([0.1, 0.1]),
            "sys_err": np.array([0.1, 0.1]),
        },
    )

    solver = neutron_ibu.NeutronUnfolderIBU()
    rates = ReactionRates(
        values=np.array([-1.0, 2.0]),
        uncertainties=np.array([0.1, 0.2]),
    )
    response = ResponseBundle(
        matrix=np.eye(2),
        energy_bins=np.array([0.0, 1.0, 2.0]),
    )

    with pytest.raises(ValueError, match="must be non-negative"):
        solver.solve(rates, response, prior_flux=np.array([1.0, 1.0]))
