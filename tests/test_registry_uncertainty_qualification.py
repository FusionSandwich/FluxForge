"""Estimator qualification and row-unit invariance counterexamples."""

from dataclasses import asdict
import json

import numpy as np
import pytest

from fluxforge.unfolding import (
    GravelUnfolder,
    MaxedUnfolder,
    MLSeedUnfolder,
    RMLEUnfolder,
)
from fluxforge.unfolding.base import estimate_unfolding_uncertainties
from fluxforge.unfolding.base import unavailable_uncertainty_metadata
from fluxforge.io.artifacts import write_unfold_result, read_unfold_result
from fluxforge.core.schemas import validate_or_raise


@pytest.mark.parametrize("scale", [1e-120, 1e-12, 1, 1e12, 1e120])
def test_linear_identity_uncertainty_is_invariant_to_measurement_row_units(scale):
    result = estimate_unfolding_uncertainties(
        np.eye(2) * scale, measurement_uncertainty=np.array([0.1, 0.2]) * scale
    )
    np.testing.assert_allclose(result, [0.1, 0.2], rtol=1e-14, atol=0)


def test_linear_overdetermined_estimator_is_invariant_to_individual_row_units():
    response = np.array([[1.0, 2.0], [2.0, 1.0], [1.0, 1.0]])
    sigma = np.array([0.1, 0.2, 0.3])
    precision = np.diag(1 / sigma**2)
    expected = np.sqrt(np.diag(np.linalg.inv(response.T @ precision @ response)))
    units = np.array([1e-12, 1e12, 1e-4])
    result = estimate_unfolding_uncertainties(
        response * units[:, None], measurement_uncertainty=sigma * units
    )
    np.testing.assert_allclose(result, expected, rtol=1e-13, atol=0)


def test_rank_metadata_uses_error_weighted_operator_after_row_unit_changes():
    units = np.array([1e-12, 1e12])
    metadata = unavailable_uncertainty_metadata(
        "MAXED",
        np.diag(units),
        measurement_uncertainty=units * [0.1, 0.1],
        converged=True,
    )
    assert metadata["response_rank"] == 2
    assert "rank deficient" not in metadata["uncertainty_unavailable_reason"]
    assert metadata["uncertainty_qualified"] is False
    missing = unavailable_uncertainty_metadata("MAXED", np.diag(units), converged=True)
    assert missing["response_rank"] is None
    assert missing["response_rank_basis"] == "unavailable"


@pytest.mark.parametrize(
    "response", [np.ones((2, 2)), np.ones((1, 2)), np.zeros((2, 2))]
)
def test_linear_rank_deficiency_does_not_report_zero_nullspace_uncertainty(response):
    with pytest.raises(ValueError, match="Rank-deficient"):
        estimate_unfolding_uncertainties(
            response, measurement_uncertainty=np.ones(len(response))
        )


@pytest.mark.parametrize(
    "kwargs", [{}, {"measured": np.ones(2)}, {"measurement_uncertainty": [0, 0.1]}]
)
def test_linear_utility_does_not_guess_missing_or_exact_constraint_variance(kwargs):
    with pytest.raises(ValueError):
        estimate_unfolding_uncertainties(np.eye(2), **kwargs)


@pytest.mark.parametrize(
    "factory,kwargs",
    [
        (
            GravelUnfolder,
            {"max_iterations": 1, "tolerance": 0.0, "chi2_tolerance": 0.0},
        ),
        (MaxedUnfolder, {"max_iterations": 1}),
        (MLSeedUnfolder, {"confidence_threshold": 2.0}),
    ],
)
def test_nonconverged_registry_estimators_have_unavailable_uncertainty(factory, kwargs):
    result = factory().unfold(
        np.array([3.0, 7.0]),
        np.eye(2),
        initial_flux=np.ones(2),
        measurement_uncertainty=np.ones(2),
        **kwargs,
    )
    assert result.converged is False
    assert result.uncertainties is None
    assert factory.definition().supports_uncertainties is False
    assert result.parameters_used["uncertainty_qualified"] is False
    assert (
        "did not converge" in result.parameters_used["uncertainty_unavailable_reason"]
    )
    exported = asdict(result)
    exported["flux"] = result.flux.tolist()
    assert exported["uncertainties"] is None


@pytest.mark.parametrize("factory", [GravelUnfolder, MaxedUnfolder, MLSeedUnfolder])
def test_rank_deficient_registry_results_preserve_unavailability_and_reason(factory):
    result = factory().unfold(
        np.array([2.0, 2.0]),
        np.ones((2, 3)),
        initial_flux=np.ones(3),
        measurement_uncertainty=np.ones(2),
        max_iterations=10,
    )
    assert result.uncertainties is None
    assert result.parameters_used["response_rank"] == 1
    assert "rank deficient" in result.parameters_used["uncertainty_unavailable_reason"]


def test_unavailable_export_roundtrips_null_with_reason(tmp_path):
    path = tmp_path / "unavailable.json"
    write_unfold_result(
        path,
        boundaries_eV=[1, 2, 3],
        reactions=["a", "b"],
        flux=[3, 7],
        covariance=None,
        chi2=0.0,
        method="maxed",
        diagnostics={
            "uncertainty_status": "unavailable",
            "uncertainty_unavailable_reason": "Estimator-specific propagation unavailable",
        },
    )
    payload = read_unfold_result(path)
    validate_or_raise(payload)
    assert payload["covariance"] is None
    assert json.loads(path.read_text())["covariance"] is None
    for diagnostics in ({}, ["malformed"], "malformed", None):
        missing_reason = dict(payload, diagnostics=diagnostics)
        with pytest.raises(ValueError, match="recorded reason"):
            validate_or_raise(missing_reason)
    with pytest.raises(ValueError, match="recorded reason"):
        write_unfold_result(
            path,
            boundaries_eV=[1, 2, 3],
            reactions=["a", "b"],
            flux=[3, 7],
            covariance=None,
            chi2=0.0,
            method="maxed",
        )


def test_report_csv_keeps_missing_uncertainty_blank_and_explains_it(tmp_path):
    from argparse import Namespace
    import csv
    from fluxforge.cli.app import cmd_report

    path = tmp_path / "unfold.json"
    reason = "Estimator-specific propagation unavailable"
    write_unfold_result(
        path,
        boundaries_eV=[1, 2, 3],
        reactions=["a", "b"],
        flux=[3, 7],
        covariance=None,
        chi2=0.0,
        method="maxed",
        diagnostics={
            "uncertainty_status": "unavailable",
            "uncertainty_unavailable_reason": reason,
        },
    )
    cmd_report(
        Namespace(
            spectrum_file=None,
            peaks_file=None,
            lines_file=None,
            rates_file=None,
            unfold_file=path,
            validation_file=None,
            output=tmp_path / "report.json",
            validate=True,
        )
    )
    with (tmp_path / "report_tables/unfold_flux_groups.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert all(row["flux_uncertainty"] == "" for row in rows)
    assert all(row["uncertainty_unavailable_reason"] == reason for row in rows)
    assert reason in (tmp_path / "report.txt").read_text(encoding="utf-8")


def test_rmle_count_identity_does_not_publish_arbitrary_percentage():
    counts = np.array([25.0, 400.0])
    result = RMLEUnfolder().unfold(
        counts,
        np.eye(2),
        initial_flux=counts,
        measurement_uncertainty=np.sqrt(counts),
        regularization_strength=0,
        auto_regularization=False,
        regularization_type="l2",
        max_iterations=20,
    )
    np.testing.assert_array_equal(result.flux, counts)
    assert result.converged and result.iterations == 0
    assert result.uncertainties is None
    assert result.parameters_used["uncertainty_qualified"] is False
    assert RMLEUnfolder.definition().supports_uncertainties is False
    assert (
        "percentage" in result.parameters_used["backend_uncertainty_unavailable_reason"]
    )


@pytest.mark.parametrize("samples", [0, 1, 4])
def test_poisson_count_uncertainty_is_unavailable_for_default_and_unqualified_mc(
    samples,
):
    from fluxforge.solvers.rmle import (
        SpectrumData,
        ResponseMatrix,
        PoissonRMLEConfig,
        PoissonPenalty,
        poisson_rmle_unfolding,
    )

    counts = np.array([25.0, 400.0])
    result = poisson_rmle_unfolding(
        SpectrumData(counts=counts),
        ResponseMatrix(matrix=np.eye(2)),
        PoissonRMLEConfig(
            alpha=0,
            penalty=PoissonPenalty.NONE,
            initial_solution=counts,
            mc_samples=samples,
            random_seed=42,
            max_iterations=20,
        ),
    )
    np.testing.assert_array_equal(result.solution, counts)
    assert result.uncertainty is None and result.covariance is None
    assert result.diagnostics["uncertainty_qualified"] is False
    assert result.diagnostics["uncertainty_unavailable_reason"]
    if samples:
        assert result.diagnostics["mc_replicates_completed"] == samples


def test_poisson_failed_optimizer_fallback_does_not_republish_linear_proxy(monkeypatch):
    from fluxforge.solvers import rmle

    def fail(*args, **kwargs):
        raise RuntimeError("forced optimizer failure")

    monkeypatch.setattr(rmle.optimize, "minimize", fail)
    result = rmle.poisson_rmle_unfolding(
        rmle.SpectrumData(counts=np.array([25.0, 400.0])),
        rmle.ResponseMatrix(matrix=np.eye(2)),
        rmle.PoissonRMLEConfig(alpha=0),
    )
    assert result.diagnostics["poisson_fallback"] is True
    assert result.uncertainty is None and result.covariance is None
    assert "fallback" in result.diagnostics["uncertainty_unavailable_reason"]


def test_gaussian_failed_covariance_does_not_fabricate_ten_percent(monkeypatch):
    from fluxforge.solvers import rmle

    def fail(*args, **kwargs):
        raise np.linalg.LinAlgError("forced covariance failure")

    monkeypatch.setattr(rmle.linalg, "inv", fail)
    result = rmle.rmle_unfolding(
        rmle.SpectrumData(counts=np.array([25.0, 400.0])),
        rmle.ResponseMatrix(matrix=np.eye(2)),
        regularization=rmle.RegularizationType.NONE,
    )
    assert result.uncertainty is None and result.covariance is None
    assert result.diagnostics["uncertainty_status"] == "unavailable"
    assert "percentage" in result.diagnostics["uncertainty_unavailable_reason"]


def test_gaussian_nonfinite_covariance_is_unavailable_despite_finite_flux():
    from fluxforge.solvers import rmle

    with np.errstate(over="ignore", invalid="ignore"):
        result = rmle.rmle_unfolding(
            rmle.SpectrumData(
                counts=np.array([1e200, 2e200]), uncertainty=np.array([1e200, 1e200])
            ),
            rmle.ResponseMatrix(matrix=np.eye(2) * 1e200),
            regularization=rmle.RegularizationType.NONE,
            param_selection=rmle.ParameterSelection.FIXED,
            enforce_positivity=False,
        )
    np.testing.assert_allclose(result.solution, [1, 2])
    assert result.uncertainty is None and result.covariance is None
    assert result.diagnostics["uncertainty_status"] == "unavailable"
