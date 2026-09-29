"""Synthetic acceptance checks for the source-bound group-integral GLS API."""

import json
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fluxforge.analysis.physical_gls import (
    MonitorRow, SourceBinding, unfold_gls_physical,
)
from fluxforge.analysis.flux_unfold import FluxWireReaction, unfold_gls


def fixture_inputs():
    rows = [
        MonitorRow("co_bare", "Co-1", "bare", "Co-59(n,g)Co-60", "Co-60"),
        MonitorRow("co_cd", "Co-Cd-1", "Cd", "Co-59(n,g)Co-60", "Co-60"),
        MonitorRow("ni_bare", "Ni-1", "bare", "Ni-58(n,p)Co-58", "Co-58"),
    ]
    # Deliberately asymmetric groups; the two Co rows have distinct responses.
    response = np.array([[2.0, 1.0], [0.1, 1.0], [0.5, 2.0]]) * 1e-24
    truth = np.array([3.0, 4.0]) * 1e12
    sources = {
        key: SourceBinding(f"synthetic://{key}", "a" * 64, unit)
        for key, unit in {
            "row_identities": "identity", "energy_edges": "eV",
            "rates": "reactions/target_atom/s", "response": "cm2",
            "prior": "n/cm2/s", "prior_covariance": "(n/cm2/s)^2",
            "observation_covariance": "(reactions/target_atom/s)^2",
        }.items()
    }
    return dict(
        rows=rows, energy_edges_eV=np.array([1.0, 3.0, 20.0]),
        measured_rates=response @ truth, response_matrix=response,
        prior_flux=np.array([1.0, 8.0]) * 1e12,
        prior_covariance=np.diag([100.0, 100.0]) * 1e24,
        observation_covariance=np.diag([0.01, 0.01, 0.01]) * 1e-24,
        sources=sources, source_commit="synthetic-test-commit",
    ), truth


def test_known_truth_wrong_prior_correlated_covariance_and_receipt():
    inputs, truth = fixture_inputs()
    inputs["observation_covariance"][0, 1] = 0.002e-24
    inputs["observation_covariance"][1, 0] = 0.002e-24
    result = unfold_gls_physical(**inputs)
    np.testing.assert_allclose(result.flux, truth, rtol=5e-3)
    np.testing.assert_allclose(result.predicted_rates, inputs["measured_rates"], rtol=5e-3)
    assert result.rows[0].reaction_id == result.rows[1].reaction_id
    assert result.response_rank == 2 and result.response_nullity == 0
    assert result.prior_innovation_chi2 > result.postfit_observation_chi2
    receipt = result.receipt()
    assert receipt["prior_flux"] == inputs["prior_flux"].tolist()
    assert receipt["rows"][1]["cover"] == "Cd"
    assert receipt["scientific_admission"] is False
    assert "not verified" in receipt["source_qualification"]
    assert len(receipt["input_hashes"]["response"]) == 64
    json.dumps(receipt)


def test_rate_unit_scale_invariance_without_absolute_floor():
    inputs, _ = fixture_inputs()
    baseline = unfold_gls_physical(**inputs)
    factor = 1e-12
    scaled = dict(inputs)
    scaled["response_matrix"] = inputs["response_matrix"] * factor
    scaled["measured_rates"] = inputs["measured_rates"] * factor
    scaled["observation_covariance"] = inputs["observation_covariance"] * factor**2
    adjusted = unfold_gls_physical(**scaled)
    np.testing.assert_allclose(adjusted.flux, baseline.flux, rtol=1e-12)
    np.testing.assert_allclose(adjusted.posterior_covariance,
                               baseline.posterior_covariance, rtol=1e-12)
    assert adjusted.solver_flags["absolute_covariance_floor"] is False


def test_group_subdivision_conserves_forward_fold():
    inputs, _ = fixture_inputs()
    response = inputs["response_matrix"]
    flux = inputs["prior_flux"]
    split_flux = np.array([flux[0] * 0.25, flux[0] * 0.75, flux[1]])
    split_response = np.column_stack((response[:, 0], response[:, 0], response[:, 1]))
    np.testing.assert_allclose(split_response @ split_flux, response @ flux, rtol=1e-15)
    inputs["energy_edges_eV"] = np.array([1.0, 1.5, 3.0, 20.0])
    inputs["response_matrix"] = split_response
    inputs["prior_flux"] = split_flux
    inputs["prior_covariance"] = np.eye(3) * 1e24
    result = unfold_gls_physical(**inputs)
    assert result.response_rank == 2 and result.response_nullity == 1
    assert np.all(result.flux_uncertainty > 0)


def test_holdout_is_sealed_from_fit_and_uses_its_own_residual():
    inputs, _ = fixture_inputs()
    inputs["holdout_ids"] = ["ni_bare"]
    first = unfold_gls_physical(**inputs)
    changed = dict(inputs)
    changed["measured_rates"] = inputs["measured_rates"].copy()
    changed["measured_rates"][2] *= 100
    second = unfold_gls_physical(**changed)
    np.testing.assert_array_equal(first.flux, second.flux)
    np.testing.assert_array_equal(first.holdout_predictions, second.holdout_predictions)
    assert first.holdout_residuals[0] != second.holdout_residuals[0]
    assert first.fit_ids == ("co_bare", "co_cd")
    assert first.holdout_ids == ("ni_bare",)


def test_rank_deficiency_keeps_prior_uncertainty_in_unseen_group():
    inputs, _ = fixture_inputs()
    inputs["response_matrix"] = inputs["response_matrix"].copy()
    inputs["response_matrix"][:, 1] = 0
    inputs["prior_covariance"] = np.diag([1.0, 9.0]) * 1e24
    first = unfold_gls_physical(**inputs)
    inputs["prior_flux"] = np.array([1.0, 80.0]) * 1e12
    second = unfold_gls_physical(**inputs)
    assert first.response_rank == 1 and first.response_nullity == 1
    assert first.flux[1] == 8e12 and second.flux[1] == 80e12
    np.testing.assert_allclose(first.posterior_covariance[1, 1], 9e24)
    assert first.effective_residual_df > 0


def test_reject_duplicate_identity_and_condition_correlated_holdout():
    inputs, _ = fixture_inputs()
    inputs["rows"][1] = inputs["rows"][0]
    with pytest.raises(ValueError, match="unique"):
        unfold_gls_physical(**inputs)
    inputs, _ = fixture_inputs()
    inputs["holdout_ids"] = ["ni_bare"]
    inputs["observation_covariance"][0, 2] = 1e-27
    inputs["observation_covariance"][2, 0] = 1e-27
    correlated = unfold_gls_physical(**inputs)
    from fluxforge.analysis.holdout_validation import predict_holdouts
    expected = predict_holdouts(
        inputs["response_matrix"], inputs["measured_rates"],
        inputs["prior_flux"], inputs["prior_covariance"],
        inputs["observation_covariance"], [2],
    )
    np.testing.assert_allclose(correlated.holdout_predictions, expected.holdout_mean)
    np.testing.assert_allclose(
        correlated.holdout_predictive_covariance, expected.holdout_covariance
    )
    changed = dict(inputs)
    changed["measured_rates"] = inputs["measured_rates"].copy()
    changed["measured_rates"][2] *= 2
    second = unfold_gls_physical(**changed)
    np.testing.assert_array_equal(correlated.flux, second.flux)


def test_physical_gls_correlated_holdout_matches_joint_gaussian() -> None:
    inputs, _ = fixture_inputs()
    inputs["holdout_ids"] = ["ni_bare"]
    rates = inputs["measured_rates"].copy()
    rates[:2] *= [1.03, 0.98]
    rates[2] *= 1.07
    inputs["measured_rates"] = rates
    # Shared detector calibration induces fit/holdout covariance.
    shared = np.outer(0.03 * rates, 0.03 * rates)
    inputs["observation_covariance"] += shared
    result = unfold_gls_physical(**inputs)

    a = inputs["response_matrix"]
    p = inputs["prior_flux"]
    s = a @ inputs["prior_covariance"] @ a.T + inputs["observation_covariance"]
    fit, hold = [0, 1], [2]
    fit_innovation = rates[fit] - a[fit] @ p
    expected_mean = a[hold] @ p + s[np.ix_(hold, fit)] @ np.linalg.solve(
        s[np.ix_(fit, fit)], fit_innovation
    )
    expected_cov = s[np.ix_(hold, hold)] - s[np.ix_(hold, fit)] @ np.linalg.solve(
        s[np.ix_(fit, fit)], s[np.ix_(fit, hold)]
    )
    np.testing.assert_allclose(result.holdout_predictions, expected_mean, rtol=1e-9)
    np.testing.assert_allclose(result.holdout_predictive_covariance, expected_cov, rtol=1e-8)
    changed = dict(inputs)
    changed["measured_rates"] = rates.copy()
    changed["measured_rates"][2] *= 10
    np.testing.assert_array_equal(result.flux, unfold_gls_physical(**changed).flux)


def test_correlated_response_error_contributes_to_conditional_holdout() -> None:
    inputs, _ = fixture_inputs()
    inputs["holdout_ids"] = ["ni_bare"]
    rates = inputs["measured_rates"]
    response_error = np.outer(0.04 * rates, 0.04 * rates)
    inputs["response_error_covariance"] = response_error
    inputs["sources"]["response_error_covariance"] = SourceBinding(
        "synthetic://shared-nuclear-data", "b" * 64,
        "(reactions/target_atom/s)^2",
    )
    result = unfold_gls_physical(**inputs)
    a = inputs["response_matrix"]
    p = inputs["prior_flux"]
    s = (a @ inputs["prior_covariance"] @ a.T
         + inputs["observation_covariance"] + response_error)
    fit, hold = [0, 1], [2]
    expected_cov = s[np.ix_(hold, hold)] - s[np.ix_(hold, fit)] @ np.linalg.solve(
        s[np.ix_(fit, fit)], s[np.ix_(fit, hold)]
    )
    np.testing.assert_allclose(result.holdout_predictive_covariance, expected_cov)


def test_response_error_increases_uncertainty_and_requires_source():
    inputs, _ = fixture_inputs()
    baseline = unfold_gls_physical(**inputs)
    inputs["response_error_covariance"] = np.eye(3) * 0.2e-24
    with pytest.raises(ValueError, match="source bindings"):
        unfold_gls_physical(**inputs)
    inputs["sources"]["response_error_covariance"] = SourceBinding(
        "synthetic://response-error", "b" * 64,
        "(reactions/target_atom/s)^2",
    )
    uncertain = unfold_gls_physical(**inputs)
    assert np.all(uncertain.flux_uncertainty > baseline.flux_uncertainty)


def test_legacy_placeholder_exposes_actual_prior_and_diagnostics():
    reaction = FluxWireReaction(
        sample_id="Co-Cd-1", reaction_id="Co-59(n,g)Co-60",
        isotope="Co60", activity_bq=1.0,
        reaction_rate=2e-12, reaction_rate_unc=1e-13,
    )
    result = unfold_gls([reaction], n_groups=3)
    assert result.diagnostic_only
    assert result.prior_flux.shape == (3,)
    assert result.response_matrix.shape == (1, 3)
    assert result.response_rank == 1
    np.testing.assert_allclose(result.predicted_rates,
                               result.response_matrix @ result.flux)
    np.testing.assert_allclose(result.postfit_residuals,
                               result.measured_rates - result.predicted_rates)
    assert result.postfit_observation_chi2 >= 0


def test_reject_tiny_negative_variance_and_malformed_source_hash():
    inputs, _ = fixture_inputs()
    inputs["prior_covariance"] = np.diag([1.0, -1e-13]) * 1e24
    with pytest.raises(ValueError, match="positive semidefinite"):
        unfold_gls_physical(**inputs)
    with pytest.raises(ValueError, match="SHA256"):
        SourceBinding("synthetic://bad", "aa" * 31 + "  ", "cm2")


def test_row_identity_hash_is_unambiguous_with_delimiters():
    inputs, _ = fixture_inputs()
    first = unfold_gls_physical(**inputs)
    modified = dict(inputs)
    modified["rows"] = list(inputs["rows"])
    original = modified["rows"][0]
    modified["rows"][0] = MonitorRow(
        original.observation_id + "\t", original.sample_id + "\n",
        original.cover, original.reaction_id, original.product_id,
    )
    second = unfold_gls_physical(**modified)
    assert first.input_hashes["row_identities"] != second.input_hashes["row_identities"]


def test_zero_psd_components_and_positive_definite_total():
    inputs, _ = fixture_inputs()
    inputs["prior_covariance"] = np.zeros((2, 2))
    inputs["observation_covariance"] = np.diag([1.0, 0.0, 1.0]) * 1e-24
    inputs["response_error_covariance"] = np.diag([0.0, 1.0, 0.0]) * 1e-24
    inputs["sources"]["response_error_covariance"] = SourceBinding(
        "synthetic://response-error", "b" * 64,
        "(reactions/target_atom/s)^2",
    )
    result = unfold_gls_physical(**inputs)
    np.testing.assert_array_equal(result.flux, inputs["prior_flux"])
    np.testing.assert_array_equal(result.posterior_covariance, np.zeros((2, 2)))
    inputs["response_error_covariance"] = np.zeros((3, 3))
    with pytest.raises(ValueError, match="Total fit covariance"):
        unfold_gls_physical(**inputs)


def test_legacy_workflow_serializes_actual_gls_prior(tmp_path, monkeypatch):
    from fluxforge.examples import rafm_workflow as workflow

    reaction = FluxWireReaction(
        sample_id="Co-bare-1", reaction_id="Co-59(n,g)Co-60",
        isotope="Co60", activity_bq=1.0,
        reaction_rate=2e-12, reaction_rate_unc=1e-13,
    )

    class StubIterative:
        def __init__(self, **_):
            self.energy_edges = np.array([1.0, 2.0])

        def add_reaction(self, **_):
            pass

        def set_initial_guess(self, *_args, **_kwargs):
            pass

        def unfold(self, method):
            return SimpleNamespace(method=method, metadata={}, converged=False)

    monkeypatch.setattr(workflow, "SpectrumUnfolder", StubIterative)
    monkeypatch.setattr(workflow, "parse_prior_spectrum",
                        lambda _path, edges: np.full(len(edges) - 1, 999.0))
    monkeypatch.setattr(workflow, "method_overlay_plot", lambda *_: None)
    for name in ("plot_spectrum_comparison", "plot_spectrum_uncertainty_bands",
                 "plot_measured_vs_predicted", "plot_response_matrix"):
        monkeypatch.setattr(workflow, name, lambda *_args, **_kwargs: None)
    original_save = workflow.save_unfolding_artifacts

    def save_gls_only(result, prior_flux, output_root, reference_label=None):
        if result.method == "GLS":
            original_save(result, prior_flux, output_root, reference_label)

    monkeypatch.setattr(workflow, "save_unfolding_artifacts", save_gls_only)
    results = workflow.run_flux_wire_unfolding(
        [reaction], Path("unused.csv"), tmp_path / "unfolding"
    )
    payload = json.loads((tmp_path / "unfolding" / "gls.json").read_text())
    actual_prior = unfold_gls([reaction], n_groups=20).prior_flux
    np.testing.assert_array_equal(payload["metadata"]["prior_flux"], actual_prior)
    assert payload["metadata"]["prior_flux_sha256"] == hashlib.sha256(
        np.asarray(actual_prior, dtype="<f8").tobytes()
    ).hexdigest()
    assert payload["initial_guess_source"] == "internal equal-lethargy prior"
    assert payload["chi_squared"] is None
    assert payload["metadata"]["diagnostic_only"] is True
    review = json.loads((tmp_path / "unfolding" / "method_overlay_review.json").read_text())
    assert review["admitted"] is False
    assert not (tmp_path / "plots" / "unfolding" / "method_overlay.png").exists()
    np.testing.assert_array_equal(results["GLS"].response_matrix,
                                  results["GLS"].metadata["response_matrix"])


def test_known_truth_cross_method_forward_fold_on_same_operator():
    """FluxForge iterations and independent SciPy NNLS see identical R and y."""
    from scipy.optimize import nnls
    from fluxforge.solvers.iterative import gravel, mlem

    inputs, truth = fixture_inputs()
    response = inputs["response_matrix"]
    rates = inputs["measured_rates"]
    prior = inputs["prior_flux"]
    gls_result = unfold_gls_physical(**inputs)
    gravel_result = gravel(
        response.tolist(), rates.tolist(), initial_flux=prior.tolist(),
        measurement_uncertainty=np.sqrt(np.diag(inputs["observation_covariance"])).tolist(),
        max_iters=1000, tolerance=1e-10,
    )
    mlem_result = mlem(
        response.tolist(), rates.tolist(), initial_flux=prior.tolist(),
        max_iters=1000, tolerance=1e-10, chi2_tolerance=0.0,
    )
    # SciPy's Lawson-Hanson NNLS is an external numerical baseline, not SpecKit.
    nnls_scaled, _ = nnls(response / 1e-24, rates / 1e-12)
    solutions = {
        "GLS": gls_result.flux,
        "GRAVEL": np.asarray(gravel_result.flux),
        "MLEM": np.asarray(mlem_result.flux),
        "SciPy_NNLS": nnls_scaled * 1e12,
    }
    assert gls_result.response_rank == np.linalg.matrix_rank(response)
    for name, flux in solutions.items():
        np.testing.assert_allclose(response @ flux, rates, rtol=5e-3, err_msg=name)
        np.testing.assert_allclose(flux, truth, rtol=5e-3, err_msg=name)


def test_provisional_ni57_is_excluded_with_its_rate_preserved(tmp_path):
    from fluxforge.examples.rafm_workflow import run_flux_wire_unfolding

    reaction = FluxWireReaction(
        sample_id="Ni-RAFM-1_25cm", reaction_id="Ni-58(n,2n)Ni-57",
        isotope="Ni57", activity_bq=147.817,
        reaction_rate=2e-12, reaction_rate_unc=3e-13,
    )
    assert run_flux_wire_unfolding([reaction], Path("unused.csv"), tmp_path) == {}
    review = json.loads((tmp_path / "input_admission_review.json").read_text())
    excluded = review["excluded_reactions"]
    assert len(excluded) == 1
    assert excluded[0]["admission_status"] == "provisional_excluded"
    assert excluded[0]["reaction_rate_per_atom_s"] == 2e-12
    assert excluded[0]["reaction_rate_unc_per_atom_s"] == 3e-13


def test_method_overlay_rejects_diagnostic_gls(tmp_path):
    from fluxforge.examples.rafm_workflow import adapt_unfold_result, method_overlay_plot

    reaction = FluxWireReaction(
        sample_id="Co-1", reaction_id="Co-59(n,g)Co-60",
        isotope="Co60", activity_bq=1.0, reaction_rate=2e-12,
    )
    result = adapt_unfold_result(
        "GLS", np.array([1.0, 2.0]), np.array([1.0]),
        np.array([0.1]), [reaction], np.array([[1.0]]),
        np.array([1.0]), 0.0,
    )
    with pytest.raises(ValueError, match="physically admitted"):
        method_overlay_plot({"GLS": result}, tmp_path / "overlay.png")
    assert not (tmp_path / "overlay.png").exists()
