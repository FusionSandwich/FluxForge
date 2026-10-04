"""Known-truth and rejection controls for the opt-in overlap layer."""

import numpy as np
import pytest

from fluxforge.analysis.multiplet_validation import QualificationPolicy, qualify_doublet
from fluxforge.analysis.peakfit import gaussian, gaussian_with_step_bg


X = np.arange(60, 141, dtype=float)
AREAS = np.array([6000.0, 4000.0])


def truth(centers=(98.0, 102.0), areas=AREAS, sigma=2.0, continuum=40.0):
    y = np.full_like(X, continuum)
    for center, area in zip(centers, areas):
        y += gaussian(X, area / (sigma * np.sqrt(2 * np.pi)), center, sigma)
    return y


def fit(y, centers=(98.0, 102.0), **kw):
    args = dict(
        sigma_bounds=(1.8, 2.2),
        resolution_evidence="independent detector control",
        component_evidence=("independent line A", "independent line B"),
        count_basis="synthetic",
        counts_uncertainty=np.sqrt(np.maximum(y, 1.0)),
    )
    args.update(kw)
    return qualify_doublet(X, y, centers, **args)


@pytest.mark.parametrize("centers", [(94.0, 106.0), (98.0, 102.0)])
def test_separated_and_partial_doublet_area_covariance(centers):
    y = truth(centers)
    result = fit(y, centers, counts_uncertainty=np.sqrt(y))
    assert result.status == "qualified", result.reasons
    assert result.component_admission == [True, True]
    np.testing.assert_allclose(result.doublet.component_areas, AREAS, rtol=1e-5)
    assert result.doublet.total_area == pytest.approx(10000.0, rel=1e-5)
    covariance = np.asarray(result.doublet.area_covariance)
    assert covariance[0, 1] != 0
    assert result.doublet.total_area_uncertainty**2 == pytest.approx(covariance.sum())
    assert np.all(np.diag(covariance) > 0)
    assert result.single.diagnostics["residual_lag1"] > 0.3
    assert result.single.diagnostics["goodness_p"] < 0.001


def test_uncertainties_track_bounded_poisson_truth_ensemble():
    rng = np.random.default_rng(240)
    y = truth()
    pulls, totals = [], []
    for _ in range(24):
        result = fit(rng.poisson(y), counts_uncertainty=np.sqrt(y))
        assert result.status == "qualified", result.reasons
        covariance = np.asarray(result.doublet.area_covariance)
        pulls.append(
            (np.asarray(result.doublet.component_areas) - AREAS)
            / np.sqrt(np.diag(covariance))
        )
        totals.append(
            (result.doublet.total_area - AREAS.sum())
            / result.doublet.total_area_uncertainty
        )
    pulls = np.asarray(pulls)
    assert np.all(np.abs(pulls.mean(axis=0)) < 0.6)
    assert np.all((pulls.std(axis=0) > 0.55) & (pulls.std(axis=0) < 1.6))
    assert abs(np.mean(totals)) < 0.6
    assert 0.55 < np.std(totals) < 1.6


def test_unresolved_doublet_never_admitted_even_at_high_counts():
    centers = (99.8, 100.2)
    result = fit(truth(centers), centers)
    assert result.status == "non_identifiable"
    assert not any(result.component_admission)


def test_displaced_single_control_rejects_extra_component():
    y = truth((100.35,), (10000.0,))
    result = fit(y)
    assert result.status in {"single_preferred", "rejected"}
    assert not any(result.component_admission)
    assert result.single.status == "qualified"
    assert abs(result.single.parameters[1] - 0.35) < 1e-4


def test_residual_improvement_never_establishes_nuclide_identity():
    result = fit(truth(), component_evidence=(None, None))
    assert result.doublet.status == "qualified"
    assert result.doublet.diagnostics["delta_bic_single_minus_doublet"] > 10
    assert result.status == "evidence_required"
    assert not any(result.component_admission)


def test_shared_resolution_mismatch_and_unmodelled_third_line_rejected():
    for y in (truth(sigma=3.0), truth() + gaussian(X, 100.0, 116.0, 1.8)):
        result = fit(y)
        assert result.status != "qualified"
        assert not any(result.component_admission)
        assert "structured_or_excess_residuals" in result.doublet.reasons


def test_broad_single_cannot_be_admitted_as_a_doublet():
    # Counterexample from independent Sol review: the constrained single
    # rejects sigma=2.9, while a two-component sigma~2 fit lowers residuals.
    result = fit(truth((100.0,), (2000.0,), sigma=2.9))
    assert result.doublet.status == "qualified"
    assert result.broad_single.status == "qualified"
    assert result.status == "response_ambiguous"
    assert not any(result.component_admission)
    assert result.broad_single.parameters[2] == pytest.approx(2.9, rel=1e-5)


@pytest.mark.parametrize("continuum", ["constant", "linear"])
def test_explicit_continuum_choices(continuum):
    result = fit(truth(), continuum=continuum)
    assert result.status == "qualified"
    assert result.provenance["continuum"] == continuum


def test_evidenced_step_response_uses_existing_helper_and_excludes_step_area():
    y = truth() + gaussian_with_step_bg(X, 0.0, 100.0, 2.0, 15.0, -15.0)
    assert fit(y, continuum="step").status == "unsupported"
    result = fit(
        y, continuum="step", response_evidence="independent Compton-step control"
    )
    assert result.status == "qualified", result.reasons
    np.testing.assert_allclose(result.doublet.component_areas, AREAS, rtol=1e-5)
    assert result.doublet.parameters[-1] == pytest.approx(30.0)
    assert (
        fit(y, response="tail", response_evidence="tail observation").status
        == "unsupported"
    )


def test_overparameterized_and_optimizer_failed_fits_are_explicit():
    result = fit(truth(), policy=QualificationPolicy(min_dof=100))
    assert result.status == "overparameterized"
    failed = fit(truth(), max_evaluations=1)
    assert failed.status == "fit_failed"
    assert failed.doublet.covariance is None
    assert failed.doublet.total_area_uncertainty is None


def test_signed_adjusted_count_covariance_is_used_without_clipping():
    y = truth(continuum=1.0)
    y[0] = -2.0
    covariance = np.eye(len(X)) * 100
    covariance += np.ones_like(covariance) * 2.0
    result = fit(
        y,
        count_basis="background_adjusted",
        counts_covariance=covariance,
        counts_uncertainty=None,
    )
    assert result.provenance["noise_basis"] == "declared_covariance"
    assert result.doublet.residuals[0] < -2.0
    assert (
        fit(y, count_basis="background_adjusted", counts_uncertainty=None).status
        == "invalid_input"
    )
    covariance[0, 1] += 10
    assert (
        fit(y, count_basis="background_adjusted", counts_covariance=covariance).status
        == "invalid_input"
    )


def test_invalid_noise_coordinates_and_unknown_basis_preserve_failure():
    assert fit(truth(), counts_uncertainty=np.zeros_like(X)).status == "invalid_input"
    assert fit(truth(), count_basis="vendor_comparison").status == "unsupported"
    assert fit(truth(), resolution_evidence="").status == "invalid_input"
    assert fit(truth(), centers=(61.0, 62.0)).status == "invalid_input"
    result = qualify_doublet(
        X * 0.5,
        truth(),
        (49, 51),
        sigma_bounds=(0.9, 1.1),
        resolution_evidence="control",
    )
    assert result.status == "invalid_input"


def test_low_count_wls_is_not_admitted():
    result = fit(truth(continuum=0.1))
    assert result.status == "rejected"
    assert "low_count_wls_not_qualified" in result.reasons


def test_measured_raw_counts_cannot_use_fractional_expectations_or_tiny_noise():
    assert fit(truth(), count_basis="raw_sample").status == "invalid_input"
    measured = np.random.default_rng(240).poisson(truth())
    result = fit(
        measured, count_basis="raw_sample", counts_uncertainty=np.full_like(X, 0.1)
    )
    assert result.status == "invalid_input"
    assert "raw_sample_variance_below_observed_poisson_noise" in result.reasons
    result = fit(measured, count_basis="raw_sample", counts_uncertainty=None)
    assert result.status == "qualified", result.reasons
    assert min(np.sqrt(np.diag(result.doublet.area_covariance))) > 50
    assert fit(truth(), component_evidence=None).status == "invalid_input"
    covariance = np.outer(np.sqrt(measured), np.sqrt(measured)) * 0.9
    np.fill_diagonal(covariance, measured)
    result = fit(
        measured,
        count_basis="raw_sample",
        counts_uncertainty=None,
        counts_covariance=covariance,
    )
    assert result.status == "invalid_input"
    assert "raw_covariance_undercuts_independent_poisson_noise" in result.reasons


def test_undersampled_response_and_blank_resolution_evidence_do_not_qualify():
    assert fit(truth(), sigma_bounds=(0.5, 0.7)).status == "unsupported"
    assert fit(truth(), resolution_evidence="  ").status == "invalid_input"
