"""Continuation regressions for #211/#195/#199/#203/#204/#187."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from tests.test_monitor_response import EDGES, mini_db, _unfolder
from fluxforge.physics.monitor_response import CoverLayer, MonitorResponseSpec
from fluxforge.workflows.spectrum_unfolding import FluxWireMeasurement, SpectrumUnfolder
from fluxforge.analysis.holdout_validation import predict_holdouts
from fluxforge.analysis.physical_gls import SourceBinding, unfold_gls_physical
from fluxforge.uncertainty.reaction_rate_budget import (
    RateUncertaintyBudget,
    UncertaintyComponent,
    REQUIRED_COMPONENTS,
    rate_covariance,
)


def covered(mini_db):
    u = _unfolder(mini_db)
    s = MonitorResponseSpec(
        "co-cd", "co-cd", "Co-59(n,g)Co-60", cover=CoverLayer("Cd", 0.0508)
    )
    u.add_reaction(
        s.reaction, 1.0, 0.05, rate_per_atom=1e-12, cover="Cd", response_spec=s
    )
    return u


def test_warm_cache_rebuilds_changed_physics(mini_db):
    u = covered(mini_db)
    old = u._build_response_matrix()[0].copy()
    m = u.measurements[0]
    m.response_spec = replace(
        m.response_spec, cover=replace(m.response_spec.cover, density_g_cm3=4.325)
    )
    new = u._build_response_matrix()[0]
    assert not np.array_equal(old, new)
    assert (
        u._response_row_metadata[0]["cover"]["number_density_per_cm3"]
        == m.response_spec.cover.number_density_per_cm3
    )


def test_warm_cache_revalidates_missing_cover(mini_db):
    u = covered(mini_db)
    u._build_response_matrix()
    u.measurements[0].response_spec = None
    with pytest.raises(ValueError, match="CoverLayer"):
        u.unfold(method="MLEM", max_iterations=1)


def test_warm_cache_binds_grid_and_evaluated_values(mini_db):
    u = covered(mini_db)
    old = u._build_response_matrix()[0].copy()
    u.energy_edges = EDGES.copy()
    u.energy_edges[2] = 0.4
    grid_row = u._build_response_matrix()[0].copy()
    assert not np.array_equal(old, grid_row)
    mini_db.get_cross_section(u.measurements[0].reaction).cross_sections *= 2
    np.testing.assert_allclose(u._build_response_matrix()[0], grid_row * 2, rtol=1e-12)


def test_rate_only_change_preserves_operator_cache(mini_db, monkeypatch):
    u = covered(mini_db)
    row = u._build_response_matrix()[0]
    monkeypatch.setattr(
        "fluxforge.workflows.spectrum_unfolding.build_monitor_response",
        lambda *a, **k: pytest.fail("Rate-only change rebuilt operator"),
    )
    u.measurements[0].rate_per_atom = 2e-12
    assert u._build_response_matrix()[0] is row


@pytest.mark.parametrize("abundance", [1.0, 0.999999, 0.99])
def test_explicit_enrichment_preserved(abundance):
    m = FluxWireMeasurement(
        "Ni-58(n,p)Co-58",
        1.0,
        sample_mass_g=1.0,
        irradiation_time=3600.0,
        isotope_abundance=abundance,
    )
    assert m.effective_isotope_abundance == abundance


def synthetic(monkeypatch):
    monkeypatch.setattr(
        "fluxforge.workflows.spectrum_unfolding.IRDFFDatabase",
        lambda **k: SimpleNamespace(),
    )
    u = SpectrumUnfolder(custom_energy_edges=np.array([1.0, 2.0, 3.0]), verbose=False)
    response = np.array([[1.0, 0.2], [0.3, 1.0]])
    prior = np.array([1.0, 2.0])
    rates = response @ prior
    for i, r in enumerate(rates):
        u.add_reaction(str(i), r, r * 0.1, rate_per_atom=r, sample_id=str(i))
    u._build_response_matrix = lambda: (response, ["0", "1"], np.zeros_like(response))
    u.set_initial_guess(prior)
    return u


def test_unconverged_draws_do_not_qualify_converged_estimator(monkeypatch):
    u = synthetic(monkeypatch)
    result = u.unfold(
        method="MLEM",
        max_iterations=1,
        tolerance=1e-10,
        uncertainty_method="monte_carlo",
        n_uncertainty_samples=8,
        uncertainty_seed=42,
    )
    assert result.converged
    assert np.all(np.isnan(result.flux_uncertainty))
    assert result.metadata["flux_uncertainty_attempted"] == 8
    assert result.metadata["flux_uncertainty_converged_draws"] == 0
    assert result.metadata["flux_uncertainty_qualification"] == "unavailable"


def test_explicit_capped_estimator_records_statuses(monkeypatch):
    u = synthetic(monkeypatch)
    result = u.unfold(
        method="MLEM",
        max_iterations=1,
        tolerance=1e-10,
        uncertainty_method="monte_carlo",
        uncertainty_estimator="capped",
        n_uncertainty_samples=8,
        uncertainty_seed=42,
    )
    assert np.all(np.isfinite(result.flux_uncertainty))
    assert result.metadata["flux_uncertainty_qualification"] == "capped_diagnostic"
    assert len(result.metadata["flux_uncertainty_draw_status"]) == 8


def test_mixed_invalid_draws_are_not_selected_away(monkeypatch):
    u = synthetic(monkeypatch)
    calls = []

    def fake(*a, **k):
        i = len(calls)
        calls.append(i)
        if i == 2:
            raise ValueError("failed numerical draw")
        return SimpleNamespace(
            flux=[1.0, 2.0],
            converged=(i != 3),
            iterations=1,
            stop_reason="relative_change" if i != 3 else "max_iterations",
            chi_squared=0.0,
        )

    monkeypatch.setattr("fluxforge.workflows.spectrum_unfolding.mlem", fake)
    result = u.unfold(
        method="MLEM",
        uncertainty_method="monte_carlo",
        uncertainty_estimator="capped",
        n_uncertainty_samples=4,
    )
    assert len(calls) == 5
    assert np.all(np.isnan(result.flux_uncertainty))
    assert result.metadata["flux_uncertainty_finite_draws"] == 3
    assert result.metadata["flux_uncertainty_attempted"] == 4


@pytest.mark.parametrize(
    "bad", [np.array([[1.0, 2.0], [2.0, 1.0]]), np.array([[1.0, 0.2], [0.3, 1.0]])]
)
def test_standalone_holdout_rejects_invalid_covariance(bad):
    with pytest.raises(ValueError):
        predict_holdouts(
            np.zeros((2, 1)),
            np.array([0.0, 1.0]),
            np.zeros(1),
            np.zeros((1, 1)),
            bad,
            [1],
        )


def test_noiseless_conditional_holdout_is_supported():
    result = predict_holdouts(
        np.ones((2, 1)), np.ones(2), np.zeros(1), np.ones((1, 1)), np.zeros((2, 2)), [1]
    )
    np.testing.assert_array_equal(result.holdout_covariance, [[0.0]])
    assert result.holdout_standardized_chi2 == 0.0


def test_physical_gls_receipt_defines_total_error_statistic():
    from tests.test_physical_gls import fixture_inputs

    args, _ = fixture_inputs()
    args["response_error_covariance"] = args["observation_covariance"] * 3
    args["sources"]["response_error_covariance"] = SourceBinding(
        "synthetic://response_error", "b" * 64, "(reactions/target_atom/s)^2"
    )
    result = unfold_gls_physical(**args)
    residual = result.fit_residuals
    expected = residual @ np.linalg.solve(
        args["observation_covariance"] + args["response_error_covariance"], residual
    )
    assert result.receipt()["postfit_total_error_chi2"] == pytest.approx(expected)
    assert "response" in result.receipt()["postfit_chi2_definition"]


def test_activity_coverage_cannot_be_double_counted():
    activity = UncertaintyComponent(
        "activity", 0.1, source="itemized report", covers=("detector_efficiency",)
    )
    with pytest.raises(ValueError, match="overlap"):
        RateUncertaintyBudget(
            "a",
            1.0,
            [
                activity,
                UncertaintyComponent("detector_efficiency", 0.03, source="cert"),
            ],
        )


def test_strict_complete_budget_and_signed_sensitivity():
    with pytest.raises(ValueError, match="incomplete"):
        rate_covariance(
            [RateUncertaintyBudget("a", 1.0, [UncertaintyComponent("activity", 0.1)])],
            require_complete=True,
        )
    a = UncertaintyComponent.from_input(
        "irradiation_history", 0.2, -3.0, source="clock cert", correlation_group="clock"
    )
    b = UncertaintyComponent.from_input(
        "irradiation_history", 0.2, 3.0, source="clock cert", correlation_group="clock"
    )
    assert rate_covariance(
        [RateUncertaintyBudget("a", 2.0, [a]), RateUncertaintyBudget("b", 3.0, [b])]
    )[0, 1] == pytest.approx(-2.16)


def test_correlated_replicates_keep_shared_uncertainty():
    u = SpectrumUnfolder.__new__(SpectrumUnfolder)
    c = np.array([[0.05, 0.04], [0.04, 0.05]])
    result = u._aggregate_duplicate_reaction_rows(
        np.ones((2, 1)),
        ["r", "r"],
        np.array([1.0, 1.0]),
        np.sqrt(np.diag(c)),
        measurement_covariance=c,
    )
    assert result["measurement_covariance"][0, 0] == pytest.approx(0.045)
    assert result["uncertainties"][0] == pytest.approx(np.sqrt(0.045))


@pytest.mark.parametrize("change", ["archive", "missing", "pin", "invalid_cover"])
def test_warm_cache_rechecks_source_and_constructor(mini_db, change):
    u = covered(mini_db)
    u._build_response_matrix()
    if change == "archive":
        mini_db.archive_path.write_text(mini_db.archive_path.read_text() + "\n")
    elif change == "missing":
        mini_db.archive_path.unlink()
    elif change == "pin":
        mini_db.expected_archive_sha256 = "0" * 64
    else:
        object.__setattr__(
            u.measurements[0].response_spec.cover, "density_g_cm3", float("nan")
        )
    with pytest.raises(ValueError):
        u._build_response_matrix()


def test_grid_length_change_rebuilds_shape(mini_db):
    u = covered(mini_db)
    u._build_response_matrix()
    u.energy_edges = np.delete(EDGES, 2)
    assert u._build_response_matrix()[0].shape == (1, len(EDGES) - 2)
    assert u.n_groups == len(EDGES) - 2


def test_converged_mc_records_every_draw(monkeypatch):
    u = synthetic(monkeypatch)
    r = u.unfold(
        method="MLEM",
        max_iterations=2000,
        tolerance=1e-10,
        uncertainty_method="monte_carlo",
        n_uncertainty_samples=64,
        uncertainty_seed=42,
    )
    assert r.metadata["flux_uncertainty_converged_draws"] == 64
    assert r.metadata["flux_uncertainty_usable"] == 64
    assert r.metadata["flux_uncertainty_qualification"] == "converged_conditional"
    np.testing.assert_allclose(r.flux_uncertainty, [0.12564006, 0.18362894], rtol=2e-5)


def test_correlated_mc_preserves_off_diagonal(monkeypatch):
    u = synthetic(monkeypatch)
    c = np.outer([0.14, 0.23], [0.14, 0.23])

    def fake(response, measurements, *args, **kwargs):
        return SimpleNamespace(
            flux=np.array(measurements),
            converged=True,
            iterations=1,
            stop_reason="relative_change",
            chi_squared=0.0,
        )

    monkeypatch.setattr("fluxforge.workflows.spectrum_unfolding.mlem", fake)
    r = u.unfold(
        method="MLEM",
        uncertainty_method="monte_carlo",
        n_uncertainty_samples=2000,
        uncertainty_seed=123,
        rate_covariance=c,
    )
    actual = np.array(r.metadata["flux_uncertainty_covariance"])
    np.testing.assert_allclose(actual / c, np.ones((2, 2)), rtol=0.08)
    assert np.linalg.det(actual) == pytest.approx(0.0, abs=1e-16)


def test_incomplete_budget_gate_rejects_caller_covariance(monkeypatch):
    u = synthetic(monkeypatch)
    with pytest.raises(ValueError, match="complete source/component"):
        u.unfold(
            method="MLEM",
            require_complete_rate_budget=True,
            rate_covariance=np.diag([0.14**2, 0.23**2]),
        )


def test_complete_budget_binds_rate_and_identity(monkeypatch):
    u = synthetic(monkeypatch)
    for m in u.measurements:
        m.rate_uncertainty_budget = RateUncertaintyBudget(
            m.sample_id,
            m.rate_per_atom,
            [
                UncertaintyComponent(
                    name, 0.1 if name == "activity" else 0.0, source="synthetic source"
                )
                for name in REQUIRED_COMPONENTS
            ],
        )
    r = u.unfold(method="MLEM", require_complete_rate_budget=True, max_iterations=1)
    assert (
        "scientific admission separate" in r.metadata["rate_covariance_qualification"]
    )
    u.measurements[0].rate_uncertainty_budget.row_id = "wrong observation"
    with pytest.raises(ValueError, match="current observation"):
        u.unfold(method="MLEM")


def test_complete_coverage_requires_sources_and_revalidates():
    b = RateUncertaintyBudget(
        "a",
        1.0,
        [UncertaintyComponent(n, 0.01, source="source") for n in REQUIRED_COMPONENTS],
    )
    assert b.complete
    b.components[0] = replace(b.components[0], source="")
    assert not b.complete
    with pytest.raises(ValueError, match="incomplete"):
        rate_covariance([b], require_complete=True)
    object.__setattr__(b.components[0], "relative", float("nan"))
    with pytest.raises(ValueError, match="finite"):
        rate_covariance([b])


def test_fully_shared_replicates_do_not_shrink_and_keep_cross_terms():
    u = SpectrumUnfolder.__new__(SpectrumUnfolder)
    c = np.array([[0.04, 0.04, 0.02], [0.04, 0.04, 0.02], [0.02, 0.02, 0.09]])
    r = u._aggregate_duplicate_reaction_rows(
        np.array([[1.0], [1.0], [2.0]]),
        ["r", "r", "s"],
        np.ones(3),
        np.sqrt(np.diag(c)),
        measurement_covariance=c,
    )
    np.testing.assert_allclose(
        r["measurement_covariance"], [[0.04, 0.02], [0.02, 0.09]]
    )


@pytest.mark.parametrize("scale", [1e-40, 1.0, 1e40])
def test_covariance_validation_is_scale_aware(scale):
    good = np.array([[2.0, 0.3], [0.3, 1.0]]) * scale
    r = predict_holdouts(
        np.zeros((2, 1)), np.zeros(2), np.zeros(1), np.zeros((1, 1)), good, [1]
    )
    assert r.holdout_covariance[0, 0] == pytest.approx(0.955 * scale, rel=1e-12, abs=0)
    with pytest.raises(ValueError, match="semidefinite"):
        predict_holdouts(
            np.zeros((2, 1)),
            np.zeros(2),
            np.zeros(1),
            np.zeros((1, 1)),
            np.array([[1.0, 2.0], [2.0, 1.0]]) * scale,
            [1],
        )


@pytest.mark.parametrize("scope", ["global", "observation"])
def test_rafm_itemized_activity_replaces_opaque_term(scope):
    from fluxforge.examples.rafm_workflow import (
        RAFMMetadata,
        TimingInfo,
        build_flux_wire_reactions,
    )

    sample = "Co-Cd-RAFM-1_25cm"
    spec = {
        "activity": {
            "relative": 0.02,
            "source": "itemized synthetic report",
            "covers": ["detector_efficiency", "gamma_yield"],
        }
    }
    config = (
        {"rate_uncertainty_components": spec}
        if scope == "global"
        else {"rate_uncertainty_budgets": {sample + "|Co-59(n,g)Co-60": spec}}
    )
    metadata = RAFMMetadata(
        config=config,
        sample_schedule={},
        sample_schedules={},
        flux_wire_metadata={"co-cd-rafm-1": [{"mass_mg": 10.0}]},
        pairing_aliases={},
        sample_gamma_library={},
    )
    timing = TimingInfo(
        "flux_wires", False, "synthetic", None, 7200.0, 1000.0, None, None, "synthetic"
    )
    reaction = build_flux_wire_reactions(
        sample,
        "co-cd-rafm-1",
        {"Co60": {"activity_eoi_bq": 500.0, "activity_eoi_unc_bq": 100.0}},
        timing,
        metadata,
    )[0]
    b = reaction.uncertainty_budget
    assert [c.name for c in b.components].count("activity") == 1
    assert next(c for c in b.components if c.name == "activity").relative == 0.02
    assert "detector_efficiency" not in b.missing
    assert b.total_relative == pytest.approx(0.25)
    assert not b.complete


def test_main_nonconvergence_cannot_qualify_mc(monkeypatch):
    u = synthetic(monkeypatch)
    calls = []

    def fake(*args, **kwargs):
        calls.append(1)
        return SimpleNamespace(
            flux=[1.0, 2.0],
            converged=len(calls) > 1,
            iterations=1,
            stop_reason="relative_change" if len(calls) > 1 else "max_iterations",
            chi_squared=0.0,
        )

    monkeypatch.setattr("fluxforge.workflows.spectrum_unfolding.mlem", fake)
    r = u.unfold(
        method="MLEM", uncertainty_method="monte_carlo", n_uncertainty_samples=4
    )
    assert r.metadata["flux_uncertainty_converged_draws"] == 4
    assert not r.metadata["flux_uncertainty_main_converged"]
    assert np.all(np.isnan(r.flux_uncertainty))


def test_public_result_mutation_cannot_poison_cached_operator(mini_db):
    u = covered(mini_db)
    original = u._build_response_matrix()[0].copy()
    r = u.unfold(method="MLEM", max_iterations=1)
    r.response_matrix[:] = 0
    r.reactions_used[:] = ["wrong"]
    r.energy_edges[:] = 0
    r.metadata["response_rows"][0]["reaction"] = "wrong"
    current, names, _ = u._build_response_matrix()
    np.testing.assert_array_equal(current, original)
    assert names == ["Co-59(n,g)Co-60"]
    assert u._response_row_metadata[0]["reaction"] == names[0]
